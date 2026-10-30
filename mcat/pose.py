"""AITEPose image inference followed by recursive KDU_final2, without plots."""
import warnings
import cv2
import numpy as np
import torch
from inference.kdu_final import load_model, read_model_config
from models.pose_network import RegressFlow3D_hci_3vm2_best_lite
from utils.transform_hpe import flip_coord


class AITEPose:
    def __init__(self, config):
        device = config['device']
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        pose = config['pose']
        self.joint_num = pose['joints']
        self.input_h, self.input_w = pose['input_height'], pose['input_width']
        self.model = RegressFlow3D_hci_3vm2_best_lite(
            resnet=pose['resnet'], joint_num=self.joint_num).to(self.device)
        state = torch.load(config['weights']['pose'], map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        if state and all(key.startswith('module.') for key in state):
            state = {key[7:]: value for key, value in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError('Incomplete AITEPose checkpoint: ' + ', '.join(missing))
        if unexpected:
            warnings.warn('Unused training-only checkpoint keys: ' + ', '.join(unexpected))
        self.model.eval()
        self.smoother = load_model(config['weights']['kdu'], device=self.device,
                                   config_path=config['kdu_config'])
        self.window_size_s = read_model_config(config['kdu_config'])['MODEL_WINDOW_SIZE']
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
        self.joint_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.pred_seq_s = torch.zeros(1, self.window_size_s, 51, device=self.device)
        self.window_seq_s = torch.zeros(1, self.window_size_s - 1, 51, device=self.device)
        self.reset_smoother()

    def preprocess(self, img, bbox):
        'Crop the image to the bounding box, resize it, and normalize pixels.'
        bbox = bbox.int()
        img = img.copy()
        x1, y1, x2, y2 = bbox.tolist()
        self.pad_left = max(-x1, 0)
        self.pad_top = max(-y1, 0)
        pad_right = max(x2 - img.shape[1], 0)
        pad_bottom = max(y2 - img.shape[0], 0)
        if not (self.pad_left == 0 and self.pad_top == 0 and (pad_right == 0) and (pad_bottom == 0)):
            img = cv2.copyMakeBorder(img, self.pad_top, pad_bottom, self.pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            x1 += self.pad_left
            y1 += self.pad_top
            x2 += self.pad_left
            y2 += self.pad_top
        input_shape = img.shape
        img = img[y1:y2, x1:x2, :]
        self.img = img
        img = np.array([cv2.resize(img, (self.input_w, self.input_h))])
        img = (np.transpose(img, (0, 3, 1, 2)) / 255.0 - self.mean) / self.std
        return torch.from_numpy(img.astype(np.float32)).to(self.device)

    def inference(self, img):
        with torch.no_grad():
            output = self.model(img)
            img_f = torch.flip(img, [3])
            output_f = self.model(img_f)
            flipped_jts, flipped_maxvals = flip_coord((output_f.pred_jts, output_f.maxvals), self.joint_pairs, width_dim=72, shift=True, flatten=False)
        return ((output['pred_jts'] + flipped_jts) / 2, (output['maxvals'] + flipped_maxvals) / 2)

    def post_process(self, coords, bbox):
        target_coords = torch.zeros(self.joint_num, 3, requires_grad=False).to(self.device)
        target_coords[:, 0] = (coords[0, :, 0] + 0.5) * (bbox[2] - bbox[0]) + bbox[0]
        target_coords[:, 1] = (coords[0, :, 1] + 0.5) * (bbox[3] - bbox[1]) + bbox[1]
        target_coords[:, 2] = coords[0, :, 2]
        return target_coords

    def predict(self, img, bbox):
        img_p = self.preprocess(img, bbox)
        coords, scores = self.inference(img_p)
        coords = self.post_process(coords, bbox)
        return (coords, scores)

    def reset_smoother(self):
        """Discard pose history for a new person/sequence; retain model weights."""
        self.pred_seq_s.zero_()
        self.window_seq_s.zero_()
        self._smooth_frames = 0

    def smooth(self, coords):
        """Smooth [17, 3] millimetre poses; pass through the first 26 frames.

            Raw history and smoothed history are separate, chronological windows.
            Each call advances the raw and refined temporal windows by one frame.
            """
        if not isinstance(coords, torch.Tensor) or coords.shape != (17, 3):
            raise ValueError('smooth expects a torch.Tensor with shape [17, 3]')
        with torch.no_grad():
            raw_pose = coords.detach().to(device=self.device, dtype=torch.float32).reshape(1, 51) / 1000
            if self._smooth_frames < self.window_size_s - 1:
                index = self._smooth_frames
                self.pred_seq_s[:, index, :] = raw_pose
                self.window_seq_s[:, index, :] = raw_pose
                self._smooth_frames += 1
                return coords
            self.pred_seq_s[:, -1, :] = raw_pose
            pro_coords = self.smoother(self.pred_seq_s, self.window_seq_s)
            self.pred_seq_s[:, :-1, :] = self.pred_seq_s[:, 1:, :].clone()
            self.window_seq_s[:, :-1, :] = self.window_seq_s[:, 1:, :].clone()
            self.window_seq_s[:, -1, :] = pro_coords
            self._smooth_frames += 1
            return pro_coords.reshape(17, 3) * 1000
