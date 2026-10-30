"""Live MCAT controller. No trajectory logs, frame archives or plotting."""
from concurrent.futures import ThreadPoolExecutor
import time

import cv2
import numpy as np
import torch

from awopt_final import AWOptConfig, AdaptiveWristOptimizer
from mcat.config import DEFAULT_CONFIG, load_config, require_assets, validate_max_frames
from mcat.geometry import wrist_to_command
from mcat.hand_observations import AsyncHandObservation
from utils.camera import pixel2cam, cam2pixel, bbox_transform, get_bbox_from_kps
from utils.hand_open import HandStateManager


class HPE_3D:
    def __init__(self, config_path=DEFAULT_CONFIG, max_frames=None, headless=False, dry_run=False):
        self.max_frames = validate_max_frames(max_frames)
        self.config = load_config(config_path)
        require_assets(self.config)
        self.headless, self.dry_run = headless, dry_run
        self.capture = self.left_hand_recognizer = self.right_hand_recognizer = None
        self.ros = None
        self.closed = False
        try:
            from mcat.pose import AITEPose
            from ultralytics import YOLO
            import mediapipe as mp
            self.mp = mp
            self.model = AITEPose(self.config)
            self.detector_model = YOLO(str(self.config['weights']['detector']), task='detect')
            self.capture = cv2.VideoCapture(self.config['camera']['source'])
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.capture.isOpened():
                raise RuntimeError('Cannot open interaction camera')
            self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            for side in ('left', 'right'):
                options = mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(self.config['weights']['hands'])),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    min_hand_presence_confidence=self.config['hands']['presence_confidence'])
                setattr(self, side + '_hand_recognizer', mp.tasks.vision.HandLandmarker.create_from_options(options))
            if not dry_run:
                import rospy
                from piper_msgs.msg import PosCmd
                self.ros, self.command_type = rospy, PosCmd
                if not rospy.core.is_initialized():
                    rospy.init_node(self.config['ros']['node'], anonymous=True)
                self.publishers = {side: rospy.Publisher(self.config['ros'][side + '_command'],
                                                       PosCmd, queue_size=1) for side in ('left', 'right')}
            self.grippers = {side: HandStateManager(hand=side[0], **self.config['gripper'])
                             for side in ('left', 'right')}
            self.optimizers = {side: AdaptiveWristOptimizer(AWOptConfig(**self.config['awopt']))
                               for side in ('left', 'right')}
            self.world_hands = dict(left=None, right=None)
            self.hand_times = dict(left=-1, right=-1)
            self.hand_epoch_ms = self.timestamp_ms = -1
            self.anchor = self.previous_bbox = self.origin = None
            self.stable_count = 0
        except BaseException:
            self.close()
            raise

    def close(self):
        if getattr(self, 'closed', False):
            return
        self.closed = True
        error = None
        for name in ('left_hand_recognizer', 'right_hand_recognizer', 'capture'):
            obj = getattr(self, name, None)
            if obj is not None:
                setattr(self, name, None)
                try:
                    (obj.release if name == 'capture' else obj.close)()
                except Exception as exc:
                    error = error or exc
        if not getattr(self, 'headless', True):
            cv2.destroyAllWindows()
        if error:
            raise error

    def detect_current_hand(self, frame_rgb, center, recognizer, timestamp_ms):
        height, width = frame_rgb.shape[:2]
        cx, cy = float(center[0]), float(center[1])
        if not np.isfinite([cx, cy]).all():
            return None, None, None, None
        crop = (max(0, min(width, int(cx - 50))), max(0, min(height, int(cy - 50))),
                max(0, min(width, int(cx + 50))), max(0, min(height, int(cy + 50))))
        x1, y1, x2, y2 = crop
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None, None, None, None
        hand_rgb = np.ascontiguousarray(frame_rgb[y1:y2, x1:x2], dtype=np.uint8)
        result = recognizer.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=hand_rgb), timestamp_ms)
        if not result.hand_landmarks:
            return None, None, None, None
        landmarks = result.hand_landmarks[0]
        world = result.hand_world_landmarks[0] if result.hand_world_landmarks else None
        uv = np.array([x1 + landmarks[0].x*(x2-x1), y1 + landmarks[0].y*(y2-y1)])
        if not np.isfinite(uv).all() or not (0 <= uv[0] < width and 0 <= uv[1] < height):
            return None, None, None, None
        return uv, landmarks, world, crop

    def consume_hand_result(self, result, side, now_ms):
        if result is None:
            return
        uv, landmarks, world, crop, source_ms, source_pixel, elapsed = result
        if source_ms < self.hand_epoch_ms or not 0 <= now_ms-source_ms <= self.config['hands']['freshness_ms']:
            return
        if landmarks is None or crop is None:
            return
        self.world_hands[side], self.hand_times[side] = world, source_ms

    def reset_tracking(self, timestamp_ms):
        self.model.reset_smoother()
        for opt in self.optimizers.values():
            opt.reset()
        self.hand_epoch_ms = timestamp_ms
        self.world_hands = dict(left=None, right=None)
        self.hand_times = dict(left=-1, right=-1)
        self.anchor = self.previous_bbox = self.origin = None
        self.stable_count = 0
        # Hold the gripper state across missing observations.

    def infer(self, stop_event=None):
        try:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix='mcat_hand') as pool:
                self.hand_workers = {side: AsyncHandObservation(
                    pool, self.detect_current_hand, getattr(self, side + '_hand_recognizer'))
                    for side in ('left', 'right')}
                return self._infer_frames(stop_event)
        finally:
            # The pool has joined its tasks before the recognizers are closed.
            self.close()

    def _infer_frames(self, stop_event):
        count = 0
        cfg = self.config
        focal, principal = cfg['camera']['focal'], cfg['camera']['principal']
        ratio = self.model.input_h / self.model.input_w
        while (self.ros is None or not self.ros.is_shutdown()) and (stop_event is None or not stop_event.is_set()):
            if not self.headless and cv2.waitKey(1) & 0xff == ord('q'):
                return 'operator_stop'
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError('Interaction camera read failed')
            self.timestamp_ms = max(int(time.monotonic()*1000), self.timestamp_ms + 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not self.headless:
                cv2.imshow('MCAT camera (q: stop)', frame)
            if self.anchor is None:
                result = self.detector_model.predict(frame, save=False, imgsz=cfg['detection']['image_size'],
                    classes=0, conf=cfg['detection']['confidence'], verbose=False,
                    device=str(self.model.device))[0]
                if result.boxes.xyxy.numel() == 0:
                    self.reset_tracking(self.timestamp_ms)
                    continue
                bbox = result.boxes.xywh[0]
            else:
                bbox = self.anchor
            bbox, _, _ = bbox_transform(bbox.clone(), ratio, bbox[2], bbox[3], self.width, self.height)
            coords, _ = self.model.predict(rgb, bbox)
            camera = coords.clone()
            camera[:, 2] -= (camera[:, 1]-camera[0, 1]) * cfg['camera']['depth_tilt']
            camera[:, 2] = camera[:, 2]*cfg['camera']['depth_scale_mm'] + cfg['camera']['root_depth_mm']
            camera = pixel2cam(camera, focal, principal)
            if not torch.isfinite(camera).all() or torch.any(camera[:17, 2] <= 50):
                continue
            root = camera[0].clone()
            raw = (camera-root)[:17].clone()
            pixels = cam2pixel(raw+root, focal, principal)
            joints = self.model.smooth(raw)
            if not torch.isfinite(joints).all():
                self.reset_tracking(self.timestamp_ms)
                continue
            if self.anchor is None:
                if self.previous_bbox is not None and (bbox-self.previous_bbox).abs().mean() < cfg['detection']['anchor_threshold_px']:
                    self.stable_count += 1
                self.previous_bbox = bbox
                if self.stable_count > cfg['detection']['anchor_stable_frames']:
                    self.anchor = get_bbox_from_kps(pixels, ratio).to(self.model.device)
            if self.anchor is not None:
                if self.origin is None:
                    self.origin = joints[0].detach().cpu().numpy().copy()
                pixels_np, joints_np = pixels.detach().cpu().numpy(), joints.detach().cpu().numpy()
                # MediaPipe observes image ROIs; it never changes AWopt inputs.
                for side, index in (('right', 16), ('left', 13)):
                    result = self.hand_workers[side].poll(rgb, pixels_np[index], self.timestamp_ms)
                    self.consume_hand_result(result, side, self.timestamp_ms)
                try:
                    wrists = {side: self.optimizers[side].update(*(joints_np[j] for j in indices))[0]
                              for side, indices in (('right', (14, 15, 16)), ('left', (11, 12, 13)))}
                except ValueError as exc:
                    if self.ros is not None:
                        self.ros.logwarn_throttle(2., 'AWopt input invalid: %s' % exc)
                    continue
                for side in ('right', 'left'):
                    fresh = (self.world_hands[side] is not None and
                             0 <= self.timestamp_ms-self.hand_times[side] <= cfg['hands']['gripper_hold_ms'])
                    gripper = self.grippers[side].update(self.world_hands[side] if fresh else None)
                    values = wrist_to_command(wrists[side], self.origin, side, cfg['mapping'])
                    if not self.dry_run:
                        message = self.command_type()
                        for key, value in values.items():
                            setattr(message, key, value)
                        message.gripper = gripper
                        self.publishers[side].publish(message)
            count += 1
            if self.max_frames is not None and count >= self.max_frames:
                return 'frame_limit'
        return 'operator_stop'
