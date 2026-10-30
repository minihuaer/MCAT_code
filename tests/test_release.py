"""Hardware-free release checks using bundled KDU weights."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from inference.kdu_final import load_model
from mcat.config import ROOT, load_config, require_assets
from mcat.geometry import wrist_to_command
from mcat.pose import AITEPose
from mcat.teleop import HPE_3D
from utils.hand_open import HandStateManager


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.kdu = load_model()

    def test_weights_integrity_and_parameters(self):
        manifest = json.loads((ROOT/'weights/manifest.json').read_text())
        for name, info in manifest.items():
            if info['status'] == 'included':
                self.assertEqual(hashlib.sha256((ROOT/'weights'/name).read_bytes()).hexdigest(), info['sha256'])
        self.assertEqual(self.kdu.infer_net.span, 2)
        self.assertEqual(self.kdu.infer_net.phase_compensator.alpha_v, .9)
        self.assertFalse(self.kdu.training)

    def test_real_pose_weights_and_image_inference(self):
        cfg = load_config()
        cfg['device'] = 'cpu'
        pose = AITEPose(cfg)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        coords, scores = pose.predict(image, torch.tensor([120., 40., 520., 440.]))
        self.assertEqual(tuple(coords.shape), (18, 3))
        self.assertEqual(tuple(scores.shape), (1, 18, 1))
        self.assertTrue(torch.isfinite(coords).all())
        self.assertTrue(torch.isfinite(scores).all())

    def test_streaming_units_feedback_reset(self):
        pose = AITEPose.__new__(AITEPose)
        pose.device, pose.window_size_s, pose.smoother = torch.device('cpu'), 27, self.kdu
        pose.pred_seq_s, pose.window_seq_s = torch.zeros(1, 27, 51), torch.zeros(1, 26, 51)
        pose.reset_smoother()
        frames = torch.randn(40, 17, 3, generator=torch.Generator().manual_seed(6))*200
        original = frames.clone()
        raw = frames.reshape(1, -1, 51)/1000
        history = raw[:, :26].clone()
        with torch.inference_mode():
            for t, frame in enumerate(frames):
                output = pose.smooth(frame)
                if t < 26:
                    self.assertIs(output, frame)
                else:
                    expected = self.kdu(raw[:, t-26:t+1], history)
                    torch.testing.assert_close(output, expected.reshape(17, 3)*1000, rtol=0, atol=0)
                    history = torch.cat((history[:, 1:], expected[:, None]), 1)
                    torch.testing.assert_close(pose.window_seq_s, history, rtol=0, atol=0)
        torch.testing.assert_close(frames, original)
        pose.reset_smoother()
        self.assertEqual(pose._smooth_frames, 0)
        self.assertEqual(torch.count_nonzero(pose.window_seq_s), 0)

    def test_gripper_transition_and_missing_hand_hold(self):
        for hand, threshold in (('l', 43), ('r', 48)):
            manager = HandStateManager(hand=hand, confidence_threshold=.83, frame_threshold=2, open=.1, close=.04)
            def landmarks(distance):
                result = [SimpleNamespace(x=0., y=0., z=0.) for _ in range(21)]
                for j in (7, 11, 15): result[j].x = distance/1000
                return result
            with patch('builtins.print'):
                self.assertEqual(manager.update(landmarks(threshold+5)), 0.)
                self.assertEqual(manager.update(landmarks(threshold+5)), .1)
                self.assertEqual(manager.update(None), .1)
                self.assertEqual(manager.update(landmarks(threshold-5)), .1)
                self.assertEqual(manager.update(landmarks(threshold-5)), .04)
                self.assertEqual(manager.update(None), .04)

    def test_mapping_limits_and_paper_orientation(self):
        cfg = load_config()['mapping']
        command = wrist_to_command([-250, -275, -250], [0, 0, 0], 'right', cfg)
        self.assertAlmostEqual(command['x'], .25)
        self.assertAlmostEqual(command['y'], 0.)
        self.assertAlmostEqual(command['z'], .25)
        radial_pitch = np.pi*(.58-.25)/(4*(.58-.12)) + 3*np.pi/4
        self.assertAlmostEqual(command['pitch'], radial_pitch + (.25-.12)/(.3-.12)*(np.pi/2-radial_pitch))
        command = wrist_to_command([1e6, -1e6, -1e6], [0, 0, 0], 'left', cfg)
        self.assertEqual([command[k] for k in ('x','y','z')], [.5,.4,.7])
        self.assertEqual(command['pitch'], np.pi/2)

    def test_missing_assets_reported_before_hardware(self):
        cfg = load_config()
        cfg['weights']['pose'] = ROOT/'weights/definitely_missing.pth'
        with self.assertRaisesRegex(FileNotFoundError, 'definitely_missing'):
            require_assets(cfg)

    def test_live_loop_publishes_both_arms_without_debug_history(self):
        hpe = HPE_3D.__new__(HPE_3D)
        hpe.config = load_config()
        hpe.config['detection']['anchor_stable_frames'] = 0
        hpe.headless, hpe.dry_run, hpe.max_frames = True, False, 3
        hpe.ros = SimpleNamespace(is_shutdown=lambda: False)
        hpe.capture = Mock()
        hpe.capture.read.return_value = True, np.zeros((480,640,3),dtype=np.uint8)
        hpe.width, hpe.height = 640,480
        coords = torch.zeros(18,3)
        coords[:,0],coords[:,1] = 320.,240.
        coords[10,1] = 120.
        coords[13] = torch.tensor([220.,200.,.05])
        coords[16] = torch.tensor([420.,200.,.05])
        hpe.model = SimpleNamespace(input_h=384,input_w=288,device=torch.device('cpu'),
            predict=Mock(return_value=(coords,None)), smooth=Mock(side_effect=lambda x: x))
        boxes = SimpleNamespace(xyxy=torch.ones(1,4),xywh=torch.tensor([[320.,240.,300.,400.]]))
        hpe.detector_model = SimpleNamespace(predict=Mock(return_value=[SimpleNamespace(boxes=boxes)]))
        hpe.timestamp_ms = hpe.hand_epoch_ms = -1
        hpe.anchor = hpe.previous_bbox = hpe.origin = None
        hpe.stable_count = 0
        hpe.world_hands = dict(left=None,right=None)
        hpe.hand_times = dict(left=-1,right=-1)
        hpe.hand_workers = {s: SimpleNamespace(poll=lambda *args: None) for s in ('left','right')}
        from awopt_final import AdaptiveWristOptimizer
        hpe.optimizers = {s: AdaptiveWristOptimizer() for s in ('left','right')}
        hpe.grippers = {s: HandStateManager(hand=s[0],**hpe.config['gripper']) for s in ('left','right')}
        hpe.command_type = SimpleNamespace
        hpe.publishers = {s: Mock() for s in ('left','right')}
        self.assertEqual(hpe._infer_frames(None),'frame_limit')
        self.assertEqual(hpe.model.smooth.call_count,3)
        for publisher in hpe.publishers.values():
            self.assertEqual(publisher.publish.call_count,2)
            message = publisher.publish.call_args.args[0]
            self.assertTrue(all(np.isfinite(getattr(message,k)) for k in ('x','y','z','roll','pitch','yaw','gripper')))
        self.assertFalse(any(name.startswith('templist') or name.endswith('list') for name in vars(hpe)))


if __name__ == '__main__':
    unittest.main()
