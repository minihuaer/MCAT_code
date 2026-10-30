"""Hardware-free checks for synchronization and feedback semantics."""
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import sys
import threading
import tempfile
import time
from types import SimpleNamespace
import numpy as np
from collect import HPEController, CaptureStopped, check_sync, rpy, vectors, wait_ready, record_with_hpe, wait_for_enter, record

class CaptureChecks(unittest.TestCase):

    def setUp(self):
        self.config = json.loads((Path(__file__).parent / 'config.json').read_text())
        self.messages = {}
        for arm in ('left', 'right'):
            for kind in ('pose', 'joints', 'command', 'status'):
                self.messages[arm + '_' + kind] = {'monotonic_s': 10.0, 'message': {}}
            self.messages[arm + '_pose']['message'] = {'position': dict(x=0.1, y=0.2, z=0.3), 'orientation': dict(x=0, y=0, z=0, w=1)}
            self.messages[arm + '_joints']['message'] = {'name': list(reversed(self.config['joint_names'])), 'position': list(reversed([1, 2, 3, 4, 5, 6, 0.04]))}
            self.messages[arm + '_command']['message'] = dict(x=0.4, y=0.5, z=0.6, roll=0, pitch=1.57, yaw=0, gripper=0.08)

    def test_loads_clean_mcat_entry_with_runtime_config(self):
        factory = Mock(return_value=Mock())
        module = SimpleNamespace(HPE_3D=factory)
        settings = dict(config=Path("teleop.yaml"), max_frames=12, headless=True)
        with patch("collect.importlib.import_module", return_value=module) as loader:
            hpe = HPEController(settings)
            hpe.load()
        loader.assert_called_once_with("hci2_KDU_AWopt_final")
        factory.assert_called_once_with(config_path=Path("teleop.yaml"), max_frames=12, headless=True)
        hpe.close()
        factory.return_value.close.assert_called_once()

    def test_reorders_joints_and_uses_feedback(self):
        row = vectors(self.messages, self.config)
        self.assertEqual(row['observation.joint_position'][:7], [1, 2, 3, 4, 5, 6, 0.04])
        self.assertEqual(row['observation.state'][:7], [0.1, 0.2, 0.3, 0, 0, 0, 0.04])
        self.assertEqual(row['action'][6], 0.08)

    def test_quaternion(self):
        result = rpy(dict(x=0, y=0, z=math.sqrt(0.5), w=math.sqrt(0.5)))
        self.assertAlmostEqual(result[2], math.pi / 2)
        with self.assertRaises(ValueError):
            rpy(dict(x=0, y=0, z=0, w=0))

    def test_nonfinite_rejected(self):
        self.messages['left_command']['message']['x'] = float('nan')
        with self.assertRaises(ValueError):
            vectors(self.messages, self.config)

    def test_missing_stale_and_skew(self):
        check_sync(self.messages, 10.0, 10.01, self.config)
        for modification in ('missing', 'stale', 'skew'):
            msgs = copy.deepcopy(self.messages)
            if modification == 'missing':
                del msgs['left_pose']
            else:
                msgs['left_pose']['monotonic_s'] = 9.0 if modification == 'stale' else 9.83
            with self.assertRaises(ValueError):
                check_sync(msgs, 10.0, 10.01, self.config)

    def test_held_command_uses_separate_age_limit(self):
        self.messages['left_command']['monotonic_s'] = 9.6
        check_sync(self.messages, 10.0, 10.01, self.config)
        self.messages['left_command']['monotonic_s'] = 9.4
        with self.assertRaises(ValueError):
            check_sync(self.messages, 10.0, 10.01, self.config)

    def run_wait(self, schedule, timeout=0, stable=1, include_commands=True, health=None):
        clock = [0.0]
        camera = Mock()
        camera.get.side_effect = lambda: (0, clock[0], 0, None)
        messages = Mock()

        def snapshot():
            result = copy.deepcopy(self.messages)
            for event in result.values():
                event['monotonic_s'] = clock[0]
            schedule(result, clock[0])
            return result
        messages.snapshot.side_effect = snapshot
        with patch('collect.time.monotonic', side_effect=lambda: clock[0]), patch('collect.time.sleep', side_effect=lambda dt: clock.__setitem__(0, clock[0] + dt)), patch('builtins.print'):
            wait_ready(camera, messages, self.config, timeout, stable, health or Mock(), lambda: False, include_commands=include_commands)
        return clock[0]

    def test_hardware_preflight_does_not_require_commands(self):

        def no_commands(result, now):
            del result['left_command'], result['right_command']
        self.assertEqual(self.run_wait(no_commands, timeout=20, stable=0, include_commands=False), 0)

    def test_delayed_both_arm_commands_wait_beyond_old_timeout(self):

        def delayed(result, now):
            if now < 25:
                del result['left_command']
            if now < 30:
                del result['right_command']
        self.assertGreaterEqual(self.run_wait(delayed), 31)

    def test_stability_resets_on_output_gap(self):

        def interrupted(result, now):
            if 0.5 <= now < 2:
                result['left_command']['monotonic_s'] = -10.0
        self.assertGreaterEqual(self.run_wait(interrupted), 3)

    def test_configured_command_timeout(self):

        def missing(result, now):
            del result['left_command']
        with self.assertRaisesRegex(RuntimeError, 'Readiness timeout'):
            self.run_wait(missing, timeout=2)

    def test_hpe_exit_interrupts_wait(self):
        with self.assertRaisesRegex(RuntimeError, 'HPE exited'):
            self.run_wait(lambda result, now: None, health=Mock(side_effect=RuntimeError('HPE exited')))

    def test_enter_wait_observes_stop_without_input(self):
        with patch('collect.select.select', return_value=([], [], [])), patch('builtins.print'):
            health = Mock(side_effect=[None, CaptureStopped('q')])
            with self.assertRaises(CaptureStopped):
                wait_for_enter(health, lambda: False)

    def run_session(self, infer, recorder):
        hpe = HPEController({})
        hpe.instance = Mock()
        hpe.instance.infer.side_effect = infer
        hpe.load = Mock()
        ros = Mock()
        ros.core.is_initialized.return_value = True
        with patch.dict(sys.modules, {'rospy': ros}), patch('collect.HPEController', return_value=hpe), patch('collect.record', side_effect=recorder):
            try:
                record_with_hpe(SimpleNamespace(), {})
            finally:
                self.assertTrue(hpe.stop_event.is_set())
                hpe.instance.close.assert_called_once()

    def test_recorder_completion_stops_main_thread_inference(self):
        main_thread = threading.current_thread()

        def infer(stop_event):
            self.assertIs(threading.current_thread(), main_thread)
            self.assertTrue(stop_event.wait(1))

        def recorder(args, config, hpe):
            self.assertIsNot(threading.current_thread(), main_thread)
        self.run_session(infer, recorder)

    def test_frame_limit_stops_recorder_normally(self):

        def recorder(args, config, hpe):
            self.assertTrue(hpe.stop_event.wait(1))
            self.assertEqual(hpe.stop_reason, 'frame_limit')
            hpe.check()
        self.run_session(lambda **kwargs: 'frame_limit', recorder)

    def test_frame_limit_validation(self):
        from mcat.config import validate_max_frames
        for value in (None, 1, 3000):
            self.assertEqual(validate_max_frames(value), value)
        for value in (0, -1, True, False, 1.5, '10'):
            with self.assertRaises(ValueError):
                validate_max_frames(value)

    def test_q_stops_waiting_recorder(self):

        def recorder(args, config, hpe):
            self.assertTrue(hpe.stop_event.wait(1))
            hpe.check()
        self.run_session(lambda **kwargs: 'operator_stop', recorder)

    def test_inference_error_propagates_to_recorder_and_caller(self):

        def recorder(args, config, hpe):
            self.assertTrue(hpe.stop_event.wait(1))
            hpe.check()
        with self.assertRaisesRegex(ValueError, 'inference failed'):
            self.run_session(Mock(side_effect=ValueError('inference failed')), recorder)

    def test_recorder_failure_propagates_to_caller(self):

        def infer(stop_event):
            self.assertTrue(stop_event.wait(1))
        with self.assertRaisesRegex(OSError, 'disk full'):
            self.run_session(infer, Mock(side_effect=OSError('disk full')))

    def record_two_frames_then_stop(self, failure, stop_reason='operator_stop'):
        config = self.config
        messages = Mock()
        messages.lock = threading.Lock()
        messages.error = None

        def snapshot():
            result = copy.deepcopy(self.messages)
            for event in result.values():
                event['monotonic_s'] = time.monotonic()
            return result
        messages.snapshot.side_effect = snapshot
        camera = Mock()
        seq = [0]

        def image():
            seq[0] += 1
            return (seq[0], time.monotonic(), time.time(), np.zeros((config['height'], config['width'], 3), dtype=np.uint8))
        camera.get.side_effect = image
        hpe = Mock(stop_reason=stop_reason)
        hpe.check.side_effect = [None, None, failure]
        ros = Mock()
        ros.is_shutdown.return_value = False
        ros.core.is_initialized.return_value = True
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {'rospy': ros, 'roslib': Mock(), 'roslib.message': Mock()}), patch('collect.Camera', return_value=camera), patch('collect.Messages', return_value=messages), patch('collect.wait_ready'), patch('builtins.print'):
            args = SimpleNamespace(root=Path(folder), task='test', auto_start=True, episodes=1, seconds=1.0, wait=20.0, command_wait=0.0, ready_seconds=0.0)
            if isinstance(failure, CaptureStopped):
                record(args, config, hpe)
            else:
                with self.assertRaisesRegex(RuntimeError, 'model failed'):
                    record(args, config, hpe)
            meta = json.loads((Path(folder) / 'episode_000000' / 'episode.json').read_text())
            self.assertEqual(meta['frames'], 2)
            camera.close.assert_called_once()
            return meta

    def test_operator_stop_saves_written_prefix(self):
        meta = self.record_two_frames_then_stop(CaptureStopped('q'))
        self.assertTrue(meta['complete'])
        self.assertEqual(meta['stop_reason'], 'operator_stop')

    def test_frame_limit_saves_written_prefix_with_reason(self):
        meta = self.record_two_frames_then_stop(CaptureStopped('limit'), 'frame_limit')
        self.assertTrue(meta['complete'])
        self.assertEqual(meta['frames'], 2)
        self.assertEqual(meta['stop_reason'], 'frame_limit')

    def test_hpe_failure_marks_episode_incomplete(self):
        meta = self.record_two_frames_then_stop(RuntimeError('model failed'))
        self.assertFalse(meta['complete'])
if __name__ == '__main__':
    unittest.main()
