#!/usr/bin/env python3
"""Import-only check; never opens cameras or publishes robot commands."""
import argparse
import importlib
from importlib.metadata import version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ros', action='store_true')
    args = parser.parse_args()
    modules = ['numpy', 'torch', 'torchvision', 'cv2', 'yaml', 'easydict', 'mediapipe', 'ultralytics']
    if args.ros:
        modules += ['rospy', 'roslib.message', 'piper_msgs.msg', 'geometry_msgs.msg', 'sensor_msgs.msg']
    failed = False
    for name in modules:
        try:
            module = importlib.import_module(name)
            if name == 'mediapipe':
                assert module.tasks.vision.HandLandmarkerOptions
            if name == 'piper_msgs.msg':
                assert module.PosCmd and module.PiperStatusMsg
            print('OK', name, getattr(module, '__version__', ''))
        except Exception as exc:
            failed = True
            print('FAIL', name, str(exc))
    if not failed:
        import torch
        print('CUDA available:', torch.cuda.is_available(), '| torch build:', torch.version.cuda)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
