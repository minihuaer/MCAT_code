#!/usr/bin/env python3
"""MCAT: KDU_final2 + AWopt live dual-arm teleoperation."""
import argparse
from mcat.config import DEFAULT_CONFIG, load_config, require_assets
from mcat.teleop import HPE_3D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG), help='Runtime YAML')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--headless', action='store_true', help='Disable the live camera window')
    parser.add_argument('--dry-run', action='store_true', help='Run perception without ROS imports or publishing')
    parser.add_argument('--check', action='store_true', help='Check configuration/weight files without hardware')
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        require_assets(config)
    except FileNotFoundError as exc:
        parser.exit(2, str(exc)+'\n')
    if args.check:
        print('Configuration and required assets present. Hardware/dependencies are not checked.')
        return
    hpe = HPE_3D(args.config, max_frames=args.max_frames, headless=args.headless, dry_run=args.dry_run)
    try:
        hpe.infer()
    except KeyboardInterrupt:
        pass
    finally:
        hpe.close()


if __name__ == '__main__':
    main()
