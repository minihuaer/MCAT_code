# MCAT

This repository provides the open-source implementation of **MCAT A Monocular Camera-Based Adaptive Teleoperation System for Stable Dual-Arm Robot Demonstrations**.

MCAT uses a single RGB camera to estimate human poses with AITEPose, refine temporal motion with **KDUnet**, and optimize wrist trajectories with the **Adaptive Wrist Optimizer (AWopt)**. The resulting trajectories drive dual-arm Cartesian control, while MediaPipe hand observations provide gripper commands.

The release includes online inference and control, robot demonstration collection with an overhead camera, dataset validation, and offline LeRobot conversion.

[Demonstration video](<MCAT A Monocular Camera-Based Adaptive Teleoperation System for Stable Dual-Arm Robot Demonstrations.mp4>) · [Model weights and demonstration dataset](<https://drive.google.com/drive/folders/1djJRXoZotVpSuHGWg_BF-laWxtBuRFnr?usp=sharing>)

## Downloads

Model weights and the collected demonstration dataset demo are available on [Google Drive](<https://drive.google.com/drive/folders/1djJRXoZotVpSuHGWg_BF-laWxtBuRFnr?usp=sharing>).

Download the model files and place them in `weights/` using the filenames below. Download and extract the demonstration dataset to a local data directory, such as `data/demo/`. Model binaries and demonstration datasets are distributed through Google Drive; the Git repository contains the source code, configurations, and documentation.

| Local model path | Component |
| --- | --- |
| `weights/abest_model_152_80.1034-lite.pth` | AITEPose, ResNet-152 with 18 joints |
| `weights/kdu_final2.pth` | KDUnet pose refinement |
| `weights/hand_landmarker.task` | MediaPipe hand landmark detection |
| `weights/yolov8n.pt` | YOLOv8n person detection |

The default pose checkpoint is `weights/abest_model_152_80.1034-lite.pth`. Asset paths are configured in `configs/teleop.yaml`. See the [weights guide](weights/README.md) for loading and checksum details and the [dataset guide](scripts/hci_dataset/README.md) for the demonstration fields, collection, and conversion commands.

## Repository layout

```text
hci2_KDU_AWopt_final.py        Live teleoperation entry point
configs/teleop.yaml           Camera, weights, device, gripper, AWopt, ROS, and mapping
configs/KDU_final2.yaml       KDUnet inference configuration
mcat/                        Pose inference, control loop, hand detection, and mapping
inference/                   KDUnet and causal temporal attention
awopt_final.py               Adaptive Wrist Optimizer
models/                      AITEPose network definitions
utils/                       Camera geometry, flip augmentation, and gripper control
weights/                     Local model files, download guide, and checksum manifest
scripts/hci_dataset/          Demonstration collection, validation, and conversion
scripts/check_environment.py  Dependency check without hardware access
tests/                       Algorithm and control-pipeline tests
```

Default asset paths resolve relative to the repository root, independently of the working directory.

## Installation

Follow the [environment setup](docs/ENVIRONMENT.md), download the model weights, and run:

```bash
python scripts/check_environment.py
python hci2_KDU_AWopt_final.py --check
```

`--check` validates configuration and required files without connecting to hardware. It lists missing assets and exits with code 2 when files are unavailable.

## Live teleoperation

1. Configure the interaction camera and its calibrated intrinsics in `configs/teleop.yaml`.
2. Start the Piper ROS1 driver and Cartesian IK controller, and source their workspace.
3. Check the command topics for both arms and run:

```bash
python hci2_KDU_AWopt_final.py
# Disable the preview window:
python hci2_KDU_AWopt_final.py --headless
# Run perception without ROS command publication:
python hci2_KDU_AWopt_final.py --dry-run --max-frames 300
```

The default command topics are `/piper_left/pin_pos_cmd` and `/piper_right/pin_pos_cmd`, with message type `piper_msgs/PosCmd`. `--max-frames` counts successfully processed pose frames, including preparation. Frames without a detected person are not counted.

Press `q` in the preview window or Ctrl-C to stop. Shutdown releases the camera and hand recognizers. Robot homing and motor enable/disable are managed by the robot driver.

## Robot demonstration collection

The recorder uses a separate overhead camera and subscribes to robot feedback and final control targets. It starts the MCAT controller in the same process:

```bash
python scripts/hci_dataset/collect.py record \
  --root ./data/raw/fold --task "fold the cloth" --seconds 30 --episodes 1
```

Configure the overhead camera and ROS topics in `scripts/hci_dataset/config.json`. The default `/dev/video2` is an example and must refer to a different physical camera from the interaction camera.

The [Google Drive download](<https://drive.google.com/drive/folders/1djJRXoZotVpSuHGWg_BF-laWxtBuRFnr?usp=sharing>) includes a collected demonstration dataset demo. See [collection and conversion](scripts/hci_dataset/README.md) for data semantics and instructions for recording additional demonstrations.

## Method and tests

The [method guide](docs/METHOD.md) describes KDUnet, AWopt, coordinate conventions, and control. The [testing guide](docs/VALIDATION.md) provides software and dataset checks.

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s scripts/hci_dataset -p 'test_*.py' -v
```

## Acknowledgments

See [component attribution](THIRD_PARTY_NOTICES.md) for third-party dependencies and copyright notices.
