## Live control and collection environment

Use a dedicated environment with Linux x86_64, Python 3.9, PyTorch 2.3.1, torchvision 0.18.1, NumPy 1.26.4, and OpenCV 4.10. The commands below install the dependencies for live control and demonstration collection.

```bash
conda create -n mcat python=3.9.19 pip -y
conda activate mcat
cd MCAT_code
# CUDA 12.1 wheels; a separate local CUDA Toolkit is not required to run them.
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
python scripts/check_environment.py
```

For CPU debugging, replace the PyTorch index with `https://download.pytorch.org/whl/cpu`. The NVIDIA driver must support the selected wheel; check it with `nvidia-smi` and the environment checker. CPU inference is supported, but real-time performance is not guaranteed. The Torch/torchvision pairing follows the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).

Core dependency versions are pinned to avoid NumPy 2 ABI conflicts with binary extensions. MediaPipe 0.10.14 provides the required `tasks.vision.HandLandmarker` API. [MediaPipe release](https://pypi.org/project/mediapipe/0.10.14/)

YOLO and MediaPipe declare dependencies on `opencv-python` and `opencv-contrib-python`, respectively; both are pinned to the same version here. Do not additionally install headless OpenCV or uninstall only one of these distributions, since they share `cv2` files. Recreate the isolated environment if that installation is damaged. Although this repository does not use Matplotlib or Seaborn for plotting, the YOLO package still declares them as dependencies. [YOLO v8.3.0 dependencies](https://github.com/ultralytics/ultralytics/blob/v8.3.0/pyproject.toml)

`--headless` disables only the OpenCV preview. MediaPipe's Linux binaries may still require system graphics libraries. Typical Ubuntu dependencies include `libgl1`, `libegl1`, and `libglib2.0-0`. The current user also needs permission to access the camera device.

## ROS1 / Piper

Robot control requires the existing ROS1 workspace to provide `rospy`, `roslib`, `geometry_msgs`, `sensor_msgs`, and the actual `piper_msgs/PosCmd` and `PiperStatusMsg` definitions used by the driver. Do not substitute unrelated PyPI packages for these generated messages.

For a typical ROS Noetic session:

```bash
conda activate mcat
source /opt/ros/noetic/setup.bash
source "$PIPER_WS/devel/setup.bash"
python scripts/check_environment.py --ros
```

Set `PIPER_WS` to the absolute path of the actual Piper workspace first. Noetic's usual system environment is Ubuntu 20.04/Python 3.8, while this isolated environment uses Python 3.9. The controller uses Python ROS communication/message interfaces and does not import `cv_bridge`. Verify that generated messages load in the selected interpreter. Any additional extensions tied to the system Python must be rebuilt for the matching interpreter. This repository does not install or replace ROS or upgrade the robot driver.

Start the driver and IK using the device's existing deployment procedure. Machine-specific launch orchestration is not included. Required interfaces are:

| Purpose | Default topic | Type |
| --- | --- | --- |
| Command target | `/piper_{left,right}/pin_pos_cmd` | `piper_msgs/PosCmd` |
| End-effector feedback | `/piper_{left,right}/end_pose` | `geometry_msgs/Pose` |
| Joint feedback | `/piper_{left,right}/joint_states_single` | `sensor_msgs/JointState` |
| Status feedback | `/piper_{left,right}/arm_status` | `piper_msgs/PiperStatusMsg` |

The controller publishes the first interface; the recorder subscribes to the feedback interfaces as well. `PosCmd` must contain floating-point x/y/z/roll/pitch/yaw/gripper fields. Additional mode fields retain their message defaults. Topic names must match the actual driver. `--dry-run` does not require ROS but still requires all visual models and a camera.

## Separate conversion environment

LeRobot 0.4.4 requires Python ≥3.10 and different Torch/OpenCV dependencies. Install it in a separate environment, not in `mcat`. [Official dependency specification](https://github.com/huggingface/lerobot/blob/v0.4.4/pyproject.toml)

```bash
conda create -n mcat-convert python=3.10 pip -y
conda activate mcat-convert
python -m pip install -r requirements-convert.txt
python -c 'from lerobot.datasets.lerobot_dataset import LeRobotDataset; print("LeRobot import OK")'
```

Do not source ROS setup files in the conversion terminal, to avoid external PYTHONPATH interference. The official writer generates the v3 dataset structure without control-side models, cameras, or robots. Run the synthetic conversion example in the collection guide to verify H.264 encoding before converting real demonstrations.
