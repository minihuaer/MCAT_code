# Demonstration collection and conversion

The recorder collects demonstration images, robot feedback, and control targets. Start it with `collect.py record`; the live teleoperation entry point runs independently of dataset recording.

## Download the demonstration dataset

A collected demonstration dataset demo is available with the model weights in the [MCAT Google Drive folder](<https://drive.google.com/drive/folders/1djJRXoZotVpSuHGWg_BF-laWxtBuRFnr?usp=sharing>). Download and extract the dataset to a local directory such as `data/demo/`. Keep downloaded demonstration data under `data/` so that it remains separate from the source repository.

The commands below describe collecting and converting demonstrations. Apply the raw-data checker to datasets with the `episode_*/episode.json` layout described here; use the LeRobot API for datasets with `data/`, `meta/`, and `videos/` directories.

## Recorded fields

| Field | Contents |
| --- | --- |
| `observation.images.cam_high` | Separate overhead camera; default 320×240 RGB |
| `observation.state` | Left xyz/roll/pitch/yaw/measured gripper, followed by right arm; 14 values |
| `action` | Left target xyz/roll/pitch/yaw/target gripper, followed by right arm; 14 values |
| `observation.joint_position` | Left six joints and gripper, followed by right arm; 14 values |
| `events.jsonl` | Complete received ROS messages and host receipt timestamps during collection |
| `frames.jsonl` | Per-frame message snapshots, image index, actual timestamps, and derived vectors |

Position and gripper opening use metres; angles use radians. The state gripper value comes from measured joint feedback, not the target. Actions are final absolute end-effector targets, not increments, and are not guaranteed to have been executed by downstream IK or the robot. Each arm's feedback is expressed in its own base frame.

`joint_names` defaults to joint0–joint6. Values are reordered by name, with the seventh joint used as the gripper. Pose feedback defaults to `geometry_msgs/Pose`; `PoseStamped` can also be selected in the configuration. `joint_ctrl_single` is recorded as optional raw feedback and is not required to start an episode.

## Configuration and startup

Edit `config.json` in this directory:

- Set `camera` to the actual overhead camera. `/dev/video2` is an example; a stable `/dev/v4l/by-id/...` name is preferable.
- Check both arms' pose, joints, command, and status topics and message types.
- Check `joint_names`, frame rate, image dimensions, and synchronization thresholds.
- Configure the interaction camera and controller in `configs/teleop.yaml`. The two camera sources must refer to different devices.

After starting the ROS driver and IK and sourcing the workspace:

```bash
python scripts/hci_dataset/collect.py record \
  --root ./data/raw/fold --task "fold the cloth" --seconds 30 --episodes 1
# Disable preview and select a runtime configuration:
python scripts/hci_dataset/collect.py record --headless \
  --hpe-config configs/teleop.yaml --root ./data/raw/fold --seconds 30
# Subscribe only, if another terminal already runs the controller:
python scripts/hci_dataset/collect.py record --external-hpe --root ./data/raw/fold
```

Integrated mode runs HPE on the main thread and records data in a background thread. Do not start a second controller simultaneously. The recorder waits for the camera and feedback, then for both arms' commands to remain valid for one second. Only then does it create an episode and start the `--seconds` timer. Command readiness has no default timeout; use `--command-wait 180` to impose a limit. No empty actions are written while waiting.

Recording starts automatically by default. `--no-auto-start` requires Enter before each episode. Automatic multi-episode mode does not reset the scene between episodes. Duration and episode count can also be set in `RUN_CONFIG` at the top of the script.

Pressing `q` in the interaction window or Ctrl-C stops both components and saves the fully written valid prefix. Inference, recording, or disk failures mark an episode incomplete; conversion skips it. Reaching the requested duration stops integrated HPE and releases resources. Stopping collection does not home, disable, or plan robot motion.

Repeated use of a root directory allocates new episode directories without overwriting earlier ones. Do not run concurrent writers against the same root. Use a new root after changing camera, FPS, or field settings. Manually reject unsuccessful demonstrations even if their files are complete, for example by renaming their directories to `rejected_*`.

## Synchronization

Sampling uses a fixed FPS and the latest received image and messages, without future interpolation. Synchronization is based on receipt times on one host, not camera exposure or hardware timestamps.

Default validity limits are 0.2 s for images/feedback, 0.5 s for commands, and 0.15 s for image-feedback receipt skew. Held targets may produce repeated actions across frames. Duplicate images, missed sampling deadlines, or stale feedback end the episode and mark it incomplete; frames are not fabricated. Reduce recording FPS if the camera cannot sustain it, and verify a short recording first.

Raw layout:

```text
data/raw/fold/episode_000000/
  episode.json
  frames.jsonl
  events.jsonl
  images/000000.png ...
```

PNG files are intentional demonstration images, not controller debug screenshots. Actual timestamps and raw messages preserve provenance.

## Validation and LeRobot conversion

Validate in the collection environment:

```bash
python scripts/hci_dataset/collect.py check --root ./data/raw/fold
```

Convert in the separate `mcat-convert` environment:

```bash
python scripts/hci_dataset/convert.py \
  --raw-root ./data/raw/fold --root ./data/lerobot/fold --repo-id local/mcat_fold
```

The output directory must not already exist. The converter uses the official LeRobot 0.4.4 `create/add_frame/save_episode/finalize` API to generate v3 data/meta/videos and provenance. It does not upload the dataset. Here 0.4.4 is the software version and v3 is the data format. Check that `provenance/conversion.json` has `complete=true` before using the result.

## Hardware-free checks

These commands do not access a camera or robot. Generated episodes are marked synthetic:

```bash
python -m unittest discover -s scripts/hci_dataset -p 'test_*.py' -v
python scripts/hci_dataset/collect.py mock --root ./data/mock/raw --seconds 0.2 --episodes 2
python scripts/hci_dataset/collect.py check --root ./data/mock/raw
# In the conversion environment:
python scripts/hci_dataset/convert.py --raw-root ./data/mock/raw \
  --root ./data/mock/lerobot --repo-id local/mcat_mock --allow-synthetic
```

Synthetic conversion is rejected by default. `--allow-synthetic` is for pipeline checks only; synthetic episodes should not enter the real training dataset.
