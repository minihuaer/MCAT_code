#!/usr/bin/env python3
"""Passive ROS1 recorder. Only record needs ROS; mock/check need numpy + OpenCV."""
import argparse
import json
import math
import os
import importlib
import select
import sys
from pathlib import Path
import threading
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent

# ==================== Runtime defaults: edit here or override them on the command line ====================
RUN_CONFIG = {
    "mode": "record",  # record: hardware collection; mock: synthetic data; check: validate existing data
    "root": HERE.parent.parent / "data/raw",
    "task": "fold the cloth",
    "episodes": 1,  # Record one episode by default, then stop HPE and collection
    "seconds": 33.0,  # Recording duration in seconds, measured after readiness and episode start
    "wait": 20.0,  # Readiness timeout for overhead camera and robot feedback, excluding HPE commands
    "command_wait": 0.0,  # HPE command readiness timeout; 0 waits indefinitely until Ctrl-C
    "ready_seconds": 1.0,  # Required interval of continuously valid data before recording
    "auto_start": True,  # Start automatically when ready; False requires Enter before each episode
    "config": HERE / "config.json",  # Camera, topics, frame rate, and synchronization thresholds
}

HPE_CONFIG = {
    "enabled": True,
    "config": HERE.parent.parent / "configs/teleop.yaml",
    "max_frames": None,
    "headless": False,
}

ARMS = ("left", "right")
REQUIRED = [arm + "_" + kind for arm in ARMS for kind in ("pose", "joints", "command", "status")]


def dump(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def ros_dict(value):
    if hasattr(value, "__slots__"):
        return {key: ros_dict(getattr(value, key)) for key in value.__slots__}
    if isinstance(value, (list, tuple)):
        return [ros_dict(item) for item in value]
    if isinstance(value, bytes):
        return list(value)
    return value


def rpy(q):
    x, y, z, w = [float(q[key]) for key in ("x", "y", "z", "w")]
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("Invalid feedback quaternion")
    x, y, z, w = [v / norm for v in (x, y, z, w)]
    return [math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            math.asin(max(-1., min(1., 2*(w*y-z*x)))),
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))]


def vectors(messages, config):
    state, action, joints = [], [], []
    for arm in ARMS:
        pose = messages[arm + "_pose"]["message"]
        pose = pose.get("pose", pose)  # Also accepts configured PoseStamped.
        joint = messages[arm + "_joints"]["message"]
        if len(joint["name"]) != len(set(joint["name"])):
            raise ValueError("Duplicate joint names")
        by_name = dict(zip(joint["name"], joint["position"]))
        ordered = [by_name[name] for name in config["joint_names"]]
        joints.extend(ordered)
        state.extend([pose["position"][k] for k in ("x", "y", "z")] +
                     rpy(pose["orientation"]) + [ordered[6]])
        cmd = messages[arm + "_command"]["message"]
        action.extend([cmd[k] for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")])
    result = {"observation.state": state, "action": action, "observation.joint_position": joints}
    if any(len(v) != 14 or not np.isfinite(v).all() for v in result.values()):
        raise ValueError("Expected finite 14D state/action/joint vectors")
    return result


def check_sync(messages, camera_time, now, config, include_commands=True):
    times = [camera_time]
    if not 0 <= now - camera_time <= config["max_age_s"]:
        raise ValueError("Camera stale")
    for key in REQUIRED:
        if not include_commands and key.endswith("command"):
            continue
        if key not in messages:
            raise ValueError("Waiting for " + key)
        stamp = messages[key]["monotonic_s"]
        limit = config["max_command_age_s"] if key.endswith("command") else config["max_age_s"]
        if not 0 <= now - stamp <= limit:
            raise ValueError("Stale topic: " + key)
        # Commands are held targets; pose/joints must be close to camera receipt time.
        if key.endswith(("pose", "joints")):
            times.append(stamp)
    if max(times) - min(times) > config["max_skew_s"]:
        raise ValueError("Camera/feedback receipt skew exceeds limit")


class CaptureStopped(Exception):
    """Cooperative operator stop, distinct from an inference/data failure."""


class HPEController:
    """HPE runs on the main thread; recorder observes the shared stop/error state."""
    def __init__(self, settings):
        self.settings = settings
        self.instance = None
        self.stop_event = threading.Event()
        self.error = None
        self.stop_reason = "operator_stop"

    def load(self):
        root = HERE.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        module = importlib.import_module("hci2_KDU_AWopt_final")
        self.instance = module.HPE_3D(
            config_path=self.settings.get("config", root / "configs/teleop.yaml"),
            max_frames=self.settings.get("max_frames"),
            headless=self.settings.get("headless", False))
        print("Loaded MCAT controller in this process", flush=True)

    def check(self):
        if self.error is not None:
            raise RuntimeError("HPE inference failed: " + str(self.error)) from self.error
        if self.stop_event.is_set():
            raise CaptureStopped("HPE/recording session stopped")

    def close(self):
        if self.instance is not None:
            self.instance.close()


def wait_for_enter(health_check, is_shutdown):
    print("Arrange scene, then press Enter to record (Ctrl-C or HPE q quits): ", end="", flush=True)
    while True:
        health_check()
        if is_shutdown():
            raise CaptureStopped("ROS shut down")
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if ready:
            if sys.stdin.readline() == "":
                raise RuntimeError("No terminal input; use auto_start=True or run in a terminal")
            return


def record_with_hpe(args, config):
    import rospy

    # ROS and GUI/model initialization must happen on the main thread, once.
    if not rospy.core.is_initialized():
        rospy.init_node("hci_dataset_recorder", anonymous=True, disable_signals=True)
    hpe = HPEController(HPE_CONFIG)
    errors = []
    worker = None
    try:
        hpe.load()

        def capture_worker():
            try:
                record(args, config, hpe)
            except CaptureStopped:
                pass
            except BaseException as exc:
                errors.append(exc)
            finally:
                hpe.stop_event.set()

        worker = threading.Thread(target=capture_worker, name="hci_recorder")
        worker.start()
        try:
            result = hpe.instance.infer(stop_event=hpe.stop_event)
            if result == "frame_limit":
                hpe.stop_reason = "frame_limit"
                print("HPE reached max_frames; stopping recording.", flush=True)
            if result not in ("operator_stop", "frame_limit") and not hpe.stop_event.is_set():
                raise RuntimeError("HPE inference returned unexpectedly")
        except KeyboardInterrupt:
            print("Stopping HPE and recorder...", flush=True)
        except BaseException as exc:
            hpe.error = exc
            raise
        finally:
            hpe.stop_event.set()
    finally:
        hpe.stop_event.set()
        if worker is not None:
            worker.join()
        hpe.close()
    if errors:
        raise errors[0]


def wait_ready(camera, messages, config, timeout, stable_seconds, health_check,
               is_shutdown, include_commands=True):
    """No episode or timer is created during warmup; reset stability on any gap."""
    started = time.monotonic()
    valid_since = None
    next_log = started
    stage = "HPE commands + feedback" if include_commands else "camera + robot feedback"
    while True:
        health_check()
        if is_shutdown():
            raise RuntimeError("ROS shut down while waiting for " + stage)
        try:
            image = camera.get()
            if image is None:
                raise ValueError("Waiting for overhead image")
            snapshot = messages.snapshot()
            now = time.monotonic()
            check_sync(snapshot, image[1], now, config, include_commands=include_commands)
            if include_commands:
                vectors(snapshot, config)
            if valid_since is None:
                valid_since = now
            if now - valid_since >= stable_seconds:
                print("Ready:", stage, flush=True)
                return
            status = "data valid, waiting for readiness interval"
        except (ValueError, KeyError) as exc:
            valid_since = None
            status = str(exc)
        now = time.monotonic()
        if timeout > 0 and now - started >= timeout:
            raise RuntimeError("Readiness timeout (%s): %s" % (stage, status))
        if now >= next_log:
            print("Waiting [%s]: %s (%.1fs, Ctrl-C to quit)" % (stage, status, now-started), flush=True)
            next_log = now + 5
        time.sleep(0.05)


class Camera:
    def __init__(self, config):
        source = config["camera"]
        if str(source).isdigit():
            source = int(source)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError("Cannot open overhead camera: " + str(source))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
        self.cap.set(cv2.CAP_PROP_FPS, config["fps"])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        seq = 0
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.error = "Overhead camera read failed"
                return
            with self.lock:
                self.latest = (seq, time.monotonic(), time.time(), frame)
            seq += 1

    def get(self):
        if self.error:
            raise RuntimeError(self.error)
        with self.lock:
            return self.latest

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.cap.release()


class Messages:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = {}
        self.events = None
        self.error = None

    def callback(self, msg, key):
        event = {"key": key, "monotonic_s": time.monotonic(), "unix_s": time.time(),
                 "message": ros_dict(msg)}
        with self.lock:
            self.latest[key] = event
            if self.events:
                try:
                    self.events.write(json.dumps(event, allow_nan=False) + "\n")
                except Exception as exc:
                    self.error = str(exc)

    def snapshot(self):
        with self.lock:
            if self.error:
                raise RuntimeError("Raw event write failed: " + self.error)
            return dict(self.latest)


def new_episode(root, config, task):
    root.mkdir(parents=True, exist_ok=True)
    index = 0
    while (root / ("episode_%06d" % index)).exists():
        index += 1
    path = root / ("episode_%06d" % index)
    path.mkdir()
    (path / "images").mkdir()
    dump(path / "episode.json", {"format": "hci_raw_v1", "complete": False,
                                  "config": config, "task": task, "created_unix_s": time.time()})
    return path


def record(args, config, hpe=None):
    import rospy
    from roslib.message import get_message_class

    if not rospy.core.is_initialized():
        rospy.init_node("hci_dataset_recorder", anonymous=True, disable_signals=True)
    hpe = hpe or HPEController(HPE_CONFIG)
    messages = Messages()
    subscriptions = []
    for key, spec in config["topics"].items():
        cls = get_message_class(spec["type"])
        if cls is None:
            raise RuntimeError("ROS runtime has no message type " + spec["type"])
        subscriptions.append(rospy.Subscriber(spec["topic"], cls, messages.callback,
                                              callback_args=key, queue_size=100))
    camera = Camera(config)
    try:
        wait_ready(camera, messages, config, args.wait, 0., hpe.check,
                   rospy.is_shutdown, include_commands=False)
        wait_ready(camera, messages, config, args.command_wait, args.ready_seconds,
                   hpe.check, rospy.is_shutdown)
        for _ in range(args.episodes):
            if not args.auto_start:
                wait_for_enter(hpe.check, rospy.is_shutdown)
            # Arranging the scene can interrupt HPE output. Recheck after Enter,
            # before creating an episode or starting its duration clock.
            wait_ready(camera, messages, config, args.command_wait, args.ready_seconds,
                       hpe.check, rospy.is_shutdown)
            path = new_episode(args.root, config, args.task)
            metadata = json.loads((path / "episode.json").read_text())
            metadata["hpe"] = {key: str(value) if isinstance(value, Path) else value
                               for key, value in HPE_CONFIG.items()}
            metadata["readiness"] = {"command_wait": args.command_wait, "ready_seconds": args.ready_seconds}
            dump(path / "episode.json", metadata)
            count, previous_seq, reason = 0, -1, "duration"
            started = time.monotonic()
            complete = False
            try:
                with (path / "frames.jsonl").open("w") as frames, (path / "events.jsonl").open("w") as events:
                    try:
                        with messages.lock:
                            messages.events = events
                        while count < round(args.seconds * config["fps"]):
                            target = started + count / config["fps"]
                            time.sleep(max(0, target - time.monotonic()))
                            if rospy.is_shutdown():
                                raise RuntimeError("ROS shut down")
                            hpe.check()
                            if time.monotonic() - target > config["max_lateness_s"]:
                                raise RuntimeError("Recording fell behind schedule; lower fps/resolution")
                            image = camera.get()
                            snapshot = messages.snapshot()
                            now = time.monotonic()
                            check_sync(snapshot, image[1], now, config)
                            if image[0] == previous_seq:
                                raise RuntimeError("Repeated camera frame; lower recording fps below camera fps")
                            data = vectors(snapshot, config)
                            bgr = cv2.resize(image[3], (config["width"], config["height"]))
                            image_name = "images/%06d.png" % count
                            if not cv2.imwrite(str(path / image_name), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                                raise IOError("Failed writing image")
                            row = {"frame_index": count, "timestamp": count / config["fps"],
                                   "sample_monotonic_s": now, "camera_monotonic_s": image[1],
                                   "camera_unix_s": image[2], "camera_sequence": image[0],
                                   "image": image_name, "messages": snapshot, **data}
                            frames.write(json.dumps(row, allow_nan=False) + "\n")
                            frames.flush()
                            previous_seq = image[0]
                            count += 1
                        complete = count > 0
                    except (KeyboardInterrupt, CaptureStopped):
                        complete, reason = count > 0, hpe.stop_reason
                    finally:
                        with messages.lock:
                            messages.events = None
                            if messages.error:
                                raise RuntimeError(messages.error)
                        frames.flush()
                        events.flush()
                        os.fsync(frames.fileno())
                        os.fsync(events.fileno())
            except BaseException:
                complete = False
                raise
            finally:
                meta = json.loads((path / "episode.json").read_text())
                meta.update(complete=complete, frames=count, stop_reason=reason if complete else "error")
                dump(path / "episode.json", meta)
                print("%s: %d frames, complete=%s" % (path, count, complete))
            if reason in ("operator_stop", "frame_limit"):
                break
    finally:
        camera.close()
        for subscription in subscriptions:
            subscription.unregister()


def mock(args, config):
    """Deterministic synthetic input for testing conversion without hardware/ROS."""
    for episode in range(args.episodes):
        path = new_episode(args.root, config, args.task)
        count = max(2, round(args.seconds * config["fps"]))
        with (path / "frames.jsonl").open("w") as frames, (path / "events.jsonl").open("w") as events:
            for i in range(count):
                stamp = 1000 + i / config["fps"]
                messages = {}
                for arm in ARMS:
                    payloads = {
                        "pose": {"position": {"x": .2, "y": i*.001, "z": .3},
                                 "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}},
                        "joints": {"name": config["joint_names"], "position": [0.1]*6+[.04],
                                   "velocity": [0.0]*6, "effort": [0.0]*7},
                        "command": dict(x=.21, y=i*.001, z=.31, roll=0., pitch=1.57, yaw=0., gripper=.04, mode1=0, mode2=0),
                        "status": {"ctrl_mode": 1, "err_code": 0}}
                    for kind, payload in payloads.items():
                        key = arm + "_" + kind
                        messages[key] = dict(key=key, monotonic_s=stamp, unix_s=stamp, message=payload)
                        events.write(json.dumps(messages[key]) + "\n")
                bgr = np.zeros((config["height"], config["width"], 3), dtype=np.uint8)
                bgr[:, :, 2] = 180
                cv2.putText(bgr, "%d:%d" % (episode, i), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                name = "images/%06d.png" % i
                if not cv2.imwrite(str(path / name), bgr):
                    raise IOError(name)
                row = dict(frame_index=i, timestamp=i/config["fps"], sample_monotonic_s=stamp,
                           camera_monotonic_s=stamp, camera_unix_s=stamp, camera_sequence=i,
                           image=name, messages=messages, **vectors(messages, config))
                frames.write(json.dumps(row) + "\n")
        meta = json.loads((path / "episode.json").read_text())
        meta.update(complete=True, frames=count, synthetic=True)
        dump(path / "episode.json", meta)
        print(path)


def iter_episode(path, meta):
    config = meta["config"]
    count, previous_time, previous_seq = 0, None, -1
    with (path / "frames.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["frame_index"] != count or abs(row["timestamp"] - count/config["fps"]) > 1e-6:
                raise ValueError("Invalid frame index/timestamp: " + str(path))
            check_sync(row["messages"], row["camera_monotonic_s"], row["sample_monotonic_s"], config)
            if row["camera_sequence"] <= previous_seq:
                raise ValueError("Repeated/out-of-order camera frame")
            if previous_time is not None and not 0 < row["sample_monotonic_s"] - previous_time <= 1/config["fps"] + config["max_lateness_s"]:
                raise ValueError("Invalid sampling gap")
            expected = vectors(row["messages"], config)
            for key, value in expected.items():
                if not np.allclose(row[key], value, atol=1e-7):
                    raise ValueError("Raw messages disagree with " + key)
            image_path = (path / row["image"]).resolve()
            if path.resolve() not in image_path.parents:
                raise ValueError("Image outside episode")
            bgr = cv2.imread(str(image_path))
            if bgr is None or bgr.shape != (config["height"], config["width"], 3):
                raise ValueError("Missing/invalid image: " + str(image_path))
            previous_time, previous_seq = row["sample_monotonic_s"], row["camera_sequence"]
            count += 1
            yield row, bgr
    if count != meta["frames"] or count == 0:
        raise ValueError("Empty episode or frame count mismatch")


def episodes(root):
    result = []
    for path in sorted(root.glob("episode_*")):
        meta = json.loads((path / "episode.json").read_text())
        if not meta["complete"]:
            print("Skipping incomplete episode:", path)
            continue
        result.append((path, meta))
    if not result:
        raise ValueError("No completed episodes in " + str(root))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Command-line options override the RUN_CONFIG defaults above.
    parser.add_argument("mode", nargs="?", choices=["record", "mock", "check"], default=RUN_CONFIG["mode"])
    parser.add_argument("--config", type=Path, default=RUN_CONFIG["config"])
    parser.add_argument("--root", type=Path, default=RUN_CONFIG["root"])
    parser.add_argument("--task", default=RUN_CONFIG["task"])
    parser.add_argument("--episodes", type=int, default=RUN_CONFIG["episodes"])
    parser.add_argument("--seconds", type=float, default=RUN_CONFIG["seconds"])
    parser.add_argument("--wait", type=float, default=RUN_CONFIG["wait"])
    parser.add_argument("--command-wait", type=float, default=RUN_CONFIG["command_wait"])
    parser.add_argument("--ready-seconds", type=float, default=RUN_CONFIG["ready_seconds"])
    parser.add_argument("--auto-start", action=argparse.BooleanOptionalAction, default=RUN_CONFIG["auto_start"])
    parser.add_argument("--hpe-config", type=Path, default=HPE_CONFIG["config"])
    parser.add_argument("--headless", action="store_true", help="Disable HPE camera preview")
    parser.add_argument("--external-hpe", action="store_true", help="Subscribe to an already running controller")
    args = parser.parse_args()
    HPE_CONFIG.update(config=args.hpe_config.resolve(), headless=args.headless, enabled=not args.external_hpe)
    if args.mode == "record" and HPE_CONFIG["enabled"]:
        root = HERE.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from mcat.config import load_config
        hpe_config = load_config(HPE_CONFIG["config"])
        overhead = json.loads(args.config.read_text())["camera"]
        def camera_path(value):
            return Path("/dev/video" + str(value) if isinstance(value, int) else value).resolve()
        if camera_path(overhead) == camera_path(hpe_config["camera"]["source"]):
            parser.error("Interaction and overhead camera must be different devices")
    if any(not math.isfinite(v) or v < 0 for v in (args.wait, args.command_wait, args.ready_seconds)):
        parser.error("Wait times and readiness interval must be finite and nonnegative")
    args.root = args.root.resolve()
    args.config = args.config.resolve()
    config = json.loads(args.config.read_text())
    if args.episodes < 1 or args.seconds <= 0 or config["fps"] < 1 or config["width"] % 2 or config["height"] % 2:
        parser.error("Use positive episode count/duration/fps and even image dimensions")
    if len(config["joint_names"]) != 7 or any(key not in config["topics"] for key in REQUIRED):
        parser.error("Config needs seven joint names and all required topics")
    if args.mode == "check":
        for path, meta in episodes(args.root):
            count = sum(1 for _ in iter_episode(path, meta))
            print("OK", path, count)
    elif args.mode == "record" and HPE_CONFIG["enabled"]:
        record_with_hpe(args, config)
    else:
        globals()[args.mode](args, config)


if __name__ == "__main__":
    main()
