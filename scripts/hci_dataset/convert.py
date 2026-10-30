#!/usr/bin/env python3
"""Convert completed HCI raw episodes with the installed LeRobot 0.4.4 writer."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from collect import ARMS, dump, episodes, iter_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="New output directory; must not exist")
    parser.add_argument("--repo-id", default="local/hci_fold")
    parser.add_argument("--allow-synthetic", action="store_true", help="Only for testing")
    args = parser.parse_args()
    if importlib.metadata.version("lerobot") != "0.4.4":
        parser.error("Run in an environment with lerobot==0.4.4")
    if args.root.exists():
        parser.error("Output already exists. Choose a new directory; raw episodes can be reused.")
    source = episodes(args.raw_root)
    config = source[0][1]["config"]
    for path, meta in source:
        if meta["config"] != config:
            raise ValueError("Mixed configurations: " + str(path))
        if meta.get("synthetic") and not args.allow_synthetic:
            parser.error("Synthetic episode requires --allow-synthetic")
        # Validate the entire input before creating a destination.
        count = sum(1 for _ in iter_episode(path, meta))
        with (path / "events.jsonl").open() as stream:
            for line in stream:
                json.loads(line)
        print("Validated", path.name, count)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = [arm + "_" + axis for arm in ARMS for axis in ("x", "y", "z", "rx", "ry", "rz", "gripper")]
    action_names = [arm + "_cmd_" + axis for arm in ARMS for axis in ("x", "y", "z", "rx", "ry", "rz", "gripper")]
    joint_names = [arm + "_" + name for arm in ARMS for name in config["joint_names"]]
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": names},
        "action": {"dtype": "float32", "shape": (14,), "names": action_names},
        "observation.joint_position": {"dtype": "float32", "shape": (14,), "names": joint_names},
        "observation.images.cam_high": {"dtype": "video", "shape": (config["height"], config["width"], 3),
                                        "names": ["height", "width", "channels"]},
    }
    dataset = LeRobotDataset.create(repo_id=args.repo_id, root=args.root, fps=config["fps"],
                                   robot_type="piper_bimanual_hci", features=features,
                                   use_videos=True, vcodec="h264", image_writer_threads=2)
    manifest = {"complete": False, "lerobot_version": "0.4.4", "raw_root": str(args.raw_root.resolve()),
                "action_semantics": "latest received absolute HPE pin_pos_cmd target; not guaranteed executed",
                "state_semantics": "measured end pose xyz/rpy and measured gripper; metres/radians",
                "clock": "fixed fps dataset timeline; actual host receipt clocks retained in provenance",
                "config": config, "episodes": []}
    provenance = args.root / "provenance"
    provenance.mkdir()
    dump(provenance / "conversion.json", manifest)
    success = False
    try:
        for path, meta in source:
            for row, bgr in iter_episode(path, meta):
                frame = {key: np.asarray(row[key], dtype=np.float32)
                         for key in ("observation.state", "action", "observation.joint_position")}
                frame["observation.images.cam_high"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frame["task"] = meta["task"]
                dataset.add_frame(frame)
            dataset.save_episode()
            dest = provenance / path.name
            dest.mkdir()
            for name in ("episode.json", "frames.jsonl", "events.jsonl"):
                shutil.copy2(path / name, dest / name)
            manifest["episodes"].append({"raw_episode": path.name, "episode_index": len(manifest["episodes"]),
                                         "frames": meta["frames"], "synthetic": meta.get("synthetic", False)})
            print("Saved", path.name)
        success = True
    finally:
        dataset.finalize()
        dataset.stop_image_writer()
    manifest["complete"] = success
    dump(provenance / "conversion.json", manifest)
    print("Complete:", args.root)


if __name__ == "__main__":
    main()
