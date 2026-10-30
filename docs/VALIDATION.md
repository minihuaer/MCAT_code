# Testing and dataset validation

## Software tests

Download the model weights and configure the Python environment before running:

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s scripts/hci_dataset -p 'test_*.py' -v
```

The method and release tests cover:

- AWopt equations, parameter constraints, normalization, and motion-adaptive filtering.
- KDU unit conversion, causal history updates, and state reset.
- AITEPose checkpoint loading and finite image-inference outputs.
- Model file integrity against the SHA-256 manifest.
- Gripper transitions, command holding, and Cartesian mapping.
- Simulated dual-arm command publication with bounded controller state.

The collection tests cover feedback ordering, action semantics, observation freshness, synchronization, readiness, controller initialization, coordinated stopping, error propagation, and preservation of fully written data.

These tests exercise software components without opening cameras or publishing commands to a robot. Control interfaces are simulated where needed.

## Configuration and dependencies

```bash
python hci2_KDU_AWopt_final.py --check
python scripts/check_environment.py --ros
```

The first command checks the runtime configuration and required model files. The second checks imports for the visual pipeline and ROS/Piper interfaces. Run it in the environment used on the robot computer; it does not open a camera or publish commands.

## Dataset checks

Use a new output directory for synthetic test episodes:

```bash
python scripts/hci_dataset/collect.py mock \
  --root ./data/mock/raw --seconds 0.2 --episodes 2
python scripts/hci_dataset/collect.py check --root ./data/mock/raw
```

In the separate conversion environment:

```bash
python scripts/hci_dataset/convert.py \
  --raw-root ./data/mock/raw --root ./data/mock/lerobot \
  --repo-id local/mcat_mock --allow-synthetic
```

Synthetic episodes are explicitly marked and require `--allow-synthetic` for conversion. Validate collected raw demonstrations with the same `check` command using their own root directory. For conversion output, check `provenance/conversion.json` for `complete=true`.

See the [dataset guide](../scripts/hci_dataset/README.md) for field definitions, synchronization settings, and demonstration downloads.
