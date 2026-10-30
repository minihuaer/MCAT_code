# Model weights

Download the model weights from the [MCAT Google Drive folder](<https://drive.google.com/drive/folders/1djJRXoZotVpSuHGWg_BF-laWxtBuRFnr?usp=sharing>). The same folder also provides the collected demonstration dataset demo.

Place the downloaded model files directly in this directory:

| Filename | Component |
| --- | --- |
| `abest_model_152_80.1034-lite.pth` | AITEPose, ResNet-152 with 18 joints |
| `kdu_final2.pth` | KDUnet pose refinement |
| `hand_landmarker.task` | MediaPipe HandLandmarker |
| `yolov8n.pt` | YOLOv8n person detector |

The pose checkpoint selected by `configs/teleop.yaml` is `weights/abest_model_152_80.1034-lite.pth`. All relative asset paths are resolved from the repository root. Set custom paths in that configuration when needed.

Model binaries are distributed through Google Drive and excluded from Git. `manifest.json` records the byte size and SHA-256 checksum of each configured release asset.

After downloading, check file availability:

```bash
python hci2_KDU_AWopt_final.py --check
```

The release tests also verify the checksums in `manifest.json`:

```bash
python -m unittest discover -s tests -p 'test_release.py' -v
```

KDU checkpoint loading is strict. AITEPose requires all inference parameters and reports any unused training-only parameters in the checkpoint.
