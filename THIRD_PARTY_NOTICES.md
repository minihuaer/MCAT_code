# Component attribution

MCAT uses the following software components and dependencies:

- AITEPose for human pose estimation, together with camera geometry and pose augmentation utilities. Applicable author and BIT807s copyright notices are included in the source files.
- `models/layers/regression_nf_3d_cbam_lite.py` includes the Shanghai Jiao Tong University / Jiefeng Li copyright notice.
- KDUnet for causal temporal pose refinement, implemented in `inference/`.
- AWopt for wrist trajectory refinement, implemented in `awopt_final.py`.
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) for hand landmark detection through its HandLandmarker API.
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/tree/v8.3.0) for person detection.
- [LeRobot](https://github.com/huggingface/lerobot/tree/v0.4.4) for dataset conversion through its official writer API.

Model weights and the demonstration dataset are distributed through the download link in the [README](README.md). The active model inventory and checksums are provided in `weights/manifest.json`. Third-party components retain their applicable licenses and copyright notices.
