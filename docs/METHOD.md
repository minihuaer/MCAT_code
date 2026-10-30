# Paper method and implementation

The position pipeline is RGB camera → YOLO person localization → AITEPose → root-relative XYZ → KDU_final2 → AWopt → dual-arm Cartesian mapping. MediaPipe observations feed only the gripper branch.

## KDUnet

`inference/kdu_final.py` implements the single-step KDUnet update. `inference/attention.py` provides causal temporal attention and pooling. Network parameters are configured in `configs/KDU_final2.yaml`.

KDU consumes 27 raw pose frames and 26 recursively refined historical poses. The long-term encoder reads refined poses, the short-term motion encoder reads raw velocities, and the deviation branch reads differences between raw and refined history. Gated fusion drives velocity prediction and observation correction. Velocity and position compensation share the positive directional lag measured from the base pose. Only the final KDU prediction is written back into its history.

External XYZ coordinates are in millimetres. Inputs are divided by 1000 before network inference and outputs are multiplied by 1000 afterward. The first 26 raw poses initialize the histories; subsequent predictions are causal. AWopt output is never fed back into KDU.

## AWopt

`awopt_final.py` implements the paper's equations. Right and left arms independently use shoulder/elbow/wrist indices `(14,15,16)` and `(11,12,13)`, respectively.

The first valid frame fixes the normalization length `L=norm(wrist-elbow)`. A causal window including the current frame provides `z=dz/L` and `q=(dx²+dy²)/L²` for the relation `q=alpha+beta1*z+beta2*z²`. The implementation uses the paper's permitted diagonal working precision `Q=I` and eliminates the unknown intercept by centering.

Parameters update only when the minimum eigenvalue of the unregularized information matrix `S` exceeds the excitation threshold. A 2×2 system combines parameter regularization with a linearized depth-consistency term referenced to the previous final output. Scale bounds are enforced by minimizing the same quadratic objective over the coefficient box. Accepted parameters correct the current wrist depth.

The most recent five shoulder/elbow xy displacements determine motion intensity and the exponential decay rate. The output filter then averages depth-corrected wrist XYZ history. The estimation and output windows default to 120 and five frames, respectively, and are configured in `configs/teleop.yaml`.

## Camera and robot mapping

Pose reconstruction applies camera-dependent depth-tilt correction and a configurable root-depth estimate before recovering XYZ with the camera intrinsics. The default root-depth estimate is 1700 mm. Set these parameters for the camera setup in `configs/teleop.yaml`.

Image projections for the hand regions of interest are computed before KDU. AWopt uses the current root-relative KDU prediction and does not use delayed MediaPipe wrist positions.

Cartesian mapping first changes axes to `[-Z, X, -Y]`, converts millimetres to metres, then applies the per-arm affine transformation and workspace limits. In the Piper/ROS frame, z is height and xy is the horizontal plane. Horizontal yaw is `atan2(y,x)` and pitch is interpolated using z.

## State and control

YOLO localizes the person during initialization. After the localization stability counter exceeds its threshold, the controller uses an anchor crop and begins publishing. This mode assumes a fixed operator and workspace; it does not perform multi-person identity tracking during the anchor phase. Restart the session when changing the operator or workspace.

When person detection returns no person, KDU/AWopt histories and stale hand observations are cleared. Each hand recognizer has at most one background job, and the control loop does not wait for hand inference. The gripper state machine uses consecutive hand-state decisions, observation freshness limits, and command holding during missing observations.

The controller retains only bounded method histories, latest observations, and current state. The dataset recorder stores demonstration images, robot observations, and control targets separately from the live controller.
