"""Pose flip augmentation. Copyright (c) 2025 BIT807s."""
import torch

def flip_coord(preds, joint_pairs, width_dim, shift=False, flatten=True):
    """Flip pose heatmap according to joint pairs.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap of joints.
    joint_pairs : list
        List of joint pairs.
    shift : bool
        Whether to shift the output.

    Returns
    -------
    numpy.ndarray
        Flipped heatmap.

    """
    pred_jts, pred_scores = preds
    if flatten:
        assert pred_jts.dim() == 2 and pred_scores.dim() == 3
        num_batches = pred_jts.shape[0]
        num_joints = pred_jts.shape[1] // 3
        pred_jts = pred_jts.reshape(num_batches, num_joints, 3)
    else:
        assert pred_jts.dim() == 3 and pred_scores.dim() == 3
        num_batches = pred_jts.shape[0]
        num_joints = pred_jts.shape[1]
    if shift:
        pred_jts[:, :, 0] = -pred_jts[:, :, 0] - 1 / (width_dim * 4)
    else:
        pred_jts[:, :, 0] = -1 / width_dim - pred_jts[:, :, 0]
    for pair in joint_pairs:
        dim0, dim1 = pair
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        pred_jts[:, idx] = pred_jts[:, inv_idx]
        pred_scores[:, idx] = pred_scores[:, inv_idx]
    return (pred_jts, pred_scores)
