"""AITEPose inference network; checkpoint parameter names are preserved."""
import torch
from torch import nn
from easydict import EasyDict
from .layers.regression_nf_3d_cbam_lite import RegressFlow3D_cbam_lite

class MLP(nn.Module):

    def __init__(self, in_planes, out_planes, mid_planes=64, act=nn.LeakyReLU()):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_planes, mid_planes), nn.LeakyReLU(), nn.Linear(mid_planes, mid_planes), nn.LeakyReLU(), nn.Linear(mid_planes, out_planes), act)

    def forward(self, x):
        return self.mlp(x)

class RegressFlow3D_hci_3vm2_best_lite(nn.Module):
    'AITEPose regression backbone with a learned coordinate correction head.'

    def __init__(self, resnet, joint_num, norm_layer=nn.BatchNorm2d):
        super(RegressFlow3D_hci_3vm2_best_lite, self).__init__()
        self.num_joints = joint_num
        self.pre_rle = RegressFlow3D_cbam_lite(resnet, joint_num)
        self.pred_mlp = MLP(3 * self.num_joints, 3 * self.num_joints, act=nn.Sigmoid())

    def forward(self, x):
        BATCH_SIZE = x.shape[0]
        self.rle_out = self.pre_rle(x)
        pred_jts = self.rle_out.pred_jts
        sigma = self.rle_out.sigma
        scores = self.rle_out.maxvals
        # Apply the coordinate correction head.
        pred_jts_o = pred_jts.clone()
        # Return predicted coordinates, uncertainty, and confidence.
        output = EasyDict(pred_jts=pred_jts + self.pred_mlp(pred_jts.reshape(BATCH_SIZE, -1)).reshape(BATCH_SIZE, self.num_joints, 3), sigma=sigma, maxvals=scores.float(), pred_jts_o=pred_jts_o, pro_jts=pred_jts)
        return output
