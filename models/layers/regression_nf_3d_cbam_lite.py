"""Pose regression backbone.
Copyright (c) Shanghai Jiao Tong University. All rights reserved.
Written by Jiefeng Li (jeff.lee.sjtu@gmail.com).
"""
import torch
from torch import nn
from easydict import EasyDict
from .Resnet_cbam import ResNet_cbam

class Linear(nn.Module):

    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())
        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm
        if self.bias:
            y = y + self.linear.bias
        return y

class RegressFlow3D_cbam_lite(nn.Module):

    def __init__(self, resnet, joint_num, norm_layer=nn.BatchNorm2d):
        super(RegressFlow3D_cbam_lite, self).__init__()
        self.fc_dim = [-1]
        self.num_joints = joint_num
        self.preact = ResNet_cbam(f'resnet{resnet}')
        self.feature_channel = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}[resnet]
        self.root_idx = 0
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fcs, out_channel = self._make_fc_layer()
        self.fc_coord = Linear(out_channel, self.num_joints * 3)
        self.fc_sigma = nn.Linear(out_channel, self.num_joints * 3)
        self.fc_layers = [self.fc_coord, self.fc_sigma]

    def _make_fc_layer(self):
        fc_layers = []
        num_deconv = len(self.fc_dim)
        input_channel = self.feature_channel
        for i in range(num_deconv):
            if self.fc_dim[i] > 0:
                fc = nn.Linear(input_channel, self.fc_dim[i])
                bn = nn.BatchNorm1d(self.fc_dim[i])
                fc_layers.append(fc)
                fc_layers.append(bn)
                fc_layers.append(nn.ReLU(inplace=True))
                input_channel = self.fc_dim[i]
            else:
                fc_layers.append(nn.Identity())
        return (nn.Sequential(*fc_layers), input_channel)

    def forward(self, x, labels=None):
        BATCH_SIZE = x.shape[0]
        feat = self.preact(x)
        _, _, f_h, f_w = feat.shape
        feat = self.avg_pool(feat).reshape(BATCH_SIZE, -1)
        out_coord = self.fc_coord(feat).reshape(BATCH_SIZE, self.num_joints, 3)
        assert out_coord.shape[2] == 3
        out_sigma = self.fc_sigma(feat).reshape(BATCH_SIZE, self.num_joints, -1)
        pred_jts = out_coord.reshape(BATCH_SIZE, self.num_joints, 3)
        if not self.training:
            pred_jts[:, :, 2] = pred_jts[:, :, 2] - pred_jts[:, self.root_idx:self.root_idx + 1, 2]
        sigma = out_sigma.reshape(BATCH_SIZE, self.num_joints, -1).sigmoid() + 1e-09
        scores = 1 - sigma
        scores = torch.mean(scores, dim=2, keepdim=True)
        output = EasyDict(pred_jts=pred_jts, sigma=sigma, maxvals=scores)
        return output
