"""Online KDU_final with the KDU_final2 YAML configuration.

The network combines causal temporal attention, recursive motion-state
estimation, and adaptive position/velocity compensation. Inputs and outputs
are in metres; raw and refined histories are maintained by the caller.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pathlib import Path

import yaml

from .attention import TimeScaleAdaptiveMotionAttention, TimeScaleTemporalPool


def _encoder(input_dim, model_dim):
    return nn.Sequential(nn.Linear(input_dim, model_dim), nn.LayerNorm(model_dim), nn.GELU())


class PaperPhaseCompensator(nn.Module):
    """Parallel velocity/position corrections using the SAME base-pose lag."""

    def __init__(self, model_dim, num_joints, alpha_v, max_velocity, max_position,
                 epsilon=1e-8):
        super().__init__()
        if not 0 <= alpha_v <= 1:
            raise ValueError('RAW_VELOCITY_CURRENT_WEIGHT must be in [0, 1]')
        if min(max_velocity, max_position, epsilon) <= 0:
            raise ValueError('Compensation bounds and epsilon must be positive')
        self.num_joints = num_joints
        self.alpha_v = alpha_v
        self.max_velocity = max_velocity
        self.max_position = max_position
        self.epsilon = epsilon
        self.velocity_encoder = _encoder(num_joints * 3, model_dim)
        self.lag_encoder = _encoder(num_joints, model_dim)
        self.fusion = nn.Sequential(
            nn.Linear(3 * model_dim, model_dim), nn.GELU(),
            nn.Linear(model_dim, model_dim), nn.GELU())
        self.gain_head = nn.Linear(model_dim, 2 * num_joints)

    def forward(self, raw_window, base_pose, predicted_velocity, fused_state):
        observed_velocity = (
            self.alpha_v * (raw_window[:, -1] - raw_window[:, -2])
            + (1 - self.alpha_v) * 0.5 * (raw_window[:, -1] - raw_window[:, -3]))
        direction = F.normalize(observed_velocity.reshape(-1, self.num_joints, 3),
                                dim=-1, eps=self.epsilon)
        innovation = (raw_window[:, -1] - base_pose).reshape(-1, self.num_joints, 3)
        lag = F.relu((innovation * direction).sum(dim=-1))
        velocity_error = observed_velocity - predicted_velocity
        context = self.fusion(torch.cat((fused_state,
            self.velocity_encoder(velocity_error), self.lag_encoder(lag)), dim=-1))
        gamma_v, gamma_p = torch.sigmoid(self.gain_head(context)).chunk(2, dim=-1)
        activity = torch.tanh(lag / self.max_position)
        error = velocity_error.reshape(-1, self.num_joints, 3)
        norm = error.norm(dim=-1, keepdim=True)
        bounded_error = (self.max_velocity * torch.tanh(norm / self.max_velocity)
                         * error / norm.clamp_min(self.epsilon))
        correction_v = (gamma_v * activity).unsqueeze(-1) * bounded_error
        correction_p = (gamma_p * activity * self.max_position).unsqueeze(-1) * direction
        return {
            'pose': base_pose + correction_v.flatten(1) + correction_p.flatten(1),
            'observed_velocity': observed_velocity,
            'velocity_error': velocity_error,
            'phase_lag': lag,
            'phase_activity': activity,
            'phase_velocity_gain': gamma_v,
            'phase_gain': gamma_p,
            'phase_velocity_correction': correction_v.flatten(1),
            'position_phase_correction': correction_p.flatten(1),
        }


class KDUNetStepFinal(nn.Module):
    """One causal update; poses use flattened [B, 3J] storage."""

    def __init__(self, **cfg):
        super().__init__()
        self.window_size = int(cfg['MODEL_WINDOW_SIZE'])
        self.input_dim = int(cfg.get('INPUT_DIM', 51))
        self.high_window_size = int(cfg.get('HIGH_WINDOW_SIZE', 5))
        self.span = int(cfg.get('LOW_VELOCITY_SPAN', 3))
        self.rho = float(cfg.get('LOW_VELOCITY_PHASE_LEAD', 0.25))
        self.cv = float(cfg.get('MAX_VELOCITY_ADJUSTMENT', 0.01))
        d = int(cfg.get('MODEL_DIM', 192))
        heads = int(cfg.get('ATTENTION_HEADS', 8))
        if self.input_dim < 3 or self.input_dim % 3:
            raise ValueError('INPUT_DIM must contain complete xyz joints')
        if self.window_size < 3 or not 1 <= self.high_window_size < self.window_size:
            raise ValueError('Require N >= 3 and 1 <= HIGH_WINDOW_SIZE < N')
        if not 1 <= self.span < self.window_size - 1:
            raise ValueError('LOW_VELOCITY_SPAN must be in [1, N-2]')
        if not 0 <= self.rho <= 1 or self.cv <= 0:
            raise ValueError('Require rho in [0,1] and positive velocity bound')
        if d <= 0 or heads <= 0 or d % heads:
            raise ValueError('MODEL_DIM must be positive and divisible by ATTENTION_HEADS')
        self.num_joints = self.input_dim // 3
        for name, length, prefix, layers in (
            ('stable', self.window_size - 1, 'LOW', int(cfg.get('LOW_TAMA_LAYERS', 2))),
            ('motion', self.high_window_size, 'HIGH', int(cfg.get('HIGH_TAMA_LAYERS', 2))),
            ('deviation', self.high_window_size, 'HIGH', int(cfg.get('DEVIATION_TAMA_LAYERS', 1)))):
            if layers < 1:
                raise ValueError('Each temporal encoder needs at least one ATM block')
            low = prefix == 'LOW'
            tau = float(cfg.get(prefix + '_ATTENTION_TAU', 12.0 if low else 1.25))
            encoder = TimeScaleAdaptiveMotionAttention(
                length, self.input_dim, d, layers, heads, tau,
                float(cfg.get(prefix + '_MIN_TAU', 4.0 if low else 0.25)),
                float(cfg.get(prefix + '_MAX_TAU', 64.0 if low else 4.0)),
                float(cfg.get('DROPOUT', 0.2)))
            pool = TimeScaleTemporalPool(length, tau,
                float(cfg.get(prefix + '_POOL_TEMPERATURE', 1.3 if low else 0.7)))
            setattr(self, name + '_encoder', nn.Sequential(encoder, pool))
        self.high_fusion = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.feature_gate = nn.Linear(2 * d, d)
        self.velocity_head = nn.Linear(d, self.input_dim)
        self.innovation_encoder = _encoder(self.input_dim, d)
        self.correction_head = nn.Linear(2 * d, self.num_joints)
        self.phase_compensator = PaperPhaseCompensator(
            d, self.num_joints, float(cfg.get('RAW_VELOCITY_CURRENT_WEIGHT', 0.8)),
            float(cfg.get('MAX_PHASE_VELOCITY_CORRECTION', 0.035)),
            float(cfg.get('MAX_PHASE_CORRECTION', 0.030)),
            float(cfg.get('NORMALIZATION_EPS', 1e-8)))

    def forward(self, raw_window, smooth_history):
        if raw_window.ndim != 3 or tuple(raw_window.shape[1:]) != (self.window_size, self.input_dim):
            raise ValueError('raw_window must have shape [B, N, INPUT_DIM]')
        if smooth_history.ndim != 3 or tuple(smooth_history.shape) != (
                raw_window.shape[0], self.window_size - 1, self.input_dim):
            raise ValueError('smooth_history must have shape [B, N-1, INPUT_DIM]')
        raw_window, smooth_history = raw_window.float(), smooth_history.float()
        h_l = self.stable_encoder(smooth_history)
        velocity = raw_window[:, 1:] - raw_window[:, :-1]
        h_m = self.motion_encoder(velocity[:, -self.high_window_size:])
        deviation = raw_window[:, :-1] - smooth_history
        h_d = self.deviation_encoder(deviation[:, -self.high_window_size:])
        h_h = self.high_fusion(torch.cat((h_m, h_d), dim=-1))
        gate = torch.sigmoid(self.feature_gate(torch.cat((h_l, h_h), dim=-1)))
        h_f = (1 - gate) * h_l + gate * h_h
        baseline = ((1 - self.rho) * (smooth_history[:, -1] - smooth_history[:, -1-self.span]) / self.span
                    + self.rho * (smooth_history[:, -1] - smooth_history[:, -2]))
        predicted_velocity = baseline + self.cv * torch.tanh(self.velocity_head(h_f))
        prior = smooth_history[:, -1] + predicted_velocity
        innovation = raw_window[:, -1] - prior
        gain = torch.sigmoid(self.correction_head(torch.cat(
            (h_f, self.innovation_encoder(innovation)), dim=-1)))
        gain_xyz = gain.repeat_interleave(3, dim=-1)
        base = prior + gain_xyz * innovation
        result = self.phase_compensator(raw_window, base, predicted_velocity, h_f)
        result.update(prior_pose=prior, predicted_velocity=predicted_velocity,
                      pre_phase_pose=base, correction_gain=gain_xyz,
                      innovation=innovation, feature_gate=gate)
        return result


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / 'configs/KDU_final2.yaml'
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / 'weights/kdu_final2.pth'


def read_model_config(config_path=DEFAULT_CONFIG_PATH):
    """Read inference parameters; PRETRAINED is a training initializer only."""
    with Path(config_path).open() as handle:
        config = yaml.safe_load(handle)['MODEL']
    if config.get('TYPE') != 'KDU_final' or config.get('INPUT_DIM') != 51:
        raise ValueError('Expected MODEL.TYPE=KDU_final and INPUT_DIM=51')
    return config


class KDUFinalLite(nn.Module):
    """Single-step pose refinement with caller-owned raw and refined histories."""

    def __init__(self, model_config):
        super().__init__()
        self.model_config = dict(model_config)
        self.infer_net = KDUNetStepFinal(**model_config)

    def forward(self, raw_window, smooth_history):
        return self.infer_net(raw_window, smooth_history)['pose']


def load_model(weights_path=None, device='cpu', config_path=DEFAULT_CONFIG_PATH):
    config_path = Path(config_path).expanduser().resolve()
    weights_path = Path(weights_path or DEFAULT_WEIGHTS_PATH).expanduser().resolve()
    model = KDUFinalLite(read_model_config(config_path))
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    if 'state_dict' in state:
        state = state['state_dict']
    if state and all(key.startswith('module.') for key in state):
        state = {key.removeprefix('module.'): value for key, value in state.items()}
    # Reject incompatible or incomplete checkpoints.
    model.load_state_dict(state, strict=True)
    model.config_path = config_path
    model.weights_path = weights_path
    return model.to(device).eval()


