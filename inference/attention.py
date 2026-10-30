"""Causal time-scale attention shared by the paper KDUnet encoders."""
import math
import torch
from torch import nn

def _inverse_sigmoid(value: float) -> float:
    return math.log(value / (1.0 - value))

class TimeScaleCausalAttentionBlock(nn.Module):
    """Causal attention with a learnable, head-wise temporal decay.

    The additive bias for query ``i`` and historical key ``j`` is
    ``-(i-j)/tau``.  A large tau retains long-range motion intent, whereas a
    small tau produces a strongly recent, high-frequency receptive field.
    """

    def __init__(self, model_dim: int, num_heads: int, initial_tau: float, min_tau: float, max_tau: float, dropout: float, mlp_ratio: int=4) -> None:
        super().__init__()
        if not 0.0 < min_tau < initial_tau < max_tau:
            raise ValueError('attention tau must satisfy 0 < min < initial < max')
        self.num_heads = num_heads
        self.min_tau = min_tau
        self.max_tau = max_tau
        tau_ratio = (initial_tau - min_tau) / (max_tau - min_tau)
        self.tau_logits = nn.Parameter(torch.full((num_heads,), _inverse_sigmoid(tau_ratio)))
        self.input_projection = nn.Linear(model_dim, model_dim)
        self.attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(nn.Linear(model_dim, model_dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout), nn.Linear(model_dim * mlp_ratio, model_dim), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(model_dim)

    def temporal_tau(self) -> torch.Tensor:
        return self.min_tau + (self.max_tau - self.min_tau) * torch.sigmoid(self.tau_logits)

    def _attention_bias(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        position = torch.arange(length, device=x.device)
        distance = position[:, None] - position[None, :]
        future = distance < 0
        tau = self.temporal_tau().to(dtype=x.dtype).view(-1, 1, 1)
        bias = -distance.clamp_min(0).to(dtype=x.dtype).unsqueeze(0) / tau
        bias = bias.masked_fill(future.unsqueeze(0), float('-inf'))
        return bias.unsqueeze(0).expand(x.shape[0], -1, -1, -1).reshape(x.shape[0] * self.num_heads, length, length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(x)
        attended, _ = self.attention(projected, projected, projected, attn_mask=self._attention_bias(x), need_weights=False)
        x = self.norm1(x + attended)
        return self.norm2(x + self.mlp(x))

class TimeScaleAdaptiveMotionAttention(nn.Module):
    """Project a motion sequence and encode it at one explicit time scale."""

    def __init__(self, sequence_length: int, input_dim: int, model_dim: int, num_layers: int, num_heads: int, initial_tau: float, min_tau: float, max_tau: float, dropout: float) -> None:
        super().__init__()
        if sequence_length < 1:
            raise ValueError('sequence_length must be positive')
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position_embedding = nn.Parameter(torch.empty(1, sequence_length, model_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.embedding_norm = nn.LayerNorm(model_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TimeScaleCausalAttentionBlock(model_dim=model_dim, num_heads=num_heads, initial_tau=initial_tau, min_tau=min_tau, max_tau=max_tau, dropout=dropout) for _ in range(num_layers)])

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        expected = (self.sequence_length, self.input_dim)
        if sequence.ndim != 3 or tuple(sequence.shape[1:]) != expected:
            raise ValueError('Expected [B, {}, {}], got {}'.format(*expected, tuple(sequence.shape)))
        encoded = self.embedding_dropout(self.embedding_norm(self.input_projection(sequence) + self.position_embedding))
        for block in self.blocks:
            encoded = block(encoded)
        return encoded

class TimeScaleTemporalPool(nn.Module):
    """Temporal pooling whose smoothness is tied to the branch time scale."""

    def __init__(self, sequence_length: int, initial_tau: float, temperature: float, adjustment_scale: float=0.5) -> None:
        super().__init__()
        if initial_tau <= 0.0 or temperature <= 0.0:
            raise ValueError('pool tau and temperature must be positive')
        self.sequence_length = sequence_length
        self.temperature = temperature
        self.adjustment_scale = adjustment_scale
        distance = torch.arange(sequence_length - 1, -1, -1).float()
        self.register_buffer('base_logits', -distance / initial_tau)
        self.logit_adjustment = nn.Parameter(torch.zeros(sequence_length))

    def weights(self) -> torch.Tensor:
        logits = self.base_logits + self.adjustment_scale * torch.tanh(self.logit_adjustment)
        return torch.softmax(logits / self.temperature, dim=0)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[1] != self.sequence_length:
            raise ValueError('Expected [B, {}, D], got {}'.format(self.sequence_length, tuple(sequence.shape)))
        return (sequence * self.weights().view(1, -1, 1)).sum(dim=1)
