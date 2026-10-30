"""Paper AWopt: causal affine depth estimation and motion-adaptive filtering.

Inputs are root-relative XYZ in mm. Each arm maintains independent state.
AWOptConfig defines the estimation and output-filter parameters. Q = I on
a finite causal estimation window. The scale constraint minimizes the
quadratic objective on its beta2 box.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AWOptConfig:
    estimation_window: int = 120
    output_window: int = 5
    regularization: float = 1e-3  # lambda
    depth_regularization: float = 0.1  # mu
    excitation_threshold: float = 1e-4
    scale_min: float = 0.5
    scale_max: float = 2.0
    shoulder_decay: float = 0.5
    elbow_decay: float = 0.5
    rho_min: float = 0.1
    rho_max: float = 2.0
    motion_sensitivity: float = 0.02  # kappa, inverse mm


def depth_linearization(beta, z, elbow_z, previous_output_z):
    """All depths normalized by L; return g, gradient j, and target b."""
    s = np.sqrt(-beta[1])
    g = elbow_z + s * z - beta[0] / (2 * s)
    j = np.array([-1 / (2 * s), -z / (2 * s) - beta[0] / (4 * s**3)])
    b = previous_output_z - g + j @ beta
    return g, j, b


def constrained_solve(A, rhs, config):
    """Exact box-constrained quadratic minimizer; beta1 is unrestricted."""
    beta = np.linalg.solve(A, rhs)
    lo, hi = -config.scale_max**2, -config.scale_min**2
    if not lo <= beta[1] <= hi:
        beta[1] = np.clip(beta[1], lo, hi)
        beta[0] = (rhs[0] - A[0, 1] * beta[1]) / A[0, 0]
    return beta


class AdaptiveWristOptimizer:
    def __init__(self, config=None):
        self.config = config or AWOptConfig()
        c = self.config
        for name in ('estimation_window', 'output_window'):
            value = getattr(c, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not all(np.isfinite(v) for v in vars(c).values()) or not (
            c.regularization > 0 and c.depth_regularization >= 0
            and c.excitation_threshold >= 0 and 0 < c.scale_min < 1 < c.scale_max
            and c.shoulder_decay > 0 and c.elbow_decay > 0
            and 0 <= c.rho_min < c.rho_max and c.motion_sensitivity > 0
        ):
            raise ValueError('Invalid AWopt configuration')
        self.reset()

    def reset(self):
        self.L = None
        self.beta = np.array([0., -1.])  # identity affine correction
        self.samples = deque(maxlen=self.config.estimation_window)
        self.corrected = deque(maxlen=self.config.output_window)
        self.motion = deque(maxlen=5)
        self.previous_shoulder = self.previous_elbow = self.previous_output = None

    def update(self, shoulder, elbow, wrist):
        """Return current filtered root-relative wrist and diagnostics.

        Invalid input raises before changing state. Reset for a new sequence,
        subject, coordinate system or tracking discontinuity.
        """
        shoulder, elbow, wrist = (np.asarray(p, dtype=float) for p in (shoulder, elbow, wrist))
        if any(p.shape != (3,) or not np.isfinite(p).all() for p in (shoulder, elbow, wrist)):
            raise ValueError('Expected finite root-relative XYZ points')
        d = wrist - elbow
        length = float(np.linalg.norm(d))
        if not np.isfinite(length) or length <= 1e-12:
            raise ValueError('Forearm vector must have nonzero finite length')
        if self.L is None:
            self.L = length
        z = d[2] / self.L
        q = float(d[:2] @ d[:2]) / self.L**2
        self.samples.append((z, q))
        samples = np.asarray(self.samples)
        F = np.column_stack((samples[:, 0], samples[:, 0]**2))
        y = samples[:, 1]
        # Q = I: centered products equal F.T @ Q_perp @ F and F.T @ Q_perp @ y.
        Fc, yc = F - F.mean(axis=0), y - y.mean()
        S, c = Fc.T @ Fc, Fc.T @ yc
        minimum_eigenvalue = float(np.linalg.eigvalsh(S)[0])
        updated = minimum_eigenvalue > self.config.excitation_threshold
        if updated:
            A = S + self.config.regularization * np.eye(2)
            rhs = c + self.config.regularization * self.beta
            if self.previous_output is not None:
                _, j, b = depth_linearization(
                    self.beta, z, elbow[2] / self.L, self.previous_output[2] / self.L)
                A += self.config.depth_regularization * np.outer(j, j)
                rhs += self.config.depth_regularization * b * j
            self.beta = constrained_solve(A, rhs, self.config)

        s = np.sqrt(-self.beta[1])
        h = -self.beta[0] / (2 * s)
        corrected = wrist.copy()
        corrected[2] = elbow[2] + self.L * (s * z + h)
        self.corrected.append(corrected)

        displacement = (0., 0.) if self.previous_shoulder is None else (
            np.linalg.norm(shoulder[:2] - self.previous_shoulder[:2]),
            np.linalg.norm(elbow[:2] - self.previous_elbow[:2]))
        self.motion.append(displacement)
        motion = sum(np.exp(-k * self.config.shoulder_decay) * ds
                     + np.exp(-k * self.config.elbow_decay) * de
                     for k, (ds, de) in enumerate(reversed(self.motion)))
        rho = self.config.rho_min + (self.config.rho_max - self.config.rho_min) * np.tanh(
            self.config.motion_sensitivity * motion)
        weights = np.exp(-rho * np.arange(len(self.corrected)))
        output = weights @ np.asarray(list(reversed(self.corrected))) / weights.sum()
        self.previous_shoulder, self.previous_elbow = shoulder.copy(), elbow.copy()
        self.previous_output = output.copy()
        return output, dict(status='ok', updated=updated, beta=self.beta.copy(),
                            scale=s, offset=h, normalization_length=self.L,
                            min_eigenvalue=minimum_eigenvalue, motion=motion,
                            rho=rho, corrected=corrected.copy())
