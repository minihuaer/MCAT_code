"""Paper Cartesian mapping in the Piper frame: z up, xy horizontal."""
import numpy as np


def wrist_to_command(wrist_mm, origin_mm, side, config):
    if side not in ('left', 'right'):
        raise ValueError('side must be left or right')
    relative = np.asarray(wrist_mm, dtype=float) - np.asarray(origin_mm, dtype=float)
    if relative.shape != (3,) or not np.isfinite(relative).all():
        raise ValueError('Expected finite wrist/origin XYZ')
    robot = relative[[2, 0, 1]] * [-1, 1, -1] / 1000
    xyz = np.clip(robot * config['gain'] + config['offset_' + side],
                  config['lower'], config['upper'])
    x, y, z = xyz
    yaw, pitch = np.arctan2(y, x), np.pi / 2
    if z < config['height_threshold']:
        low = (np.pi * (config['radial_max'] - np.hypot(x, y))
               / (4 * (config['radial_max'] - config['radial_min'])) + 3*np.pi/4)
        pitch = low + (z - config['lower'][2]) * (np.pi/2 - low) / (
            config['height_threshold'] - config['lower'][2])
        z = max(0.15*np.cos(pitch), min(config['upper'][2], z))
    return dict(x=float(x), y=float(y), z=float(z), roll=0., pitch=float(pitch), yaw=float(yaw))
