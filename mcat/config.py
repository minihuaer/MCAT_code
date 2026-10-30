"""Repository-relative runtime settings and asset validation."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/teleop.yaml'


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    # Asset paths are always relative to the release root, independent of cwd.
    for key, value in config['weights'].items():
        value = Path(value).expanduser()
        config['weights'][key] = value if value.is_absolute() else ROOT / value
    value = Path(config['kdu_config']).expanduser()
    config['kdu_config'] = value if value.is_absolute() else ROOT / value
    if config['camera']['focal'][0] <= 0 or config['camera']['focal'][1] <= 0:
        raise ValueError('Camera focal lengths must be positive')
    return config


def require_assets(config):
    paths = dict(config['weights'], kdu_config=config['kdu_config'])
    missing = [f'{key}: {path}' for key, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing release assets (see weights/README.md):\n' + '\n'.join(missing))


def validate_max_frames(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise ValueError('max_frames must be a positive integer or None')
    return value
