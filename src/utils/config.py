import os
import yaml
from typing import Any, Dict


class DotDict(dict):
    """Allows dot access to dictionary keys."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = DotDict(v)


def load_config(config_path: str, overrides: Dict[str, Any] = None) -> DotDict:
    """Loads a YAML configuration file and merges optional overrides."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                keys = k.split(".")
                curr = config_data
                for sub_key in keys[:-1]:
                    curr = curr.setdefault(sub_key, {})
                curr[keys[-1]] = v

    return DotDict(config_data)

