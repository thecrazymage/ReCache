"""Config loading."""

import yaml
from munch import Munch, munchify


def config_from_path(path: str) -> Munch:
    """Load a YAML config as a `Munch` for attribute-style access."""
    with open(path, 'r') as f:
        return munchify(yaml.safe_load(f))
