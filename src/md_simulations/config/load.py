import os
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from md_simulations.config.base import SimulationConfig

_ADAPTER: TypeAdapter = TypeAdapter(SimulationConfig)


def load_config(path: str | Path) -> SimulationConfig:
    """Load and validate a simulation config from a YAML file.

    The concrete config type is selected by the ``engine`` discriminator field.

    :param path: Path to the YAML config file.
    :return: A validated engine-specific config model."""
    data = yaml.safe_load(Path(path).read_text())
    return _ADAPTER.validate_python(data)


def resolve_data_root(config: SimulationConfig, cli_data_root: str | None = None) -> Path:
    """Resolve the data root by precedence: CLI > ``MD_DATA_ROOT`` > config.

    :param config: The loaded simulation config.
    :param cli_data_root: Value passed on the command line (highest priority).
    :return: The resolved data-root directory.
    :raises ValueError: If no data root is provided by any source."""
    if cli_data_root:
        return Path(cli_data_root).expanduser()
    if env := os.environ.get("MD_DATA_ROOT"):
        return Path(env).expanduser()
    if config.data_root:
        return Path(config.data_root).expanduser()
    raise ValueError(
        "No data root: pass --data-root, set MD_DATA_ROOT, or set data_root in the config."
    )
