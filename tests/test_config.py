from pathlib import Path
from typing import Callable

import pytest

from md_simulations.config import load_config, resolve_data_root
from md_simulations.config.base import AmberConfig, SlurmConfig

from conftest import SAMPLE_CONFIGS, SAMPLE_SLURM


@pytest.mark.parametrize("engine", sorted(SAMPLE_CONFIGS))
def test_load_dispatches_on_engine(
    engine: str, sample_configs: dict[str, dict], write_yaml: Callable[..., Path]
) -> None:
    """load_config selects the right engine model from the discriminator field.

    :param engine: Engine name (parametrised).
    :param sample_configs: Per-engine sample payloads fixture.
    :param write_yaml: YAML-writing factory fixture.
    :return: None."""
    config = load_config(write_yaml(sample_configs[engine]))
    assert config.engine == engine
    assert config.system == "t"
    assert config.output_subdir == "t/run"


def test_slurm_block_parses(
    sample_configs: dict[str, dict], write_yaml: Callable[..., Path]
) -> None:
    """A YAML slurm block round-trips into a SlurmConfig.

    :param sample_configs: Per-engine sample payloads fixture.
    :param write_yaml: YAML-writing factory fixture.
    :return: None."""
    payload = sample_configs["amber"] | {"slurm": dict(SAMPLE_SLURM)}
    config = load_config(write_yaml(payload))
    assert isinstance(config.slurm, SlurmConfig)
    assert config.slurm.job_name == "md-t"


def test_data_root_precedence(tmp_path: Path, monkeypatch) -> None:
    """CLI data root wins over MD_DATA_ROOT and the config value.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :return: None."""
    config = AmberConfig(system="t", output_subdir="t/run", input_pdb="a.pdb")
    monkeypatch.setenv("MD_DATA_ROOT", str(tmp_path / "env"))
    assert resolve_data_root(config, str(tmp_path / "cli")) == tmp_path / "cli"
    assert resolve_data_root(config, None) == tmp_path / "env"


def test_output_dir_resolution(tmp_path: Path) -> None:
    """output_dir joins the data root with the config's output_subdir.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    config = AmberConfig(system="t", output_subdir="t/run", input_pdb="a.pdb")
    assert config.output_dir(tmp_path) == tmp_path / "t" / "run"
