from pathlib import Path
from typing import Callable

import pytest
from pydantic import ValidationError

from md_simulations.config import load_config, resolve_data_root
from md_simulations.config.base import AmberConfig, SlurmConfig
from md_simulations.workspace import Workspace

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


def test_data_root_workspace_between_env_and_config(tmp_path: Path, monkeypatch) -> None:
    """The workspace data root sits below MD_DATA_ROOT and above the config value.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :return: None."""
    config = AmberConfig(
        system="t", output_subdir="t/run", input_pdb="a.pdb", data_root=str(tmp_path / "cfg")
    )
    workspace = Workspace(root=tmp_path, data_root="data")
    monkeypatch.delenv("MD_DATA_ROOT", raising=False)
    assert resolve_data_root(config, None, workspace) == tmp_path / "data"
    assert resolve_data_root(config, None) == tmp_path / "cfg"
    monkeypatch.setenv("MD_DATA_ROOT", str(tmp_path / "env"))
    assert resolve_data_root(config, None, workspace) == tmp_path / "env"
    assert resolve_data_root(config, str(tmp_path / "cli"), workspace) == tmp_path / "cli"


def test_data_root_error_names_workspace(monkeypatch) -> None:
    """With no source at all the error mentions md-workspace.yaml.

    :param monkeypatch: Pytest monkeypatch fixture.
    :return: None."""
    monkeypatch.delenv("MD_DATA_ROOT", raising=False)
    config = AmberConfig(system="t", output_subdir="t/run", input_pdb="a.pdb")
    with pytest.raises(ValueError, match="md-workspace.yaml"):
        resolve_data_root(config)


def test_output_dir_resolution(tmp_path: Path) -> None:
    """output_dir joins the data root with the config's output_subdir.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    config = AmberConfig(system="t", output_subdir="t/run", input_pdb="a.pdb")
    assert config.output_dir(tmp_path) == tmp_path / "t" / "run"


def test_constraints_defaults_to_hbonds() -> None:
    """The constraints scheme defaults to the value every shipped trajectory used.

    :return: None."""
    config = AmberConfig(system="t", output_subdir="t/run", input_pdb="a.pdb")
    assert config.constraints == "hbonds"


@pytest.mark.parametrize("kind", ["hbonds", "allbonds", "none"])
def test_constraints_round_trips(
    kind: str, sample_configs: dict[str, dict], write_yaml: Callable[..., Path]
) -> None:
    """Every accepted constraints scheme survives a YAML round-trip.

    Spelled as a string rather than a null so an unquoted ``none`` in YAML cannot be mistaken
    for a missing value.

    :param kind: Constraint scheme (parametrised).
    :param sample_configs: Per-engine sample payloads fixture.
    :param write_yaml: YAML-writing factory fixture.
    :return: None."""
    payload = sample_configs["amber"] | {"constraints": kind}
    assert load_config(write_yaml(payload)).constraints == kind


def test_constraints_rejects_an_unknown_scheme(
    sample_configs: dict[str, dict], write_yaml: Callable[..., Path]
) -> None:
    """A misspelled constraints scheme fails at load rather than silently at build time.

    :param sample_configs: Per-engine sample payloads fixture.
    :param write_yaml: YAML-writing factory fixture.
    :return: None."""
    payload = sample_configs["amber"] | {"constraints": "hangles"}
    with pytest.raises(ValidationError):
        load_config(write_yaml(payload))


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_fixture_config_loads() -> None:
    """The vendored trp-cage CGSchNet config parses with no data root and no mlcg.

    Pins the fixture against drift from the ``simulations`` repo's config: the engine, the
    replica count and the bead count of the vendored PDB are what the cgschnet regression
    tests assume.

    :return: None."""
    config = load_config(FIXTURES / "trpcage-cgschnet.yaml")
    assert config.engine == "cgschnet"
    assert config.input_pdb == "pdbs/2JOF_5B-CG.pdb"
    assert config.model_energy_unit == "kcal/mol"
    assert config.n_replicas == 10
    pdb = (FIXTURES / "2JOF_5B-CG.pdb").read_text().splitlines()
    assert sum(line.startswith(("ATOM", "HETATM")) for line in pdb) == 97
