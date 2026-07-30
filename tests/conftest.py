import json
from pathlib import Path
from typing import Callable

import pytest
import yaml

# Minimal, self-contained config payloads (one per engine). Tests build configs
# from these fixtures rather than depending on the shipped configs/ files.
SAMPLE_CONFIGS: dict[str, dict] = {
    "amber": {
        "engine": "amber",
        "system": "t",
        "output_subdir": "t/run",
        "input_pdb": "a.pdb",
    },
    "pure_liquid": {
        "engine": "pure_liquid",
        "system": "t",
        "output_subdir": "t/run",
        "input_pdb": "a.pdb",
    },
    "martini": {
        "engine": "martini",
        "system": "t",
        "output_subdir": "t/run",
        "input_gro": "a.gro",
        "top": "a.top",
    },
    "gromacs_input": {
        "engine": "gromacs_input",
        "system": "t",
        "output_subdir": "t/run",
        "input_gro": "a.gro",
        "input_top": "a.top",
    },
    "awsem": {
        "engine": "awsem",
        "system": "t",
        "output_subdir": "t/run",
        "input_pdb": "a.pdb",
    },
    "cgschnet": {
        "engine": "cgschnet",
        "system": "t",
        "output_subdir": "t/run",
        "input_pdb": "cg.pdb",
        "model_file": "m.pt",
        "configurations_file": "c.pt",
    },
    "gromacs_native": {
        "engine": "gromacs_native",
        "system": "t",
        "output_subdir": "t/run",
        "input_gro": "a.gro",
        "top": "a.top",
        "mdp": "m.mdp",
    },
}

# Resource block merged into a sample when a config needs a SLURM section.
SAMPLE_SLURM = {"job_name": "md-t", "partition": "a30", "gres": "gpu:1"}


@pytest.fixture
def sample_configs() -> dict[str, dict]:
    """Return per-engine sample config payloads (deep-copied per test).

    :return: Mapping of engine name to a config dict."""
    return {k: dict(v) for k, v in SAMPLE_CONFIGS.items()}


@pytest.fixture
def write_yaml(tmp_path: Path) -> Callable[[dict, str], Path]:
    """Return a factory that serialises a config dict to a YAML file.

    :param tmp_path: Pytest temporary directory.
    :return: ``write(data, name="config.yaml") -> Path`` helper."""

    def _write(data: dict, name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data))
        return path

    return _write


@pytest.fixture
def cg_mapping_file(tmp_path: Path) -> Path:
    """Write a tiny CGSchNet slicing-map JSON and return its path.

    :param tmp_path: Pytest temporary directory.
    :return: Path to the mapping JSON."""
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"beads": [["ALA", "CA", 1], ["GLY", "CA", 8]]}))
    return path
