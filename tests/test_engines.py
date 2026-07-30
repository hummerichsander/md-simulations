import logging
from pathlib import Path

import pytest

from md_simulations.config.base import (
    AmberConfig,
    GromacsInputConfig,
    MartiniConfig,
    PureLiquidConfig,
)
from md_simulations.engines import build_engine
from md_simulations.engines.amber import AmberEngine
from md_simulations.engines.gromacs_input import GromacsInputEngine
from md_simulations.engines.martini_openmm import MartiniEngine
from md_simulations.engines.pure_liquid import PureLiquidEngine

_LOG = logging.getLogger("test")
_ROOT = Path("/data")

_CASES = [
    (AmberConfig(system="t", output_subdir="t", input_pdb="a.pdb"), AmberEngine),
    (PureLiquidConfig(system="t", output_subdir="t", input_pdb="a.pdb"), PureLiquidEngine),
    (MartiniConfig(system="t", output_subdir="t", input_gro="a.gro", top="a.top"), MartiniEngine),
    (
        GromacsInputConfig(system="t", output_subdir="t", input_gro="a.gro", input_top="a.top"),
        GromacsInputEngine,
    ),
]


@pytest.mark.parametrize("config,cls", _CASES, ids=[c[0].engine for c in _CASES])
def test_build_engine_dispatch(config, cls: type) -> None:
    """The factory instantiates the engine class matching each config's engine.

    :param config: A config instance (parametrised).
    :param cls: Expected engine class (parametrised).
    :return: None."""
    assert isinstance(build_engine(config, _ROOT, _LOG), cls)


def test_gromacs_input_velocity_flag_from_config() -> None:
    """The gromacs_input engine mirrors initialize_velocities from its config.

    :return: None."""
    on = GromacsInputConfig(
        system="t", output_subdir="t", input_gro="a.gro", input_top="a.top",
        initialize_velocities=True,
    )
    off = GromacsInputConfig(
        system="t", output_subdir="t", input_gro="a.gro", input_top="a.top",
        initialize_velocities=False,
    )
    assert build_engine(on, _ROOT, _LOG).initialize_velocities is True
    assert build_engine(off, _ROOT, _LOG).initialize_velocities is False
