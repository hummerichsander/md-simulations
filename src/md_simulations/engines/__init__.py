"""Simulation engines and the factory that dispatches configs to them."""

import logging
from pathlib import Path

from md_simulations.config.base import SimulationConfig

# Third-party modules that only the ``simulate`` extra installs. A ModuleNotFoundError
# for one of these gets the install hint; any other missing module is left untouched.
_SIMULATE_MODULES = frozenset({"openmm", "beamtime", "tqdm", "mdtraj", "parmed"})

# Shared with the CLI and mirrored by ``torch_potentials.build.EXTRA_HINT``, so the
# guidance is identical whichever entry point the user came through.
EXTRA_HINT = (
    "md_simulations.engines needs the 'simulate' extra: install with "
    "`pip install 'md-simulations[simulate]'` (or `uv sync --extra simulate`)."
)

try:
    from md_simulations.engines.base import BuiltSystem, Engine, OpenMMEngine
except ModuleNotFoundError as exc:
    if (exc.name or "").split(".")[0] in _SIMULATE_MODULES:
        raise ModuleNotFoundError(EXTRA_HINT, name=exc.name) from exc
    raise

__all__ = ["EXTRA_HINT", "BuiltSystem", "Engine", "OpenMMEngine", "build_engine"]


def build_engine(
    config: SimulationConfig, data_root: Path, logger: logging.Logger, resume: bool = False
) -> Engine:
    """Instantiate the engine matching ``config.engine``.

    Engines with optional/heavy dependencies (AWSEM, native GROMACS) are
    imported lazily so the core install stays light.

    :param config: A validated engine-specific config model.
    :param data_root: Root directory for resolving input/output paths.
    :param logger: Logger for progress messages.
    :param resume: Continue from a checkpoint (replica exchange only).
    :return: The instantiated :class:`Engine`."""
    match config.engine:
        case "amber":
            from md_simulations.engines.amber import AmberEngine

            return AmberEngine(config, data_root, logger, resume)
        case "pure_liquid":
            from md_simulations.engines.pure_liquid import PureLiquidEngine

            return PureLiquidEngine(config, data_root, logger, resume)
        case "martini":
            from md_simulations.engines.martini_openmm import MartiniEngine

            return MartiniEngine(config, data_root, logger, resume)
        case "gromacs_input":
            from md_simulations.engines.gromacs_input import GromacsInputEngine

            return GromacsInputEngine(config, data_root, logger, resume)
        case "awsem":
            from md_simulations.engines.awsem import AwsemEngine

            return AwsemEngine(config, data_root, logger, resume)
        case "cgschnet":
            from md_simulations.engines.cgschnet import CGSchNetEngine

            return CGSchNetEngine(config, data_root, logger, resume)
        case "gromacs_native":
            from md_simulations.engines.gromacs_native import GromacsNativeEngine

            return GromacsNativeEngine(config, data_root, logger, resume)
        case other:  # pragma: no cover - guarded by pydantic discriminator
            raise ValueError(f"Unknown engine: {other!r}")
