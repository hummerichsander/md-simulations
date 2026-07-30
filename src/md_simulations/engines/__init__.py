"""Simulation engines and the factory that dispatches configs to them."""

import logging
from pathlib import Path

from md_simulations.config.base import SimulationConfig
from md_simulations.engines.base import BuiltSystem, Engine, OpenMMEngine

__all__ = ["BuiltSystem", "Engine", "OpenMMEngine", "build_engine"]


def build_engine(config: SimulationConfig, data_root: Path, logger: logging.Logger) -> Engine:
    """Instantiate the engine matching ``config.engine``.

    Engines with optional/heavy dependencies (AWSEM, native GROMACS) are
    imported lazily so the core install stays light.

    :param config: A validated engine-specific config model.
    :param data_root: Root directory for resolving input/output paths.
    :param logger: Logger for progress messages.
    :return: The instantiated :class:`Engine`."""
    match config.engine:
        case "amber":
            from md_simulations.engines.amber import AmberEngine

            return AmberEngine(config, data_root, logger)
        case "pure_liquid":
            from md_simulations.engines.pure_liquid import PureLiquidEngine

            return PureLiquidEngine(config, data_root, logger)
        case "martini":
            from md_simulations.engines.martini_openmm import MartiniEngine

            return MartiniEngine(config, data_root, logger)
        case "gromacs_input":
            from md_simulations.engines.gromacs_input import GromacsInputEngine

            return GromacsInputEngine(config, data_root, logger)
        case "awsem":
            from md_simulations.engines.awsem import AwsemEngine

            return AwsemEngine(config, data_root, logger)
        case "cgschnet":
            from md_simulations.engines.cgschnet import CGSchNetEngine

            return CGSchNetEngine(config, data_root, logger)
        case "gromacs_native":
            from md_simulations.engines.gromacs_native import GromacsNativeEngine

            return GromacsNativeEngine(config, data_root, logger)
        case other:  # pragma: no cover - guarded by pydantic discriminator
            raise ValueError(f"Unknown engine: {other!r}")
