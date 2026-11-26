"""Engine-agnostic building blocks shared by all simulation engines.

Every re-exported name is resolved on first access, not at import, so the submodules that
need no MD stack (``md_simulations.core.forces``, ``core.replica_exchange``,
``core.logging``) import on an install without openmm: ``prep/migrate_forces.py`` and
``prep/cgschnet_dataset.py`` read forces through ``core.forces`` under extras that carry no
openmm. The names backed by openmm, beamtime or tqdm raise a :class:`ModuleNotFoundError`
naming the ``simulate`` extra when accessed without it."""

from importlib import import_module
from typing import Any

# Third-party modules that only the ``simulate`` extra installs. A ModuleNotFoundError
# for one of these gets the install hint; any other missing module is left untouched.
_SIMULATE_MODULES = frozenset({"openmm", "beamtime", "tqdm", "mdtraj"})

# Same wording as ``md_simulations.engines.EXTRA_HINT``.
EXTRA_HINT = (
    "md_simulations.core needs the 'simulate' extra: install with "
    "`pip install 'md-simulations[simulate]'` (or `uv sync --extra simulate`)."
)

__all__ = [
    "EXTRA_HINT",
    "minimize_and_equilibrate",
    "freeze_atoms",
    "resolve_frozen_atoms",
    "create_integrator",
    "RelativeTimeFormatter",
    "setup_logger",
    "ForceReporter",
    "FORCES_FILE",
    "ForceWriter",
    "count_frames",
    "read_forces",
    "add_standard_reporters",
    "run_simulation_loop",
    "save_final_structure",
    "save_initial_structure",
    "save_structure",
    "write_simulation_summary",
    "write_gromacs_native_summary",
    "BeamtimeReporter",
    "record_outputs",
    "ReplicaExchange",
    "read_checkpoint",
    "resume_replica_runs",
    "start_replica_runs",
    "start_run",
]

_C = "md_simulations.core"

# Public name -> (module, attribute). Nothing here is imported until it is looked up.
_LAZY: dict[str, tuple[str, str]] = {
    "minimize_and_equilibrate": (f"{_C}.equilibration", "minimize_and_equilibrate"),
    "freeze_atoms": (f"{_C}.freezing", "freeze_atoms"),
    "resolve_frozen_atoms": (f"{_C}.freezing", "resolve_frozen_atoms"),
    "create_integrator": (f"{_C}.integrators", "create_integrator"),
    "RelativeTimeFormatter": (f"{_C}.logging", "RelativeTimeFormatter"),
    "setup_logger": (f"{_C}.logging", "setup_logger"),
    "ReplicaExchange": (f"{_C}.remd", "ReplicaExchange"),
    "read_checkpoint": (f"{_C}.remd", "read_checkpoint"),
    "FORCES_FILE": (f"{_C}.forces", "FORCES_FILE"),
    "ForceWriter": (f"{_C}.forces", "ForceWriter"),
    "count_frames": (f"{_C}.forces", "count_frames"),
    "read_forces": (f"{_C}.forces", "read_forces"),
    "ForceReporter": (f"{_C}.reporters", "ForceReporter"),
    "add_standard_reporters": (f"{_C}.reporters", "add_standard_reporters"),
    "run_simulation_loop": (f"{_C}.runner", "run_simulation_loop"),
    "save_final_structure": (f"{_C}.runner", "save_final_structure"),
    "save_initial_structure": (f"{_C}.runner", "save_initial_structure"),
    "save_structure": (f"{_C}.runner", "save_structure"),
    "BeamtimeReporter": (f"{_C}.tracking", "BeamtimeReporter"),
    "record_outputs": (f"{_C}.tracking", "record_outputs"),
    "resume_replica_runs": (f"{_C}.tracking", "resume_replica_runs"),
    "start_replica_runs": (f"{_C}.tracking", "start_replica_runs"),
    "start_run": (f"{_C}.tracking", "start_run"),
    "write_gromacs_native_summary": (f"{_C}.summary", "write_gromacs_native_summary"),
    "write_simulation_summary": (f"{_C}.summary", "write_simulation_summary"),
}


def __getattr__(name: str) -> Any:
    """Resolve a re-exported name on first access.

    :param name: Attribute name being looked up.
    :return: The resolved object.
    :raises AttributeError: If ``name`` is not an attribute of this module.
    :raises ModuleNotFoundError: With :data:`EXTRA_HINT` if the ``simulate`` extra is missing."""
    if (target := _LAZY.get(name)) is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module, attr = target
    try:
        resolved = getattr(import_module(module), attr)
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] in _SIMULATE_MODULES:
            raise ModuleNotFoundError(EXTRA_HINT, name=exc.name) from exc
        raise

    globals()[name] = resolved  # cache so __getattr__ runs once per name
    return resolved


def __dir__() -> list[str]:
    """List the module's public names, including the lazily-resolved ones.

    :return: Sorted public attribute names."""
    return sorted(__all__)
