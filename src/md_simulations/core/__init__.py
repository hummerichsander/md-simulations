"""Engine-agnostic building blocks shared by all simulation engines."""

# Third-party modules that only the ``simulate`` extra installs. A ModuleNotFoundError
# for one of these gets the install hint; any other missing module is left untouched.
_SIMULATE_MODULES = frozenset({"openmm", "beamtime", "tqdm", "mdtraj", "parmed"})

# Same wording as ``md_simulations.engines.EXTRA_HINT``.
EXTRA_HINT = (
    "md_simulations.core needs the 'simulate' extra: install with "
    "`pip install 'md-simulations[simulate]'` (or `uv sync --extra simulate`)."
)

try:
    from md_simulations.core.equilibration import minimize_and_equilibrate
    from md_simulations.core.freezing import freeze_atoms, resolve_frozen_atoms
    from md_simulations.core.integrators import create_integrator
    from md_simulations.core.logging import RelativeTimeFormatter, setup_logger
    from md_simulations.core.remd import ReplicaExchange, read_checkpoint
    from md_simulations.core.forces import FORCES_FILE, ForceWriter, count_frames, read_forces
    from md_simulations.core.reporters import ForceReporter, add_standard_reporters
    from md_simulations.core.runner import (
        run_simulation_loop,
        save_final_structure,
        save_initial_structure,
        save_structure,
    )
    from md_simulations.core.tracking import (
        BeamtimeReporter,
        record_outputs,
        resume_replica_runs,
        start_replica_runs,
        start_run,
    )
    from md_simulations.core.summary import (
        write_gromacs_native_summary,
        write_simulation_summary,
    )
except ModuleNotFoundError as exc:
    if (exc.name or "").split(".")[0] in _SIMULATE_MODULES:
        raise ModuleNotFoundError(EXTRA_HINT, name=exc.name) from exc
    raise

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
