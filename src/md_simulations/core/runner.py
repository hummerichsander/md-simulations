import logging
from pathlib import Path

import openmm.app as app
from tqdm import tqdm


def run_simulation_loop(
    simulation: app.Simulation,
    num_steps: int,
    output_freq: int,
    desc: str = "MD Progress",
) -> None:
    """Run *simulation* in chunks of *output_freq* steps with a tqdm progress bar.

    Stepping in equal-sized chunks (rather than a single call to
    ``simulation.step(num_steps)``) allows reporters to flush output
    incrementally and keeps the progress bar responsive.

    :param simulation: Simulation to advance. Reporters should already be
        attached via :func:`add_standard_reporters` before this call.
    :param num_steps: Total number of integration steps to run.
    :param output_freq: Chunk size (steps per iteration). A trailing remainder
        smaller than one chunk is run in a final short step.
    :param desc: Label shown on the tqdm progress bar.
    :return: None."""
    num_chunks = num_steps // output_freq
    remainder = num_steps % output_freq
    pbar = tqdm(total=num_chunks + (1 if remainder else 0), desc=desc)
    for _ in range(num_chunks):
        simulation.step(output_freq)
        pbar.update(1)
    if remainder:
        simulation.step(remainder)
        pbar.update(1)
    pbar.close()


def save_structure(
    simulation: app.Simulation,
    path: Path,
    logger: logging.Logger | None = None,
    label: str = "structure",
) -> Path:
    """Write the current simulation frame to *path* as a PDB.

    :param simulation: The simulation whose current positions to write.
    :param path: Destination PDB path.
    :param logger: Optional logger for a confirmation message.
    :param label: Human-readable label for the log message.
    :return: The written PDB path."""
    _log = logger or logging.getLogger(__name__)
    _log.info(f"Saving {label} → {path}")
    positions = simulation.context.getState(getPositions=True).getPositions()
    with open(path, "w") as f:
        app.PDBFile.writeFile(simulation.topology, positions, f)
    return path


def save_initial_structure(
    simulation: app.Simulation,
    output_dir: Path,
    logger: logging.Logger | None = None,
) -> Path:
    """Write the starting frame to ``<output_dir>/initial_structure.pdb``.

    Intended to be called before minimisation/equilibration so it captures the
    true initial coordinates and doubles as a topology reference for the DCD.

    :param simulation: The freshly positioned simulation.
    :param output_dir: Directory to write the PDB into.
    :param logger: Optional logger for a confirmation message.
    :return: The written PDB path."""
    return save_structure(
        simulation, output_dir / "initial_structure.pdb", logger, "initial structure"
    )


def save_final_structure(
    simulation: app.Simulation,
    output_dir: Path,
    logger: logging.Logger | None = None,
) -> Path:
    """Write the final simulation frame to ``<output_dir>/final_structure.pdb``.

    :param simulation: Completed (or paused) simulation.
    :param output_dir: Directory to write the PDB into.
    :param logger: Optional logger for a confirmation message.
    :return: :class:`pathlib.Path` to the written PDB file."""
    return save_structure(
        simulation, output_dir / "final_structure.pdb", logger, "final structure"
    )
