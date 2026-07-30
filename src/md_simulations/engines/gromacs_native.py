import shutil
import subprocess
from pathlib import Path

from md_simulations.config import GromacsNativeConfig
from md_simulations.core import write_gromacs_native_summary
from md_simulations.engines.base import Engine


def _parse_mdp(path: Path) -> dict[str, str]:
    """Parse a GROMACS ``.mdp`` file into a lower-cased key/value dict.

    Comments (``;``) are stripped; keys keep their original ``-``/``_`` form
    lower-cased.

    :param path: Path to the mdp file.
    :return: Mapping of option name to value string."""
    params: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.split(";", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        params[key.strip().lower()] = value.strip()
    return params


class GromacsNativeEngine(Engine):
    """Native GROMACS MD via ``gmx grompp`` + ``gmx mdrun``.

    Wraps the standard two-step GROMACS flow: preprocess (mdp + gro + top →
    tpr) then run. Requires a ``gmx`` binary on PATH (e.g. ``module load
    gromacs/...`` in the SLURM job). Trajectories are written in GROMACS
    formats (``.xtc``/``.trr``/``.edr``) inside the output directory."""

    config: GromacsNativeConfig

    def _save_initial_frame(self, gro: Path, output_dir: Path) -> int | None:
        """Write the input GRO's starting frame to ``initial_structure.pdb``.

        Uses mdtraj so no ``gmx`` call is needed for the conversion.

        :param gro: Path to the input GRO coordinate file.
        :param output_dir: Run output directory.
        :return: The atom count, or None if the conversion failed."""
        import mdtraj as md

        try:
            frame = md.load(str(gro))
            frame.save_pdb(str(output_dir / "initial_structure.pdb"))
            self.logger.info(f"Saving initial structure → {output_dir / 'initial_structure.pdb'}")
            return frame.n_atoms
        except Exception as e:
            self.logger.warning(f"Could not write initial_structure.pdb from {gro}: {e}")
            return None

    def _gmx(self, *cmd: str, cwd: Path) -> None:
        """Run a ``gmx`` subcommand, streaming output and raising on failure.

        :param cmd: The ``gmx`` subcommand and its arguments.
        :param cwd: Working directory for the call.
        :return: None."""
        full = [self.config.gmx, *cmd]
        self.logger.info(f"$ {' '.join(full)}  (cwd={cwd})")
        subprocess.run(full, cwd=cwd, check=True)

    def run(self) -> Path:
        cfg = self.config
        if shutil.which(cfg.gmx) is None:
            raise RuntimeError(
                f"GROMACS binary '{cfg.gmx}' not found on PATH. "
                "Load it first (e.g. `module load gromacs/...`)."
            )

        output_dir = cfg.output_dir(self.data_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        gro = cfg.resolve(self.data_root, cfg.input_gro)
        top = cfg.resolve(self.data_root, cfg.top)
        mdp = cfg.resolve(self.data_root, cfg.mdp)
        tpr = output_dir / "topol.tpr"

        # Save the starting frame and a parameter summary up front, before the run.
        n_atoms = self._save_initial_frame(gro, output_dir)
        write_gromacs_native_summary(output_dir, cfg, _parse_mdp(mdp), n_atoms, self.logger)

        grompp = ["grompp", "-f", str(mdp), "-c", str(gro), "-p", str(top), "-o", str(tpr)]
        if cfg.index is not None:
            grompp += ["-n", str(cfg.resolve(self.data_root, cfg.index))]
        self.logger.info("Preprocessing with grompp...")
        self._gmx(*grompp, cwd=output_dir)

        self.logger.info(f"Running mdrun for up to {cfg.num_steps:,} steps (nsteps from mdp)...")
        mdrun = ["mdrun", "-s", str(tpr), "-deffnm", "md", *cfg.mdrun_extra]
        self._gmx(*mdrun, cwd=output_dir)

        self.logger.info("Simulation completed successfully!")
        self.logger.info(f"Output files: {output_dir}")
        return output_dir
