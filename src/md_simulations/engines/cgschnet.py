"""CGSchNet / mlcg machine-learned coarse-grained MD.

The trained model (SchNet + prior, e.g. the Clementi-group transferable model
from Charron et al., Nat. Chem. 2025) consumes ``mlcg.AtomicData`` and is not
TorchScriptable, so it cannot be run through OpenMM-Torch (whose ``TorchForce``
executes TorchScript in a Python-free C++ context). Instead this engine drives
dynamics with mlcg's own Langevin integrator, then converts its output into the
same ``trajectory.dcd`` / ``energies.csv`` / ``forces.txt`` / PDB / summary files
every other engine produces.

Two non-obvious requirements (see ``md-sim-cgschnet-reexport`` and the module
notes): the distributed checkpoint must be re-exported for the current
torch-geometric, and ``LangevinSimulation`` is constructed with
``specialize_priors=True`` so zero-interaction prior terms are pruned. Runs in
the model's native units (Ångström / kcal·mol⁻¹); positions and energies are
converted to the OpenMM convention (nm, kJ·mol⁻¹) on output."""

import copy
import logging
from pathlib import Path

import numpy as np

from md_simulations.config import CGSchNetConfig
from md_simulations.engines.base import Engine

_KCAL_TO_KJ = 4.184
_ANG_TO_NM = 0.1
# Boltzmann constant in the model's energy unit, per kelvin.
_KB = {"kcal/mol": 0.0019872041, "kJ/mol": 0.00831446261815324}


def _preload_nvrtc_libraries() -> None:
    """Globally preload the NVRTC libraries torch's JIT fuser dlopens at runtime.

    torch's kernel fuser NVRTC-compiles kernels on the fly and needs
    ``libnvrtc.so.13`` + ``libnvrtc-builtins.so.13.0``, which ship in pip's
    ``nvidia-*`` packages but sit outside torch's RPATH, so load them with
    ``RTLD_GLOBAL`` up front. A no-op when absent (e.g. a conda install that
    links correctly)."""
    import ctypes
    import sysconfig

    rel_libs = (
        "nvidia/cu13/lib/libnvrtc.so.13",
        "nvidia/cu13/lib/libnvrtc-builtins.so.13.0",
    )
    site_dirs = {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}
    for base in site_dirs:
        for rel in rel_libs:
            lib = Path(base) / rel
            if lib.exists():
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def _build_output_topology(pdb_path: Path, n_beads: int, logger: logging.Logger):
    """Return an mdtraj topology for trajectory output.

    Uses *pdb_path* when its atom count matches the CG bead count; otherwise
    synthesizes a generic single-chain topology of ``n_beads`` carbon beads.

    :param pdb_path: Candidate CG PDB (one atom per bead).
    :param n_beads: Number of CG beads in the simulation.
    :param logger: Logger for a fallback warning.
    :return: An :class:`mdtraj.Topology`."""
    import mdtraj as md

    if pdb_path.exists():
        top = md.load(str(pdb_path)).topology
        if top.n_atoms == n_beads:
            return top
        logger.warning(
            f"input_pdb has {top.n_atoms} atoms != {n_beads} CG beads; "
            "synthesizing a generic bead topology for output."
        )
    top = md.Topology()
    chain = top.add_chain()
    for _ in range(n_beads):
        residue = top.add_residue("CG", chain)
        top.add_atom("CA", md.element.carbon, residue)
    return top


def _replica_suffix(index: int, n_replicas: int) -> str:
    """Filename suffix for a replica: empty for a single replica, else zero-padded."""
    if n_replicas == 1:
        return ""
    width = max(2, len(str(n_replicas - 1)))
    return f"_{index:0{width}d}"


class _IncrementalTrajectoryWriter:
    """Stream mlcg's export chunks into the standard output files during the run.

    mlcg flushes its accumulated arrays to ``{prefix}_{kind}_NNNN.npy`` every
    ``export_interval`` steps and then calls this as its ``save_subroutine``.
    Each call appends any newly-written chunk to per-replica ``trajectory.dcd``
    (via mdtraj's incremental DCD writer, in the model's native Ångström),
    ``energies.csv`` and ``forces.txt`` — so output grows as the simulation
    proceeds instead of only at the end. With multiple replicas each gets its own
    suffixed files; a single replica keeps the plain names. mlcg's final remainder
    ``write()`` does not trigger the hook, so :meth:`close` must be called
    afterwards to flush it.

    mlcg's ``write()`` always dumps the full fixed-size export buffer, so the
    final chunk carries stale rows past the true frame count; chunks are therefore
    truncated to ``n_saved`` total frames.

    :param output_dir: Destination directory for the standard output files.
    :param raw_dir: Directory mlcg writes its raw ``.npy`` chunks into.
    :param prefix: Filename prefix passed to the simulation.
    :param output_freq: Steps between saved frames (for the energies step column).
    :param energy_to_kj: Factor converting the model energy unit to kJ·mol⁻¹.
    :param n_saved: Total number of valid saved frames over the whole run.
    :param n_replicas: Number of parallel replicas (first array dimension)."""

    def __init__(
        self,
        output_dir: Path,
        raw_dir: Path,
        prefix: str,
        output_freq: int,
        energy_to_kj: float,
        n_saved: int,
        n_replicas: int,
    ) -> None:
        import mdtraj as md

        self._raw_dir = raw_dir
        self._prefix = prefix
        self._output_freq = output_freq
        self._energy_to_kj = energy_to_kj
        self._n_saved = n_saved
        self._n_replicas = n_replicas
        self._consumed = 0
        self._frames = 0
        self.last_xyz_ang: np.ndarray | None = None  # (n_replicas, N, 3)

        self._dcd, self._energies, self._forces = [], [], []
        for r in range(n_replicas):
            sfx = _replica_suffix(r, n_replicas)
            self._dcd.append(
                md.formats.DCDTrajectoryFile(str(output_dir / f"trajectory{sfx}.dcd"), "w")
            )
            energies = open(output_dir / f"energies{sfx}.csv", "w")
            energies.write('#"Step","Potential Energy (kJ/mole)"\n')
            self._energies.append(energies)
            self._forces.append(open(output_dir / f"forces{sfx}.txt", "w"))

    def __call__(self, data=None, frame_index=None) -> None:
        """mlcg ``save_subroutine`` hook — flush chunks written since the last call."""
        self.flush()

    def flush(self) -> None:
        """Append every not-yet-consumed export chunk to the per-replica files."""
        chunks = sorted(self._raw_dir.glob(f"{self._prefix}_coords_*.npy"))
        for chunk in chunks[self._consumed :]:
            self._consumed += 1
            # mlcg pads the last chunk with stale rows; keep only the frames still
            # owed against the true total.
            take = min(np.load(chunk).shape[1], self._n_saved - self._frames)
            if take <= 0:
                continue
            coords = np.load(chunk)[:, :take]  # (n_replicas, frames, N, 3) in Å
            self.last_xyz_ang = coords[:, -1]

            potential_file = chunk.with_name(chunk.name.replace("_coords_", "_potential_"))
            potential = (
                np.load(potential_file)[:, :take] * self._energy_to_kj
                if potential_file.exists()
                else None
            )
            forces_file = chunk.with_name(chunk.name.replace("_coords_", "_forces_"))
            # kcal·mol⁻¹·Å⁻¹ → kJ·mol⁻¹·nm⁻¹.
            forces = (
                np.load(forces_file)[:, :take] * (self._energy_to_kj / _ANG_TO_NM)
                if forces_file.exists()
                else None
            )

            for r in range(self._n_replicas):
                self._dcd[r].write(coords[r])  # DCD stores Ångström
                if potential is not None:
                    for offset, energy in enumerate(potential[r]):
                        step = (self._frames + offset + 1) * self._output_freq
                        self._energies[r].write(f"{step},{energy}\n")
                if forces is not None:
                    for frame in forces[r]:
                        self._forces[r].write(" ".join(f"{c:g}" for c in frame.reshape(-1)) + "\n")
                self._energies[r].flush()
                self._forces[r].flush()

            self._frames += take

    def close(self) -> None:
        """Flush the final remainder chunk (hook is skipped for it) and close files."""
        self.flush()
        for r in range(self._n_replicas):
            self._dcd[r].close()
            self._energies[r].close()
            self._forces[r].close()


class CGSchNetEngine(Engine):
    """ML coarse-grained MD via mlcg's Langevin integrator (not OpenMM)."""

    config: CGSchNetConfig

    def run(self) -> Path:
        _preload_nvrtc_libraries()
        import torch

        from mlcg.simulation import LangevinSimulation

        cfg = self.config
        output_dir = cfg.output_dir(self.data_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = cfg.resolve(self.data_root, cfg.model_file)
        configs_path = cfg.resolve(self.data_root, cfg.configurations_file)

        self.logger.info(f"Loading model:          {model_path}")
        model = torch.load(str(model_path), map_location=cfg.device, weights_only=False)
        self.logger.info(f"Loading configurations: {configs_path}")
        configurations = torch.load(str(configs_path), map_location="cpu", weights_only=False)
        if not isinstance(configurations, (list, tuple)):
            configurations = [configurations]
        n_available = len(configurations)
        if not 0 <= cfg.replica < n_available:
            raise ValueError(
                f"replica {cfg.replica} out of range for {n_available} configurations."
            )
        if cfg.n_replicas < 1:
            raise ValueError(f"n_replicas must be >= 1 (got {cfg.n_replicas}).")
        # Take n_replicas starting structures from the list, tiling if necessary.
        # mlcg batches them into one forward per step — the throughput lever.
        # deepcopy each: when tiling repeats a configuration, mlcg's prior
        # specialization deepcopies the whole list (preserving aliasing) and pops
        # neighbor-list keys per element, so a shared reference would be popped
        # twice and raise KeyError. Independent copies also avoid mutating the
        # loaded configurations in place.
        initial = [
            copy.deepcopy(configurations[(cfg.replica + i) % n_available])
            for i in range(cfg.n_replicas)
        ]
        if cfg.n_replicas > n_available:
            self.logger.warning(
                f"n_replicas ({cfg.n_replicas}) exceeds available configurations "
                f"({n_available}); some starting structures are reused (decorrelated "
                "only by independent random velocities)."
            )
        n_beads = int(initial[0].pos.shape[0])
        topology = _build_output_topology(
            cfg.resolve(self.data_root, cfg.input_pdb), n_beads, self.logger
        )

        # Write the starting structures and parameter summary up front (before the
        # run) so the folder is populated immediately — as the OpenMM engines do —
        # rather than only once the whole simulation finishes.
        self._write_initial_structures(output_dir, initial, topology)
        self._write_summary(output_dir)

        beta = 1.0 / (_KB[cfg.model_energy_unit] * cfg.temperature)
        n_saved = cfg.num_steps // cfg.output_freq
        raw_dir = output_dir / "_mlcg_raw"
        raw_dir.mkdir(exist_ok=True)
        prefix = "sim"
        energy_to_kj = _KCAL_TO_KJ if cfg.model_energy_unit == "kcal/mol" else 1.0

        # Flush mlcg's raw arrays to disk in chunks (bounded memory), and stream
        # each chunk into the standard output files via a save_subroutine so the
        # trajectory/energies/forces grow during the run rather than only at the end.
        export_interval = min(cfg.num_steps, cfg.output_freq * 1000)
        writer = _IncrementalTrajectoryWriter(
            output_dir, raw_dir, prefix, cfg.output_freq, energy_to_kj, n_saved, cfg.n_replicas
        )

        self.logger.info(
            f"Running cgschnet MD (mlcg Langevin) for {cfg.num_steps:,} steps "
            f"(dt={cfg.timestep} ps, T={cfg.temperature} K, {n_beads} beads, "
            f"{cfg.n_replicas} replica(s), {n_saved} frames each)..."
        )
        # The writer holds all per-replica output files open for the whole run;
        # close it in a finally so a mid-run failure (e.g. in prior specialization)
        # doesn't leak open file handles.
        try:
            simulation = LangevinSimulation(
                friction=cfg.friction,
                dt=cfg.timestep,
                n_timesteps=cfg.num_steps,
                save_interval=cfg.output_freq,
                save_forces=True,
                save_energies=True,
                specialize_priors=True,
                device=cfg.device,
                dtype="double" if cfg.dtype == "float64" else "single",
                filename=str(raw_dir / prefix),
                export_interval=export_interval,
                save_subroutine=writer,
            )
            simulation.attach_model_and_configurations(model, initial, beta=beta)
            simulation.simulate()
        finally:
            writer.close()

        if writer.last_xyz_ang is not None:
            self._write_final_structures(output_dir, writer.last_xyz_ang, topology)
        self.logger.info(f"Trajectory output → {output_dir}")
        self.logger.info("Simulation completed successfully!")
        self.logger.info(f"Output files: {output_dir}")
        return output_dir

    def _write_initial_structures(self, output_dir: Path, initial: list, topology) -> None:
        """Write each replica's starting frame (step 0) as ``initial_structure*.pdb``.

        :param output_dir: Destination directory.
        :param initial: The per-replica starting ``AtomicData`` (positions in Ångström).
        :param topology: The mdtraj output topology."""
        import mdtraj as md

        n = len(initial)
        for r, data in enumerate(initial):
            xyz_nm = data.pos.detach().cpu().numpy()[None] * _ANG_TO_NM
            pdb = output_dir / f"initial_structure{_replica_suffix(r, n)}.pdb"
            md.Trajectory(xyz_nm, topology).save_pdb(str(pdb))
        self.logger.info(f"Initial structure(s) → {output_dir}")

    def _write_final_structures(self, output_dir: Path, xyz_ang: np.ndarray, topology) -> None:
        """Write each replica's last simulated frame as ``final_structure*.pdb``.

        :param output_dir: Destination directory.
        :param xyz_ang: Last-frame positions ``(n_replicas, N, 3)`` in Ångström.
        :param topology: The mdtraj output topology."""
        import mdtraj as md

        n = len(xyz_ang)
        for r in range(n):
            xyz_nm = xyz_ang[r][None] * _ANG_TO_NM
            pdb = output_dir / f"final_structure{_replica_suffix(r, n)}.pdb"
            md.Trajectory(xyz_nm, topology).save_pdb(str(pdb))

    def _write_summary(self, output_dir: Path) -> None:
        """Write a concise parameter summary matching the other engines' output."""
        cfg = self.config
        sim_time_ns = cfg.num_steps * cfg.timestep / 1000.0
        lines = [
            "CGSchNet / mlcg coarse-grained MD",
            "=" * 40,
            f"System:           {cfg.system}",
            f"Model:            {cfg.model_file}",
            f"Configurations:   {cfg.configurations_file}",
            f"Replicas:         {cfg.n_replicas} (start index {cfg.replica})",
            "Integrator:       mlcg Langevin (BAOAB), specialized priors",
            f"Temperature:      {cfg.temperature} K",
            f"Friction:         {cfg.friction} ps^-1",
            f"Timestep:         {cfg.timestep} ps",
            f"Steps:            {cfg.num_steps:,}",
            f"Simulated time:   {sim_time_ns:.3f} ns",
            f"Save interval:    every {cfg.output_freq} steps",
            "Units:            model Å/kcal·mol⁻¹ → output nm/kJ·mol⁻¹",
        ]
        (output_dir / "simulation_summary.txt").write_text("\n".join(lines) + "\n")
