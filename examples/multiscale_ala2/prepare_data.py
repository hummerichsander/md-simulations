#!/usr/bin/env python3
"""Build the notebook's reference frames: subsample a T-REMD run, then relax it unconstrained.

The shipped trajectory came from ``configs/ala2-amber14-implicit-remd-330K-train.yaml`` --
101 ns of temperature replica exchange whose 330 K rung is a proper canonical ensemble at
330 K, but which crosses the alpha_L barrier far more often than plain MD does (alpha_L is
1.7 % of frames here against 0.0004 % in a 10 ns plain run, so a plain run cannot show that
basin at all).

That run was integrated with ``constraints: hbonds``, which freezes all twelve X-H bonds at
their ideal length and **removes their energy term from the force field**. The potential the
notebook differentiates has ``constraints: none``, so every degree of freedom is in it -- and
the constrained frames are not samples of it: their X-H lengths are a delta at r0 rather than
a thermal distribution.

Resampling each bond length analytically would be one option (see
``md_simulations.torch_potentials.resampled``). This script does the more direct thing: it
runs a short Langevin relaxation of every frame *in the unconstrained potential*, so the fast
X-H modes thermalise under the real force field while the slow backbone torsions, which are
what the subsample was chosen for, stay put.

Defaults are calibrated for this system:

* **0.5 fs timestep** -- an X-H stretch has a ~11 fs period, so the 2 fs step of the
  constrained run would be unstable once the bonds are free.
* **gamma = 50/ps, 500 steps (250 fs)** -- the stretch modes relax in ~20 fs at this friction,
  so 250 fs is ~12 relaxation times; the X-H width is converged (identical at 2000 steps)
  while phi moves by only ~13 degrees rms, well inside a basin.

Usage::

    python prepare_data.py --repo /path/to/md-simulations          # ~2 min on 16 cores
    python prepare_data.py --repo ... --stride 20 --workers 4      # quicker, coarser
"""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path

import mdtraj as md
import numpy as np
import openmm.unit as u
from openmm import Context, LangevinMiddleIntegrator, Platform
from openmm.app import PDBFile

from md_simulations.config import load_config
from md_simulations.engines import build_engine

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "ala2-amber14-implicit-330K.yaml"
SOURCE = "alaX/ala2_AMBER14_implicit_remd_330K-train"

_WORKER: dict = {}


def build_system(repo: Path, constraints: str):
    """Build the example's OpenMM system under a given constraint scheme.

    :param repo: Path to the md-simulations checkout (for the data root).
    :param constraints: ``"none"`` or ``"hbonds"``.
    :return: The engine's ``BuiltSystem``.
    """
    config = load_config(CONFIG).model_copy(update={"constraints": constraints})
    logger = logging.getLogger("prepare_data")
    return build_engine(config, HERE / "data", logger).build_system()


def atom_order(topology) -> list[tuple[str, str]]:
    """Return ``(residue name, atom name)`` per atom, the identity used to match orders.

    :param topology: An OpenMM or mdtraj topology.
    :return: One ``(residue, atom)`` pair per atom, in topology order.
    """
    atoms = topology.atoms() if callable(getattr(topology, "atoms", None)) else topology.atoms
    return [(a.residue.name, a.name) for a in atoms]


def _init_worker(system, temperature: float, friction: float, timestep: float) -> None:
    """Give each process its own single-threaded OpenMM context.

    :param system: The unconstrained ``openmm.System``.
    :param temperature: Thermostat temperature in K.
    :param friction: Collision rate in 1/ps.
    :param timestep: Integration step in ps.
    """
    integrator = LangevinMiddleIntegrator(
        temperature * u.kelvin, friction / u.picosecond, timestep * u.picoseconds
    )
    _WORKER["integrator"] = integrator
    _WORKER["context"] = Context(
        system, integrator, Platform.getPlatformByName("CPU"), {"Threads": "1"}
    )
    _WORKER["temperature"] = temperature


def _relax(item: tuple[int, np.ndarray, int]) -> tuple[int, np.ndarray]:
    """Relax one frame in the unconstrained potential.

    :param item: ``(index, positions in nm, number of steps)``.
    :return: ``(index, relaxed positions in nm)``.
    """
    index, positions, steps = item
    context = _WORKER["context"]
    context.setPositions(positions)
    # a fresh seed per frame: the frames are independent draws, not one trajectory
    context.setVelocitiesToTemperature(_WORKER["temperature"] * u.kelvin, index + 1)
    _WORKER["integrator"].step(steps)
    state = context.getState(getPositions=True)
    return index, state.getPositions(asNumpy=True).value_in_unit(u.nanometer)


def xh_lengths(xyz: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Bond lengths of the X-H pairs over a set of frames.

    :param xyz: ``(F, N, 3)`` positions in nm.
    :param pairs: Atom index pairs.
    :return: ``(F, n_pairs)`` lengths in nm.
    """
    return np.stack([np.linalg.norm(xyz[:, j] - xyz[:, i], axis=-1) for i, j in pairs], axis=1)


def basin_report(xyz: np.ndarray, topology, label: str) -> None:
    """Print the phi/psi basin populations of a set of frames.

    :param xyz: ``(F, N, 3)`` positions in nm.
    :param topology: mdtraj topology matching ``xyz``.
    :param label: Name to print the populations under.
    """
    traj = md.Trajectory(xyz.astype(np.float32), topology)
    phi = np.degrees(md.compute_phi(traj)[1][:, 0])
    psi = np.degrees(md.compute_psi(traj)[1][:, 0])
    basins = {
        "C7eq/beta": (phi < 0) & ((psi > 100) | (psi < -150)),
        "alpha_R": (phi < 0) & (psi > -100) & (psi < 50),
        "alpha_L": (phi > 0) & (psi > -50) & (psi < 100),
        "C7ax": (phi > 0) & ((psi > 100) | (psi < -100)),
    }
    populations = "  ".join(f"{n} {m.mean() * 100:5.2f}% ({m.sum()})" for n, m in basins.items())
    print(f"  {label:22s} {populations}")


def main() -> None:
    """Subsample the REMD run, relax it unconstrained and write the notebook's data files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="md-simulations checkout")
    parser.add_argument("--stride", type=int, default=5, help="keep every stride-th REMD frame")
    parser.add_argument("--steps", type=int, default=500, help="relaxation steps per frame")
    parser.add_argument("--timestep", type=float, default=0.0005, help="ps")
    parser.add_argument("--friction", type=float, default=50.0, help="1/ps")
    parser.add_argument("--temperature", type=float, default=330.0, help="K")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    free = build_system(args.repo, "none")
    held = build_system(args.repo, "hbonds")
    order = atom_order(free.topology)
    pairs = [
        tuple(held.system.getConstraintParameters(i)[:2])
        for i in range(held.system.getNumConstraints())
    ]
    print(
        f"{len(pairs)} X-H bonds are constrained in the source run, free here "
        f"({free.system.getNumConstraints()} constraints)"
    )

    # --- subsample, after checking this really is the same molecule ---
    source = args.repo / "data" / SOURCE
    source_topology = md.load_topology(source / "initial_structure.pdb")
    source_order = atom_order(source_topology)
    if sorted(source_order) != sorted(order):
        raise SystemExit(f"{source} is a different molecule than {CONFIG.name}")
    permutation = [source_order.index(name) for name in order]

    trajectory = md.load(str(source / "trajectory.dcd"), top=source_topology, stride=args.stride)
    xyz = trajectory.xyz[:, permutation, :].astype(np.float64)
    mdtraj_topology = md.Topology.from_openmm(free.topology)
    print(f"{len(xyz)} frames kept (every {args.stride}th of {source.name})")
    basin_report(xyz, mdtraj_topology, "before relaxation")

    # --- relax every frame in the unconstrained potential ---
    before = xh_lengths(xyz, pairs)
    work = [(i, frame, args.steps) for i, frame in enumerate(xyz)]
    with mp.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(free.system, args.temperature, args.friction, args.timestep),
    ) as pool:
        relaxed = np.empty_like(xyz)
        for done, (index, positions) in enumerate(
            pool.imap_unordered(_relax, work, chunksize=16), 1
        ):
            relaxed[index] = positions
            if done % 2000 == 0:
                print(f"  relaxed {done}/{len(work)}")

    after = xh_lengths(relaxed, pairs)
    print(
        f"  X-H length: {before.mean():.5f} +- {before.std():.5f} nm (constrained) "
        f"-> {after.mean():.5f} +- {after.std():.5f} nm (free)"
    )
    basin_report(relaxed, mdtraj_topology, "after relaxation")

    # --- write ---
    out = HERE / "data"
    np.save(out / "ala2_330K_xyz.npy", relaxed.astype(np.float32))
    with open(out / "ala2_topology.pdb", "w") as handle:
        PDBFile.writeFile(free.topology, relaxed[0] * u.nanometer, handle, keepIds=True)
    print(f"wrote {out / 'ala2_330K_xyz.npy'} {relaxed.shape} and {out / 'ala2_topology.pdb'}")


if __name__ == "__main__":
    main()
