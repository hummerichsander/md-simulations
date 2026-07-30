"""Backend-swap stub: the same ``Potential`` frontend against ASE's LAMMPS engine.

ASE is kept under the hood precisely so the engine is swappable. Switching from
OpenMM to LAMMPS changes ONLY the calculator you pass to ``Potential`` -- the
mdtraj + torch frontend (build the trajectory, call ``pot(x)``, backprop) is
byte-for-byte identical to the OpenMM example.

This file is a stub: it requires a working LAMMPS build with the Python interface
(``ase.calculators.lammpslib.LAMMPSlib``) plus the appropriate potential files,
which we do not ship. The structure is what matters."""

import mdtraj as md
import numpy as np
import torch
from ase.build import bulk

from md_simulations.torch_potentials import Potential


def _traj_from_symbols(symbols: list[str], xyz_nm: np.ndarray) -> md.Trajectory:
    """Build a single-residue mdtraj Trajectory from symbols and nm coordinates."""
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("SYS", chain)
    for s in symbols:
        top.add_atom(s, md.element.Element.getBySymbol(s), res)
    return md.Trajectory(np.asarray(xyz_nm, dtype=np.float32)[None], top)


def main() -> None:
    """Show the one-line backend swap to ``LAMMPSlib``.

    :return: None."""
    try:
        from ase.calculators.lammpslib import LAMMPSlib
    except ImportError:
        print("LAMMPSlib not available; this is a structural stub only.")
        return

    ar = bulk("Ar", "fcc", a=5.26, cubic=True)
    xyz_nm = ar.get_positions() / 10.0
    lengths_nm = np.diag(np.asarray(ar.get_cell())) / 10.0  # cubic -> (3,) nm
    traj = _traj_from_symbols(["Ar"] * len(ar), xyz_nm)

    # --- The ONLY engine-specific part: build the ASE calculator (Layer 0/1). ---
    cmds = [
        "pair_style lj/cut 8.0",
        "pair_coeff 1 1 0.0103 3.40",  # epsilon (eV), sigma (A) for Argon
    ]
    calc = LAMMPSlib(lmpcmds=cmds, atom_types={"Ar": 1}, keep_alive=True)
    # ---------------------------------------------------------------------------

    # The frontend is unchanged from every other backend.
    pot = Potential(traj.topology, calc)
    x = torch.tensor(xyz_nm, dtype=torch.float64, requires_grad=True)
    energy = pot(x, unitcell_lengths=torch.tensor(lengths_nm))
    energy.backward()

    print(f"energy = {energy.item():.6f} kJ/mol")
    print(f"forces = -x.grad (kJ/mol/nm), shape {tuple(x.grad.shape)}")


if __name__ == "__main__":
    main()
