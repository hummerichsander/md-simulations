"""Score coarse-grained configurations with the CGSchNet/mlcg potential.

The counterpart to ``openmm_all_atom.py`` for the native-torch route: cgschnet has no
ASE calculator, so ``build_potential_from_file`` returns a
:class:`~md_simulations.torch_potentials.cgschnet.CGSchNetPotential` directly. The
frontend is unchanged -- positions in nm, energies in kJ/mol, forces from
``backward()`` -- which is the point of the shared ``PotentialLike`` contract.

Two things this demonstrates that the ASE path cannot do: a whole batch of frames in
one forward, and a second derivative.

Needs the ``cgschnet-potentials`` extra (Python 3.12) and the data root:

    MD_DATA_ROOT=data uv run python examples/torch_potentials/cgschnet_cg_energy.py"""

from pathlib import Path

import mdtraj as md
import torch

from md_simulations.torch_potentials import build_potential_from_file

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "trpcage-cgschnet-300K.yaml"
DATA_ROOT = REPO / "data"
BEAD_PDB = DATA_ROOT / "pdbs" / "2JOF_5B-CG.pdb"


def main() -> None:
    """Build the CG potential and evaluate energies, forces and a Hessian-vector product.

    :return: None."""
    # The topology fixes the bead order the potential's inputs are interpreted in. Here
    # it is the model's own PDB, so no reordering is needed; a projected topology with a
    # different within-residue order works too -- the potential derives the permutation.
    topology = md.load_topology(str(BEAD_PDB))
    pot = build_potential_from_file(topology, CONFIG, data_root=DATA_ROOT)
    print(f"{pot.n_atoms} beads, {len(pot.terms)} energy terms, permutation={pot.permutation}")

    # A batch of frames: jitter the reference structure so the frames differ.
    reference = torch.tensor(md.load(str(BEAD_PDB)).xyz[0], dtype=torch.float32)
    torch.manual_seed(0)
    frames = reference.unsqueeze(0) + 0.01 * torch.randn(4, *reference.shape)

    with torch.no_grad():
        energies = pot(frames)  # one mlcg forward for all 4 frames
    print(f"energies (kJ/mol): {[round(float(e), 2) for e in energies]}")

    # Forces: the nm->A and kcal->kJ conversions are autograd ops, so the gradient comes
    # back in kJ/mol/nm and in the caller's bead order without any manual bookkeeping.
    x = frames.clone().requires_grad_(True)
    pot(x).sum().backward()
    print(f"max |force|: {x.grad.abs().max():.1f} kJ/mol/nm")

    # Second derivatives, which the ASE bridge cannot provide. Contracted with a random
    # probe: the energy is translation invariant, so grad.sum() is identically zero.
    y = frames[:1].clone().requires_grad_(True)
    gradient = torch.autograd.grad(pot(y).sum(), y, create_graph=True)[0]
    probe = torch.randn_like(y)
    hessian_vector = torch.autograd.grad((gradient * probe).sum(), y)[0]
    print(f"max |Hessian-vector product|: {hessian_vector.abs().max():.1f} kJ/mol/nm^2")


if __name__ == "__main__":
    main()
