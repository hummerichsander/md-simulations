"""Regression tests for the CG potential against mlcg's own simulation output.

The strongest tests in the suite. ``_mlcg_raw/sim_{coords,potential,forces}_*.npy`` were
produced by mlcg's ``LangevinSimulation`` driving a *specialized* model through
``GradientsOut`` force heads; we reproduce them with plain autograd on an *unwrapped,
pruned* model. Two independent code paths, so agreement is real evidence rather than a
tautology -- and any drift in mlcg's internals fails here loudly.

Skipped unless both mlcg and the (git-ignored) data root are present."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("mlcg", reason="requires the 'cgschnet-potentials' extra")

import mdtraj as md  # noqa: E402

from md_simulations.torch_potentials import build_potential_from_file  # noqa: E402
from md_simulations.torch_potentials.bridge import KCALMOL_TO_KJMOL, NM_TO_ANG  # noqa: E402
from md_simulations.torch_potentials.cgschnet import bead_permutation  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
CONFIG = REPO / "configs" / "trpcage-cgschnet-300K.yaml"
RAW = DATA_ROOT / "trp-cage" / "CGSchNet_300K_2" / "_mlcg_raw"
BEAD_PDB = DATA_ROOT / "pdbs" / "2JOF_5B-CG.pdb"

N_FRAMES = 8
# kcal/mol/A -> kJ/mol/nm. Never hand-written in production code: there the nm->A and
# kcal->kJ factors are autograd ops, so the force conversion is derived. Spelling it
# out here is the point -- it checks that derivation from the outside.
FORCE_KCAL_ANG_TO_KJMOL_NM = KCALMOL_TO_KJMOL * NM_TO_ANG
# Measured max deviations over these frames at float32: 4.2e-3 kJ/mol on ~-435 kJ/mol
# energies, and 0.32 kJ/mol/nm against forces reaching ~3.9e3. Pinned at ~10x observed
# so float32 summation-order noise cannot flake the suite, while a real regression
# (wrong bead order, a dropped term, a bad offset) is orders of magnitude larger.
ENERGY_ATOL = 0.05
ENERGY_RTOL = 1e-4
FORCE_RTOL = 1e-3


def _require_data() -> None:
    """Skip the test when the git-ignored reference data is not checked out.

    :return: None."""
    for path in (RAW, BEAD_PDB, CONFIG):
        if not path.exists():
            pytest.skip(f"{path} not present (the data root is git-ignored)")


def _reorder_within_residues(topology: "md.Topology", order: tuple[str, ...]) -> "md.Topology":
    """Rebuild ``topology`` with each residue's beads emitted in ``order``.

    Lets the test synthesize a differently-ordered caller topology (the projected
    ``N, CA, C, O, CB`` layout a downstream consumer actually has) without depending on
    a file outside this repo.

    :param topology: The source topology.
    :param order: Bead names in the desired within-residue order; unlisted names go last.
    :return: A new mdtraj ``Topology`` with the same beads in a new order."""
    rank = {name: i for i, name in enumerate(order)}
    rebuilt = md.Topology()
    chain = rebuilt.add_chain()
    for residue in topology.residues:
        new_residue = rebuilt.add_residue(residue.name, chain, resSeq=residue.resSeq)
        beads = sorted(residue.atoms, key=lambda a: rank.get(a.name, len(order)))
        for bead in beads:
            rebuilt.add_atom(bead.name, bead.element, new_residue)
    return rebuilt


@pytest.fixture(scope="module")
def reference() -> dict:
    """mlcg's own coordinates, energies and forces for the first frames.

    :return: A dict of ``coords`` (nm, model bead order), ``energy`` (kJ/mol) and
        ``forces`` (kJ/mol/nm, model bead order)."""
    _require_data()
    return {
        "coords": np.load(RAW / "sim_coords_0000.npy")[0, :N_FRAMES] / NM_TO_ANG,
        "energy": np.load(RAW / "sim_potential_0000.npy")[0, :N_FRAMES] * KCALMOL_TO_KJMOL,
        "forces": np.load(RAW / "sim_forces_0000.npy")[0, :N_FRAMES]
        * FORCE_KCAL_ANG_TO_KJMOL_NM,
    }


@pytest.fixture(scope="module")
def model_potential():
    """A potential bound to the model's own bead order (identity permutation).

    :return: A ``CGSchNetPotential``."""
    _require_data()
    topology = md.load_topology(str(BEAD_PDB))
    return build_potential_from_file(topology, CONFIG, data_root=DATA_ROOT)


@pytest.fixture(scope="module")
def float64_potential():
    """A separate float64 potential for the finite-difference check.

    Built independently rather than copied: ``to()`` casts modules in place and the
    module-scoped fixtures share their model dict, so converting a copy would silently
    leave every later test running against a float64 model.

    :return: A ``CGSchNetPotential`` in double precision."""
    _require_data()
    topology = md.load_topology(str(BEAD_PDB))
    return build_potential_from_file(
        topology, CONFIG, data_root=DATA_ROOT
    ).to(dtype=torch.float64)


@pytest.fixture(scope="module")
def projected_potential():
    """A potential bound to a reordered (``N, CA, C, O, CB``) caller topology.

    :return: A ``CGSchNetPotential`` whose permutation is not the identity."""
    _require_data()
    topology = _reorder_within_residues(
        md.load_topology(str(BEAD_PDB)), ("N", "CA", "C", "O", "CB")
    )
    return build_potential_from_file(topology, CONFIG, data_root=DATA_ROOT)


def test_energies_match_mlcg(model_potential, reference: dict) -> None:
    """Energies reproduce mlcg's simulation output frame by frame.

    :param model_potential: Potential in the model's bead order.
    :param reference: mlcg's reference arrays.
    :return: None."""
    x = torch.tensor(reference["coords"], dtype=torch.float32)
    with torch.no_grad():
        energy = model_potential(x).numpy()

    assert energy.shape == (N_FRAMES,)
    deviation = np.abs(energy - reference["energy"]).max()
    print(f"\nmax energy deviation: {deviation:.2e} kJ/mol")
    np.testing.assert_allclose(
        energy, reference["energy"], rtol=ENERGY_RTOL, atol=ENERGY_ATOL
    )


def test_forces_match_mlcg(model_potential, reference: dict) -> None:
    """Autograd forces reproduce mlcg's ``GradientsOut`` force heads.

    :param model_potential: Potential in the model's bead order.
    :param reference: mlcg's reference arrays.
    :return: None."""
    x = torch.tensor(reference["coords"], dtype=torch.float32, requires_grad=True)
    model_potential(x).sum().backward()
    forces = -x.grad.numpy()
    expected = reference["forces"]

    scale = np.abs(expected).max()
    deviation = np.abs(forces - expected).max()
    print(f"\nmax force deviation: {deviation:.2e} of max|f| {scale:.2e} kJ/mol/nm")
    assert deviation <= FORCE_RTOL * scale

    cosine = float(
        (forces.ravel() @ expected.ravel())
        / (np.linalg.norm(forces) * np.linalg.norm(expected))
    )
    assert cosine > 1 - 1e-6


def test_wrong_bead_order_is_catastrophically_wrong(projected_potential, reference: dict) -> None:
    """Negative control: without the permutation the energies are nowhere near right.

    Every other permutation test would still pass if ``bead_permutation`` silently
    returned the identity, because the potential and the reference would agree with each
    other. This is the test that makes the bead-order machinery falsifiable: it feeds
    model-order coordinates to a potential expecting caller order, so the permutation is
    applied when it should not be.

    :param projected_potential: Potential expecting the reordered bead layout.
    :param reference: mlcg's reference arrays.
    :return: None."""
    x = torch.tensor(reference["coords"], dtype=torch.float32)
    with torch.no_grad():
        energy = projected_potential(x).numpy()

    deviation = np.abs(energy - reference["energy"]).max()
    print(f"\nmis-ordered deviation: {deviation:.2e} kJ/mol")
    assert deviation > 1e3 * ENERGY_ATOL


def test_permutation_makes_the_projected_topology_equivalent(
    model_potential, projected_potential, reference: dict
) -> None:
    """Reordered input through the permuting potential equals the model-order path.

    :param model_potential: Potential in the model's bead order.
    :param projected_potential: Potential expecting the reordered bead layout.
    :param reference: mlcg's reference arrays.
    :return: None."""
    perm = np.asarray(projected_potential.permutation)
    assert model_potential.permutation is None
    assert perm[:5].tolist() == [0, 1, 4, 2, 3]

    inverse = np.argsort(perm)
    x_model = torch.tensor(reference["coords"], dtype=torch.float32)
    x_caller = torch.tensor(reference["coords"][:, inverse], dtype=torch.float32)
    with torch.no_grad():
        np.testing.assert_allclose(
            projected_potential(x_caller).numpy(),
            model_potential(x_model).numpy(),
            rtol=1e-5,
            atol=1e-3,
        )


def test_pruning_kept_the_expected_terms(model_potential) -> None:
    """34 of 52 terms survive for trp-cage: SchNet plus 33 non-empty priors.

    :param model_potential: Potential in the model's bead order.
    :return: None."""
    assert "SchNet" in model_potential.terms
    assert len(model_potential.terms) == 34
    # The 18 pruned terms are phi/psi dihedrals of residue types absent from
    # trp-cage's sequence (DAYAQWLKDGGPSSGRPPPS).
    assert not any(term.startswith(("ASN_", "CYS_", "VAL_")) for term in model_potential.terms)


def test_finite_differences(float64_potential, reference: dict) -> None:
    """A float64 central difference agrees with the analytic gradient.

    A full ``gradcheck`` would need ~580 forwards on 97x3 coordinates; spot-checking a
    few components is enough given that :func:`test_forces_match_mlcg` already validates
    the gradient globally.

    :param float64_potential: Double-precision potential in the model's bead order.
    :param reference: mlcg's reference arrays.
    :return: None."""
    potential = float64_potential
    x = torch.tensor(reference["coords"][:1], dtype=torch.float64, requires_grad=True)
    analytic = torch.autograd.grad(potential(x).sum(), x)[0]

    step = 1e-5
    generator = torch.Generator().manual_seed(0)
    for _ in range(5):
        atom = int(torch.randint(0, potential.n_atoms, (1,), generator=generator))
        axis = int(torch.randint(0, 3, (1,), generator=generator))
        shift = torch.zeros_like(x)
        shift[0, atom, axis] = step
        with torch.no_grad():
            plus = potential(x.detach() + shift)
            minus = potential(x.detach() - shift)
        numeric = float((plus - minus) / (2 * step))
        assert numeric == pytest.approx(float(analytic[0, atom, axis]), rel=1e-4, abs=1e-4)


def test_double_backward_on_real_model(model_potential, reference: dict) -> None:
    """A Hessian-vector product on the real checkpoint is finite and non-trivial.

    Contracted with a random vector rather than summed: the energy is translationally
    invariant, so ``gradient.sum()`` is identically zero for every input and its
    derivative would be legitimately zero, testing nothing.

    The radius graph's integer edge indices are treated as constants, so the Hessian is
    exact only away from the 15 A cutoff boundary -- where the cosine envelope has
    smoothly reached zero anyway.

    :param model_potential: Potential in the model's bead order.
    :param reference: mlcg's reference arrays.
    :return: None."""
    x = torch.tensor(reference["coords"][:1], dtype=torch.float32, requires_grad=True)
    generator = torch.Generator().manual_seed(0)
    probe = torch.randn(x.shape, generator=generator)

    gradient = torch.autograd.grad(model_potential(x).sum(), x, create_graph=True)[0]
    hessian_vector = torch.autograd.grad((gradient * probe).sum(), x)[0]
    assert torch.isfinite(hessian_vector).all()
    assert hessian_vector.abs().sum() > 0


def test_chunking_matches_a_single_batch(model_potential, reference: dict) -> None:
    """Chunked evaluation of the real model agrees with one batch.

    :param model_potential: Potential in the model's bead order.
    :param reference: mlcg's reference arrays.
    :return: None."""
    x = torch.tensor(reference["coords"], dtype=torch.float32)
    with torch.no_grad():
        whole = model_potential(x)
        model_potential._max_batch_frames = 3
        chunked = model_potential(x)
        model_potential._max_batch_frames = 64
    np.testing.assert_allclose(chunked.numpy(), whole.numpy(), rtol=1e-5, atol=1e-3)


def test_bead_permutation_on_the_real_topologies() -> None:
    """The real model PDB and its projected reordering differ by the known permutation.

    :return: None."""
    _require_data()
    model_top = md.load_topology(str(BEAD_PDB))
    caller_top = _reorder_within_residues(model_top, ("N", "CA", "C", "O", "CB"))

    perm = bead_permutation(model_top, caller_top)
    assert perm[:5] == [0, 1, 4, 2, 3]
    assert sorted(perm) == list(range(model_top.n_atoms))
