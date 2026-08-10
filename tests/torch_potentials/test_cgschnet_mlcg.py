"""Cross-checks of the CG potential's internals against mlcg's own machinery.

:mod:`test_cgschnet_potential` proves the batching logic is self-consistent, and
:mod:`test_cgschnet_regression` proves the end-to-end numbers match a real simulation.
This module covers the gap between them: the *semantics* of the hand-built batch, which
neither can see. ``CGSchNetPotential._static_batch`` reimplements what mlcg does via
torch-geometric's ``collate``, so the honest test is to run both and demand equality.

Skipped unless mlcg and the (git-ignored) data root are present."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("mlcg", reason="requires the 'cgschnet-potentials' extra")

import mdtraj as md  # noqa: E402
from torch_geometric.data.collate import collate  # noqa: E402

from md_simulations.prep.cgschnet_embeddings import extract_atom_types  # noqa: E402
from md_simulations.torch_potentials import build_potential_from_file  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
CONFIG = REPO / "configs" / "trpcage-cgschnet-300K.yaml"
CONFIGURATIONS = DATA_ROOT / "trp-cage" / "configurations.pt"
BEAD_PDB = DATA_ROOT / "pdbs" / "2JOF_5B-CG.pdb"

N_FRAMES = 3


@pytest.fixture(scope="module")
def potential():
    """A potential built from the real config, in the model's own bead order.

    :return: A ``CGSchNetPotential``."""
    for path in (CONFIG, CONFIGURATIONS, BEAD_PDB):
        if not path.exists():
            pytest.skip(f"{path} not present (the data root is git-ignored)")
    return build_potential_from_file(
        md.load_topology(str(BEAD_PDB)), CONFIG, data_root=DATA_ROOT
    )


def test_hand_built_batch_matches_torch_geometric_collate(potential) -> None:
    """Our batch scaffold is identical to what mlcg's ``collate`` would produce.

    mlcg batches replicas with
    ``torch_geometric.data.collate.collate(cls, data_list, increment=True, add_batch=True)``
    (see ``Simulation.collate``). We build the same thing by hand -- to avoid a
    module-level torch-geometric import, and because that path has never been exercised
    on these *unspecialized* neighbour lists. This is the oracle that justifies the
    choice: every per-frame index offset and ``mapping_batch`` must agree exactly.

    :param potential: The fixture potential.
    :return: None."""
    ours = potential._static_batch(N_FRAMES)
    template = copy.deepcopy(potential._template)
    theirs, _, _ = collate(
        type(template),
        [copy.deepcopy(template) for _ in range(N_FRAMES)],
        increment=True,
        add_batch=True,
    )

    assert torch.equal(ours.batch, theirs.batch)
    assert torch.equal(ours.atom_types, theirs.atom_types)

    assert set(ours.neighbor_list) == set(theirs.neighbor_list)
    assert len(ours.neighbor_list) > 1
    for name, term in ours.neighbor_list.items():
        expected = theirs.neighbor_list[name]
        assert torch.equal(term["index_mapping"], expected["index_mapping"]), name
        assert torch.equal(term["mapping_batch"], expected["mapping_batch"]), name


def test_atom_types_agree_with_the_embeddings_helper(potential) -> None:
    """The template's ``atom_types`` match what ``extract_atom_types`` reports.

    An independent reader of the same file, so a wrong ``replica`` index or a silently
    reshaped field would show up here.

    :param potential: The fixture potential.
    :return: None."""
    configurations = torch.load(str(CONFIGURATIONS), map_location="cpu", weights_only=False)
    expected = extract_atom_types(configurations)

    np.testing.assert_array_equal(potential._template.atom_types.cpu().numpy(), expected)
    assert potential._template.atom_types.shape == (potential.n_atoms,)


def test_backbone_embeddings_are_residue_independent(potential) -> None:
    """``N``/``C``/``O`` beads each map to a single embedding index.

    This is the invariant ``from_config`` checks to catch an ``input_pdb`` and
    ``configurations_file`` that describe different systems; asserting it here pins the
    assumption to the real artifacts rather than to the checking code.

    :param potential: The fixture potential.
    :return: None."""
    topology = md.load_topology(str(BEAD_PDB))
    atom_types = potential._template.atom_types.cpu().numpy()

    for name in ("N", "C", "O"):
        indices = [i for i, atom in enumerate(topology.atoms) if atom.name == name]
        assert indices, name
        assert len(set(atom_types[indices].tolist())) == 1, name


def test_neighbour_lists_are_position_independent(potential) -> None:
    """The prior interaction lists are topological, not distance-based.

    The whole caching strategy rests on this: if any prior carried an ``rcut`` the
    scaffold would have to be rebuilt whenever coordinates changed.

    :param potential: The fixture potential.
    :return: None."""
    for name, term in potential._template.neighbor_list.items():
        assert term["rcut"] is None, name
        assert term["cell_shifts"] is None, name
