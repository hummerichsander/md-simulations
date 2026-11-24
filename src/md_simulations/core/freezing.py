"""Hold a subset of atoms immobile, so a run samples the conditional of the rest.

Freezing sets the mass of the selected atoms to zero. OpenMM reads a zero mass as an inverse mass
of zero, i.e. the particle is left out of the equations of motion entirely rather than made
infinitely heavy, so the remaining atoms sample the Boltzmann distribution conditioned on the
frozen coordinates *exactly*. A stiff positional restraint would instead leave a thermal smear of
width ``sqrt(kT/k)`` on the conditioning coordinates, and would add its energy to the total force
that ``ForceReporter`` writes.

The one restriction is constraints: OpenMM refuses, at context creation, any constraint that
involves a massless particle, because SHAKE/CCMA/SETTLE enforce a constraint by splitting the
correction between two mobile partners in proportion to inverse mass, and a frozen partner can
absorb none of it. :func:`freeze_atoms` therefore checks the constraints up front and raises a
message that names the offending bond, instead of letting a cryptic OpenMM exception surface once
the job is already on a GPU.
"""

import logging

import openmm
import openmm.app as app
import openmm.unit as unit

from md_simulations.config import FrozenAtomsConfig


def resolve_frozen_atoms(topology: app.Topology, spec: FrozenAtomsConfig) -> list[int]:
    """Resolve a frozen-atom selection against a built topology.

    :param topology: The topology the system was built from.
    :param spec: The selection, by atom name or by index.
    :return: The selected atom indices, ascending.
    :raises ValueError: If the selection is empty, out of range, or of an unexpected size."""
    n_atoms = topology.getNumAtoms()
    if spec.indices is not None:
        out_of_range = [i for i in spec.indices if i >= n_atoms]
        if out_of_range:
            raise ValueError(
                f"frozen_atoms.indices out of range for a topology of {n_atoms} atoms: "
                f"{out_of_range[:10]}"
            )
        indices = sorted(spec.indices)
    else:
        wanted = set(spec.names or ())
        indices = [atom.index for atom in topology.atoms() if atom.name in wanted]
        if not indices:
            raise ValueError(
                f"frozen_atoms.names {sorted(wanted)} matched no atom of the built topology; "
                "the names are matched against the topology the engine builds, not the input PDB"
            )

    if spec.expect_count is not None and len(indices) != spec.expect_count:
        raise ValueError(
            f"frozen_atoms selected {len(indices)} atoms but expect_count is {spec.expect_count}; "
            "the topology is not the one this selection was written for"
        )
    return indices


def _constraints_touching(system: openmm.System, frozen: set[int]) -> list[tuple[int, int]]:
    """List the constraints of *system* that involve a frozen atom.

    :param system: The system.
    :param frozen: Indices to be frozen.
    :return: The ``(p1, p2)`` pairs of the offending constraints."""
    pairs = []
    for i in range(system.getNumConstraints()):
        p1, p2, _ = system.getConstraintParameters(i)
        if p1 in frozen or p2 in frozen:
            pairs.append((p1, p2))
    return pairs


def freeze_atoms(
    system: openmm.System,
    topology: app.Topology,
    spec: FrozenAtomsConfig,
    logger: logging.Logger | None = None,
) -> list[int]:
    """Freeze the selected atoms of *system* in place, by setting their mass to zero.

    Also removes the centre-of-mass motion remover: with atoms frozen there is no centre-of-mass
    motion to remove, and leaving it in makes both OpenMM's ``StateDataReporter`` and
    :func:`md_simulations.core.tracking.degrees_of_freedom` subtract three degrees of freedom that
    the system does not have.

    :param system: The system to modify in place.
    :param topology: The topology the system was built from.
    :param spec: The selection, by atom name or by index.
    :param logger: Optional logger.
    :return: The frozen atom indices, ascending.
    :raises ValueError: If a constraint involves a frozen atom, or the selection is invalid."""
    log = logger or logging.getLogger(__name__)
    indices = resolve_frozen_atoms(topology, spec)

    touching = _constraints_touching(system, set(indices))
    if touching:
        p1, p2 = touching[0]
        raise ValueError(
            f"{len(touching)} of the {system.getNumConstraints()} constraints involve an atom to "
            f"be frozen (e.g. {p1}-{p2}); OpenMM cannot constrain a massless particle. Set "
            "`constraints: none` and a timestep small enough for explicit X-H stretches "
            "(~0.0005 ps)."
        )

    for i in indices:
        system.setParticleMass(i, 0.0 * unit.dalton)

    for i in reversed(range(system.getNumForces())):
        if isinstance(system.getForce(i), openmm.CMMotionRemover):
            system.removeForce(i)
            log.info("Removed the centre-of-mass motion remover (atoms are frozen).")

    free = system.getNumParticles() - len(indices)
    log.info(f"Froze {len(indices)} atoms, {free} remain mobile ({3 * free} degrees of freedom).")
    log.debug(f"Frozen atom indices: {indices}")
    return indices
