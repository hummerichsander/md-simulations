"""ASE ``Calculator`` wrapping an OpenMM ``Context`` (the reference backend).

ASE ships no official OpenMM calculator, so we provide one. Its only job is to
take the geometry ASE hands over, push it into the OpenMM ``Context``, and store
``energy`` and ``forces`` in ``self.results``.

The critical detail is **units**. OpenMM works in nm / kJ.mol^-1, ASE in
A / eV. We convert carefully at this one boundary and keep the ASE convention
(A, eV, eV/A) everywhere downstream. Unit bugs here are silent -- everything
looks plausible but is off by a constant factor -- so each conversion carries a
docstring explaining its factor."""

import ase.units as u
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from openmm import Context, Platform, System, VerletIntegrator, unit


class OpenMMCalculator(Calculator):
    """ASE calculator that evaluates energy and forces via an OpenMM ``Context``.

    One ``Context`` holds one ``System``; single-point evaluation only needs its
    (otherwise unused) integrator to have been supplied at construction time."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        context: Context,
        groups: int | set[int] = -1,
        **kwargs: object,
    ) -> None:
        """Store the OpenMM context to evaluate.

        :param context: An OpenMM ``Context`` built from the target ``System``.
        :param groups: OpenMM force groups to include in the energy/force
            evaluation, as a bitmask or a set of group indices. Defaults to
            ``-1`` (all groups). Only meaningful if the underlying ``System``
            assigns its forces to distinct groups via ``Force.setForceGroup``.
        :param kwargs: Forwarded to :class:`ase.calculators.calculator.Calculator`."""
        super().__init__(**kwargs)
        self.context = context
        self.groups = groups

    @classmethod
    def from_system(
        cls,
        system: System,
        positions: object,
        box_vectors: object | None = None,
        groups: int | set[int] = -1,
        platform: str | None = None,
        **kwargs: object,
    ) -> "OpenMMCalculator":
        """Build a calculator from an OpenMM ``System`` (a ``Context`` is created here).

        Saves callers the boilerplate of wiring up a throwaway integrator and a
        ``Context``. The integrator is never stepped -- single-point evaluation
        only reads state -- and the positions set here are just a valid initial
        state; :class:`~md_simulations.torch_potentials.potential.Potential`
        overwrites them per call.

        :param system: The OpenMM ``System`` to evaluate.
        :param positions: Initial atomic positions (an OpenMM-acceptable quantity,
            e.g. nm-valued positions from a topology).
        :param box_vectors: Optional periodic box vectors to set on the context.
        :param groups: OpenMM force groups to include (see :meth:`__init__`).
        :param platform: Optional OpenMM platform name (e.g. ``"CUDA"``, ``"CPU"``,
            ``"Reference"``); ``None`` lets OpenMM choose.
        :param kwargs: Forwarded to :class:`ase.calculators.calculator.Calculator`.
        :return: A ready :class:`OpenMMCalculator` wrapping the new context."""
        integrator = VerletIntegrator(0.001)  # required by Context, never stepped
        if platform is None:
            context = Context(system, integrator)
        else:
            context = Context(system, integrator, Platform.getPlatformByName(platform))
        context.setPositions(positions)
        if box_vectors is not None:
            context.setPeriodicBoxVectors(*box_vectors)
        return cls(context, groups=groups, **kwargs)

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ) -> None:
        """Push geometry into OpenMM and store energy and forces in ``self.results``.

        :param atoms: ASE ``Atoms`` to evaluate (uses the attached one if ``None``).
        :param properties: Requested properties (energy and forces are always set).
        :param system_changes: ASE list of changes triggering the recompute."""
        super().calculate(atoms, properties, system_changes)

        # ASE positions are in A; tagging with unit.angstrom lets OpenMM convert
        # to its internal nm. Periodic box vectors, if any, are handled the same way.
        self.context.setPositions(self.atoms.get_positions() * unit.angstrom)
        if self.atoms.cell.rank == 3 and self.atoms.pbc.all():
            self.context.setPeriodicBoxVectors(*(self.atoms.get_cell() * unit.angstrom))

        groups = self.groups
        if isinstance(groups, set):
            # OpenMM accepts a bitmask int; fold the index set into one.
            groups = sum(1 << i for i in groups)

        state = self.context.getState(getEnergy=True, getForces=True, groups=groups)
        e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        f = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

        # kJ/mol -> eV via the factor (u.kJ / u.mol), ASE's eV-based value of a kJ/mol.
        self.results["energy"] = e * (u.kJ / u.mol)
        # kJ/mol/nm -> eV/A: apply the energy factor, then /u.nm (u.nm == 10 A).
        self.results["forces"] = np.asarray(f) * (u.kJ / u.mol) / u.nm
