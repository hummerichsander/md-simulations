from typing import Literal

import openmm.app as app
import openmm.unit as unit

from md_simulations.config import AmberConfig
from md_simulations.engines.base import BuiltSystem, OpenMMEngine
from md_simulations.forcefields import resolve_forcefield_files


def resolve_constraints(kind: Literal["hbonds", "allbonds", "none"]) -> object | None:
    """Translate the config's ``constraints`` value into OpenMM's own constant.

    :param kind: The configured constraint scheme.
    :return: The matching ``openmm.app`` constant, or ``None`` for no constraints."""
    match kind:
        case "hbonds":
            return app.HBonds
        case "allbonds":
            return app.AllBonds
        case "none":
            return None
        case _:
            raise ValueError(f"unknown constraints scheme: {kind!r}")


class AmberEngine(OpenMMEngine):
    """All-atom AMBER MD.

    In explicit-solvent mode the solute is placed in a periodic water box and
    long-range electrostatics use PME. In implicit-solvent mode no water is
    added; a Generalized Born model (supplied via the force-field list, e.g.
    ``implicit/gbn2.xml``) is used with a non-periodic cutoff."""

    config: AmberConfig

    def build_system(self) -> BuiltSystem:
        cfg = self.config
        pdb_file = self.config.resolve(self.data_root, cfg.input_pdb)

        self.logger.info(f"Loading PDB structure: {pdb_file}")
        pdb = app.PDBFile(str(pdb_file))

        self.logger.info(f"Loading force field: {cfg.forcefield}")
        forcefield = app.ForceField(*resolve_forcefield_files(cfg.forcefield))

        modeller = app.Modeller(pdb.topology, pdb.positions)

        # Strip crystallographic waters: implicit solvent has no water box, and
        # explicit mode re-solvates below, so input waters are never wanted.
        modeller.deleteWater()

        self.logger.info("Adding hydrogens...")
        modeller.addHydrogens(forcefield, pH=cfg.ph)

        self.logger.info(f"Constraints: {cfg.constraints}")

        if cfg.implicit_solvent:
            self.logger.info(
                f"Implicit solvent (GB): no water box, salt = {cfg.salt_conc} mol/L."
            )
            kwargs = dict(
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=cfg.nonbonded_cutoff * unit.nanometer,
                constraints=resolve_constraints(cfg.constraints),
            )
            # OpenMM rejects a zero salt concentration as an unused argument.
            if cfg.salt_conc > 0.0:
                kwargs["implicitSolventSaltConc"] = cfg.salt_conc * (unit.moles / unit.liter)
            system = forcefield.createSystem(modeller.topology, **kwargs)  # type: ignore[arg-type]
            return BuiltSystem(modeller.topology, system, modeller.positions)

        self.logger.info(f"Adding solvent ({cfg.water_model}) with {cfg.padding} nm padding...")
        modeller.addSolvent(
            forcefield, model=cfg.water_model, padding=cfg.padding * unit.nanometer  # type: ignore[arg-type]
        )

        self.logger.info("Creating OpenMM system...")
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,  # type: ignore[arg-type]
            nonbondedCutoff=cfg.nonbonded_cutoff * unit.nanometer,  # type: ignore[arg-type]
            constraints=resolve_constraints(cfg.constraints),
            ewaldErrorTolerance=0.0005,
        )
        return BuiltSystem(modeller.topology, system, modeller.positions)
