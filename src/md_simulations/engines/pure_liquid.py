import openmm
import openmm.app as app
import openmm.unit as unit

from md_simulations.config import PureLiquidConfig
from md_simulations.engines.base import BuiltSystem, OpenMMEngine
from md_simulations.forcefields import resolve_forcefield_files


class PureLiquidEngine(OpenMMEngine):
    """NPT MD for a pre-built, fully periodic box.

    No hydrogens are added and no solvent is added — the input PDB is assumed
    to be a complete, fully-solvated (or pure-liquid) periodic box. A Monte
    Carlo barostat is attached for constant-pressure sampling."""

    config: PureLiquidConfig

    def build_system(self) -> BuiltSystem:
        cfg = self.config
        pdb_file = cfg.resolve(self.data_root, cfg.input_pdb)

        self.logger.info(f"Loading PDB structure: {pdb_file}")
        pdb = app.PDBFile(str(pdb_file))

        self.logger.info(f"Loading force field: {cfg.forcefield}")
        forcefield = app.ForceField(*resolve_forcefield_files(cfg.forcefield))

        self.logger.info("Creating OpenMM system...")
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.2 * unit.nanometer,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )

        self.logger.info(
            f"Adding Monte Carlo barostat ({cfg.pressure} bar, {cfg.temperature} K)..."
        )
        system.addForce(
            openmm.MonteCarloBarostat(
                cfg.pressure * unit.bar,  # type: ignore[arg-type]
                cfg.temperature * unit.kelvin,  # type: ignore[arg-type]
            )
        )

        box = pdb.topology.getPeriodicBoxVectors()
        return BuiltSystem(pdb.topology, system, pdb.positions, box_vectors=box)
