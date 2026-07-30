import openmm
import openmm.app as app
import openmm.unit as unit

from md_simulations.config import GromacsInputConfig
from md_simulations.engines.base import BuiltSystem, OpenMMEngine


class GromacsInputEngine(OpenMMEngine):
    """OpenMM MD built directly from GROMACS-format inputs (e.g. FreeSolv).

    Reads a ``.gro``/``.top`` pair, builds the system from the deposited
    topology/parameters (no ForceField XML, no added hydrogens/solvent), and
    uses settings chosen to mimic a GROMACS setup as closely as practical
    (LJ switching, LangevinMiddle integrator as an analogue of GROMACS ``sd``)."""

    config: GromacsInputConfig

    @property
    def initialize_velocities(self) -> bool:  # type: ignore[override]
        return self.config.initialize_velocities

    def build_system(self) -> BuiltSystem:
        cfg = self.config
        gro_file = cfg.resolve(self.data_root, cfg.input_gro)
        top_file = cfg.resolve(self.data_root, cfg.input_top)

        self.logger.info(f"Loading GROMACS coordinates: {gro_file}")
        gro = app.GromacsGroFile(str(gro_file))

        self.logger.info(f"Loading GROMACS topology: {top_file}")
        top = app.GromacsTopFile(str(top_file), periodicBoxVectors=gro.getPeriodicBoxVectors())

        if gro.getPeriodicBoxVectors() is None:
            self.logger.warning("No periodic box found. PME will fail for non-periodic systems.")

        self.logger.info("Creating OpenMM system from GROMACS topology...")
        system = top.createSystem(
            nonbondedMethod=app.PME,  # type: ignore[arg-type]
            nonbondedCutoff=cfg.nonbonded_cutoff * unit.nanometer,  # type: ignore[arg-type]
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=cfg.ewald_error_tolerance,
            removeCMMotion=True,
            switchDistance=cfg.switch_distance * unit.nanometer,  # type: ignore[arg-type]
        )

        if cfg.use_barostat:
            self.logger.info(
                f"Adding MonteCarloBarostat at {cfg.pressure} bar and {cfg.temperature} K"
            )
            system.addForce(
                openmm.MonteCarloBarostat(
                    cfg.pressure * unit.bar,  # type: ignore[arg-type]
                    cfg.temperature * unit.kelvin,  # type: ignore[arg-type]
                    cfg.barostat_interval,
                )
            )

        return BuiltSystem(top.topology, system, gro.positions)
