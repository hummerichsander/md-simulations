import math

import openmm
import openmm.app as app
import openmm.unit as unit

from md_simulations.config import MartiniConfig
from md_simulations.engines.base import BuiltSystem, OpenMMEngine


class MartiniTopFile(app.GromacsTopFile):
    """GromacsTopFile subclass that adds support for MARTINI3-specific potentials.

    Handles restricted bending (GROMACS angle function type 10):
        V(θ) = k/2 · (cos θ − cos θ₀)² / sin²θ

    OpenMM's built-in GromacsTopFile only accepts angle types 1, 2, and 5.
    Type 10 is stored separately during parsing so the parent __init__ does not
    raise, then added as a CustomAngleForce in createSystem().
    """

    def _processDefaults(self, line: str) -> None:
        # MARTINI3 [ defaults ] has only 2 fields (nbfunc comb-rule); OpenMM
        # requires at least 3.  Supply gen-pairs=no and fudge values of 1.0.
        fields = line.split()
        if len(fields) == 2:
            line = line.rstrip() + " no 1.0 1.0"
        super()._processDefaults(line)

    def _processAngle(self, line: str) -> None:
        fields = line.split()
        if len(fields) >= 4 and fields[3] == "10":
            if not hasattr(self._currentMoleculeType, "restricted_angles"):
                self._currentMoleculeType.restricted_angles = []
            self._currentMoleculeType.restricted_angles.append(fields)
        else:
            super()._processAngle(line)

    def createSystem(self, **kwargs) -> openmm.System:
        system = super().createSystem(**kwargs)

        reb_force = None
        deg_to_rad = math.pi / 180.0
        base = 0

        for mol_name, mol_count in self._molecules:
            mol = self._moleculeTypes[mol_name]
            n_atoms = len(mol.atoms)
            restricted = getattr(mol, "restricted_angles", [])

            if restricted:
                if reb_force is None:
                    reb_force = openmm.CustomAngleForce(
                        "0.5*k*(cos(theta)-cos(theta0))^2/sin(theta)^2"
                    )
                    reb_force.addPerAngleParameter("theta0")
                    reb_force.addPerAngleParameter("k")
                    system.addForce(reb_force)

                for _ in range(mol_count):
                    for fields in restricted:
                        ai, aj, ak = [int(x) - 1 for x in fields[:3]]
                        theta0 = float(fields[4]) * deg_to_rad
                        k = float(fields[5])  # kJ/mol
                        reb_force.addAngle(base + ai, base + aj, base + ak, [theta0, k])
                    base += n_atoms
            else:
                base += n_atoms * mol_count

        return system


class MartiniEngine(OpenMMEngine):
    """MARTINI3 coarse-grained MD from a GROMACS GRO + TOP file pair."""

    config: MartiniConfig

    def build_system(self) -> BuiltSystem:
        cfg = self.config
        gro_file = cfg.resolve(self.data_root, cfg.input_gro)
        top_file = cfg.resolve(self.data_root, cfg.top)
        include_dir = cfg.resolve(self.data_root, cfg.include_dir)

        self.logger.info(f"Loading GRO structure: {gro_file}")
        gro = app.GromacsGroFile(str(gro_file))

        self.logger.info(f"Loading GROMACS topology: {top_file} (include dir: {include_dir})")
        top = MartiniTopFile(
            str(top_file),
            periodicBoxVectors=gro.getPeriodicBoxVectors(),
            includeDir=str(include_dir),
        )

        self.logger.info("Creating OpenMM system...")
        system = top.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=cfg.cutoff * unit.nanometer,  # type: ignore[arg-type]
            constraints=None,
            ewaldErrorTolerance=0.0001,
        )

        if cfg.pressure is not None:
            self.logger.info(f"Adding Monte Carlo barostat at {cfg.pressure} bar, {cfg.temperature} K...")
            system.addForce(
                openmm.MonteCarloBarostat(
                    cfg.pressure * unit.bar,  # type: ignore[arg-type]
                    cfg.temperature * unit.kelvin,  # type: ignore[arg-type]
                )
            )

        return BuiltSystem(
            top.topology, system, gro.positions, box_vectors=gro.getPeriodicBoxVectors()
        )
