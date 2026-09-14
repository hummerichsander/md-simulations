#!/usr/bin/env python3
"""Grid umbrella sampling of the alanine-dipeptide Ramachandran (phi, psi) plane.

Runs one harmonic-biased window per (phi, psi) grid point to explore the
Ramachandran surface. Standalone CLI (not YAML-config driven): the grid/biasing
parameters are specific to this workflow. Shares logging and the integrator with
:mod:`md_simulations.core`."""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
from tqdm import tqdm

from md_simulations.core import create_integrator, setup_logger

logger = logging.getLogger("md_simulations.umbrella")


class DihedralReporter:
    """Reporter that records phi and psi dihedral angles during simulation.

    :param file: Path to the output text file.
    :param reportInterval: Steps between successive dihedral writes.
    :param phi_atoms: Atom indices defining the phi torsion.
    :param psi_atoms: Atom indices defining the psi torsion."""

    def __init__(
        self,
        file: str,
        reportInterval: int,
        phi_atoms: tuple[int, ...],
        psi_atoms: tuple[int, ...],
    ) -> None:
        self._out = open(file, "w")
        self._reportInterval = reportInterval
        self._phi_atoms = phi_atoms
        self._psi_atoms = psi_atoms
        self._out.write("# Step\tPhi(deg)\tPsi(deg)\n")

    def __del__(self) -> None:
        try:
            self._out.close()
        except Exception:
            pass

    def describeNextReport(self, simulation: app.Simulation) -> dict:
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return {"steps": steps, "periodic": None, "include": ["positions"]}

    def report(self, simulation: app.Simulation, state: openmm.State) -> None:
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        phi_deg = np.degrees(self._calculate_dihedral(positions, self._phi_atoms))
        psi_deg = np.degrees(self._calculate_dihedral(positions, self._psi_atoms))
        self._out.write(f"{simulation.currentStep}\t{phi_deg:.2f}\t{psi_deg:.2f}\n")

    @staticmethod
    def _calculate_dihedral(positions: np.ndarray, atoms: tuple[int, ...]) -> float:
        """Compute a dihedral angle (radians) from four atom positions.

        :param positions: Atomic positions in nanometres.
        :param atoms: The four atom indices defining the dihedral.
        :return: The dihedral angle in radians."""
        b1 = positions[atoms[1]] - positions[atoms[0]]
        b2 = positions[atoms[2]] - positions[atoms[1]]
        b3 = positions[atoms[3]] - positions[atoms[2]]

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))

        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        return np.arctan2(y, x)


def identify_dihedral_atoms(
    topology: app.Topology,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Identify phi/psi dihedral atom indices for the central residue.

    Uses the first residue that has both a previous and a next residue.

    :param topology: OpenMM topology of the solvated system.
    :return: A pair (phi_atoms, psi_atoms), each a 4-tuple of atom indices.
    :raises RuntimeError: If no suitable phi/psi atoms are found."""

    def get_atom_index(residue, name: str) -> int:
        for atom in residue.atoms():
            if atom.name == name:
                return atom.index
        raise ValueError(f"Atom '{name}' not found in residue {residue.name} {residue.id}")

    for chain in topology.chains():
        residues = list(chain.residues())
        for i in range(1, len(residues) - 1):
            res_prev, res, res_next = residues[i - 1], residues[i], residues[i + 1]
            try:
                phi_atoms = (
                    get_atom_index(res_prev, "C"),
                    get_atom_index(res, "N"),
                    get_atom_index(res, "CA"),
                    get_atom_index(res, "C"),
                )
                psi_atoms = (
                    get_atom_index(res, "N"),
                    get_atom_index(res, "CA"),
                    get_atom_index(res, "C"),
                    get_atom_index(res_next, "N"),
                )
                logger.info(
                    "Using residue %s %s for Ramachandran dihedrals: phi %s, psi %s",
                    res.name,
                    res.id,
                    phi_atoms,
                    psi_atoms,
                )
                return phi_atoms, psi_atoms
            except ValueError:
                continue

    raise RuntimeError("Could not identify phi/psi dihedral atoms in topology.")


def add_umbrella_potential(
    system: openmm.System,
    phi_atoms: tuple[int, ...],
    psi_atoms: tuple[int, ...],
    phi_0_deg: float,
    psi_0_deg: float,
    force_constant: float,
) -> openmm.System:
    """Add harmonic umbrella biases on the phi and psi Ramachandran angles.

    :param system: System to modify in place (forces are appended).
    :param phi_atoms: Atom indices for the phi torsion.
    :param psi_atoms: Atom indices for the psi torsion.
    :param phi_0_deg: Centre of the phi umbrella in degrees.
    :param psi_0_deg: Centre of the psi umbrella in degrees.
    :param force_constant: Force constant k in kJ/mol/rad².
    :return: The same system object, for convenience."""
    phi_0 = math.radians(phi_0_deg)
    psi_0 = math.radians(psi_0_deg)
    k = force_constant * unit.kilojoule_per_mole

    for atoms, theta0 in ((phi_atoms, phi_0), (psi_atoms, psi_0)):
        force = openmm.CustomTorsionForce("0.5 * k * (theta - theta0)^2")
        force.addGlobalParameter("k", k)
        force.addPerTorsionParameter("theta0")
        force.addTorsion(int(atoms[0]), int(atoms[1]), int(atoms[2]), int(atoms[3]), [theta0])
        system.addForce(force)

    logger.info(
        "Added umbrella potentials: phi0 = %.2f deg, psi0 = %.2f deg, k = %.2f kJ/mol/rad^2",
        phi_0_deg,
        psi_0_deg,
        force_constant,
    )
    return system


def setup_system(
    pdb_file: str,
    forcefield_files: list[str],
    ph: float = 7.0,
    padding: float = 1.0,
    water_model: str = "tip3p",
) -> tuple[app.Modeller, openmm.System, app.ForceField]:
    """Set up a solvated all-atom system for umbrella sampling.

    :param pdb_file: Path to the input PDB file.
    :param forcefield_files: OpenMM force-field XML files.
    :param ph: pH used when adding hydrogens.
    :param padding: Water-box padding around the solute (nm).
    :param water_model: Water model name recognised by OpenMM.
    :return: A tuple (Modeller, System, ForceField)."""
    logger.info(f"Loading PDB structure: {pdb_file}")
    pdb = app.PDBFile(pdb_file)

    logger.info(f"Loading force field: {forcefield_files}")
    forcefield = app.ForceField(*forcefield_files)

    logger.info("Building modeller and adding hydrogens...")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield, pH=ph)

    logger.info(f"Adding solvent ({water_model}) with {padding} nm padding...")
    modeller.addSolvent(forcefield, model=water_model, padding=padding * unit.nanometer)

    logger.info("Creating OpenMM system...")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        ewaldErrorTolerance=0.0005,
    )
    return modeller, system, forcefield


def run_single_umbrella_window(
    modeller: app.Modeller,
    system: openmm.System,
    integrator: openmm.Integrator,
    phi_atoms: tuple[int, ...],
    psi_atoms: tuple[int, ...],
    phi_target: float,
    psi_target: float,
    output_dir: Path,
    args: argparse.Namespace,
) -> bool:
    """Run a single umbrella-sampling window at a target (phi, psi).

    Guards against NaN/exploding energies at every stage (minimisation,
    optional gradual heating, equilibration, production) and skips the window
    on failure rather than aborting the whole grid.

    :param modeller: The base modeller (shared across windows).
    :param system: The per-window system with umbrella forces added.
    :param integrator: A fresh integrator for this window.
    :param phi_atoms: Atom indices for the phi torsion.
    :param psi_atoms: Atom indices for the psi torsion.
    :param phi_target: Target phi angle in degrees.
    :param psi_target: Target psi angle in degrees.
    :param output_dir: Directory for this window's output files.
    :param args: Parsed CLI arguments.
    :return: True on success, False if the window was skipped."""
    window_name = f"phi_{phi_target:+07.2f}_psi_{psi_target:+07.2f}"
    logger.info(f"Starting umbrella window: φ={phi_target:.1f}°, ψ={psi_target:.1f}°")

    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    logger.debug("Minimizing energy (may take longer for strained configurations)...")
    try:
        simulation.minimizeEnergy(maxIterations=5000, tolerance=10.0)
        energy_val = (
            simulation.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoules_per_mole)
        )
        logger.debug(f"Energy after minimization: {energy_val:.2f} kJ/mol")
        if math.isnan(energy_val):
            logger.warning(f"NaN energy after minimization at {window_name}. Skipping window.")
            return False
        if energy_val > 1e6:
            logger.warning(
                f"Extremely high energy ({energy_val:.2e} kJ/mol) after minimization at "
                f"{window_name}. Skipping window."
            )
            return False
    except Exception as e:
        logger.warning(f"Minimization failed at {window_name}: {e}. Skipping window.")
        return False

    logger.debug(f"Initializing velocities at {args.temperature} K...")
    simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin)

    if args.num_equilibration_steps > 0:
        logger.debug(f"Equilibrating for {args.num_equilibration_steps} steps...")

        if args.gradual_heating:
            heating_steps = min(5000, args.num_equilibration_steps // 4)
            logger.debug(f"Gradual heating for {heating_steps} steps...")
            try:
                for i in range(10):
                    integrator.setTemperature(args.temperature * (i + 1) / 10.0 * unit.kelvin)
                    simulation.step(heating_steps // 10)
                    if i % 3 == 0:
                        energy_val = (
                            simulation.context.getState(getEnergy=True)
                            .getPotentialEnergy()
                            .value_in_unit(unit.kilojoules_per_mole)
                        )
                        if math.isnan(energy_val) or energy_val > 1e6:
                            logger.warning(
                                f"Energy issue during heating at {window_name}. Skipping."
                            )
                            return False
                integrator.setTemperature(args.temperature * unit.kelvin)
                remaining_eq_steps = args.num_equilibration_steps - heating_steps
            except Exception as e:
                logger.warning(f"Heating failed at {window_name}: {e}")
                return False
        else:
            remaining_eq_steps = args.num_equilibration_steps

        try:
            for chunk in range(remaining_eq_steps // 1000):
                simulation.step(1000)
                if chunk % 5 == 0:
                    energy_val = (
                        simulation.context.getState(getEnergy=True)
                        .getPotentialEnergy()
                        .value_in_unit(unit.kilojoules_per_mole)
                    )
                    if math.isnan(energy_val) or energy_val > 1e6:
                        logger.warning(
                            f"Energy issue during equilibration at {window_name}. Skipping."
                        )
                        return False
            if remainder := remaining_eq_steps % 1000:
                simulation.step(remainder)
        except Exception as e:
            logger.warning(f"Equilibration failed at {window_name}: {e}")
            return False

        equilibrated_pdb = output_dir / f"{window_name}_equilibrated.pdb"
        logger.debug(f"Saving equilibrated structure to: {equilibrated_pdb}")
        equilibrated_positions = simulation.context.getState(getPositions=True).getPositions()
        with open(equilibrated_pdb, "w") as f:
            app.PDBFile.writeFile(simulation.topology, equilibrated_positions, f)

    if args.num_umbrella_samples > 0:
        trajectory_dcd = output_dir / f"{window_name}.dcd"
        dihedrals_txt = output_dir / f"{window_name}_dihedrals.txt"
        simulation.reporters.append(app.DCDReporter(str(trajectory_dcd), args.output_freq))
        simulation.reporters.append(
            DihedralReporter(str(dihedrals_txt), args.output_freq, phi_atoms, psi_atoms)
        )
        logger.debug(f"Running production for {args.num_umbrella_samples} steps...")
        try:
            for chunk in range(args.num_umbrella_samples // args.output_freq):
                simulation.step(args.output_freq)
                if chunk % 10 == 0:
                    energy_val = (
                        simulation.context.getState(getEnergy=True)
                        .getPotentialEnergy()
                        .value_in_unit(unit.kilojoules_per_mole)
                    )
                    if math.isnan(energy_val):
                        logger.warning(
                            f"NaN energy during production at {window_name}. Stopping window."
                        )
                        return False
        except Exception as e:
            logger.warning(f"Simulation failed at {window_name}: {e}")
            return False
    else:
        logger.debug("Skipping production run (num_umbrella_samples = 0)")

    logger.debug(f"Successfully completed window {window_name}")
    return True


def run_grid_umbrella_sampling(args: argparse.Namespace) -> None:
    """Run umbrella sampling over a grid of (phi, psi) target angles.

    :param args: Parsed CLI arguments.
    :return: None."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Setting up base system...")
    modeller, system_base, _ = setup_system(
        args.input_pdb, args.forcefield, args.ph, args.padding, args.water_model
    )

    logger.info("Identifying phi and psi dihedral atoms...")
    phi_atoms, psi_atoms = identify_dihedral_atoms(modeller.topology)

    topology_pdb = output_dir / "topology.pdb"
    logger.info(f"Saving topology to: {topology_pdb}")
    with open(topology_pdb, "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

    phi_values = np.linspace(-180, 180, args.num_grid_points, endpoint=False)
    psi_values = np.linspace(-180, 180, args.num_grid_points, endpoint=False)

    logger.info(
        f"Grid size: {args.num_grid_points} x {args.num_grid_points} "
        f"= {args.num_grid_points**2} windows"
    )
    logger.info(f"Samples per window: {args.num_umbrella_samples}")
    logger.info(f"Total steps: {args.num_grid_points**2 * args.num_umbrella_samples:,}")

    successful_windows: list[tuple[float, float]] = []
    failed_windows: list[tuple[float, float]] = []

    total_windows = len(phi_values) * len(psi_values)
    progress_bar = tqdm(total=total_windows, desc="Grid Progress")

    for phi_target in phi_values:
        for psi_target in psi_values:
            # Deep-copy the base system per window via XML (round)-trip serialization.
            system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system_base))
            add_umbrella_potential(
                system, phi_atoms, psi_atoms, phi_target, psi_target, args.force_constant
            )
            integrator = create_integrator(args.temperature, args.friction, args.timestep)
            success = run_single_umbrella_window(
                modeller,
                system,
                integrator,
                phi_atoms,
                psi_atoms,
                phi_target,
                psi_target,
                output_dir,
                args,
            )
            (successful_windows if success else failed_windows).append((phi_target, psi_target))
            progress_bar.update(1)

    progress_bar.close()

    summary_file = output_dir / "grid_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Grid Umbrella Sampling Summary\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Total windows: {total_windows}\n")
        f.write(f"Successful: {len(successful_windows)}\n")
        f.write(f"Failed (NaN/errors): {len(failed_windows)}\n")
        f.write("\nSuccessful windows:\n")
        for phi, psi in successful_windows:
            f.write(f"  φ={phi:+7.2f}°, ψ={psi:+7.2f}°\n")
        f.write("\nFailed windows:\n")
        for phi, psi in failed_windows:
            f.write(f"  φ={phi:+7.2f}°, ψ={psi:+7.2f}°\n")

    logger.info("=" * 60)
    logger.info("Grid umbrella sampling completed!")
    logger.info(f"Successful windows: {len(successful_windows)}/{total_windows}")
    logger.info(f"Failed windows: {len(failed_windows)}/{total_windows}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


def main() -> None:
    """Entry point: parse arguments and run grid umbrella sampling."""
    parser = argparse.ArgumentParser(
        description="OpenMM umbrella sampling for the alanine-dipeptide Ramachandran plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_pdb", help="Input PDB file path.")
    parser.add_argument(
        "--num-grid-points", type=int, default=36, help="Grid points per dimension (phi and psi)."
    )
    parser.add_argument(
        "--num-umbrella-samples", type=int, default=100_000, help="Production MD steps per window."
    )
    parser.add_argument(
        "--num-equilibration-steps",
        type=int,
        default=10_000,
        help="Equilibration steps per window.",
    )
    parser.add_argument(
        "--gradual-heating", action="store_true", help="Gradually heat during equilibration."
    )
    parser.add_argument(
        "--force-constant", type=float, default=100.0, help="Umbrella force constant (kJ/mol/rad²)."
    )
    parser.add_argument(
        "--output-freq",
        "-f",
        type=int,
        default=1000,
        help="Steps between trajectory/dihedral writes.",
    )
    parser.add_argument(
        "--forcefield",
        nargs="+",
        default=["amber14-all.xml", "amber14/tip3pfb.xml"],
        help="Force field XML files.",
    )
    parser.add_argument("--ph", type=float, default=7.0, help="pH for adding hydrogens.")
    parser.add_argument("--padding", type=float, default=1.0, help="Water-box padding (nm).")
    parser.add_argument("--water-model", default="tip3p", help="Water model.")
    parser.add_argument("--temperature", "-T", type=float, default=300.0, help="Temperature (K).")
    parser.add_argument("--friction", type=float, default=1.0, help="Friction coefficient (ps⁻¹).")
    parser.add_argument("--timestep", "-dt", type=float, default=0.002, help="Timestep (ps).")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity level.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_pdb):
        print(f"Error: Input PDB file '{args.input_pdb}' not found!")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("md_simulations.umbrella", output_dir, getattr(logging, args.log_level))

    logger.info("=" * 60)
    logger.info("OpenMM Grid Umbrella Sampling - Ramachandran Plot")
    logger.info("=" * 60)
    logger.info(f"Input PDB: {args.input_pdb}")
    logger.info(f"Grid size: {args.num_grid_points} x {args.num_grid_points}")
    logger.info(f"Equilibration steps: {args.num_equilibration_steps:,}")
    logger.info(f"Gradual heating: {args.gradual_heating}")
    logger.info(f"Production steps per window: {args.num_umbrella_samples:,}")
    logger.info(f"Force constant: {args.force_constant} kJ/mol/rad²")
    logger.info(f"Temperature: {args.temperature} K")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)

    try:
        run_grid_umbrella_sampling(args)
    except Exception as e:
        logger.error(f"Error during simulation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
