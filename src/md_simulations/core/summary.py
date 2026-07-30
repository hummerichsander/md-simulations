import logging
from datetime import datetime, timezone
from pathlib import Path

import openmm
import openmm.app as app
import openmm.unit as unit

SUMMARY_FILENAME = "simulation_summary.txt"

_NONBONDED_METHODS = {
    0: "NoCutoff",
    1: "CutoffNonPeriodic",
    2: "CutoffPeriodic",
    3: "Ewald",
    4: "PME",
    5: "LJPME",
}


def _fmt_time(ns: float) -> str:
    """Format a duration given in nanoseconds for a methods section.

    :param ns: Duration in nanoseconds.
    :return: Human-readable string (ps / ns / ns + µs)."""
    if ns >= 1000:
        return f"{ns:.4g} ns ({ns / 1000:.4g} µs)"
    if ns >= 1:
        return f"{ns:.4g} ns"
    return f"{ns * 1000:.4g} ps"


def _render(title: str, sections: dict[str, list[tuple[str, str]]]) -> str:
    """Render titled key/value sections into a fixed-width text block.

    :param title: Title shown in the header banner.
    :param sections: Mapping of section name to a list of (key, value) pairs.
    :return: The formatted summary text."""
    bar = "=" * 62
    lines = [bar, f" Simulation summary — {title}", bar]
    for section, rows in sections.items():
        rows = [(k, v) for k, v in rows if v is not None]
        if not rows:
            continue
        lines.append("")
        lines.append(f"[{section}]")
        width = max(len(k) for k, _ in rows)
        for k, v in rows:
            lines.append(f"{(k + ':').ljust(width + 2)}{v}")
    lines.append("")
    lines.append(bar)
    lines.append("")
    return "\n".join(lines)


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string.

    :return: Timestamp like ``2026-07-15T13:20:00Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _inspect_system(simulation: app.Simulation) -> dict:
    """Read ensemble/nonbonded facts back from a built OpenMM simulation.

    :param simulation: The simulation to introspect.
    :return: Dict of discovered facts (best-effort; missing keys omitted)."""
    system = simulation.system
    info: dict = {"n_particles": system.getNumParticles(), "n_constraints": system.getNumConstraints()}

    barostat = None
    nonbonded = None
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, (openmm.MonteCarloBarostat, openmm.MonteCarloAnisotropicBarostat)):
            barostat = force
        elif isinstance(force, openmm.NonbondedForce):
            nonbonded = force

    if barostat is not None:
        info["ensemble"] = "NPT"
        try:
            info["pressure_bar"] = barostat.getDefaultPressure().value_in_unit(unit.bar)
        except Exception:
            pass
    else:
        info["ensemble"] = "NVT"

    if nonbonded is not None:
        info["nonbonded_method"] = _NONBONDED_METHODS.get(nonbonded.getNonbondedMethod())
        try:
            info["cutoff_nm"] = nonbonded.getCutoffDistance().value_in_unit(unit.nanometer)
        except Exception:
            pass

    try:
        if system.usesPeriodicBoundaryConditions():
            box = simulation.context.getState().getPeriodicBoxVectors(asNumpy=True)
            box = box.value_in_unit(unit.nanometer)
            info["box_nm"] = (box[0][0], box[1][1], box[2][2])
    except Exception:
        pass

    return info


def write_simulation_summary(
    output_dir: Path,
    config,
    simulation: app.Simulation,
    logger: logging.Logger | None = None,
) -> Path:
    """Write a concise, publication-oriented parameter summary for an OpenMM run.

    Combines the declared config with facts read back from the built system
    (particle count, ensemble, nonbonded method, box). Written early in the run
    so it is available as soon as the job starts.

    :param output_dir: Run output directory.
    :param config: The engine-specific config model.
    :param simulation: The built simulation (positions/box already set).
    :param logger: Optional logger for a confirmation message.
    :return: Path to the written summary file."""
    _log = logger or logging.getLogger(__name__)
    info = _inspect_system(simulation)

    total_ns = config.num_steps * config.timestep / 1000.0
    frames = config.num_steps // config.output_freq if config.output_freq else 0
    frame_ps = config.output_freq * config.timestep

    box = info.get("box_nm")
    box_str = f"{box[0]:.3g} x {box[1]:.3g} x {box[2]:.3g} nm" if box else None

    ensemble = info.get("ensemble", "NVT")
    integrator = "LangevinMiddle" if config.integrator == "langevin_middle" else "Langevin"
    implicit = getattr(config, "implicit_solvent", False)

    sections: dict[str, list[tuple[str, str]]] = {
        "Provenance": [
            ("Generated", _now()),
            ("Software", f"OpenMM {openmm.version.version}"),
            ("Engine", config.engine),
            ("System", config.system),
        ],
        "System": [
            ("Input", getattr(config, "input_pdb", None) or getattr(config, "input_gro", None)),
            ("Topology", getattr(config, "top", None) or getattr(config, "input_top", None)),
            ("Force field", ", ".join(getattr(config, "forcefield", []) or []) or None),
            ("Water model", None if implicit else getattr(config, "water_model", None)),
            ("Solvent padding", None if implicit else _nm(getattr(config, "padding", None))),
            ("Implicit solvent", "yes (GB)" if implicit else None),
            ("Salt concentration", f"{config.salt_conc} mol/L" if getattr(config, "salt_conc", 0.0) else None),
            ("Particles", f"{info['n_particles']:,}"),
            ("Constraints", f"{info['n_constraints']:,}"),
            ("Periodic", "yes" if box else "no"),
            ("Initial box", box_str),
        ],
        "Ensemble & integrator": [
            ("Ensemble", ensemble),
            ("Integrator", f"{integrator} (Langevin thermostat)"),
            ("Temperature", f"{config.temperature} K"),
            ("Friction", f"{config.friction} ps^-1"),
            ("Timestep", f"{config.timestep * 1000:.4g} fs"),
            ("Barostat", f"Monte Carlo, {info.get('pressure_bar', getattr(config, 'pressure', '?'))} bar" if ensemble == "NPT" else None),
        ],
        "Nonbonded": [
            ("Method", info.get("nonbonded_method")),
            ("Cutoff", _nm(info.get("cutoff_nm"))),
            ("Switch distance", _nm(getattr(config, "switch_distance", None)) if config.engine == "gromacs_input" else None),
        ],
        "Production": [
            ("Steps", f"{config.num_steps:,}"),
            ("Simulated time", _fmt_time(total_ns)),
            ("Output interval", f"{config.output_freq:,} steps ({frame_ps:.4g} ps)"),
            ("Trajectory frames", f"{frames:,}"),
            ("Outputs", "trajectory.dcd, energies.csv, forces.txt, initial_structure.pdb, final_structure.pdb"),
        ],
    }

    path = output_dir / SUMMARY_FILENAME
    path.write_text(_render(f"{config.system} ({config.engine})", sections), encoding="utf-8")
    _log.info(f"Wrote simulation summary → {path}")
    return path


def _nm(value: float | None) -> str | None:
    """Format an optional length in nanometres.

    :param value: Length in nm, or None.
    :return: Formatted string, or None."""
    return None if value is None else f"{value} nm"


def write_gromacs_native_summary(
    output_dir: Path,
    config,
    mdp_params: dict[str, str],
    n_atoms: int | None,
    logger: logging.Logger | None = None,
) -> Path:
    """Write a parameter summary for a native GROMACS run.

    Pulls run parameters from the parsed ``.mdp`` file rather than an OpenMM
    system, since GROMACS owns the integration.

    :param output_dir: Run output directory.
    :param config: The gromacs_native config model.
    :param mdp_params: Parsed mdp key/value pairs (lower-cased keys).
    :param n_atoms: Atom count of the input structure, if known.
    :param logger: Optional logger for a confirmation message.
    :return: Path to the written summary file."""
    _log = logger or logging.getLogger(__name__)

    def mdp(*keys: str) -> str | None:
        for k in keys:
            if k in mdp_params:
                return mdp_params[k]
        return None

    dt = mdp("dt")
    nsteps = mdp("nsteps")
    total_ns = None
    if dt and nsteps:
        try:
            total_ns = float(dt) * float(nsteps) / 1000.0
        except ValueError:
            pass

    sections: dict[str, list[tuple[str, str]]] = {
        "Provenance": [
            ("Generated", _now()),
            ("Software", f"GROMACS (external: {config.gmx})"),
            ("Engine", config.engine),
            ("System", config.system),
        ],
        "System": [
            ("Coordinates", config.input_gro),
            ("Topology", config.top),
            ("MDP", config.mdp),
            ("Atoms", f"{n_atoms:,}" if n_atoms is not None else None),
        ],
        "Ensemble & integrator": [
            ("Integrator", mdp("integrator")),
            ("Timestep", f"{float(dt) * 1000:.4g} fs" if dt else None),
            ("Thermostat", mdp("tcoupl")),
            ("Temperature", mdp("ref-t", "ref_t")),
            ("Barostat", mdp("pcoupl")),
            ("Pressure", mdp("ref-p", "ref_p")),
            ("Constraints", mdp("constraints")),
        ],
        "Nonbonded": [
            ("Cutoff scheme", mdp("cutoff-scheme", "cutoff_scheme")),
            ("Coulomb", mdp("coulombtype")),
            ("rcoulomb", mdp("rcoulomb")),
            ("rvdw", mdp("rvdw")),
        ],
        "Production": [
            ("Steps", f"{int(nsteps):,}" if nsteps and nsteps.isdigit() else nsteps),
            ("Simulated time", _fmt_time(total_ns) if total_ns is not None else None),
            ("Extra mdrun flags", " ".join(config.mdrun_extra) or None),
            ("Outputs", "md.xtc/trr, md.edr, md.log, initial_structure.pdb"),
        ],
    }

    path = output_dir / SUMMARY_FILENAME
    path.write_text(_render(f"{config.system} ({config.engine})", sections), encoding="utf-8")
    _log.info(f"Wrote simulation summary → {path}")
    return path
