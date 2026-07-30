from pathlib import Path

from md_simulations.config.base import SimulationConfig


def render_slurm(
    config: SimulationConfig,
    config_path: str | Path,
    data_root: str | None = None,
    venv_activate: str = ".venv/bin/activate",
) -> str:
    """Render a submittable SLURM batch script for *config*.

    Resource directives come from the config's ``slurm`` block. The job runs
    from the submission directory (the project root), activates the project
    virtualenv, and invokes ``md-sim run`` on the given config file. When no
    explicit ``data_root`` is given the job defaults ``MD_DATA_ROOT`` to the
    project's local ``data/`` directory (still overridable via the environment).
    Native-GROMACS configs additionally emit ``module load`` lines.

    :param config: A validated engine-specific config model (must set ``slurm``).
    :param config_path: Path to the YAML config the job should run.
    :param data_root: Optional explicit ``--data-root`` for the run command.
    :param venv_activate: Path to the virtualenv activate script, relative to
        the submission directory.
    :return: The full contents of the SLURM batch script.
    :raises ValueError: If the config has no ``slurm`` block."""
    s = config.slurm
    if s is None:
        raise ValueError(
            f"Config for system '{config.system}' has no 'slurm:' block to generate a job from."
        )

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={s.job_name}",
        f"#SBATCH --nodes={s.nodes}",
        f"#SBATCH --gres={s.gres}",
        f"#SBATCH --time={s.time}",
        f"#SBATCH --partition={s.partition}",
        f"#SBATCH --mem={s.mem}",
    ]
    if s.nodelist:
        lines.append(f"#SBATCH --nodelist={s.nodelist}")
    lines += [
        "#SBATCH --output=./slurm/output/%x-%j.out",
        "#SBATCH --error=./slurm/output/%x-%j.err",
        "",
        "set -euo pipefail",
        "",
        "# Run from the directory the job was submitted from (the project root).",
        'cd "${SLURM_SUBMIT_DIR:-.}"',
        "mkdir -p ./slurm/output",
        "",
    ]
    for mod in s.modules:
        lines.append(f"module load {mod}")
    if s.modules:
        lines.append("")

    lines.append(f"source {venv_activate}")
    lines.append("")

    run_cmd = f"md-sim run {config_path}"
    if data_root:
        run_cmd += f" --data-root {data_root}"
    else:
        # Default the data root to the project's local data/ directory.
        lines.append('export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"')
        lines.append("")
    lines.append(run_cmd)
    lines.append("")
    lines.append('echo "Simulation finished."')
    lines.append("")
    return "\n".join(lines)
