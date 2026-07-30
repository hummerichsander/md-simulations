from md_simulations.config.base import AmberConfig, GromacsNativeConfig, SlurmConfig
from md_simulations.slurm import render_slurm


def _amber() -> AmberConfig:
    """Build an AMBER config with a SLURM block for rendering tests.

    :return: A minimal AmberConfig with resources set."""
    return AmberConfig(
        system="cln",
        output_subdir="cln/AMBER14_340K",
        input_pdb="pdbs/1UAO.pdb",
        slurm=SlurmConfig(job_name="md-cln-amber14", partition="a30", gres="gpu:1"),
    )


def test_render_amber_slurm() -> None:
    """The generated script carries its resources and the run command.

    :return: None."""
    script = render_slurm(_amber(), "configs/cln-amber14.yaml", data_root="/data")
    assert script.startswith("#!/bin/bash")
    assert "#SBATCH --job-name=md-cln-amber14" in script
    assert "#SBATCH --partition=a30" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "md-sim run configs/cln-amber14.yaml --data-root /data" in script


def test_gromacs_native_slurm_loads_module() -> None:
    """A native-GROMACS config emits its module-load line.

    :return: None."""
    config = GromacsNativeConfig(
        system="fsp",
        output_subdir="fsp/x",
        input_gro="a.gro",
        top="a.top",
        mdp="a.mdp",
        slurm=SlurmConfig(job_name="j", modules=["gromacs"]),
    )
    assert "module load gromacs" in render_slurm(config, "c.yaml")


def test_slurm_runs_from_submit_dir_and_defaults_data_root() -> None:
    """Without an explicit data root the job runs from SLURM_SUBMIT_DIR and
    defaults MD_DATA_ROOT to the project's local data/ directory.

    :return: None."""
    script = render_slurm(_amber(), "configs/cln-amber14.yaml")
    assert 'cd "${SLURM_SUBMIT_DIR:-.}"' in script
    assert 'export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"' in script
    assert "--data-root" not in script
