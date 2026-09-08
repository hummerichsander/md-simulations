from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class SlurmConfig(BaseModel):
    """SLURM batch-job resource request for a simulation.

    Mirrors the ``#SBATCH`` directives used across the original per-system
    scripts. Only the fields that varied between jobs are exposed.
    """

    job_name: str
    partition: str = "a30"
    gres: str = "gpu:1"
    time: str = "0-40:00:00"
    mem: str = "20G"
    nodes: int = 1
    nodelist: str | None = None
    modules: list[str] = Field(default_factory=list)


class _Common(BaseModel):
    """Fields shared by every engine configuration.

    Path fields (``input_*``) and ``output_subdir`` are resolved against the
    data root at run time via :meth:`resolve` / :meth:`output_dir`.
    """

    system: str
    output_subdir: str
    data_root: str | None = None

    num_steps: int = 10_000_000
    output_freq: int = 1_000
    report_interval: int = 100
    temperature: float = 300.0
    friction: float = 1.0
    timestep: float = 0.002
    integrator: Literal["langevin", "langevin_middle"] = "langevin"
    minimize_energy: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    slurm: SlurmConfig | None = None

    def resolve(self, data_root: Path, path: str) -> Path:
        """Resolve *path* against *data_root* (absolute paths pass through).

        :param data_root: Root directory holding inputs and outputs.
        :param path: Path string, absolute or relative to *data_root*.
        :return: Resolved absolute :class:`pathlib.Path`."""
        p = Path(path)
        return p if p.is_absolute() else data_root / p

    def output_dir(self, data_root: Path) -> Path:
        """Return the output directory ``<data_root>/<output_subdir>``.

        :param data_root: Root directory holding inputs and outputs.
        :return: Resolved output directory path."""
        return self.resolve(data_root, self.output_subdir)


class AmberConfig(_Common):
    """All-atom AMBER simulation, explicit solvent (PME) or implicit GB."""

    engine: Literal["amber"] = "amber"
    input_pdb: str
    forcefield: list[str] = Field(
        default_factory=lambda: ["amber14-all.xml", "amber14/tip3pfb.xml"]
    )
    ph: float = 7.0
    padding: float = 1.0
    water_model: str = "tip3p"
    nonbonded_cutoff: float = 1.0
    implicit_solvent: bool = False
    salt_conc: float = 0.0
    # "hbonds" is what every trajectory in this repo was produced with, and a 2 fs timestep needs
    # it. Set "none" only to build a *scoring* potential: OpenMM deletes the harmonic bond term of
    # every constrained bond, so under "hbonds" the energy is nearly independent of the X-H bond
    # lengths, and reweighting a generative model against it charges the model for coordinates the
    # target has no opinion about. An unconstrained system needs a ~0.5 fs timestep to integrate.
    constraints: Literal["hbonds", "allbonds", "none"] = "hbonds"


class PureLiquidConfig(_Common):
    """NPT simulation of a pre-built, fully periodic box (no solvation)."""

    engine: Literal["pure_liquid"] = "pure_liquid"
    input_pdb: str
    forcefield: list[str] = Field(default_factory=lambda: ["charmm36.xml", "charmm36/water.xml"])
    pressure: float = 1.0


class MartiniConfig(_Common):
    """MARTINI3 coarse-grained simulation from GROMACS GRO/TOP inputs."""

    engine: Literal["martini"] = "martini"
    input_gro: str
    top: str
    include_dir: str = "."
    cutoff: float = 1.1
    pressure: float | None = None
    timestep: float = 0.020
    friction: float = 1.0


class GromacsInputConfig(_Common):
    """OpenMM run directly from GROMACS-format inputs (e.g. FreeSolv)."""

    engine: Literal["gromacs_input"] = "gromacs_input"
    input_gro: str
    input_top: str
    nonbonded_cutoff: float = 1.0
    switch_distance: float = 0.9
    ewald_error_tolerance: float = 1.0e-6
    use_barostat: bool = False
    pressure: float = 1.01325
    barostat_interval: int = 25
    initialize_velocities: bool = False
    temperature: float = 298.15
    integrator: Literal["langevin", "langevin_middle"] = "langevin_middle"


class AwsemConfig(_Common):
    """OpenAWSEM coarse-grained simulation (requires the ``awsem`` extra)."""

    engine: Literal["awsem"] = "awsem"
    input_pdb: str
    prepare: bool = False
    cis_proline: bool = False
    chains: str = "A"
    parameters_dir: str | None = None
    k_awsem: float = 1.0

    use_contact: bool = True
    use_burial: bool = True
    use_rama: bool = True
    use_beta: bool = True
    use_helical: bool = False
    use_electrostatics: bool = True
    k_contact: float = 4.184
    k_beta: float = 4.184

    use_associative_memory: bool = False
    memory_pdbs: list[str] = Field(default_factory=list)
    memory_chains: list[str] = Field(default_factory=list)
    memory_weights: list[float] = Field(default_factory=list)
    memory_target_starts: list[int] = Field(default_factory=list)
    memory_fragment_starts: list[int] = Field(default_factory=list)
    memory_lengths: list[int] = Field(default_factory=list)
    k_am: float = 0.8368
    am_min_seq_sep: int = 2
    am_max_seq_sep: int = 9
    am_well_width: float = 0.1


class CGSchNetConfig(_Common):
    """ML coarse-grained (CGSchNet / Clementi-group mlcg) simulation.

    The model consumes ``mlcg.AtomicData`` and is not TorchScriptable, so it
    cannot run under OpenMM-Torch; dynamics are driven by mlcg's own Langevin
    integrator (:class:`mlcg.simulation.LangevinSimulation`). Provide:

    - ``model_file``: a re-exported model+prior (see ``md-sim-cgschnet-reexport``;
      the distributed checkpoints are pickled against an older torch-geometric).
    - ``configurations_file``: an mlcg configurations ``.pt`` (a ``List[AtomicData]``
      carrying initial positions, atom types, masses and prior neighbor lists).
    - ``input_pdb``: a CG PDB (one atom per bead) used only as the topology for
      trajectory/PDB output; a generic bead topology is synthesized if it does
      not match the bead count.

    Requires the ``cgschnet`` extra."""

    engine: Literal["cgschnet"] = "cgschnet"
    input_pdb: str
    model_file: str
    configurations_file: str

    device: str = "cuda"
    dtype: Literal["float32", "float64"] = "float32"
    model_energy_unit: Literal["kcal/mol", "kJ/mol"] = "kcal/mol"
    # Starting index into the configurations list for the first replica.
    replica: int = 0
    # Number of replicas to propagate in parallel (batched in one forward each
    # step — the main throughput lever, since a single CG molecule badly
    # underuses the GPU). Starting structures are taken from the configurations
    # list (tiled if ``n_replicas`` exceeds its length). Each replica is written
    # to its own trajectory/energies/forces files when > 1.
    n_replicas: int = 1

    # NN potentials are started from a valid CG frame and run directly.
    minimize_energy: bool = False


class GromacsNativeConfig(_Common):
    """Native GROMACS run via ``gmx grompp`` + ``gmx mdrun`` (``gromacs`` extra)."""

    engine: Literal["gromacs_native"] = "gromacs_native"
    input_gro: str
    top: str
    mdp: str
    index: str | None = None
    gmx: str = "gmx"
    mdrun_extra: list[str] = Field(default_factory=list)


SimulationConfig = Annotated[
    Union[
        AmberConfig,
        PureLiquidConfig,
        MartiniConfig,
        GromacsInputConfig,
        AwsemConfig,
        CGSchNetConfig,
        GromacsNativeConfig,
    ],
    Field(discriminator="engine"),
]
