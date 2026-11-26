from pathlib import Path
from typing import Annotated, ClassVar, Literal, Union
import math

from pydantic import BaseModel, Field, model_validator


class BeamtimeConfig(BaseModel):
    """Where a simulation is logged below the Beamtime log dir."""

    # The group the run goes into; the system by default, a replica set below it otherwise.
    group: str | None = None
    name: str | None = None


def _multiples(*values: int) -> bool:
    """Whether every pair of *values* divides one another.

    :param values: Positive integers.
    :return: ``True`` if the smaller of each pair divides the larger."""
    return all(max(a, b) % min(a, b) == 0 for a in values for b in values)


class ReplicaExchangeConfig(BaseModel):
    """Temperature replica exchange (T-REMD) with the target temperature at the bottom.

    The ladder is either given as ``temperatures`` or built geometrically from ``temperature`` up
    to ``t_max`` with ``n_replicas`` rungs. ``output_freq`` of the parent config applies to the
    target temperature; the other temperatures write as set by ``other_output``.
    """

    temperatures: list[float] | None = None
    t_max: float | None = None
    n_replicas: int | None = None
    # Steps between exchange attempts.
    exchange_interval: int = Field(gt=0)
    other_output: Literal["none", "energies", "full"] = "energies"
    # Defaults to the parent's output_freq.
    other_output_freq: int | None = Field(default=None, gt=0)
    # Seeds the exchange RNG; the integrators are seeded seed + 1 + replica index.
    seed: int | None = None
    checkpoint_interval: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check(self) -> "ReplicaExchangeConfig":
        """Require exactly one ladder form, a sane ladder and a checkpoint interval on the grid.

        :return: The validated config."""
        geometric = (self.t_max, self.n_replicas)
        if self.temperatures is not None:
            if any(v is not None for v in geometric):
                raise ValueError("give either `temperatures` or `t_max` + `n_replicas`, not both")
            if len(self.temperatures) < 2:
                raise ValueError("replica exchange needs at least 2 temperatures")
            if any(a >= b for a, b in zip(self.temperatures, self.temperatures[1:])):
                raise ValueError("`temperatures` must be strictly ascending")
        else:
            if any(v is None for v in geometric):
                raise ValueError("give either `temperatures` or both `t_max` and `n_replicas`")
            if self.n_replicas < 2:  # type: ignore[operator]
                raise ValueError("replica exchange needs at least 2 replicas")
        if self.checkpoint_interval and self.checkpoint_interval % self.exchange_interval:
            raise ValueError("`checkpoint_interval` must be a multiple of `exchange_interval`")
        return self


class FrozenAtomsConfig(BaseModel):
    """Atoms held immobile during dynamics, so the run samples the conditional of the rest.

    The atoms are frozen by setting their mass to zero, which takes them out of the equations of
    motion exactly -- unlike a stiff positional restraint, which leaves a thermal smear on the
    very coordinates the conditional is defined by. OpenMM refuses a constraint that involves a
    massless particle, so a build whose constraints touch a frozen atom is rejected at freeze time
    (for an all-atom protein with frozen backbone atoms that means ``constraints: none``, and a
    timestep small enough for explicit X-H stretches).

    Give exactly one of ``names`` or ``indices``.
    """

    # Atom names to freeze, matched against the *built* topology, e.g. ["N", "CA", "CB", "C", "O"].
    names: list[str] | None = None
    # Zero-based atom indices into the built topology.
    indices: list[int] | None = None
    # Guards a name selection against a topology that is not the one it was written for.
    expect_count: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check(self) -> "FrozenAtomsConfig":
        """Check that exactly one selection form is given.

        :return: The validated config."""
        if (self.names is None) == (self.indices is None):
            raise ValueError("give exactly one of `names` or `indices`")
        if self.names is not None and not self.names:
            raise ValueError("`names` must not be empty")
        if self.indices is not None:
            if not self.indices:
                raise ValueError("`indices` must not be empty")
            if min(self.indices) < 0:
                raise ValueError("`indices` must be non-negative")
            if len(set(self.indices)) != len(self.indices):
                raise ValueError("`indices` must not repeat")
        return self


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
    # Atoms to hold immobile; the run then samples the conditional distribution of the others.
    frozen_atoms: FrozenAtomsConfig | None = None
    # forces.npy is the force-matching training data; runs that are not producing it can skip it.
    write_forces: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    replica_exchange: ReplicaExchangeConfig | None = None

    beamtime: BeamtimeConfig = Field(default_factory=BeamtimeConfig)

    # Engines that drive their own integration cannot run through the OpenMM replica-exchange driver.
    supports_replica_exchange: ClassVar[bool] = True

    @model_validator(mode="after")
    def _check_replica_exchange(self) -> "_Common":
        """Check the replica-exchange block against the temperature, step counts and engine.

        :return: The validated config."""
        rex = self.replica_exchange
        if rex is None:
            return self
        if not self.supports_replica_exchange:
            raise ValueError(f"replica_exchange is not supported by the {self.engine} engine")  # type: ignore[attr-defined]
        if rex.temperatures is not None:
            if not math.isclose(rex.temperatures[0], self.temperature):
                raise ValueError(
                    f"the lowest replica_exchange temperature ({rex.temperatures[0]}) must equal "
                    f"`temperature` ({self.temperature})"
                )
        elif rex.t_max <= self.temperature:  # type: ignore[operator]
            raise ValueError(f"`t_max` must exceed `temperature` ({self.temperature})")
        freqs = [self.output_freq, rex.exchange_interval]
        if rex.other_output != "none":
            freqs.append(rex.other_output_freq or self.output_freq)
        if not _multiples(*freqs):
            raise ValueError(
                "`exchange_interval`, `output_freq` and `other_output_freq` must be multiples of "
                f"one another, got {freqs}"
            )
        if self.num_steps % rex.exchange_interval:
            raise ValueError("`num_steps` must be a multiple of `exchange_interval`")
        return self

    @property
    def beamtime_group(self) -> str:
        """The Beamtime group of the run: ``beamtime.group``, else the system."""
        return self.beamtime.group or self.system

    @property
    def beamtime_name(self) -> str:
        """The Beamtime run name: ``beamtime.name``, else the last part of ``output_subdir``."""
        return self.beamtime.name or Path(self.output_subdir).name

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
    supports_replica_exchange: ClassVar[bool] = False
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
    supports_replica_exchange: ClassVar[bool] = False
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
