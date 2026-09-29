"""Bridge simulation configs to differentiable torch potentials.

Turns a validated :class:`SimulationConfig` into a ready ASE ``Calculator`` (and,
via :func:`build_potential`, a torch-differentiable :class:`Potential`) so ML code
can obtain a reference potential from the same YAML that defines a simulation,
without re-specifying the system.

Note ``ase`` alone is enough for :func:`build_calculator`; only
:func:`build_potential` additionally needs ``torch``.

Two kinds of engine are supported, and they take different routes:

- **OpenMM-family** engines share the ``OpenMMEngine`` base and its ``BuiltSystem``
  contract, which carries everything an OpenMM ``Context`` needs. They export an ASE
  ``Calculator``, which :class:`Potential` makes differentiable through
  :mod:`~md_simulations.torch_potentials.bridge`. Both entry points work.
- **cgschnet** has no ASE calculator: its mlcg model is neither TorchScriptable nor
  an ASE backend, but it *is* already a differentiable torch module, so it provides a
  native :class:`~md_simulations.torch_potentials.cgschnet.CGSchNetPotential`
  directly. :func:`build_potential` handles it; :func:`build_calculator` cannot.

Both return types satisfy
:class:`~md_simulations.torch_potentials.protocol.PotentialLike`. Any further engine
adds its own branch here.

This relies on optional extras. The imports are deferred so this module still imports
on a core install, failing only when a calculator or potential is actually built."""

from __future__ import annotations

import contextlib
import logging
import os
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from md_simulations.config import load_config, resolve_data_root
from md_simulations.config.base import SimulationConfig
from md_simulations.engines import BuiltSystem, OpenMMEngine, build_engine
from md_simulations.torch_potentials.protocol import PotentialLike

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

    from md_simulations.torch_potentials.potential import Potential

# Shared by every deferred import in the package (here and in __init__.__getattr__)
# so the guidance a user hits is identical whichever entry point they came through.
EXTRA_HINT = (
    "needs the 'torch-potentials' extra: install with "
    "`uv sync --extra torch-potentials` "
    "(or `pip install md-simulations[torch-potentials]`)."
)
# cgschnet cannot reuse EXTRA_HINT: it needs the mlcg stack on top of torch, and mlcg
# pins Python ==3.12.*, so a 3.10/3.11 user following the generic hint would retry the
# install forever without ever being told why it resolves to nothing.
CGSCHNET_EXTRA_HINT = (
    "needs the 'cgschnet-potentials' extra on Python 3.12: install with "
    "`uv sync --extra cgschnet-potentials` "
    "(mlcg pins Python ==3.12.*, so it is unavailable on 3.10/3.11)."
)


@contextlib.contextmanager
def _silence_build() -> Iterator[None]:
    """Silence the verbose output emitted while building a calculator.

    The build is noisy on three channels the caller cannot otherwise control:
    ``md_simulations`` logs at INFO, ``openawsem`` prints to stdout, and Biopython
    emits deprecation warnings. This raises the ``md_simulations`` logger to
    WARNING, redirects stdout to ``os.devnull``, and ignores warnings for the
    duration, restoring all three on exit. Exceptions and WARNING+ logs still
    propagate.

    :yield: None."""
    pkg_logger = logging.getLogger("md_simulations")
    prev_level = pkg_logger.level
    pkg_logger.setLevel(logging.WARNING)
    try:
        with (
            warnings.catch_warnings(),
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
        ):
            warnings.simplefilter("ignore")
            yield
    finally:
        pkg_logger.setLevel(prev_level)


def _build_openmm_system(
    config: SimulationConfig,
    data_root: str | Path | None,
    quiet: bool,
) -> BuiltSystem:
    """Build the engine's ``BuiltSystem``, rejecting engines with no OpenMM bridge.

    :param config: A validated engine-specific config model.
    :param data_root: Data root for resolving inputs, or ``None`` for the usual precedence.
    :param quiet: Suppress the verbose build output.
    :return: The engine's ``BuiltSystem`` (system, positions, box vectors).
    :raises NotImplementedError: If the config's engine has no calculator bridge."""
    root = resolve_data_root(config, str(data_root) if data_root is not None else None)

    logger = logging.getLogger("md_simulations.torch_potentials")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    with _silence_build() if quiet else contextlib.nullcontext():
        engine = build_engine(config, root, logger)

        if not isinstance(engine, OpenMMEngine):
            raise NotImplementedError(
                f"calculator export is not yet supported for engine {config.engine!r}"
            )

        return engine.build_system()


def build_calculator(
    config: SimulationConfig,
    data_root: str | Path | None = None,
    *,
    platform: str | None = None,
    groups: int | set[int] = -1,
    quiet: bool = True,
) -> "Calculator":
    """Build an ASE calculator from a simulation config.

    :param config: A validated engine-specific config model.
    :param data_root: Data root for resolving inputs; falls back to the usual
        ``MD_DATA_ROOT`` env var / ``config.data_root`` precedence when ``None``.
    :param platform: Optional OpenMM platform name (e.g. ``"CUDA"``, ``"CPU"``).
    :param groups: OpenMM force groups to include (bitmask or set of indices);
        default ``-1`` is all groups.
    :param quiet: Suppress the verbose build output (``md_simulations`` INFO logs,
        ``openawsem`` stdout, Biopython warnings) while constructing the calculator.
        Defaults to ``True``; pass ``False`` to let it through.
    :return: A ready ASE ``Calculator`` evaluating this system's energy/forces.
    :raises NotImplementedError: If the config's engine has no calculator bridge."""
    if config.engine == "cgschnet":
        # Checked before build_engine: there is no ASE calculator to build, and
        # CGSchNetEngine would only be constructed to be thrown away.
        raise NotImplementedError(
            "cgschnet has no ASE calculator: its mlcg model is neither TorchScriptable "
            "nor an ASE backend. It is natively differentiable instead, so use "
            "build_potential()/build_potential_from_file() to get a CGSchNetPotential."
        )

    built = _build_openmm_system(config, data_root, quiet)

    try:
        from md_simulations.torch_potentials.calculators.openmm import OpenMMCalculator
    except ModuleNotFoundError as exc:  # pragma: no cover - needs a core-only install
        raise ModuleNotFoundError(f"building a calculator {EXTRA_HINT}") from exc

    return OpenMMCalculator.from_system(
        built.system,
        built.positions,
        box_vectors=built.box_vectors,
        groups=groups,
        platform=platform,
    )


def build_calculator_from_file(
    path: str | Path,
    data_root: str | Path | None = None,
    *,
    platform: str | None = None,
    groups: int | set[int] = -1,
    quiet: bool = True,
) -> "Calculator":
    """Load a YAML config and build its ASE calculator.

    Mirrors the ``md-sim run <config>`` ergonomics for the calculator path.

    :param path: Path to the YAML config file.
    :param data_root: See :func:`build_calculator`.
    :param platform: See :func:`build_calculator`.
    :param groups: See :func:`build_calculator`.
    :param quiet: See :func:`build_calculator`.
    :return: A ready ASE ``Calculator``."""
    return build_calculator(
        load_config(path), data_root, platform=platform, groups=groups, quiet=quiet
    )


def build_potential(
    topology: "mdtraj.Topology",  # noqa: F821
    config: SimulationConfig,
    data_root: str | Path | None = None,
    *,
    platform: str | None = None,
    groups: int | set[int] = -1,
    quiet: bool = True,
) -> PotentialLike:
    """Build a torch-differentiable potential from a simulation config.

    For OpenMM-family engines this is the one-call form of
    ``Potential(topology, build_calculator(config))``: the config fixes the force
    field, the topology fixes the atoms, and the result is callable with torch tensors
    (nm in, kJ/mol out). For ``cgschnet`` it returns a native
    :class:`~md_simulations.torch_potentials.cgschnet.CGSchNetPotential`, which is
    batched over frames and twice differentiable.

    :param topology: mdtraj ``Topology`` whose atoms match the config's system.
    :param config: A validated engine-specific config model.
    :param data_root: See :func:`build_calculator`.
    :param platform: See :func:`build_calculator`; for cgschnet, a torch device name.
    :param groups: See :func:`build_calculator`; cgschnet has no force groups, so
        anything other than the default ``-1`` is an error.
    :param quiet: See :func:`build_calculator`.
    :return: A :class:`PotentialLike` bound to ``topology`` and this config's engine.
    :raises NotImplementedError: If the config's engine has no potential bridge."""
    if config.engine == "cgschnet":
        return _build_cgschnet_potential(topology, config, data_root, platform, groups, quiet)

    try:
        from md_simulations.torch_potentials.potential import Potential
    except ModuleNotFoundError as exc:  # pragma: no cover - needs a core-only install
        raise ModuleNotFoundError(f"building a potential {EXTRA_HINT}") from exc

    return Potential(
        topology,
        build_calculator(config, data_root, platform=platform, groups=groups, quiet=quiet),
    )


def _build_cgschnet_potential(
    topology: "mdtraj.Topology",  # noqa: F821
    config: SimulationConfig,
    data_root: str | Path | None,
    platform: str | None,
    groups: int | set[int],
    quiet: bool,
) -> PotentialLike:
    """Build the native cgschnet potential, bypassing the ASE/engine path entirely.

    :param topology: The caller's coarse-grained topology.
    :param config: A validated ``CGSchNetConfig``.
    :param data_root: See :func:`build_calculator`.
    :param platform: An OpenMM-style platform name or a torch device; ``None`` defers
        to ``config.device``.
    :param groups: Must be the default ``-1``.
    :param quiet: Suppress the verbose build output while loading the checkpoint.
    :return: A :class:`~md_simulations.torch_potentials.cgschnet.CGSchNetPotential`.
    :raises ValueError: If ``groups`` selects a force-group subset.
    :raises ModuleNotFoundError: If the mlcg stack is unavailable."""
    if groups != -1:
        raise ValueError(
            f"force groups are an OpenMM concept; cgschnet has none (got groups={groups!r})."
        )

    root = resolve_data_root(config, str(data_root) if data_root is not None else None)

    logger = logging.getLogger("md_simulations.torch_potentials")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    try:
        from md_simulations.torch_potentials.cgschnet import CGSchNetPotential
    except ModuleNotFoundError as exc:  # pragma: no cover - needs a core-only install
        raise ModuleNotFoundError(f"building a potential {EXTRA_HINT}") from exc

    with _silence_build() if quiet else contextlib.nullcontext():
        try:
            return CGSchNetPotential.from_config(
                config,
                root,
                topology,
                device=None if platform is None else platform.lower(),
            )
        except ModuleNotFoundError as exc:
            # torch.load unpickles mlcg classes, so a missing mlcg surfaces here as a
            # bare "No module named 'mlcg'" from deep inside the unpickler.
            raise ModuleNotFoundError(
                f"building a cgschnet potential {CGSCHNET_EXTRA_HINT}"
            ) from exc


def build_potential_terms(
    topology: "mdtraj.Topology",  # noqa: F821
    config: SimulationConfig,
    data_root: str | Path | None = None,
    *,
    platform: str | None = None,
    quiet: bool = True,
) -> dict[str, "Potential"]:
    """Build one potential per OpenMM force, so an energy can be read term by term.

    ``build_potential`` returns the total; this returns the summands. Every force is
    assigned its own force group here, *before* the ``Context`` is created -- OpenMM
    snapshots the group assignments at construction, so a system built the usual way
    leaves every force in group 0 and ``groups={0}`` would return the total again.

    All the returned potentials share one ``Context`` but carry their own calculator.
    That is deliberate: ASE only recomputes when the positions, cell or pbc change, so
    mutating ``groups`` on a single shared calculator would silently return the
    previously evaluated term.

    Every force is included, so terms that carry no energy (``CMMotionRemover``) appear
    as a constant zero, and the terms sum to ``build_potential``'s total.

    :param topology: mdtraj ``Topology`` whose atoms match the config's system.
    :param config: A validated engine-specific config model.
    :param data_root: See :func:`build_calculator`.
    :param platform: See :func:`build_calculator`.
    :param quiet: See :func:`build_calculator`.
    :return: Force name -> :class:`Potential`, ordered as the forces appear in the
        system. Repeated force classes are suffixed ``#2``, ``#3``, ...
    :raises NotImplementedError: If the config's engine has no calculator bridge.
    :raises ValueError: If the system has more than 32 forces, OpenMM's group limit."""
    try:
        from md_simulations.torch_potentials.calculators.openmm import OpenMMCalculator
        from md_simulations.torch_potentials.potential import Potential
    except ModuleNotFoundError as exc:  # pragma: no cover - needs a core-only install
        raise ModuleNotFoundError(f"building a potential {EXTRA_HINT}") from exc

    built = _build_openmm_system(config, data_root, quiet)

    num_forces = built.system.getNumForces()
    if num_forces > 32:
        raise ValueError(
            f"system has {num_forces} forces but OpenMM allows only 32 force groups; "
            "a per-term split is not possible for this system."
        )

    names: list[str] = []
    seen: dict[str, int] = {}
    for i in range(num_forces):
        force = built.system.getForce(i)
        force.setForceGroup(i)
        name = type(force).__name__
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name}#{seen[name]}")

    # one Context for all of them; the first calculator is what creates it.
    first = OpenMMCalculator.from_system(
        built.system,
        built.positions,
        box_vectors=built.box_vectors,
        groups={0},
        platform=platform,
    )
    calculators = [first] + [
        OpenMMCalculator(first.context, groups={i}) for i in range(1, num_forces)
    ]

    return {name: Potential(topology, calc) for name, calc in zip(names, calculators)}


def build_potential_terms_from_file(
    topology: "mdtraj.Topology",  # noqa: F821
    path: str | Path,
    data_root: str | Path | None = None,
    *,
    platform: str | None = None,
    quiet: bool = True,
) -> dict[str, "Potential"]:
    """Load a YAML config and build one potential per OpenMM force.

    :param topology: mdtraj ``Topology`` whose atoms match the config's system.
    :param path: Path to the YAML config file.
    :param data_root: See :func:`build_calculator`.
    :param platform: See :func:`build_calculator`.
    :param quiet: See :func:`build_calculator`.
    :return: See :func:`build_potential_terms`."""
    return build_potential_terms(
        topology,
        load_config(path),
        data_root,
        platform=platform,
        quiet=quiet,
    )


def build_potential_from_file(
    topology: "mdtraj.Topology",  # noqa: F821
    path: str | Path,
    data_root: str | Path | None = None,
    *,
    platform: str | None = None,
    groups: int | set[int] = -1,
    quiet: bool = True,
) -> PotentialLike:
    """Load a YAML config and build its torch-differentiable potential.

    :param topology: mdtraj ``Topology`` whose atoms match the config's system.
    :param path: Path to the YAML config file.
    :param data_root: See :func:`build_calculator`.
    :param platform: See :func:`build_calculator`.
    :param groups: See :func:`build_calculator`.
    :param quiet: See :func:`build_calculator`.
    :return: A :class:`PotentialLike` bound to ``topology`` and this config's engine."""
    return build_potential(
        topology,
        load_config(path),
        data_root,
        platform=platform,
        groups=groups,
        quiet=quiet,
    )
