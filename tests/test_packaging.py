"""Pin the eager import set so the slim installs cannot regress.

``import md_simulations`` must not pull the ``simulate`` stack (openmm, beamtime, tqdm,
mdtraj) nor the ``torch-potentials`` one (torch, ase). Each check runs in a subprocess,
a clean interpreter rather than this session's polluted ``sys.modules``."""

import json
import subprocess
import sys

import pytest

from md_simulations.config.base import GromacsNativeConfig

FORBIDDEN = ("openmm", "beamtime", "tqdm", "mdtraj", "torch", "ase")

# Importing these must not pull the forbidden set either: the config/workspace layer and
# the CLI are what ``md-sim check-job`` and ``md-sim config`` run on a core install.
CORE_MODULES = (
    "md_simulations",
    "md_simulations.torch_potentials",
    "md_simulations.config",
    "md_simulations.workspace",
    "md_simulations.preflight",
    "md_simulations.cli",
)


def _run(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a clean interpreter.

    :param code: Python source for ``python -c``.
    :return: The completed process with captured text output."""
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def _loaded_forbidden(module: str) -> list[str]:
    """Import ``module`` in a subprocess and list the forbidden top-level packages loaded.

    :param module: Dotted module name.
    :return: The forbidden packages present in ``sys.modules`` afterwards."""
    proc = _run(
        f"import {module}; import sys, json; "
        "print(json.dumps(sorted(m.split('.')[0] for m in sys.modules)))"
    )
    assert proc.returncode == 0, proc.stderr
    loaded = set(json.loads(proc.stdout.splitlines()[-1]))
    return sorted(loaded & set(FORBIDDEN))


def test_eager_imports() -> None:
    """``import md_simulations`` imports none of the forbidden third-party packages.

    :return: None."""
    assert _loaded_forbidden("md_simulations") == []


@pytest.mark.parametrize("module", CORE_MODULES[1:])
def test_core_layer_imports_are_clean(module: str) -> None:
    """The torch_potentials package, config, workspace, preflight and cli import cleanly.

    :param module: Module to import (parametrised).
    :return: None."""
    assert _loaded_forbidden(module) == []


def test_annotations_do_not_import_engines() -> None:
    """``build`` keeps the engine names under TYPE_CHECKING, so nothing evaluates them.

    :return: None."""
    proc = _run(
        "import sys; import md_simulations.torch_potentials.build as b; "
        "assert 'md_simulations.engines' not in sys.modules; "
        "print(b._build_openmm_system.__annotations__['return'])"
    )
    assert proc.returncode == 0, proc.stderr
    assert "BuiltSystem" in proc.stdout


@pytest.mark.parametrize("first", ["md_simulations.engines", "md_simulations.torch_potentials"])
def test_import_order_both_ways(first: str) -> None:
    """Engines and torch_potentials import in either order without a circular import.

    :param first: Module imported first (parametrised).
    :return: None."""
    other = (
        "md_simulations.torch_potentials"
        if first == "md_simulations.engines"
        else "md_simulations.engines"
    )
    proc = _run(f"import {first}; import {other}; import md_simulations.cli")
    assert proc.returncode == 0, proc.stderr


def test_openmm_engine_dispatch_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deferred import keeps the ``OpenMMEngine`` check and the engine factory live.

    A gromacs_native engine is no OpenMMEngine and is rejected; an OpenMMEngine subclass
    returned by the factory has its ``build_system`` called.

    :param monkeypatch: Pytest monkeypatch fixture.
    :return: None."""
    import md_simulations.engines as engines
    from md_simulations.torch_potentials.build import _build_openmm_system

    cfg = GromacsNativeConfig(
        system="t", output_subdir="t", input_gro="a.gro", top="a.top", mdp="m.mdp"
    )
    with pytest.raises(NotImplementedError, match="gromacs_native"):
        _build_openmm_system(cfg, "/data", True)

    sentinel = object()

    class _Stub(engines.OpenMMEngine):
        def __init__(self) -> None:
            pass

        def build_system(self):
            return sentinel

        def run(self) -> None:
            pass

    monkeypatch.setattr(engines, "build_engine", lambda *a, **k: _Stub())
    assert _build_openmm_system(cfg, "/data", True) is sentinel


_BLOCK_SIMULATE = (
    "import sys\n"
    "class Block:\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name.split('.')[0] in ('openmm', 'beamtime'):\n"
    "            raise ModuleNotFoundError(f'No module named {name!r}', name=name)\n"
    "sys.meta_path.insert(0, Block())\n"
)


def test_engines_without_openmm_names_the_simulate_extra() -> None:
    """With openmm unimportable, ``import md_simulations.engines`` raises with the hint.

    :return: None."""
    proc = _run(
        _BLOCK_SIMULATE + "try:\n"
        "    import md_simulations.engines\n"
        "except ModuleNotFoundError as exc:\n"
        "    assert 'simulate' in str(exc), exc\n"
        "    assert exc.name in ('openmm', 'beamtime'), exc.name\n"
        "    print('ok')\n"
        "else:\n"
        "    raise SystemExit('engines imported without openmm')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["ok"]


def test_core_numpy_submodules_import_without_openmm() -> None:
    """Without openmm, ``md_simulations.core`` imports and its numpy-only submodules resolve.

    ``prep/migrate_forces.py`` and ``prep/cgschnet_dataset.py`` import ``core.forces`` and
    belong to extras that carry no openmm, so the package ``__init__`` must not import it.
    ``setup_logger`` is stdlib only and resolves too.

    :return: None."""
    proc = _run(
        _BLOCK_SIMULATE + "import md_simulations.core\n"
        "from md_simulations.core.forces import FORCES_FILE, ForceWriter, read_forces\n"
        "from md_simulations.core.replica_exchange import swap_probability\n"
        "from md_simulations.core import setup_logger\n"
        "import md_simulations.prep.migrate_forces, md_simulations.prep.cgschnet_dataset\n"
        "assert 'openmm' not in sys.modules\n"
        "print('ok')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["ok"]


@pytest.mark.parametrize("name", ["create_integrator", "start_run", "ReplicaExchange"])
def test_core_simulate_names_raise_the_hint_on_access(name: str) -> None:
    """An openmm- or beamtime-backed ``core`` name raises the ``simulate`` hint on first access.

    :param name: Re-exported name (parametrised).
    :return: None."""
    proc = _run(
        _BLOCK_SIMULATE + "import md_simulations.core as core\n"
        "try:\n"
        f"    core.{name}\n"
        "except ModuleNotFoundError as exc:\n"
        "    assert exc.args[0] == core.EXTRA_HINT, exc\n"
        "    assert exc.name in ('openmm', 'beamtime'), exc.name\n"
        "    print('ok')\n"
        "else:\n"
        f"    raise SystemExit('{name} resolved without openmm')\n"
        "try:\n"
        f"    from md_simulations.core import {name}\n"
        "except ModuleNotFoundError as exc:\n"
        "    assert exc.args[0] == core.EXTRA_HINT, exc\n"
        "else:\n"
        f"    raise SystemExit('from-import of {name} resolved without openmm')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["ok"]


def test_extra_hint_is_shared() -> None:
    """Both packages export the same install wording apart from the module name.

    :return: None."""
    from md_simulations import core, engines

    assert "[simulate]" in engines.EXTRA_HINT and "[simulate]" in core.EXTRA_HINT
