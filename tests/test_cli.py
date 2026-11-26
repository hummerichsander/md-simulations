"""``md-sim`` as one noun/verb entry point: dispatch, aliases, and the ``--json`` envelope."""

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

from md_simulations.cli import main
from md_simulations.cli._dispatch import GROUPS, dispatch
from md_simulations.cli.main import run_cli

ALIASED = [(group, verb) for group, verbs in GROUPS.items() for verb in verbs]


def _fake_module(monkeypatch, name: str, func) -> str:
    """Register a throwaway module whose ``main`` is ``func`` and return its dispatch target.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param name: Module name to register in ``sys.modules``.
    :param func: The ``main`` function.
    :return: The ``module:function`` string for :data:`GROUPS`."""
    module = types.ModuleType(name)
    module.main = func
    monkeypatch.setitem(sys.modules, name, module)
    return f"{name}:main"


def test_help_lists_commands_and_groups(capsys) -> None:
    """``md-sim --help`` names every command and noun group.

    :param capsys: Pytest capture fixture.
    :return: None."""
    assert run_cli(["--help"]) == 0
    out = capsys.readouterr().out
    for name in ("run", "config", "prep", "analysis", "sampling", "check-job"):
        assert f"\n    {name} " in out


def test_json_listing_enumerates_the_surface(capsys) -> None:
    """``md-sim --json`` and ``md-sim <group> --json`` list verbs with their aliases.

    :param capsys: Pytest capture fixture.
    :return: None."""
    assert run_cli(["--json"]) == 0
    surface = json.loads(capsys.readouterr().out)
    assert surface["ok"] is True
    assert {c["name"] for c in surface["commands"]} == {"run", "check-job"}
    groups = {g["name"]: g for g in surface["groups"]}
    assert set(groups) == {"config", "prep", "analysis", "sampling"}
    assert len(groups["prep"]["verbs"]) == 8

    assert run_cli(["analysis", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert [v["name"] for v in listing["verbs"]] == ["tica", "cvs", "basins"]
    assert listing["verbs"][0]["alias"] == "md-sim-tica"


def test_group_without_verb_lists_verbs(capsys) -> None:
    """``md-sim prep`` prints the verbs; an unknown verb is a usage error.

    :param capsys: Pytest capture fixture.
    :return: None."""
    assert run_cli(["prep"]) == 0
    assert "cgschnet-priors" in capsys.readouterr().out
    assert run_cli(["prep", "nope", "--json"]) == 2
    err = json.loads(capsys.readouterr().out)
    assert err["ok"] is False and err["error"]["type"] == "ValueError"
    assert run_cli(["config", "nope"]) == 2
    assert "unknown config verb" in capsys.readouterr().err


def test_passthrough_preserves_argv(monkeypatch) -> None:
    """Everything after the verb reaches the script, including its own ``--json``.

    :param monkeypatch: Pytest monkeypatch fixture.
    :return: None."""
    seen: list[list[str]] = []
    target = _fake_module(monkeypatch, "fake_cli_a", lambda: seen.append(list(sys.argv)))
    monkeypatch.setitem(GROUPS["prep"], "fake", target)
    argv = ["--json", "-x", "a b", "--help", "--", "--json"]
    before = list(sys.argv)
    assert run_cli(["prep", "fake", *argv]) == 0
    assert seen == [["md-sim-fake", *argv]]
    assert sys.argv == before


def test_usage_shows_alias(monkeypatch, capsys) -> None:
    """A script's argparse usage line shows ``md-sim-<verb>``, as the alias would.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param capsys: Pytest capture fixture.
    :return: None."""
    import argparse

    def script_main() -> None:
        argparse.ArgumentParser(description="d").parse_args()

    target = _fake_module(monkeypatch, "fake_cli_b", script_main)
    monkeypatch.setitem(GROUPS["analysis"], "fake", target)
    assert run_cli(["analysis", "fake", "--help"]) == 0
    assert capsys.readouterr().out.startswith("usage: md-sim-fake ")


def test_exit_code_propagates(monkeypatch, capsys) -> None:
    """A script's exit code, message exit and plain return survive the dispatcher.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param capsys: Pytest capture fixture.
    :return: None."""

    def exit_three() -> None:
        raise SystemExit(3)

    def exit_message() -> None:
        sys.exit("boom")

    def return_none() -> None:
        return None

    for verb, func, code in (("three", exit_three, 3), ("msg", exit_message, 1)):
        target = _fake_module(monkeypatch, f"fake_cli_{verb}", func)
        monkeypatch.setitem(GROUPS["prep"], verb, target)
        assert run_cli(["prep", verb]) == code
    assert "boom" in capsys.readouterr().err
    monkeypatch.setitem(GROUPS["prep"], "ok", _fake_module(monkeypatch, "fake_cli_ok", return_none))
    assert dispatch("prep", "ok", []) == 0
    with pytest.raises(SystemExit) as exc:
        monkeypatch.setattr(sys, "argv", ["md-sim", "prep", "three"])
        main()
    assert exc.value.code == 3


def test_no_torch_import_on_help() -> None:
    """Help and listings import no heavy stack; verbs are resolved by string.

    :return: None."""
    code = (
        "import sys; from md_simulations.cli import main\n"
        "for argv in (['--help'], ['prep'], ['--json'], ['sampling', '--json']):\n"
        "    sys.argv = ['md-sim', *argv]\n"
        "    try:\n"
        "        main()\n"
        "    except SystemExit:\n"
        "        pass\n"
        "bad = sorted({'torch', 'openmm', 'mdtraj', 'beamtime', 'ase'} & "
        "{m.split('.')[0] for m in sys.modules})\n"
        "pre = [m for m in sys.modules if m.startswith('md_simulations.prep')]\n"
        "print(bad, pre)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[-1] == "[] []"


def test_aliases_match_pyproject() -> None:
    """All 15 ``md-sim-*`` entry points exist and point where the dispatch table does.

    :return: None."""
    tomllib = pytest.importorskip("tomllib")
    root = Path(__file__).resolve().parent.parent
    scripts = tomllib.loads((root / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts.pop("md-sim") == "md_simulations.cli:main"
    table = {
        f"md-sim-{verb}": target for verbs in GROUPS.values() for verb, target in verbs.items()
    }
    assert len(table) == 14
    assert scripts == table


@pytest.mark.parametrize(("group", "verb"), ALIASED)
def test_alias_help_is_identical(group: str, verb: str, monkeypatch, capsys) -> None:
    """``md-sim <group> <verb> --help`` prints what ``md-sim-<verb> --help`` prints.

    :param group: Dispatch group.
    :param verb: Verb of that group.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param capsys: Pytest capture fixture.
    :return: None."""
    import importlib

    module_name, _, func_name = GROUPS[group][verb].partition(":")
    try:
        func = getattr(importlib.import_module(module_name), func_name)
    except ModuleNotFoundError as exc:
        pytest.skip(f"{module_name} needs an extra that is not installed: {exc}")
    monkeypatch.setattr(sys, "argv", [f"/venv/bin/md-sim-{verb}", "--help"])
    with pytest.raises(SystemExit) as exc:
        func()
    assert exc.value.code == 0
    direct = capsys.readouterr().out
    assert run_cli([group, verb, "--help"]) == 0
    assert capsys.readouterr().out == direct
    assert direct.startswith(f"usage: md-sim-{verb} ")


# --- md-sim run --json -------------------------------------------------------------------


class _Engine:
    """A stand-in engine that logs to the console handler and prints to stdout."""

    fail = False

    def __init__(self, config, data_root, logger, resume=False) -> None:
        """Keep the logger.

        :param config: Unused.
        :param data_root: Unused.
        :param logger: The run logger.
        :param resume: Unused.
        :return: None."""
        self.logger = logger

    def run(self) -> None:
        """Log, print, then finish or fail.

        :return: None."""
        self.logger.info("engine running")
        print("reporter line on stdout")
        if self.fail:
            raise RuntimeError("kaboom")


@pytest.fixture
def run_config(tmp_path: Path, monkeypatch) -> Path:
    """A valid AMBER config on disk, with the engine factory stubbed out.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :return: Path of the config file."""
    pytest.importorskip("openmm")
    monkeypatch.setattr("md_simulations.engines.build_engine", _Engine)
    monkeypatch.setattr(_Engine, "fail", False)
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump(
            {"engine": "amber", "system": "t", "output_subdir": "t/run", "input_pdb": "a.pdb"}
        )
    )
    return path


def test_run_json_prints_one_object(run_config: Path, tmp_path: Path, capsys) -> None:
    """Under ``--json`` stdout holds one JSON object; the run's own output moves to stderr.

    :param run_config: Config fixture.
    :param tmp_path: Pytest temporary directory.
    :param capsys: Pytest capture fixture.
    :return: None."""
    assert run_cli(["run", str(run_config), "--json", "--data-root", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert list(payload) == ["ok", "config", "data_root", "output_dir", "engine", "system", "steps"]
    assert payload["ok"] is True and payload["engine"] == "amber" and payload["system"] == "t"
    assert payload["output_dir"] == str(tmp_path / "t" / "run")
    assert "engine running" in captured.err and "reporter line" in captured.err
    assert "engine running" not in captured.out


def test_run_human_mode_keeps_logging_on_stdout(run_config: Path, tmp_path: Path, capsys) -> None:
    """Without ``--json`` nothing changes: logging and reporters print to stdout.

    :param run_config: Config fixture.
    :param tmp_path: Pytest temporary directory.
    :param capsys: Pytest capture fixture.
    :return: None."""
    assert run_cli(["run", str(run_config), "--data-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "engine running" in out and "reporter line" in out and '"ok"' not in out


def test_run_failure_json_envelope(run_config: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    """A failing run exits 1 with ``{"ok": false, "error": {"type", "message"}}`` on stdout.

    :param run_config: Config fixture.
    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param capsys: Pytest capture fixture.
    :return: None."""
    monkeypatch.setattr(_Engine, "fail", True)
    assert run_cli(["run", str(run_config), "--json", "--data-root", str(tmp_path)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["type"] == "RuntimeError"
    assert "kaboom" in payload["error"]["message"]

    assert run_cli(["run", str(tmp_path / "nope.yaml"), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["type"] == "FileNotFoundError"

    assert (
        run_cli(["run", str(run_config), "--json", "--resume", "--data-root", str(tmp_path)]) == 1
    )
    assert "replica_exchange" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_run_failure_human_mode_exits_one(run_config: Path, tmp_path: Path, monkeypatch) -> None:
    """Without ``--json`` a failed run still exits 1 after logging, as before.

    :param run_config: Config fixture.
    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :return: None."""
    monkeypatch.setattr(_Engine, "fail", True)
    assert run_cli(["run", str(run_config), "--data-root", str(tmp_path)]) == 1
