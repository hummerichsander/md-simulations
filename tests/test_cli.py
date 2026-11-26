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
    """All 15 entry points exist: ``md-sim`` plus the 14 ``md-sim-*`` aliases of the table.

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


# --- md-sim config ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_ambient_roots(monkeypatch) -> None:
    """Keep the developer's data root and workspace out of every test here.

    :param monkeypatch: Pytest monkeypatch fixture.
    :return: None."""
    monkeypatch.delenv("MD_DATA_ROOT", raising=False)
    monkeypatch.delenv("MD_WORKSPACE", raising=False)


def _json(capsys) -> dict:
    """Parse the one JSON object on captured stdout.

    :param capsys: Pytest capture fixture.
    :return: The envelope."""
    return json.loads(capsys.readouterr().out)


def _workspace(root: Path, **configs: dict) -> None:
    """Write a workspace whose ``configs/`` holds the given configs (a str is raw YAML).

    :param root: Workspace directory.
    :param configs: File stem to config dict or raw text.
    :return: None."""
    (root / "configs").mkdir(parents=True)
    (root / "md-workspace.yaml").write_text("config_dir: configs\n")
    for name, body in configs.items():
        text = body if isinstance(body, str) else yaml.safe_dump(body)
        (root / "configs" / f"{name}.yaml").write_text(text)


def test_config_group_lists_its_verbs(capsys) -> None:
    """``md-sim config --json`` lists the five verbs, none with an ``md-sim-*`` alias.

    :param capsys: Pytest capture fixture.
    :return: None."""
    assert run_cli(["config", "--json"]) == 0
    verbs = _json(capsys)["verbs"]
    assert [v["name"] for v in verbs] == ["schema", "validate", "describe", "ls", "new"]
    assert all("alias" not in v for v in verbs)
    assert run_cli(["config", "bogus", "--json"]) == 2
    assert _json(capsys)["ok"] is False


def test_schema_is_the_discriminated_union(capsys) -> None:
    """The schema has one ``oneOf`` branch per engine and keeps its ``$defs``.

    :param capsys: Pytest capture fixture.
    :return: None."""
    assert run_cli(["config", "schema", "--json"]) == 0
    out = _json(capsys)
    assert len(out["payload"]["oneOf"]) == len(out["engines"]) == 7
    assert out["payload"]["discriminator"]["propertyName"] == "engine"
    assert "AmberConfig" in out["payload"]["$defs"] and out["title"] == "SimulationConfig"


def test_schema_engine_branch(capsys) -> None:
    """``--engine amber`` emits only that branch, still resolvable; a bad name exits 2.

    :param capsys: Pytest capture fixture.
    :return: None."""
    assert run_cli(["config", "schema", "--engine", "amber", "--json"]) == 0
    out = _json(capsys)
    assert out["title"] == "AmberConfig" and "input_pdb" in out["payload"]["properties"]
    assert "oneOf" not in out["payload"]
    assert run_cli(["config", "schema", "--engine", "nope", "--json"]) == 2
    assert "amber" in _json(capsys)["error"]["message"]


def test_validate_good_file(sample_configs, write_yaml, capsys) -> None:
    """A valid config exits 0 with ``{"ok": true}``.

    :param sample_configs: Per-engine sample payloads.
    :param write_yaml: Config writer fixture.
    :param capsys: Pytest capture fixture.
    :return: None."""
    path = write_yaml(sample_configs["amber"])
    assert run_cli(["config", "validate", str(path), "--json"]) == 0
    assert _json(capsys)["ok"] is True


def test_validate_lists_every_error(sample_configs, write_yaml, capsys) -> None:
    """Two bad fields give two ``{loc, msg, type}`` records and exit 1.

    :param sample_configs: Per-engine sample payloads.
    :param write_yaml: Config writer fixture.
    :param capsys: Pytest capture fixture.
    :return: None."""
    bad = {**sample_configs["amber"], "num_steps": "many", "log_level": "LOUD"}
    assert run_cli(["config", "validate", str(write_yaml(bad)), "--json"]) == 1
    out = _json(capsys)
    assert out["ok"] is False and out["error"]["type"] == "ValidationError"
    errors = out["error"]["errors"]
    assert {tuple(e["loc"]) for e in errors} == {("amber", "num_steps"), ("amber", "log_level")}
    assert all(set(e) == {"loc", "msg", "type"} for e in errors)


def test_validate_yaml_syntax_error(tmp_path: Path, capsys) -> None:
    """Broken YAML is a ``YAMLError`` envelope, not a traceback.

    :param tmp_path: Pytest temporary directory.
    :param capsys: Pytest capture fixture.
    :return: None."""
    path = tmp_path / "broken.yaml"
    path.write_text("engine: [amber\n")
    assert run_cli(["config", "validate", str(path), "--json"]) == 1
    assert _json(capsys)["error"]["type"] == "YAMLError"
    assert run_cli(["config", "validate", str(tmp_path / "missing.yaml"), "--json"]) == 1
    assert _json(capsys)["error"]["type"] == "FileNotFoundError"


def test_validate_human_mode_goes_to_stderr(sample_configs, write_yaml, capsys) -> None:
    """Without ``--json`` the errors print on stderr and stdout stays empty.

    :param sample_configs: Per-engine sample payloads.
    :param write_yaml: Config writer fixture.
    :param capsys: Pytest capture fixture.
    :return: None."""
    bad = {**sample_configs["amber"], "num_steps": "many"}
    assert run_cli(["config", "validate", str(write_yaml(bad))]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "num_steps" in captured.err


def test_describe_resolves_paths(sample_configs, write_yaml, tmp_path: Path, capsys) -> None:
    """Describe reports resolved paths with ``exists`` flags and the derived durations.

    :param sample_configs: Per-engine sample payloads.
    :param write_yaml: Config writer fixture.
    :param tmp_path: Pytest temporary directory.
    :param capsys: Pytest capture fixture.
    :return: None."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.pdb").write_text("")
    cfg = {**sample_configs["amber"], "num_steps": 5_000_000, "output_freq": 100}
    path = write_yaml({**cfg, "beamtime": {"group": "g"}})
    assert run_cli(["config", "describe", str(path), "--data-root", str(root), "--json"]) == 0
    out = _json(capsys)
    assert out["engine"] == "amber" and out["system"] == "t"
    assert out["output_dir"] == str(root / "t" / "run")
    assert out["inputs"] == [
        {"field": "input_pdb", "path": "a.pdb", "resolved": str(root / "a.pdb"), "exists": True}
    ]
    assert (out["beamtime_group"], out["beamtime_name"]) == ("g", "run")
    assert out["total_time_ns"] == 10.0 and out["frame_spacing_ps"] == 0.2
    assert out["n_frames"] == 50_000


def test_describe_without_data_root(sample_configs, write_yaml, tmp_path: Path, capsys) -> None:
    """No data root anywhere: exit 0, paths as written, ``exists`` false, a warning.

    :param sample_configs: Per-engine sample payloads.
    :param write_yaml: Config writer fixture.
    :param tmp_path: Pytest temporary directory.
    :param capsys: Pytest capture fixture.
    :return: None."""
    path = write_yaml({**sample_configs["amber"], "input_pdb": str(tmp_path / "nope.pdb")})
    assert run_cli(["config", "describe", str(path), "--json"]) == 0
    out = _json(capsys)
    assert out["data_root"] is None and out["output_dir"] == "t/run"
    assert out["inputs"][0]["exists"] is False and "No data root" in out["warnings"][0]
    assert run_cli(["config", "describe", str(path)]) == 0
    assert "warning: No data root" in capsys.readouterr().out


def test_describe_uses_workspace_data_root(
    sample_configs, tmp_path: Path, monkeypatch, capsys
) -> None:
    """A workspace's data root applies when the cwd is inside the workspace.

    :param sample_configs: Per-engine sample payloads.
    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param capsys: Pytest capture fixture.
    :return: None."""
    _workspace(tmp_path / "lab", c=sample_configs["amber"])
    (tmp_path / "lab" / "md-workspace.yaml").write_text("data_root: dat\n")
    monkeypatch.chdir(tmp_path / "lab")
    cfg = str(tmp_path / "lab" / "configs" / "c.yaml")
    assert run_cli(["config", "describe", cfg, "--json"]) == 0
    assert _json(capsys)["data_root"] == str(tmp_path / "lab" / "dat")


def test_ls_lists_workspace_configs(sample_configs, tmp_path: Path, monkeypatch, capsys) -> None:
    """``ls`` shows engine, system and temperature, and a broken config with its error.

    :param sample_configs: Per-engine sample payloads.
    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param capsys: Pytest capture fixture.
    :return: None."""
    _workspace(
        tmp_path,
        good={**sample_configs["amber"], "temperature": 310},
        bad={**sample_configs["amber"], "num_steps": "many"},
        syntax="engine: [amber\n",
    )
    monkeypatch.chdir(tmp_path)
    assert run_cli(["config", "ls", "--json"]) == 0
    rows = {e["name"]: e for e in _json(capsys)["configs"]}
    assert (rows["good.yaml"]["engine"], rows["good.yaml"]["temperature"]) == ("amber", 310)
    assert rows["good.yaml"]["error"] is None
    assert "num_steps" in rows["bad.yaml"]["error"] and rows["bad.yaml"]["engine"] is None
    assert "invalid YAML" in rows["syntax.yaml"]["error"]
    assert run_cli(["config", "ls"]) == 0
    assert "INVALID" in capsys.readouterr().out


def test_ls_without_workspace(tmp_path: Path, monkeypatch, capsys) -> None:
    """No workspace is a ``WorkspaceNotFound`` envelope and exit 2, not a traceback.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param capsys: Pytest capture fixture.
    :return: None."""
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    assert run_cli(["config", "ls", "--json"]) == 2
    assert _json(capsys)["error"]["type"] == "WorkspaceNotFound"
    assert run_cli(["config", "ls"]) == 2
    assert "md-workspace.yaml" in capsys.readouterr().err


def _new(source: Path, out: Path, *sets: str, extra: tuple[str, ...] = ()) -> int:
    """Run ``md-sim config new`` with ``--json``.

    :param source: The config to copy.
    :param out: The output path.
    :param sets: ``KEY=VALUE`` overrides.
    :param extra: Further arguments.
    :return: The exit code."""
    argv = ["config", "new", "--from", str(source), "-o", str(out), "--json", *extra]
    for item in sets:
        argv += ["--set", item]
    return run_cli(argv)


def test_new_writes_a_valid_config(sample_configs, write_yaml, tmp_path: Path, capsys) -> None:
    """``new`` applies overrides and the result passes ``validate``.

    :param sample_configs: Per-engine sample payloads.
    :param write_yaml: Config writer fixture.
    :param tmp_path: Pytest temporary directory.
    :param capsys: Pytest capture fixture.
    :return: None."""
    out = tmp_path / "sub" / "new.yaml"
    assert _new(write_yaml(sample_configs["amber"]), out, "temperature=350", "output_subdir=x/y") == 0
    assert _json(capsys)["set"] == ["temperature", "output_subdir"]
    assert run_cli(["config", "validate", str(out), "--json"]) == 0
    written = yaml.safe_load(out.read_text())
    assert written["temperature"] == 350 and written["output_subdir"] == "x/y"


def test_set_type_coercion(sample_configs, write_yaml, tmp_path: Path) -> None:
    """Values are YAML scalars: a number stays a number, a quoted one a string, ``true`` a bool.

    :param sample_configs: Per-engine sample payloads.
    :param write_yaml: Config writer fixture.
    :param tmp_path: Pytest temporary directory.
    :return: None."""
    out = tmp_path / "new.yaml"
    sets = ("temperature=350", "system='350'", "minimize_energy=false", "friction=0.5")
    assert _new(write_yaml(sample_configs["amber"]), out, *sets) == 0
    written = yaml.safe_load(out.read_text())
    assert written["temperature"] == 350 and isinstance(written["temperature"], int)
    assert written["system"] == "350"
    assert written["minimize_energy"] is False and written["friction"] == 0.5


def test_set_nested_key(sample_configs, write_yaml, tmp_path: Path, capsys) -> None:
    """A dotted key edits a nested block rather than writing a literal dotted key.

    :param sample_configs: Per-engine sample payloads.
    :param write_yaml: Config writer fixture.
    :param tmp_path: Pytest temporary directory.
    :param capsys: Pytest capture fixture.
    :return: None."""
    rex = {"n_replicas": 4, "t_max": 400.0, "exchange_interval": 1000}
    cfg = {**sample_configs["amber"], "num_steps": 100_000, "replica_exchange": rex}
    out = tmp_path / "new.yaml"
    assert _new(write_yaml(cfg), out, "replica_exchange.n_replicas=8") == 0
    written = yaml.safe_load(out.read_text())
    assert written["replica_exchange"]["n_replicas"] == 8 and "replica_exchange.n_replicas" not in written
    capsys.readouterr()
    # A scalar in the middle of the path is rejected, not overwritten.
    assert _new(write_yaml(cfg, "b.yaml"), tmp_path / "b.yaml", "system.x=1") == 2
    assert "not a block" in _json(capsys)["error"]["message"]


def test_new_rejects_invalid_and_unknown(
    sample_configs, write_yaml, tmp_path: Path, capsys
) -> None:
    """A bad value or a misspelt field writes nothing.

    :param sample_configs: Per-engine sample payloads.
    :param write_yaml: Config writer fixture.
    :param tmp_path: Pytest temporary directory.
    :param capsys: Pytest capture fixture.
    :return: None."""
    src, out = write_yaml(sample_configs["amber"]), tmp_path / "new.yaml"
    assert _new(src, out, "num_steps=lots") == 1
    assert _json(capsys)["error"]["errors"][0]["loc"] == ["amber", "num_steps"]
    assert _new(src, out, "temprature=350") == 2
    assert "temprature" in _json(capsys)["error"]["message"]
    assert _new(src, out, "temperature") == 2
    assert not out.exists()


def test_new_refuses_to_overwrite(sample_configs, write_yaml, tmp_path: Path, capsys) -> None:
    """An existing output needs ``--force``.

    :param sample_configs: Per-engine sample payloads.
    :param write_yaml: Config writer fixture.
    :param tmp_path: Pytest temporary directory.
    :param capsys: Pytest capture fixture.
    :return: None."""
    src, out = write_yaml(sample_configs["amber"]), tmp_path / "new.yaml"
    out.write_text("keep")
    assert _new(src, out, "temperature=350") == 1
    assert _json(capsys)["error"]["type"] == "FileExistsError" and out.read_text() == "keep"
    assert _new(src, out, "temperature=350", extra=("--force",)) == 0
    assert yaml.safe_load(out.read_text())["temperature"] == 350


def test_config_commands_stay_core_only() -> None:
    """The config verbs import without openmm, beamtime or torch.

    :return: None."""
    code = (
        "import sys; import md_simulations.cli.config_cmds, md_simulations.cli.main;"
        "bad = [m for m in ('openmm', 'beamtime', 'torch') if m in sys.modules];"
        "sys.exit(len(bad))"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
