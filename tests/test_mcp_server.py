"""``md-sim mcp``: the tool handlers, the slim import, and a real stdio round trip."""

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml

from md_simulations.cli.main import run_cli
from md_simulations.mcp_server import (
    TOOL_DESCRIPTIONS,
    McpUnavailable,
    build_server,
    make_tools,
    resolve_workspace,
)
from md_simulations.workspace import WorkspaceNotFound, load_workspace

TOOLS = {
    "list_configs",
    "describe_config",
    "validate_config",
    "config_schema",
    "new_config",
    "check_job",
}

CONFIG = {"engine": "amber", "system": "x", "output_subdir": "out/x", "input_pdb": "pdbs/x.pdb"}

SCRIPT = """#!/bin/bash
#SBATCH --job-name=t
#SBATCH --output=./slurm/%x-%j.out
md-sim run configs/x.yaml
"""


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> None:
    """Keep the ambient workspace and data root out of every test.

    :param monkeypatch: Pytest monkeypatch fixture.
    :return: None."""
    monkeypatch.delenv("MD_WORKSPACE", raising=False)
    monkeypatch.delenv("MD_DATA_ROOT", raising=False)


@pytest.fixture
def lab(tmp_path: Path, monkeypatch) -> Path:
    """A workspace with one config and one script, pinned as the server would pin it.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :return: The workspace root."""
    root = tmp_path / "lab"
    (root / "configs").mkdir(parents=True)
    (root / "slurm").mkdir()
    (root / "sites").mkdir()
    (root / "configs" / "x.yaml").write_text(yaml.safe_dump(CONFIG))
    (root / "slurm" / "job.sh").write_text(SCRIPT)
    (root / "md-workspace.yaml").write_text("config_dir: configs\n")
    monkeypatch.setenv("MD_WORKSPACE", str(root))
    return root


@pytest.fixture
def tools(lab: Path) -> dict:
    """The tool handlers bound to the ``lab`` workspace.

    :param lab: The workspace root.
    :return: Tool name to handler."""
    return make_tools(load_workspace())


def _cli(capsys, *argv: str) -> dict:
    """Run ``md-sim <argv> --json`` and parse what it printed.

    :param capsys: Pytest capture fixture.
    :param argv: Arguments after ``md-sim``.
    :return: The envelope."""
    capsys.readouterr()
    run_cli([*argv, "--json"])
    return json.loads(capsys.readouterr().out)


def _same_as_cli(result: dict, envelope: dict, lab: Path) -> None:
    """Assert a tool result is the CLI envelope plus the workspace.

    :param result: The tool result.
    :param envelope: The envelope the CLI printed.
    :param lab: The workspace root.
    :return: None."""
    assert result["workspace"] == str(lab)
    assert {k: v for k, v in result.items() if k != "workspace"} == envelope


def test_tool_set_is_exact(tools: dict) -> None:
    """Exactly the six documented tools exist, each with a description.

    :param tools: Tool handlers.
    :return: None."""
    assert set(tools) == TOOLS == set(TOOL_DESCRIPTIONS)
    assert all(TOOL_DESCRIPTIONS[name].strip() for name in TOOLS)


def test_results_match_the_cli(tools: dict, lab: Path, capsys) -> None:
    """Each tool returns the payload the matching ``--json`` command prints.

    :param tools: Tool handlers.
    :param lab: The workspace root.
    :param capsys: Pytest capture fixture.
    :return: None."""
    cfg = str(lab / "configs" / "x.yaml")
    _same_as_cli(tools["list_configs"](), _cli(capsys, "config", "ls"), lab)
    _same_as_cli(tools["validate_config"]("x.yaml"), _cli(capsys, "config", "validate", cfg), lab)
    _same_as_cli(tools["describe_config"](cfg), _cli(capsys, "config", "describe", cfg), lab)
    _same_as_cli(tools["config_schema"](), _cli(capsys, "config", "schema"), lab)
    _same_as_cli(
        tools["config_schema"]("amber"), _cli(capsys, "config", "schema", "--engine", "amber"), lab
    )
    script = str(lab / "slurm" / "job.sh")
    _same_as_cli(tools["check_job"]("slurm/job.sh"), _cli(capsys, "check-job", script), lab)
    _same_as_cli(
        tools["check_job"](script, require_gpu=True),
        _cli(capsys, "check-job", "--require-gpu", script),
        lab,
    )


def test_check_job_require_gpu(tools: dict) -> None:
    """``require_gpu`` turns a missing GPU request into a failed ``gres`` check.

    :param tools: Tool handlers.
    :return: None."""
    plain = tools["check_job"]("slurm/job.sh")
    gated = tools["check_job"]("slurm/job.sh", require_gpu=True)
    status = lambda r: {c["name"]: c["status"] for c in r["checks"]}["gres"]  # noqa: E731
    assert status(plain) == "skipped"
    assert status(gated) == "fail"
    assert gated["ok"] is False


def test_new_config_writes_inside_the_config_dir(tools: dict, lab: Path) -> None:
    """``new_config`` writes a validated copy with overrides, relative to ``config_dir``.

    :param tools: Tool handlers.
    :param lab: The workspace root.
    :return: None."""
    res = tools["new_config"]("x.yaml", "y/hot.yaml", ["temperature=350"])
    assert res["ok"] is True and res["set"] == ["temperature"]
    written = yaml.safe_load((lab / "configs" / "y" / "hot.yaml").read_text())
    assert written["temperature"] == 350
    again = tools["new_config"]("x.yaml", "y/hot.yaml")
    assert again["ok"] is False and again["error"]["type"] == "FileExistsError"
    assert tools["new_config"]("x.yaml", "y/hot.yaml", force=True)["ok"] is True


def test_new_config_path_escape(tools: dict, lab: Path, tmp_path: Path) -> None:
    """``new_config`` refuses every path that leaves ``config_dir`` or hits a protected file.

    :param tools: Tool handlers.
    :param lab: The workspace root.
    :param tmp_path: Pytest temporary directory.
    :return: None."""
    (lab / "link").symlink_to(tmp_path, target_is_directory=True)
    outside = tmp_path / "evil.yaml"
    for output in (
        "../evil.yaml",
        str(outside),
        "../../etc/evil.yaml",
        "../sites/s.yaml",
        "../md-workspace.yaml",
        str(lab / "slurm" / "evil.yaml"),
        "notyaml.txt",
    ):
        res = tools["new_config"]("x.yaml", output, force=True)
        assert res["ok"] is False and res["error"]["type"] == "PathOutsideWorkspace", output
    (lab / "configs" / "esc").symlink_to(tmp_path, target_is_directory=True)
    res = tools["new_config"]("x.yaml", "esc/evil.yaml")
    assert res["error"]["type"] == "PathOutsideWorkspace"
    assert not outside.exists()
    assert not list((lab / "slurm").glob("evil*"))


def test_tool_error_is_a_result(tools: dict, lab: Path) -> None:
    """A bad input comes back as ``{"ok": false, "error"}``, never as an exception.

    :param tools: Tool handlers.
    :param lab: The workspace root.
    :return: None."""
    (lab / "configs" / "bad.yaml").write_text("engine: nope\n")
    (lab / "configs" / "broken.yaml").write_text("a: [unclosed\n")
    calls = [
        tools["validate_config"]("missing.yaml"),
        tools["validate_config"]("bad.yaml"),
        tools["validate_config"]("broken.yaml"),
        tools["describe_config"]("missing.yaml"),
        tools["config_schema"]("nonesuch"),
        tools["config_schema"]("--help"),
        tools["new_config"]("x.yaml", "z.yaml", ["no_such_field=1"]),
        tools["new_config"]("x.yaml", "z.yaml", ["malformed"]),
        tools["new_config"]("missing.yaml", "z.yaml"),
        tools["check_job"]("slurm/missing.sh"),
    ]
    for res in calls:
        assert res["ok"] is False and res["error"]["type"] and res["error"]["message"], res
        assert res["workspace"] == str(lab)
    assert tools["validate_config"]("bad.yaml")["error"]["errors"]
    assert not (lab / "configs" / "z.yaml").exists()


def test_results_are_json_serialisable(tools: dict, lab: Path) -> None:
    """Every result, success or failure, survives ``json.dumps``.

    :param tools: Tool handlers.
    :param lab: The workspace root.
    :return: None."""
    results = [
        tools["list_configs"](),
        tools["describe_config"]("x.yaml"),
        tools["validate_config"]("x.yaml"),
        tools["config_schema"](),
        tools["new_config"]("x.yaml", "n.yaml"),
        tools["check_job"]("slurm/job.sh"),
        tools["validate_config"]("missing.yaml"),
    ]
    for res in results:
        assert json.loads(json.dumps(res)) == res


def test_stdout_is_not_leaked(tools: dict, capsys) -> None:
    """A tool prints nothing itself: on a stdio server stdout is the protocol.

    :param tools: Tool handlers.
    :param capsys: Pytest capture fixture.
    :return: None."""
    tools["list_configs"]()
    tools["validate_config"]("missing.yaml")
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_concurrent_tool_calls_do_not_mix_results(tools: dict, lab: Path) -> None:
    """Tools called at once keep their own output: a client may have several calls in flight.

    :param tools: Tool handlers.
    :param lab: The workspace root.
    :return: None."""
    calls = [
        (lambda: tools["validate_config"]("x.yaml"), True),
        (lambda: tools["validate_config"]("missing.yaml"), False),
        (lambda: tools["config_schema"](), True),
        (lambda: tools["list_configs"](), True),
    ] * 4
    results: list[tuple[dict, bool]] = []
    lock = threading.Lock()

    def call(handler, expected: bool) -> None:
        res = handler()
        with lock:
            results.append((res, expected))

    threads = [threading.Thread(target=call, args=pair) for pair in calls]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert len(results) == len(calls)
    for res, expected in results:
        assert res["ok"] is expected, res
        assert res["workspace"] == str(lab)


def test_workspace_is_pinned(lab: Path, tmp_path: Path, monkeypatch) -> None:
    """``resolve_workspace`` honours ``--workspace`` and pins the result for later calls.

    :param lab: The workspace root.
    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :return: None."""
    monkeypatch.delenv("MD_WORKSPACE")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(WorkspaceNotFound):
        resolve_workspace()
    ws = resolve_workspace(str(lab))
    assert ws.root == lab.resolve()
    monkeypatch.chdir(tmp_path)
    assert load_workspace().root == ws.root  # a later cwd change no longer matters


def _run_python(code: str, **kwargs) -> subprocess.CompletedProcess:
    """Run ``code`` in a clean interpreter.

    :param code: Python source.
    :param kwargs: Extra ``subprocess.run`` arguments.
    :return: The completed process."""
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, **kwargs)


def test_server_import_is_slim() -> None:
    """Importing the server and building its handlers loads no MD stack and no ``mcp`` module.

    :return: None."""
    proc = _run_python(
        "import sys, json, md_simulations, md_simulations.mcp_server as s; "
        "bare = 'mcp' in sys.modules; "
        "print(json.dumps([bare, sorted(m.split('.')[0] for m in sys.modules)]))"
    )
    assert proc.returncode == 0, proc.stderr
    bare, loaded = json.loads(proc.stdout.splitlines()[-1])
    assert bare is False
    assert not {"openmm", "beamtime", "tqdm", "mdtraj", "torch", "ase", "mcp"} & set(loaded)


def test_missing_extra_is_a_clean_error(lab: Path, monkeypatch, capsys) -> None:
    """Without the ``mcp`` SDK, ``md-sim mcp`` exits 1 and names the extra.

    :param lab: The workspace root.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param capsys: Pytest capture fixture.
    :return: None."""
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.server", None)
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", None)
    with pytest.raises(McpUnavailable, match=r"md-simulations\[mcp\]"):
        build_server(load_workspace())
    assert run_cli(["mcp"]) == 1
    assert "mcp" in capsys.readouterr().err
    assert run_cli(["mcp", "--json"]) == 1
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["ok"] is False and "[mcp]" in envelope["error"]["message"]


def test_no_workspace_is_a_usage_error(tmp_path: Path, monkeypatch, capsys) -> None:
    """``md-sim mcp`` outside a workspace exits 2 with the workspace error.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param capsys: Pytest capture fixture.
    :return: None."""
    pytest.importorskip("mcp")
    monkeypatch.chdir(tmp_path)
    assert run_cli(["mcp"]) == 2
    assert "md-workspace.yaml" in capsys.readouterr().err


def _rpc(*messages: dict, cwd: Path, extra: list[str] | None = None) -> list[dict]:
    """Send JSON-RPC lines to ``python -m md_simulations.cli mcp`` and read the replies.

    stdin stays open until every request has been answered: the server stops at EOF, which would
    drop calls still in flight.

    :param messages: Requests (with an ``id``) and notifications, in order.
    :param cwd: Working directory of the server.
    :param extra: Extra ``md-sim mcp`` arguments.
    :return: The responses, in order.
    :raises AssertionError: If the server wrote anything to stdout that is not JSON-RPC."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "md_simulations.cli", "mcp", *(extra or [])],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd=cwd,
    )
    replies: list[dict] = []
    watchdog = threading.Timer(60, proc.kill)
    watchdog.daemon = True
    watchdog.start()
    try:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write("".join(json.dumps(m) + "\n" for m in messages))
        proc.stdin.flush()
        for _ in [m for m in messages if "id" in m]:
            replies.append(json.loads(proc.stdout.readline()))
    finally:
        watchdog.cancel()
        proc.stdin.close()
        proc.wait(timeout=30)
    return replies


def test_stdio_tools_list_and_call(lab: Path) -> None:
    """A real server answers ``tools/list`` with the six tools and runs one.

    :param lab: The workspace root.
    :return: None."""
    pytest.importorskip("mcp")
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "0"},
        },
    }
    replies = _rpc(
        init,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "validate_config", "arguments": {"path": "x.yaml"}},
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "validate_config", "arguments": {"path": "missing.yaml"}},
        },
        cwd=lab,
    )
    by_id = {r["id"]: r for r in replies}
    assert by_id[1]["result"]["serverInfo"]["name"] == "md-simulations"
    listed = by_id[2]["result"]["tools"]
    assert {t["name"] for t in listed} == TOOLS
    for tool in listed:
        assert tool["description"]
        assert tool["inputSchema"]["type"] == "object"
        for prop in tool["inputSchema"].get("properties", {}).values():
            assert prop.get("description")
    required = {t["name"]: t["inputSchema"].get("required", []) for t in listed}
    assert required["check_job"] == ["script"] and required["new_config"] == ["source", "output"]
    ok = by_id[3]["result"]
    assert not ok.get("isError")
    payload = json.loads(ok["content"][0]["text"])
    assert payload["ok"] is True and payload["workspace"] == str(lab)
    bad = json.loads(by_id[4]["result"]["content"][0]["text"])
    assert bad["ok"] is False and bad["error"]["type"] == "FileNotFoundError"
