"""``md-sim mcp``: the config surface over MCP, for an agent working in a workspace.

One surface, two front ends. Every tool runs the matching ``md-sim ... --json`` command in this
process and returns the envelope it printed, so an agent and a terminal see the same answers;
what the CLI learns later, the server gets for free. The tool set is read-and-author only
(Decision 14 of the split plan): it lists, describes, validates and scaffolds configs, returns the
JSON Schema and runs ``check-job``. It never starts a simulation and never calls ``sbatch``; run
state is the beamtime server's.

Core-only imports. The ``mcp`` SDK (the ``mcp`` extra) is imported inside :func:`serve`, so
``import md_simulations`` and a torch consumer never meet it."""

import contextlib
import io
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from md_simulations.cli.main import run_cli
from md_simulations.workspace import (
    WORKSPACE_ENV,
    WORKSPACE_FILE,
    Workspace,
    load_workspace,
)

SERVER_NAME = "md-simulations"

INSTRUCTIONS = (
    "Read-and-author tools for the MD workspace this server was started in. List, describe and "
    "validate configs, fetch the config JSON Schema, scaffold a new config from an existing one, "
    "and check an sbatch script before the user submits it. Every result carries 'ok' and the "
    "'workspace' it refers to; a failure is a result with ok=false and an 'error', not a "
    "transport error. Nothing here runs a simulation or submits a job; run state belongs to the "
    "beamtime server."
)

TOOL_DESCRIPTIONS: dict[str, str] = {
    "list_configs": (
        "List the workspace's simulation configs with engine, system and temperature. A config "
        "that fails to load is listed with its error. Same as `md-sim config ls --json`."
    ),
    "describe_config": (
        "Show what a config resolves to without touching the data: output directory, each input "
        "path with an exists flag, Beamtime group and name, duration and frame count. Same as "
        "`md-sim config describe --json`."
    ),
    "validate_config": (
        "Check a config file against the schema; on failure every error is listed with its "
        "location. Same as `md-sim config validate --json`."
    ),
    "config_schema": (
        "Return the JSON Schema of the config (one branch per engine) or of a single engine. "
        "Same as `md-sim config schema --json`; the schema is the result's 'payload'."
    ),
    "new_config": (
        "Write a validated copy of an existing config with field overrides, inside the "
        "workspace's config directory only. Unknown fields are rejected; comments in the source "
        "are not kept. Same as `md-sim config new --json`."
    ),
    "check_job": (
        "Check an sbatch script against the workspace, the site profile and the configs it runs, "
        "before the user submits it. Returns 'ok' and one entry per check. Nothing is submitted. "
        "Same as `md-sim check-job --json`."
    ),
}


class McpUnavailable(ImportError):
    """The ``mcp`` SDK is not installed (the ``mcp`` extra)."""


def resolve_workspace(path: str | None = None) -> Workspace:
    """Find the workspace once and pin it for every later call.

    ``path`` (a directory or the descriptor file) wins over ``$MD_WORKSPACE`` and the cwd. The
    result is written back to ``$MD_WORKSPACE`` so the ``md-sim`` commands the tools run use this
    tree whatever the process does afterwards.

    :param path: The ``--workspace`` argument, or ``None`` for ``$MD_WORKSPACE`` or the cwd.
    :return: The loaded workspace.
    :raises WorkspaceError: If none is found or the descriptor is invalid."""
    if path is not None:
        os.environ[WORKSPACE_ENV] = str(Path(path).expanduser())
    workspace = load_workspace()
    os.environ[WORKSPACE_ENV] = str(workspace.root / WORKSPACE_FILE)
    return workspace


def _failure(error_type: str, message: str) -> dict[str, Any]:
    """Build a failure envelope.

    :param error_type: ``error.type``.
    :param message: ``error.message``.
    :return: ``{"ok": False, "error": {...}}``."""
    return {"ok": False, "error": {"type": error_type, "message": message}}


def _invoke(argv: list[str]) -> dict[str, Any]:
    """Run ``md-sim <argv>`` in this process and return the envelope it printed.

    stdout is captured (on a stdio server it is the transport), and neither an exception nor a
    usage error escapes: both become a failure envelope.

    :param argv: The arguments after ``md-sim``, ``--json`` included.
    :return: The parsed envelope."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            run_cli(argv)
    except Exception as exc:  # a tool reports its failure, it does not break the session
        return _failure(type(exc).__name__, str(exc))
    text = out.getvalue().strip()
    try:
        result = json.loads(text)
    except ValueError:
        return _failure("UsageError", err.getvalue().strip() or text or "no output")
    return result if isinstance(result, dict) else _failure("UsageError", text)


def make_tools(workspace: Workspace) -> dict[str, Callable[..., dict[str, Any]]]:
    """Build the tool handlers, bound to one workspace.

    :param workspace: The workspace every tool reports as ``"workspace"`` and resolves against.
    :return: Tool name to handler; each returns the ``--json`` envelope of the matching command
        with ``"workspace"`` added."""
    root = str(workspace.root)
    config_dir = workspace.resolved_config_dir()

    def result(envelope: dict[str, Any]) -> dict[str, Any]:
        """Put ``ok`` and ``workspace`` first in an envelope."""
        rest = {k: v for k, v in envelope.items() if k != "ok"}
        return {"ok": bool(envelope.get("ok", False)), "workspace": root, **rest}

    def locate(path: str, *bases: Path) -> Path:
        """An absolute path as is; a relative one against the first base that holds it."""
        p = Path(path).expanduser()
        if p.is_absolute():
            return p
        return next((b / p for b in bases if (b / p).exists()), bases[0] / p)

    def list_configs() -> dict[str, Any]:
        return result(_invoke(["config", "ls", "--json"]))

    def describe_config(
        path: Annotated[str, Field(description="Config path, absolute or relative to config_dir.")],
        data_root: Annotated[
            str | None, Field(description="Resolve paths against this data root instead.")
        ] = None,
    ) -> dict[str, Any]:
        argv = ["config", "describe", "--json"]
        if data_root is not None:
            argv.append(f"--data-root={data_root}")
        return result(_invoke([*argv, str(locate(path, config_dir, workspace.root))]))

    def validate_config(
        path: Annotated[str, Field(description="Config path, absolute or relative to config_dir.")],
    ) -> dict[str, Any]:
        path_ = str(locate(path, config_dir, workspace.root))
        return result(_invoke(["config", "validate", "--json", path_]))

    def config_schema(
        engine: Annotated[
            str | None, Field(description="An engine name; omit for the whole config union.")
        ] = None,
    ) -> dict[str, Any]:
        argv = ["config", "schema", "--json"]
        if engine is not None:
            argv.append(f"--engine={engine}")
        return result(_invoke(argv))

    def new_config(
        source: Annotated[
            str, Field(description="Config to start from, absolute or relative to config_dir.")
        ],
        output: Annotated[
            str,
            Field(
                description="New config path; must be a .yaml inside config_dir (relative to it)."
            ),
        ],
        overrides: Annotated[
            list[str] | None,
            Field(description="KEY=VALUE overrides; KEY may be dotted, VALUE is a YAML scalar."),
        ] = None,
        force: Annotated[bool, Field(description="Overwrite an existing output file.")] = False,
    ) -> dict[str, Any]:
        target = locate(output, config_dir)
        try:
            resolved = target.resolve()
            inside = resolved.is_relative_to(config_dir.resolve())
            protected = resolved == (workspace.root / WORKSPACE_FILE).resolve() or (
                resolved.is_relative_to((workspace.root / "sites").resolve())
            )
        except OSError as exc:
            return result(_failure(type(exc).__name__, str(exc)))
        if not inside or protected or resolved.suffix != ".yaml":
            return result(
                _failure(
                    "PathOutsideWorkspace",
                    f"refusing to write {output!r}: new configs go to a .yaml file inside "
                    f"{config_dir}",
                )
            )
        argv = ["config", "new", "--json", f"--from={locate(source, config_dir, workspace.root)}"]
        argv += [f"--set={item}" for item in overrides or []]
        argv += [f"--output={resolved}"]
        if force:
            argv.append("--force")
        return result(_invoke(argv))

    def check_job(
        script: Annotated[
            str, Field(description="sbatch script, absolute or relative to the workspace root.")
        ],
        require_gpu: Annotated[
            bool, Field(description="Fail the gres check unless the script requests a GPU.")
        ] = False,
    ) -> dict[str, Any]:
        argv = ["check-job", "--json"]
        if require_gpu:
            argv.append("--require-gpu")
        return result(_invoke([*argv, str(locate(script, workspace.root))]))

    return {
        "list_configs": list_configs,
        "describe_config": describe_config,
        "validate_config": validate_config,
        "config_schema": config_schema,
        "new_config": new_config,
        "check_job": check_job,
    }


def build_server(workspace: Workspace) -> Any:
    """Build the MCP server with the six tools registered.

    :param workspace: The workspace to serve.
    :return: The SDK's server object (not started).
    :raises McpUnavailable: If the ``mcp`` SDK is not installed."""
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:
        raise McpUnavailable(
            f"the MCP server needs the 'mcp' extra: pip install 'md-simulations[mcp]' ({exc})"
        ) from exc
    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS)
    for name, handler in make_tools(workspace).items():
        server.tool(name=name, description=TOOL_DESCRIPTIONS[name])(handler)
    return server


def serve(workspace_path: str | None = None) -> None:
    """Resolve the workspace and serve over stdio until the client disconnects.

    :param workspace_path: The ``--workspace`` argument, or ``None``.
    :return: None.
    :raises McpUnavailable: If the ``mcp`` SDK is not installed.
    :raises WorkspaceError: If no usable workspace is found."""
    workspace = resolve_workspace(workspace_path)
    build_server(workspace).run("stdio")
