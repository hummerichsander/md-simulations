"""The ``md-sim`` dispatcher: ``md-sim <noun> <verb>`` with ``--json`` on its own commands.

``run`` and ``check-job`` are parsed here. ``prep``, ``analysis`` and ``sampling`` hand
everything after the verb to the standalone script untouched, so a script's own options
(including one named ``--json``) never meet this parser. ``config`` is the group for config
introspection; its verbs register in :data:`CONFIG_VERBS` (from ``config_cmds``)."""

import argparse
import contextlib
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from md_simulations.cli._dispatch import (
    GROUP_SUMMARIES,
    GROUPS,
    SUMMARIES,
    alias,
    dispatch,
    exit_code,
)
from md_simulations.cli._output import EXIT_USAGE, emit, fail
from md_simulations.cli.config_cmds import VERBS as CONFIG_VERBS
from md_simulations.config import load_config, resolve_data_root
from md_simulations.preflight import check_job, format_checks
from md_simulations.workspace import WorkspaceError, load_workspace

# ``config`` verbs are handlers in ``config_cmds`` (``verb -> (summary, handler(argv) -> int)``);
# the other groups' verbs are lazy ``module:main`` strings in ``_dispatch.GROUPS``.
ALL_GROUP_SUMMARIES = {"config": "Inspect and author simulation configs.", **GROUP_SUMMARIES}

TOP_COMMANDS: dict[str, str] = {
    "run": "Run a simulation from a YAML config.",
    "check-job": "Check an sbatch script against the workspace, site profile and configs.",
}

# Deliberately not in TOP_COMMANDS: that table is the `md-sim --json` surface, which
# tests/test_cli.py pins to exactly {run, check-job}.
MCP_SUMMARY = (
    "Serve config, schema, validate, scaffold and check-job tools over MCP on stdio "
    "(needs the 'mcp' extra)."
)

GROUP_ORDER = ("config", "prep", "analysis", "sampling")


class _RunError(Exception):
    """A failed ``md-sim run`` whose cause was already logged (or needs no traceback)."""

    def __init__(self, message: str, *, cause: BaseException | None = None, silent: bool = False):
        """Build the error.

        :param message: What went wrong; printed to stderr in human mode unless ``silent``.
        :param cause: The underlying exception, whose class names the JSON ``error.type``.
        :param silent: Exit 1 without printing in human mode (the logger already said it).
        :return: None."""
        super().__init__(message)
        self.error_type = type(cause).__name__ if cause is not None else "RunError"
        self.silent = silent


def _execute_run(args: argparse.Namespace) -> dict:
    """Load a config, resolve the data root, and run the simulation.

    :param args: Parsed CLI arguments.
    :return: The ``--json`` payload of a finished run.
    :raises _RunError: If a dependency is missing, ``--resume`` does not apply, or the run fails."""
    # Deferred: core and engines need the ``simulate`` extra, but ``md-sim check-job``
    # and the other config-only commands must run without it.
    try:
        from md_simulations.core import setup_logger
        from md_simulations.engines import build_engine
    except ModuleNotFoundError as exc:  # message is core/engines ``EXTRA_HINT`` on a slim install
        raise _RunError(f"md-sim run: {exc}", cause=exc) from exc

    config = load_config(args.config)
    data_root = resolve_data_root(config, args.data_root)
    output_dir = config.output_dir(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("md_simulations", output_dir, getattr(logging, config.log_level))
    logger.info("=" * 60)
    logger.info(f"md-simulations | engine={config.engine} | system={config.system}")
    logger.info("=" * 60)
    logger.info(f"Config:     {args.config}")
    logger.info(f"Data root:  {data_root}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(
        f"Steps:      {config.num_steps:,}  (dt={config.timestep} ps, T={config.temperature} K)"
    )
    logger.info("=" * 60)

    if args.resume and config.replica_exchange is None:
        message = "--resume needs a config with a replica_exchange block."
        logger.error(message)
        raise _RunError(message, silent=True)

    engine = build_engine(config, data_root, logger, resume=args.resume)
    try:
        engine.run()
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        raise _RunError(f"Simulation failed: {e}", cause=e, silent=True) from e
    return {
        "config": str(args.config),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "engine": config.engine,
        "system": config.system,
        "steps": config.num_steps,
    }


def _cmd_run(args: argparse.Namespace) -> int:
    """Run a simulation; under ``--json`` print one JSON object on stdout and nothing else.

    In JSON mode everything the run prints to stdout (the logger's console handler, the
    state reporters) goes to stderr instead.

    :param args: Parsed CLI arguments.
    :return: The exit code (0 ok, 1 failure)."""
    if not args.json:
        try:
            _execute_run(args)
        except _RunError as err:
            if err.silent:
                return 1
            sys.exit(str(err))
        return 0

    error: BaseException | None = None
    error_type: str | None = None
    payload: dict = {}
    with contextlib.redirect_stdout(sys.stderr):
        try:
            payload = _execute_run(args)
        except _RunError as err:
            error, error_type = err, err.error_type
        except Exception as exc:  # config errors and the like: reported, not a traceback
            error = exc
    if error is not None:
        fail(error, json_mode=True, error_type=error_type)
    emit(payload, json_mode=True)
    return 0


def _cmd_check_job(args: argparse.Namespace) -> int:
    """Preflight an sbatch script: 0 if no check failed, 1 if one did, 2 on a usage error.

    The workspace is found from the cwd (or ``$MD_WORKSPACE``), then from the script's directory.

    :param args: Parsed CLI arguments.
    :return: The exit code."""
    script = Path(args.script)
    try:
        try:
            workspace = load_workspace()
        except WorkspaceError:
            workspace = load_workspace(script.resolve().parent)
        checks = check_job(
            script, workspace, data_root=args.data_root, require_gpu=args.require_gpu
        )
    except (WorkspaceError, OSError) as err:
        fail(err, json_mode=args.json, code=EXIT_USAGE, prog="md-sim check-job")
    ok = all(c.ok for c in checks)
    payload = {
        "ok": ok,
        "checks": [
            {"name": c.name, "ok": c.ok, "status": c.status, "detail": c.detail} for c in checks
        ],
    }
    emit(payload, json_mode=args.json, human=format_checks(checks))
    return 0 if ok else 1


def _cmd_mcp(args: argparse.Namespace) -> int:
    """Serve the config surface over MCP on stdio until the client disconnects.

    The server module is imported here, and the ``mcp`` SDK only inside it, so no other command
    (and no torch consumer) needs the ``mcp`` extra. ``--json`` shapes a startup failure; stdout
    is the protocol once the server runs.

    :param args: Parsed CLI arguments.
    :return: The exit code (1 if the ``mcp`` extra is missing, 2 if there is no usable workspace)."""
    from md_simulations.mcp_server import McpUnavailable, serve

    try:
        serve(args.workspace)
    except McpUnavailable as exc:
        fail(exc, json_mode=args.json, prog="md-sim mcp")
    except WorkspaceError as exc:
        fail(exc, json_mode=args.json, code=EXIT_USAGE, prog="md-sim mcp")
    return 0


def _group_entries(group: str) -> list[dict]:
    """Describe a group's verbs, so a listing needs no per-group special case.

    :param group: ``config``, ``prep``, ``analysis`` or ``sampling``.
    :return: ``{"name", "command", "summary"}`` per verb, plus ``"alias"`` for the verbs that
        are also a standalone ``md-sim-*`` script (every one but ``config``'s)."""
    if group in GROUPS:
        summaries = {verb: SUMMARIES[verb] for verb in GROUPS[group]}
    else:
        summaries = {verb: summary for verb, (summary, _) in CONFIG_VERBS.items()}
    return [
        {"name": v, "command": f"md-sim {group} {v}", "summary": s}
        | ({"alias": alias(v)} if group in GROUPS else {})
        for v, s in summaries.items()
    ]


def _verb_lines(group: str) -> str:
    """Render a group's verbs for ``--help`` and the human listing.

    :param group: The group.
    :return: One ``  verb  summary`` line per verb, or a note when it has none yet."""
    entries = _group_entries(group)
    if not entries:
        return "  (no commands yet)"
    width = max(len(e["name"]) for e in entries)
    return "\n".join(f"  {e['name'].ljust(width)}  {e['summary']}" for e in entries)


def _surface() -> dict:
    """Describe every command, so an agent can enumerate the CLI without reading source.

    :return: ``{"commands": [...], "groups": [...]}`` with each group's verbs."""
    return {
        "commands": [
            {"name": n, "command": f"md-sim {n}", "summary": s} for n, s in TOP_COMMANDS.items()
        ],
        "groups": [
            {
                "name": g,
                "summary": ALL_GROUP_SUMMARIES[g],
                "verbs": _group_entries(g),
            }
            for g in GROUP_ORDER
        ],
    }


def _cmd_group(args: argparse.Namespace) -> int:
    """List a group's verbs.

    :param args: Parsed CLI arguments.
    :return: 0."""
    group = args.group
    payload = {
        "group": group,
        "summary": ALL_GROUP_SUMMARIES[group],
        "verbs": _group_entries(group),
    }
    human = f"md-sim {group}: {ALL_GROUP_SUMMARIES[group]}\n\n{_verb_lines(group)}"
    emit(payload, json_mode=args.json, human=human)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``md-sim`` parser.

    :return: The top-level parser, with a subparser per command and group."""
    parser = argparse.ArgumentParser(
        prog="md-sim",
        description="Config-driven MD simulations: md-sim <command> or md-sim <group> <verb>.",
        epilog="Run 'md-sim <group>' to list a group's verbs; every md-sim command takes --json.",
    )
    parser.add_argument(
        "--json", action="store_true", help="With no command: list every command as JSON."
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_run = sub.add_parser("run", help=TOP_COMMANDS["run"])
    p_run.add_argument("config", help="Path to the YAML simulation config.")
    p_run.add_argument(
        "--data-root",
        default=None,
        help="Root directory for inputs/outputs (overrides MD_DATA_ROOT and config).",
    )
    p_run.add_argument(
        "--resume",
        action="store_true",
        help="Continue a replica-exchange run from its checkpoint (same platform as the original).",
    )
    p_run.add_argument(
        "--json",
        action="store_true",
        help="Print the result as one JSON object on stdout; run logging goes to stderr.",
    )
    p_run.set_defaults(func=_cmd_run)

    for group in GROUP_ORDER:
        p_group = sub.add_parser(
            group,
            help=ALL_GROUP_SUMMARIES[group],
            description=f"{ALL_GROUP_SUMMARIES[group]}\n\nVerbs:\n{_verb_lines(group)}",
            epilog=f"md-sim {group} <verb> --help shows a verb's own options.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        p_group.add_argument("--json", action="store_true", help="List the verbs as JSON.")
        p_group.set_defaults(func=_cmd_group, group=group)

    p_check = sub.add_parser("check-job", help=TOP_COMMANDS["check-job"])
    p_check.add_argument("script", help="Path to the sbatch script.")
    p_check.add_argument(
        "--data-root",
        default=None,
        help="Check as if the job ran with this data root (overrides the script, MD_DATA_ROOT, "
        "the workspace and the config).",
    )
    p_check.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if the script requests no GPU (--gres=gpu:N or --gpus=N).",
    )
    p_check.add_argument("--json", action="store_true", help="Print the result as JSON.")
    p_check.set_defaults(func=_cmd_check_job)

    p_mcp = sub.add_parser("mcp", help=MCP_SUMMARY, description=MCP_SUMMARY)
    p_mcp.add_argument(
        "--workspace",
        default=None,
        metavar="PATH",
        help="The workspace directory or md-workspace.yaml to serve "
        "(default: $MD_WORKSPACE, else found upward from the cwd).",
    )
    p_mcp.add_argument(
        "--json", action="store_true", help="Print a startup failure as a JSON envelope."
    )
    p_mcp.set_defaults(func=_cmd_mcp)
    return parser


def _route_verb(group: str, verb: str, rest: list[str]) -> int:
    """Run ``md-sim <group> <verb> ...rest``, with ``rest`` untouched.

    :param group: A group name.
    :param verb: The word after the group.
    :param rest: Every argument after the verb.
    :return: The exit code."""
    if group in GROUPS and verb in GROUPS[group]:
        return dispatch(group, verb, rest)
    if group == "config" and verb in CONFIG_VERBS:
        return CONFIG_VERBS[verb][1](rest)
    known = ", ".join(e["name"] for e in _group_entries(group))
    exc = ValueError(f"unknown {group} verb {verb!r} (choose from: {known})")
    fail(exc, json_mode="--json" in rest, code=EXIT_USAGE, prog=f"md-sim {group}")


def run_cli(argv: Sequence[str]) -> int:
    """Run ``md-sim`` on an argument list.

    :param argv: The arguments after ``md-sim``.
    :return: The process exit code; nothing is raised for ``SystemExit``."""
    args_in = list(argv)
    try:
        if len(args_in) >= 2 and args_in[0] in GROUP_ORDER and not args_in[1].startswith("-"):
            return _route_verb(args_in[0], args_in[1], args_in[2:])
        parser = _build_parser()
        args = parser.parse_args(args_in)
        if args.command is None:
            if not args.json:
                parser.error("the following arguments are required: <command>")
            emit(_surface(), json_mode=True)
            return 0
        return args.func(args)
    except SystemExit as exc:
        return exit_code(exc.code)


def main() -> None:
    """Entry point for the ``md-sim`` command."""
    sys.exit(run_cli(sys.argv[1:]))
