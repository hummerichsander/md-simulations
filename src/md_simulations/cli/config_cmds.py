"""``md-sim config <verb>``: inspect and author simulation configs without running them.

Core-only: pydantic, PyYAML and the config models, no openmm, beamtime or torch. Each verb is a
``handler(argv) -> exit code`` registered in :data:`VERBS`; ``--json`` prints the envelope of
:mod:`md_simulations.cli._output`."""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter, ValidationError

from md_simulations.cli._output import EXIT_USAGE, emit, fail
from md_simulations.config import SimulationConfig, load_config, resolve_data_root
from md_simulations.workspace import (
    Workspace,
    WorkspaceError,
    WorkspaceNotFound,
    load_workspace,
)

_ADAPTER: TypeAdapter = TypeAdapter(SimulationConfig)

# Config fields that name an input file or directory, resolved against the data root.
_INPUT_FIELDS = (
    "input_pdb",
    "input_gro",
    "input_top",
    "top",
    "mdp",
    "index",
    "include_dir",
    "parameters_dir",
    "model_file",
    "configurations_file",
)


def _parser(verb: str, summary: str) -> argparse.ArgumentParser:
    """Build the parser of one verb, with ``--json``.

    :param verb: The verb name.
    :param summary: Its one-line description.
    :return: A parser whose usage line reads ``md-sim config <verb>``."""
    parser = argparse.ArgumentParser(prog=f"md-sim config {verb}", description=summary)
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    return parser


def _errors(err: ValidationError) -> list[dict[str, Any]]:
    """Flatten a pydantic error into ``{"loc", "msg", "type"}`` records.

    :param err: The validation error.
    :return: One record per error; ``loc`` is a list of keys and indices."""
    return [
        {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
        for e in err.errors(include_url=False)
    ]


def _report_invalid(
    path: str | Path, exc: ValidationError | yaml.YAMLError, *, json_mode: bool
) -> int:
    """Print an invalid config's errors in full and return the failure exit code.

    :param path: The config the errors belong to.
    :param exc: A pydantic validation error or a YAML syntax error.
    :param json_mode: Print the ``--json`` envelope instead of text on stderr.
    :return: 1."""
    if isinstance(exc, ValidationError):
        errors = _errors(exc)
        error = {"type": "ValidationError", "message": f"{path}: {len(errors)} error(s)"}
        error["errors"] = errors  # type: ignore[assignment]
        lines = [
            f"  {'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']} [{e['type']}]"
            for e in errors
        ]
    else:
        error = {"type": "YAMLError", "message": f"{path}: {exc}"}
        lines = [f"  {exc}"]
    if json_mode:
        emit({"ok": False, "error": error}, json_mode=True)
    else:
        print(f"{path}: invalid\n" + "\n".join(lines), file=sys.stderr)
    return 1


def _schema(argv: list[str]) -> int:
    """``md-sim config schema [--engine E]``: the JSON Schema of the config union or one engine.

    :param argv: Arguments after the verb.
    :return: The exit code."""
    parser = _parser("schema", VERB_SUMMARIES["schema"])
    parser.add_argument(
        "--engine", default=None, help="Emit only this engine's schema (its `engine` value)."
    )
    args = parser.parse_args(argv)
    schema = _ADAPTER.json_schema(ref_template="#/$defs/{model}")
    mapping = schema["discriminator"]["mapping"]
    title = "SimulationConfig"
    if args.engine is not None:
        if args.engine not in mapping:
            exc = ValueError(f"unknown engine {args.engine!r} (choose from: {', '.join(mapping)})")
            fail(exc, json_mode=args.json, code=EXIT_USAGE, prog="md-sim config schema")
        name = mapping[args.engine].rsplit("/", 1)[-1]
        # One branch, still self-contained: it keeps the shared $defs its $refs point into.
        schema = {**schema["$defs"][name], "$defs": schema["$defs"]}
        title = name
    emit(
        {"title": title, "engines": list(mapping), "payload": schema},
        json_mode=args.json,
        human=json.dumps(schema, indent=2),
    )
    return 0


def _validate(argv: list[str]) -> int:
    """``md-sim config validate <path>``: 0 if the file is a valid config, 1 with every error.

    :param argv: Arguments after the verb.
    :return: The exit code."""
    parser = _parser("validate", VERB_SUMMARIES["validate"])
    parser.add_argument("path", help="Path to the YAML config.")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.path)
    except (ValidationError, yaml.YAMLError) as exc:
        return _report_invalid(args.path, exc, json_mode=args.json)
    except OSError as exc:
        fail(exc, json_mode=args.json, prog="md-sim config validate")
    emit(
        {"path": args.path, "engine": config.engine, "system": config.system},
        json_mode=args.json,
        human=f"{args.path}: valid ({config.engine}, {config.system})",
    )
    return 0


def _find_workspace() -> Workspace | None:
    """Find the workspace of the cwd or ``$MD_WORKSPACE``, if there is one.

    :return: The workspace, or ``None`` when none is set up.
    :raises WorkspaceError: If a workspace file is found but cannot be used."""
    try:
        return load_workspace()
    except WorkspaceNotFound:
        return None


def _exists(path: Path) -> bool:
    """Whether a path exists, never raising on an unreadable or missing tree.

    :param path: The path to test.
    :return: ``True`` if it exists."""
    try:
        return path.exists()
    except OSError:
        return False


def _describe(argv: list[str]) -> int:
    """``md-sim config describe <path>``: what a config resolves to, without touching the data.

    :param argv: Arguments after the verb.
    :return: The exit code."""
    parser = _parser("describe", VERB_SUMMARIES["describe"])
    parser.add_argument("path", help="Path to the YAML config.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Resolve paths against this root (overrides MD_DATA_ROOT, the workspace, the config).",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.path)
    except (ValidationError, yaml.YAMLError) as exc:
        return _report_invalid(args.path, exc, json_mode=args.json)
    except OSError as exc:
        fail(exc, json_mode=args.json, prog="md-sim config describe")
    try:
        workspace = _find_workspace()
    except WorkspaceError as exc:
        fail(exc, json_mode=args.json, code=EXIT_USAGE, prog="md-sim config describe")

    warnings: list[str] = []
    try:
        data_root: Path | None = resolve_data_root(config, args.data_root, workspace)
    except ValueError as exc:
        data_root = None
        warnings.append(f"{exc} Relative paths are reported as written and as not existing.")

    def resolve(p: str) -> Path:
        """Resolve one config path; without a data root a relative one stays as written."""
        return config.resolve(data_root, p) if data_root is not None else Path(p)

    inputs = []
    for field in _INPUT_FIELDS:
        value = getattr(config, field, None)
        if value is None:
            continue
        resolved = resolve(value)
        found = _exists(resolved) and (data_root is not None or resolved.is_absolute())
        inputs.append({"field": field, "path": value, "resolved": str(resolved), "exists": found})
    payload = {
        "path": args.path,
        "engine": config.engine,
        "system": config.system,
        "data_root": str(data_root) if data_root is not None else None,
        "output_dir": str(resolve(config.output_subdir)),
        "inputs": inputs,
        "beamtime_group": config.beamtime_group,
        "beamtime_name": config.beamtime_name,
        "temperature": config.temperature,
        "num_steps": config.num_steps,
        "timestep": config.timestep,
        "total_time_ns": config.num_steps * config.timestep / 1000,
        "frame_spacing_ps": config.output_freq * config.timestep,
        "n_frames": config.num_steps // config.output_freq,
        "warnings": warnings,
    }
    lines = [f"{k}: {v}" for k, v in payload.items() if k not in ("inputs", "warnings")]
    lines += [f"input {i['field']}: {i['resolved']} (exists: {i['exists']})" for i in inputs]
    lines += [f"warning: {w}" for w in warnings]
    emit(payload, json_mode=args.json, human="\n".join(lines))
    return 0


def _ls(argv: list[str]) -> int:
    """``md-sim config ls``: the workspace's configs, a broken one listed with its error.

    :param argv: Arguments after the verb.
    :return: The exit code."""
    args = _parser("ls", VERB_SUMMARIES["ls"]).parse_args(argv)
    try:
        workspace = load_workspace()
        paths = workspace.configs()
    except WorkspaceError as exc:
        fail(exc, json_mode=args.json, code=EXIT_USAGE, prog="md-sim config ls")
    config_dir = workspace.resolved_config_dir()
    entries = []
    for path in paths:
        entry: dict[str, Any] = {
            "path": str(path),
            "name": path.relative_to(config_dir).as_posix(),
            "engine": None,
            "system": None,
            "temperature": None,
            "error": None,
        }
        try:
            config = load_config(path)
            entry.update(engine=config.engine, system=config.system, temperature=config.temperature)
        except ValidationError as exc:
            first = _errors(exc)[0]
            where = ".".join(map(str, first["loc"])) or "<root>"
            entry["error"] = (
                f"{len(_errors(exc))} validation error(s), first: {where}: {first['msg']}"
            )
        except yaml.YAMLError as exc:
            entry["error"] = f"invalid YAML: {exc}"
        except OSError as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entries.append(entry)
    width = max((len(e["name"]) for e in entries), default=0)
    lines = [
        f"{e['name'].ljust(width)}  "
        + (
            f"INVALID  {e['error']}"
            if e["error"]
            else f"{e['engine']}  {e['system']}  {e['temperature']} K"
        )
        for e in entries
    ]
    emit(
        {"config_dir": str(config_dir), "configs": entries},
        json_mode=args.json,
        human="\n".join(lines) or f"no configs in {config_dir}",
    )
    return 0


def _set_path(data: dict[str, Any], key: str, value: Any) -> None:
    """Set ``data[a][b]... = value`` for a dotted key, creating missing blocks.

    :param data: The config mapping to modify.
    :param key: A key, or a dotted path such as ``replica_exchange.n_replicas``.
    :param value: The value to store.
    :raises ValueError: If a path component below the top is not a mapping."""
    *parents, last = key.split(".")
    node = data
    for part in parents:
        child = node.get(part)
        if child is None:
            child = node[part] = {}
        if not isinstance(child, dict):
            raise ValueError(f"cannot set {key!r}: {part!r} is not a block, it is {child!r}")
        node = child
    node[last] = value


def _has_path(dumped: dict[str, Any], key: str) -> bool:
    """Whether a dotted key names a field the validated model knows.

    :param dumped: ``model_dump()`` of the validated config.
    :param key: A dotted key.
    :return: ``False`` for a key the model silently dropped (a typo)."""
    node: Any = dumped
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _new(argv: list[str]) -> int:
    """``md-sim config new --from CFG --set k=v -o PATH``: write a modified, validated copy.

    :param argv: Arguments after the verb.
    :return: The exit code."""
    parser = _parser("new", VERB_SUMMARIES["new"])
    parser.add_argument("--from", dest="source", required=True, help="Config to start from.")
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a field; repeatable. KEY may be dotted (replica_exchange.n_replicas) "
        "and VALUE is read as a YAML scalar (350 is a number, '350' a string). "
        "Unknown fields are rejected.",
    )
    parser.add_argument("-o", "--output", required=True, help="Where to write the new config.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file.")
    args = parser.parse_args(argv)
    prog = "md-sim config new"
    out = Path(args.output)
    try:
        data = yaml.safe_load(Path(args.source).read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{args.source} is not a YAML mapping")
        keys = []
        for item in args.sets:
            key, sep, raw = item.partition("=")
            if not sep or not key:
                raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
            _set_path(data, key, yaml.safe_load(raw))
            keys.append(key)
        config = _ADAPTER.validate_python(data)
        unknown = [k for k in keys if not _has_path(config.model_dump(), k)]
        if unknown:
            raise ValueError(f"unknown config field(s): {', '.join(unknown)}")
    except ValidationError as exc:
        return _report_invalid(args.source, exc, json_mode=args.json)
    except (ValueError, yaml.YAMLError, OSError) as exc:
        fail(exc, json_mode=args.json, code=EXIT_USAGE, prog=prog)
    if out.exists() and not args.force:
        fail(
            FileExistsError(f"{out} exists; pass --force to overwrite"),
            json_mode=args.json,
            prog=prog,
        )
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(data, sort_keys=False))
    except OSError as exc:
        fail(exc, json_mode=args.json, prog=prog)
    emit(
        {"path": str(out), "engine": config.engine, "system": config.system, "set": keys},
        json_mode=args.json,
        human=f"wrote {out}",
    )
    return 0


VERB_SUMMARIES: dict[str, str] = {
    "schema": "Print the JSON Schema of the config union, or of one engine with --engine.",
    "validate": "Check a config file; every error is listed with its location.",
    "describe": "Show what a config resolves to: paths, run name, duration and frame count.",
    "ls": "List the workspace's configs with engine, system and temperature.",
    "new": "Write a validated copy of a config with --set overrides (comments are not kept).",
}

# verb -> (summary, handler taking the argv after the verb and returning an exit code)
VERBS: dict[str, tuple[str, Callable[[list[str]], int]]] = {
    "schema": (VERB_SUMMARIES["schema"], _schema),
    "validate": (VERB_SUMMARIES["validate"], _validate),
    "describe": (VERB_SUMMARIES["describe"], _describe),
    "ls": (VERB_SUMMARIES["ls"], _ls),
    "new": (VERB_SUMMARIES["new"], _new),
}
