"""Preflight an sbatch script against the workspace, the site profile and the configs it runs.

The mistakes this catches are the ones an author cannot see from the script: a partition the
cluster does not have queues forever, a data root that points at the wrong tree writes a long
trajectory somewhere nobody looks, and an output directory that does not exist makes the job die
before its first line. ``check_job`` never runs, submits or modifies anything.

A check is ``skipped`` rather than guessed whenever the script does not state the fact statically
(a config path built from a shell variable, a ``--partition=$PART``): a preflight that passes a
broken script is worse than none. Relative paths in the script resolve against the directory the job
starts in, which is the workspace root (the submit directory) unless ``#SBATCH --chdir`` says otherwise.

Importing this module needs only the core dependencies (pydantic, pyyaml)."""

import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from md_simulations.config import load_config, resolve_data_root
from md_simulations.config.base import SimulationConfig
from md_simulations.workspace import Site, Workspace, WorkspaceError

Status = Literal["ok", "fail", "skipped", "warn"]

# Short sbatch options that map onto a long name the checks read.
_SHORT_OPTIONS = {
    "p": "partition",
    "J": "job-name",
    "t": "time",
    "o": "output",
    "e": "error",
    "D": "chdir",
    "G": "gpus",
    "N": "nodes",
    "n": "ntasks",
    "c": "cpus-per-task",
    "w": "nodelist",
    "x": "exclude",
    "A": "account",
    "q": "qos",
    "d": "dependency",
    "a": "array",
    "C": "constraint",
}
# Short options that take no value, so ``-N1`` style joining is not mistaken for ``-N 1``.
_SHORT_FLAGS = {"H", "h", "O", "s", "Q", "v", "W"}

_SBATCH = re.compile(r"^\s*#SBATCH\b(.*)$")
_HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")
_ASSIGNMENT = re.compile(r"^[A-Za-z_]\w*=")
_PUNCTUATION = set("();<>|&")
_PREFIXES = {"then", "do", "else", "!", "{", "exec", "time", "nohup", "srun", "if", "elif"}
_DYNAMIC = set("$`*?[{(~")
_GPU_GRES = re.compile(r"(^|,)gpu(:|,|$)")
_ZERO_GPU = re.compile(r"(^|,)gpu(:[A-Za-z0-9_.-]+)*:0(,|$)")
_INPUT_FIELDS = ("input_pdb", "input_gro", "top", "model_file")


@dataclass(frozen=True)
class Check:
    """The outcome of one preflight check.

    :param name: ``directives``, ``config``, ``inputs``, ``partition``, ``gres``, ``output`` or
        ``venv``.
    :param status: ``ok``, ``fail``, ``skipped`` (not determinable, not a pass) or ``warn``.
    :param detail: One line saying what was checked or why it failed."""

    name: str
    status: Status
    detail: str

    @property
    def ok(self) -> bool:
        """Whether the check did not fail (``warn`` and ``skipped`` do not fail a script)."""
        return self.status != "fail"


@dataclass(frozen=True)
class RunInvocation:
    """One ``md-sim run`` command found in a script.

    :param config: The config path as written, or ``None`` when it is not statically determinable.
    :param data_root: The command's own ``--data-root`` value, if it has a static one.
    :param line: The 1-based line number of the command."""

    config: str | None
    data_root: str | None
    line: int


@dataclass
class ScriptFacts:
    """What a script states statically.

    :param directives: ``#SBATCH`` options by long name (last one wins), values without quotes.
    :param late_sbatch: Line numbers of ``#SBATCH`` lines after the first command; slurm ignores them.
    :param runs: Every ``md-sim run`` invocation, including those with a dynamic config path.
    :param data_root_export: The last static ``export MD_DATA_ROOT=`` value.
    :param data_root_dynamic: Whether ``MD_DATA_ROOT`` is also exported with a value built at run time."""

    directives: dict[str, str] = field(default_factory=dict)
    late_sbatch: list[int] = field(default_factory=list)
    runs: list[RunInvocation] = field(default_factory=list)
    data_root_export: str | None = None
    data_root_dynamic: bool = False


def _is_static(value: str) -> bool:
    """Whether a shell word has no expansion in it.

    :param value: The word with its quotes already removed.
    :return: ``True`` if it contains none of ``$``, a backtick, a glob character or ``~``."""
    return not (_DYNAMIC & set(value))


def _parse_directive(rest: str, directives: dict[str, str]) -> None:
    """Add the options of one ``#SBATCH`` line to ``directives``.

    :param rest: The line after ``#SBATCH``.
    :param directives: The mapping to update; short options are stored under their long name."""
    try:
        tokens = shlex.split(rest, comments=True)
    except ValueError:
        return
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if tok.startswith("--"):
            name, eq, value = tok[2:].partition("=")
            if not eq and i < len(tokens) and not tokens[i].startswith("-"):
                value, i = tokens[i], i + 1
            directives[name] = value
        elif tok.startswith("-") and len(tok) >= 2:
            letter = tok[1]
            if letter in _SHORT_FLAGS:
                directives[letter] = ""
                continue
            value = tok[2:].removeprefix("=")
            if not value and i < len(tokens) and not tokens[i].startswith("-"):
                value, i = tokens[i], i + 1
            directives[_SHORT_OPTIONS.get(letter, letter)] = value


def _commands(tokens: list[str]) -> list[list[str]]:
    """Split a tokenised shell line into commands and drop leading keywords and assignments.

    :param tokens: Tokens from ``shlex`` with ``punctuation_chars`` set.
    :return: The commands, each starting at its program name."""
    commands: list[list[str]] = [[]]
    for tok in tokens:
        if tok and set(tok) <= _PUNCTUATION:
            commands.append([])
        else:
            commands[-1].append(tok)
    result = []
    for command in commands:
        while command and (command[0] in _PREFIXES or _ASSIGNMENT.match(command[0])):
            command = command[1:]
        if command:
            result.append(command)
    return result


def _run_args(command: list[str]) -> list[str] | None:
    """Return the arguments after ``run`` if the command is ``md-sim run``.

    :param command: One command, from its program name on.
    :return: The arguments, or ``None`` for any other command."""
    program = Path(command[0]).name
    if program == "md-sim" and command[1:2] == ["run"]:
        return command[2:]
    if program.startswith("python") and command[1:4] == ["-m", "md_simulations.cli", "run"]:
        return command[4:]
    return None


def _invocation(args: list[str], line: int) -> RunInvocation | None:
    """Read the config path and data root out of the arguments of ``md-sim run``.

    :param args: The arguments after ``run``.
    :param line: The line number, for the result.
    :return: The invocation, or ``None`` for ``--help`` (it runs nothing)."""
    config: str | None = None
    positional = False
    data_root: str | None = None
    dynamic_root = False
    i = 0
    while i < len(args):
        tok = args[i]
        i += 1
        if tok in ("-h", "--help"):
            return None
        if tok == "--data-root" or tok.startswith("--data-root="):
            value = tok.partition("=")[2] if "=" in tok else (args[i] if i < len(args) else "")
            i += 0 if "=" in tok else 1
            if _is_static(value) and value:
                data_root = value
            else:
                dynamic_root = True
        elif tok.startswith("-"):
            continue
        elif not positional:
            positional = True
            config = tok if _is_static(tok) else None
    if dynamic_root:
        # A data root we cannot read makes every path under it unknowable.
        config = None
    return RunInvocation(config=config, data_root=data_root, line=line)


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Join backslash continuations and drop comments and heredoc bodies.

    :param text: The script.
    :return: ``(line number, logical line)`` pairs, for lines that can hold a command."""
    lines: list[tuple[int, str]] = []
    pending: tuple[int, str] | None = None
    heredoc: str | None = None
    for number, raw in enumerate(text.splitlines(), start=1):
        if heredoc is not None:
            if raw.strip() == heredoc:
                heredoc = None
            continue
        line = raw if pending is None else pending[1] + raw
        start = number if pending is None else pending[0]
        if line.rstrip().endswith("\\"):
            pending = (start, line.rstrip()[:-1] + " ")
            continue
        pending = None
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if match := _HEREDOC.search(line):
            heredoc = match.group(1)
        lines.append((start, line))
    return lines


def parse_script(text: str) -> ScriptFacts:
    """Read the ``#SBATCH`` directives, ``md-sim run`` invocations and data-root exports of a script.

    Directives stop at the first line that is neither blank nor a comment, as in slurm; later
    ``#SBATCH`` lines are only recorded in ``late_sbatch``.

    :param text: The script text.
    :return: The facts the script states statically."""
    facts = ScriptFacts()
    in_header = True
    for number, raw in enumerate(text.splitlines(), start=1):
        if match := _SBATCH.match(raw):
            if in_header:
                _parse_directive(match.group(1), facts.directives)
            else:
                facts.late_sbatch.append(number)
        elif raw.strip() and not raw.lstrip().startswith("#"):
            in_header = False
    for number, line in _logical_lines(text):
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            # Unbalanced quoting: if the line looks like a run, record it as not determinable.
            if re.search(r"(md-sim|md_simulations\.cli)\s+run\b", line):
                facts.runs.append(RunInvocation(config=None, data_root=None, line=number))
            continue
        for command in _commands(tokens):
            if (args := _run_args(command)) is not None:
                if (run := _invocation(args, number)) is not None:
                    facts.runs.append(run)
            elif command[0] == "export":
                for word in command[1:]:
                    name, eq, value = word.partition("=")
                    if name == "MD_DATA_ROOT" and eq:
                        if _is_static(value) and value:
                            facts.data_root_export, facts.data_root_dynamic = value, False
                        else:
                            facts.data_root_dynamic = True
    return facts


def _worst(statuses: list[Status]) -> Status:
    """Combine sub-results: any fail, else any warn, else any skipped, else ok.

    :param statuses: The sub-results.
    :return: The combined status."""
    for status in ("fail", "warn", "skipped"):
        if status in statuses:
            return status  # type: ignore[return-value]
    return "ok"


def _base_dir(facts: ScriptFacts, workspace: Workspace) -> Path | None:
    """The directory relative paths in the script resolve against.

    :param facts: The parsed script.
    :param workspace: The workspace; its root is the submit directory.
    :return: The workspace root, the static ``--chdir`` if given, or ``None`` if ``--chdir`` is dynamic."""
    chdir = facts.directives.get("chdir")
    if chdir is None:
        return workspace.root
    return workspace.resolve(chdir) if chdir and _is_static(chdir) else None


def _in_base(base: Path | None, value: str) -> Path | None:
    """Resolve a script path, or ``None`` if the base directory is unknown.

    :param base: From ``_base_dir``.
    :param value: A static path from the script.
    :return: The absolute path, or ``None`` when a relative path cannot be placed."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return None if base is None else base / path


def _check_directives(facts: ScriptFacts) -> Check:
    """Report ``#SBATCH`` lines that slurm ignores.

    :param facts: The parsed script.
    :return: ``warn`` if any directive follows the first command, else ``ok``."""
    if facts.late_sbatch:
        lines = ", ".join(str(n) for n in facts.late_sbatch)
        return Check(
            "directives",
            "warn",
            f"#SBATCH after the first command is ignored by slurm (line {lines}); "
            "move it above the first command",
        )
    return Check("directives", "ok", f"{len(facts.directives)} #SBATCH option(s) in the header")


def _check_config(
    facts: ScriptFacts, base: Path | None
) -> tuple[Check, dict[int, SimulationConfig]]:
    """Check that each statically named config exists and validates.

    :param facts: The parsed script.
    :param base: Directory relative config paths resolve against.
    :return: The check and the loaded configs by index into ``facts.runs``."""
    loaded: dict[int, SimulationConfig] = {}
    if not facts.runs:
        return Check("config", "skipped", "no `md-sim run <config>` command in the script"), loaded
    problems: list[str] = []
    undetermined = 0
    for index, run in enumerate(facts.runs):
        path = None if run.config is None else _in_base(base, run.config)
        if path is None:
            undetermined += 1
            continue
        if not path.is_file():
            problems.append(f"{run.config}: no such file ({path})")
            continue
        try:
            loaded[index] = load_config(path)
        except (OSError, ValueError, ValidationError) as err:
            first = str(err).strip().splitlines()[0] if str(err).strip() else type(err).__name__
            problems.append(f"{run.config}: invalid config ({first})")
    if problems:
        return Check("config", "fail", "; ".join(problems)), loaded
    if not loaded:
        return (
            Check("config", "skipped", "config path not statically determinable"),
            loaded,
        )
    detail = f"{len(loaded)} config(s) found and valid"
    if undetermined:
        return (
            Check(
                "config",
                "warn",
                f"{detail}; {undetermined} more `md-sim run` with a config path not statically "
                "determinable were not checked",
            ),
            loaded,
        )
    return Check("config", "ok", detail), loaded


def _data_root(
    run: RunInvocation,
    config: SimulationConfig,
    facts: ScriptFacts,
    workspace: Workspace,
    data_root: str | None,
) -> tuple[Path | None, str]:
    """Resolve the data root a run would use.

    Precedence: the command's ``--data-root``, the ``data_root`` argument of the check, a static
    ``export MD_DATA_ROOT=`` in the script, then the order of ``resolve_data_root``.

    :param run: The invocation.
    :param config: Its loaded config.
    :param facts: The parsed script.
    :param workspace: The workspace.
    :param data_root: The ``--data-root`` given to ``check-job``.
    :return: The absolute root (``None`` if undeterminable) and a short description of why."""
    explicit = run.data_root or data_root or facts.data_root_export
    if explicit is None and facts.data_root_dynamic:
        return None, "the script sets MD_DATA_ROOT at run time"
    try:
        root = resolve_data_root(config, explicit, workspace)
    except ValueError as err:
        return None, str(err)
    return workspace.resolve(root), "data root"


def _check_inputs(
    facts: ScriptFacts,
    loaded: dict[int, SimulationConfig],
    workspace: Workspace,
    data_root: str | None,
) -> Check:
    """Check that each config's input files exist under the data root.

    :param facts: The parsed script.
    :param loaded: The configs that loaded, by index into ``facts.runs``.
    :param workspace: The workspace.
    :param data_root: The ``--data-root`` given to ``check-job``.
    :return: The check."""
    if not loaded:
        return Check("inputs", "skipped", "no config could be loaded")
    statuses: list[Status] = []
    notes: list[str] = []
    for index, config in loaded.items():
        run = facts.runs[index]
        root, why = _data_root(run, config, facts, workspace, data_root)
        if root is None:
            statuses.append("fail" if why.startswith("No data root") else "skipped")
            notes.append(f"{run.config}: {why}")
            continue
        missing = []
        for name in _INPUT_FIELDS:
            value = getattr(config, name, None)
            if value and not config.resolve(root, value).exists():
                missing.append(f"{name} {config.resolve(root, value)}")
        if missing:
            statuses.append("fail")
            notes.append(f"{run.config}: missing under {root}: " + ", ".join(missing))
            continue
        statuses.append("ok")
        notes.append(f"{run.config}: inputs found under {root}")
        ws_root = workspace.resolved_data_root()
        from_export = run.data_root is None and data_root is None and facts.data_root_export
        if from_export and ws_root is not None and root != ws_root:
            statuses.append("warn")
            notes.append(f"the script exports MD_DATA_ROOT={root}, the workspace says {ws_root}")
    return Check("inputs", _worst(statuses), "; ".join(notes))


def _load_site(workspace: Workspace) -> tuple[Site | None, str | None]:
    """Load the site profile without raising.

    :param workspace: The workspace.
    :return: ``(site, None)``, ``(None, None)`` if the workspace sets no site, or ``(None, error)``
        if the profile is set but unusable."""
    if not workspace.site:
        return None, None
    try:
        return workspace.load_site(), None
    except WorkspaceError as err:
        return None, str(err)


def _check_partition(facts: ScriptFacts, site: Site | None, site_error: str | None) -> Check:
    """Check ``--partition`` against the site's partitions.

    A workspace without a site, or a site with no ``partitions``, means "do not check" and is
    ``skipped``; a site that is set but cannot be loaded fails, since it cannot be verified against.

    :param facts: The parsed script.
    :param site: The site profile, if any.
    :param site_error: Why the profile could not be loaded.
    :return: The check."""
    if site_error:
        return Check("partition", "fail", f"cannot verify: {site_error}")
    if site is None:
        return Check("partition", "skipped", "the workspace sets no site")
    if not site.partitions:
        return Check("partition", "skipped", f"site {site.name!r} lists no partitions")
    value = facts.directives.get("partition")
    if value is None:
        return Check(
            "partition",
            "warn",
            f"no --partition, the cluster default applies (site {site.name!r} allows: "
            f"{', '.join(site.partitions)})",
        )
    if not _is_static(value):
        return Check("partition", "skipped", f"--partition={value} is not statically determinable")
    wanted = [p for p in value.split(",") if p]
    unknown = [p for p in wanted if p not in site.partitions]
    if unknown:
        return Check(
            "partition",
            "fail",
            f"--partition={value}: {', '.join(unknown)} not in site {site.name!r} partitions "
            f"({', '.join(site.partitions)})",
        )
    return Check("partition", "ok", f"--partition={value} is in site {site.name!r}")


def _gres_gpus(facts: ScriptFacts) -> str | None:
    """Describe the GPU request in the directives.

    :param facts: The parsed script.
    :return: The directive that requests a GPU (``--gres=gpu:1``), or ``None`` if there is none."""
    gres = facts.directives.get("gres", "")
    if _GPU_GRES.search(gres) and not _ZERO_GPU.search(gres):
        return f"--gres={gres}"
    for name in ("gpus", "gpus-per-node", "gpus-per-task"):
        value = facts.directives.get(name, "")
        if value and not re.fullmatch(r"(\w+:)?0", value):
            return f"--{name}={value}"
    return None


def _check_gres(facts: ScriptFacts, require_gpu: bool = False) -> Check:
    """Report whether the script requests a GPU.

    A config does not state its platform, so whether the job *needs* a GPU is not determinable from
    it. Without ``require_gpu`` this check never fails; the author or agent states the need with it.

    :param facts: The parsed script.
    :param require_gpu: Fail if no GPU is requested.
    :return: ``ok`` if a GPU is requested; else ``fail`` with ``require_gpu`` and ``skipped`` without."""
    if found := _gres_gpus(facts):
        return Check("gres", "ok", f"{found} requests a GPU")
    if require_gpu:
        return Check("gres", "fail", "no GPU requested (add e.g. --gres=gpu:1) but one is required")
    return Check("gres", "skipped", "no GPU requested; the configs do not say if the job needs one")


def _check_output(facts: ScriptFacts, base: Path | None) -> Check:
    """Check the directories of ``--output`` and ``--error``.

    Slurm does not create them, so a missing one makes the job fail before its first line; that is a
    ``warn`` if it could still be created, a ``fail`` if no writable parent exists.

    :param facts: The parsed script.
    :param base: Directory relative paths resolve against.
    :return: The check."""
    targets = [(k, facts.directives[k]) for k in ("output", "error") if k in facts.directives]
    if not targets:
        return Check("output", "skipped", "no --output/--error; slurm writes slurm-<jobid>.out")
    statuses: list[Status] = []
    notes: list[str] = []
    for key, value in targets:
        directory = os.path.dirname(value) or "."
        if not _is_static(directory) or "%" in directory:
            statuses.append("skipped")
            notes.append(f"--{key}={value} is not statically determinable")
            continue
        path = _in_base(base, directory)
        if path is None:
            statuses.append("skipped")
            notes.append(f"--{key}={value} is relative to a dynamic --chdir")
            continue
        if path.is_dir():
            statuses.append("ok" if os.access(path, os.W_OK | os.X_OK) else "fail")
            notes.append(
                f"--{key} dir {path} " + ("exists" if statuses[-1] == "ok" else "is not writable")
            )
            continue
        parent = next((p for p in path.parents if p.exists()), None)
        if parent is not None and parent.is_dir() and os.access(parent, os.W_OK | os.X_OK):
            statuses.append("warn")
            notes.append(
                f"--{key} dir {path} does not exist; slurm does not create it, mkdir it first"
            )
        else:
            statuses.append("fail")
            notes.append(f"--{key} dir {path} does not exist and cannot be created")
    return Check("output", _worst(statuses), "; ".join(notes))


def _check_venv(workspace: Workspace, site: Site | None, site_error: str | None) -> Check:
    """Check that the site's ``venv_activate`` script exists.

    :param workspace: The workspace, for resolving a relative path.
    :param site: The site profile, if any.
    :param site_error: Why the profile could not be loaded.
    :return: The check; ``skipped`` without a site or when the site sets no ``venv_activate``."""
    if site_error:
        return Check("venv", "fail", f"cannot verify: {site_error}")
    if site is None:
        return Check("venv", "skipped", "the workspace sets no site")
    if not site.venv_activate:
        return Check("venv", "skipped", f"site {site.name!r} sets no venv_activate")
    path = workspace.resolve(site.venv_activate)
    if path.is_file():
        return Check("venv", "ok", f"{path} exists")
    return Check("venv", "fail", f"venv_activate {path} does not exist")


def check_job(
    path: str | Path,
    workspace: Workspace,
    *,
    data_root: str | None = None,
    require_gpu: bool = False,
) -> list[Check]:
    """Preflight one sbatch script.

    Checks, in order: ``directives``, ``config``, ``inputs``, ``partition``, ``gres``, ``output``,
    ``venv``. Nothing is created, submitted or run.

    :param path: The script.
    :param workspace: The workspace the job will run in.
    :param data_root: A data root to check as if passed to ``md-sim run --data-root``; it takes
        precedence over a static ``export MD_DATA_ROOT=`` in the script, ``MD_DATA_ROOT`` and the
        workspace.
    :param require_gpu: Fail the ``gres`` check unless the script requests a GPU.
    :return: One ``Check`` per check; the script passes if none has status ``fail``.
    :raises OSError: If the script cannot be read."""
    facts = parse_script(Path(path).read_text())
    base = _base_dir(facts, workspace)
    site, site_error = _load_site(workspace)
    config_check, loaded = _check_config(facts, base)
    return [
        _check_directives(facts),
        config_check,
        _check_inputs(facts, loaded, workspace, data_root),
        _check_partition(facts, site, site_error),
        _check_gres(facts, require_gpu),
        _check_output(facts, base),
        _check_venv(workspace, site, site_error),
    ]


def format_checks(checks: list[Check]) -> str:
    """Render checks as one aligned line each.

    :param checks: The results of ``check_job``.
    :return: The text, without a trailing newline."""
    width = max((len(c.name) for c in checks), default=0)
    return "\n".join(f"{c.status.upper():<8}{c.name:<{width}}  {c.detail}" for c in checks)
