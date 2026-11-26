"""The workspace descriptor: where configs, data and run state live, and which site this is.

A lab repo states its layout in one ``md-workspace.yaml`` instead of the package learning it by being
run from a particular checkout. Cluster knowledge (partitions, venv activation, modules) lives in a
site profile, ``sites/<site>.yaml`` beside the descriptor; the package itself names none of it.

Example ``md-workspace.yaml``::

    config_dir: configs
    data_root: ${MD_DATA_ROOT:-./data}
    run_store: beamtime
    site: gpu04
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

WORKSPACE_FILE = "md-workspace.yaml"
WORKSPACE_ENV = "MD_WORKSPACE"

_INTERPOLATION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class WorkspaceError(Exception):
    """A workspace descriptor or site profile that cannot be used."""


class WorkspaceNotFound(WorkspaceError):
    """No ``md-workspace.yaml`` was found.

    :param searched: The directories (or the ``$MD_WORKSPACE`` target) that were looked in."""

    def __init__(self, searched: list[Path]) -> None:
        self.searched = list(searched)
        listing = ", ".join(str(p) for p in self.searched)
        super().__init__(
            f"No {WORKSPACE_FILE} found (searched: {listing}). Create one in your lab repo "
            f"or point ${WORKSPACE_ENV} at it."
        )


class Site(BaseModel):
    """Cluster-specific facts a job script must respect.

    Empty ``partitions`` means "unknown, do not check"."""

    model_config = ConfigDict(extra="forbid")

    name: str
    partitions: list[str] = []
    venv_activate: str | None = None
    slurm_output: str = "./slurm/output"
    modules: list[str] = []


class Workspace(BaseModel):
    """A lab repo's layout, as read from ``md-workspace.yaml``.

    Relative paths resolve against ``root``, the directory holding the descriptor, never the cwd."""

    model_config = ConfigDict(extra="forbid")

    root: Path = Field(exclude=True)
    config_dir: Path = Path("configs")
    data_root: str | None = None
    run_store: Path = Path("beamtime")
    site: str | None = None

    def resolve(self, p: str | Path) -> Path:
        """Resolve a path against the descriptor's directory.

        :param p: An absolute path, a ``~`` path or a path relative to ``root``.
        :return: The absolute path."""
        path = Path(p).expanduser()
        return path if path.is_absolute() else self.root / path

    def resolved_data_root(self) -> Path | None:
        """The data root as an absolute path.

        :return: The resolved ``data_root``, or ``None`` if the descriptor sets none."""
        return None if not self.data_root else self.resolve(self.data_root)

    def resolved_config_dir(self) -> Path:
        """The config directory as an absolute path.

        :return: The resolved ``config_dir``."""
        return self.resolve(self.config_dir)

    def resolved_run_store(self) -> Path:
        """The run store (beamtime log dir) as an absolute path.

        :return: The resolved ``run_store``."""
        return self.resolve(self.run_store)

    def load_site(self) -> Site:
        """Read the site profile ``<root>/sites/<site>.yaml``.

        :return: The validated site profile.
        :raises WorkspaceError: If no site is set, the profile file is missing or it is invalid."""
        if not self.site:
            raise WorkspaceError(f"No site set in {self.root / WORKSPACE_FILE}.")
        if Path(self.site).name != self.site or self.site.startswith("."):
            raise WorkspaceError(f"Invalid site name {self.site!r}: it must be a plain file stem.")
        sites_dir = self.root / "sites"
        path = sites_dir / f"{self.site}.yaml"
        if not path.is_file():
            raise WorkspaceError(f"Unknown site {self.site!r}: no {path.name} in {sites_dir}.")
        data = _read_yaml(path)
        data.setdefault("name", self.site)
        try:
            return Site.model_validate(data)
        except ValidationError as err:
            raise WorkspaceError(f"Invalid site profile {path}:\n{err}") from err

    def configs(self) -> list[Path]:
        """List the simulation configs under ``config_dir``, excluding ``generated/``.

        :return: Sorted paths of ``*.yaml`` files.
        :raises WorkspaceError: If the config directory does not exist."""
        config_dir = self.resolved_config_dir()
        if not config_dir.is_dir():
            raise WorkspaceError(f"config_dir {config_dir} does not exist.")
        return sorted(
            p
            for p in config_dir.rglob("*.yaml")
            if "generated" not in p.relative_to(config_dir).parts[:-1]
        )


def _expand(value: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` from the environment.

    No shell is involved and nothing else is interpreted. As in a shell, ``:-`` applies the default
    when the variable is unset or empty.

    :param value: The raw string.
    :return: The string with every interpolation replaced.
    :raises WorkspaceError: If a variable is unset or empty and has no default."""

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        found = os.environ.get(name)
        if found:
            return found
        if default is not None:
            return default
        raise WorkspaceError(
            f"${{{name}}} is not set; set the variable or write ${{{name}:-default}}."
        )

    return _INTERPOLATION.sub(replace, value)


def _expand_all(node: Any) -> Any:
    """Apply ``_expand`` to every string in a loaded YAML structure.

    :param node: A scalar, list or dict from ``yaml.safe_load``.
    :return: The same structure with strings expanded."""
    if isinstance(node, str):
        return _expand(node)
    if isinstance(node, list):
        return [_expand_all(item) for item in node]
    if isinstance(node, dict):
        return {key: _expand_all(item) for key, item in node.items()}
    return node


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, with ``${VAR}`` interpolation applied.

    :param path: The YAML file.
    :return: Its content, an empty dict for an empty file.
    :raises WorkspaceError: If the file is not a mapping or interpolation fails."""
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise WorkspaceError(f"{path} must be a YAML mapping.")
    try:
        return _expand_all(data)
    except WorkspaceError as err:
        raise WorkspaceError(f"{path}: {err}") from err


def _search_dirs(start: Path) -> list[Path]:
    """List the directories the upward walk visits.

    The walk ends after the git root or a filesystem boundary (mount point), whichever comes first,
    or at ``/``, so it never wanders into a parent that is not part of the project.

    :param start: The directory to start from.
    :return: The directories, innermost first."""
    dirs: list[Path] = []
    for d in (start, *start.parents):
        dirs.append(d)
        if (d / ".git").exists() or d.is_mount():
            break
    return dirs


def load_workspace(start: Path | None = None) -> Workspace:
    """Find and load ``md-workspace.yaml``.

    ``$MD_WORKSPACE`` wins when set; it names the descriptor file or the directory containing it.
    Otherwise the search walks up from ``start`` (default: the cwd) to the git root or the
    filesystem boundary.

    :param start: Directory to start the upward walk from.
    :return: The loaded workspace, ``root`` set to the descriptor's directory.
    :raises WorkspaceNotFound: If no descriptor is found.
    :raises WorkspaceError: If the descriptor is invalid."""
    if env := os.environ.get(WORKSPACE_ENV):
        target = Path(env).expanduser()
        path = target / WORKSPACE_FILE if target.is_dir() else target
        if not path.is_file():
            raise WorkspaceNotFound([target])
        return _load(path)
    searched = _search_dirs((start or Path.cwd()).resolve())
    for d in searched:
        if (d / WORKSPACE_FILE).is_file():
            return _load(d / WORKSPACE_FILE)
    raise WorkspaceNotFound(searched)


def _load(path: Path) -> Workspace:
    """Load one descriptor file.

    :param path: The ``md-workspace.yaml`` file.
    :return: The validated workspace rooted at the file's directory.
    :raises WorkspaceError: If the file is invalid."""
    data = _read_yaml(path)
    if "root" in data:
        raise WorkspaceError(f"{path}: 'root' is the descriptor's own directory; do not set it.")
    try:
        return Workspace.model_validate({**data, "root": path.resolve().parent})
    except ValidationError as err:
        raise WorkspaceError(f"Invalid workspace {path}:\n{err}") from err
