from pathlib import Path

import pytest
import yaml

from md_simulations.workspace import (
    Site,
    Workspace,
    WorkspaceError,
    WorkspaceNotFound,
    _expand,
    load_workspace,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> None:
    """Keep the ambient workspace variables out of every test.

    :param monkeypatch: Pytest monkeypatch fixture.
    :return: None."""
    monkeypatch.delenv("MD_WORKSPACE", raising=False)
    monkeypatch.delenv("MD_DATA_ROOT", raising=False)


def _write(root: Path, **fields) -> Path:
    """Write an ``md-workspace.yaml`` into ``root``.

    :param root: Directory to write into.
    :param fields: Descriptor fields.
    :return: The descriptor path."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "md-workspace.yaml"
    path.write_text(yaml.safe_dump(fields))
    return path


def test_finds_descriptor_by_walking_up(tmp_path: Path) -> None:
    """A nested start directory finds the descriptor above it.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    _write(tmp_path, site="gpu04")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    ws = load_workspace(nested)
    assert ws.root == tmp_path.resolve()
    assert ws.site == "gpu04"
    assert ws.config_dir == Path("configs")


def test_env_overrides_walk(tmp_path: Path, monkeypatch) -> None:
    """MD_WORKSPACE wins over the walk, as a directory or as the file itself.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :return: None."""
    _write(tmp_path / "near", site="near")
    far = _write(tmp_path / "far", site="far")
    monkeypatch.setenv("MD_WORKSPACE", str(far.parent))
    assert load_workspace(tmp_path / "near").site == "far"
    monkeypatch.setenv("MD_WORKSPACE", str(far))
    assert load_workspace(tmp_path / "near").site == "far"
    monkeypatch.setenv("MD_WORKSPACE", str(tmp_path / "nothing"))
    with pytest.raises(WorkspaceNotFound):
        load_workspace(tmp_path / "near")


def test_missing_descriptor_lists_searched(tmp_path: Path) -> None:
    """A missing descriptor raises with every directory that was searched.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "a"
    nested.mkdir()
    with pytest.raises(WorkspaceNotFound) as info:
        load_workspace(nested)
    assert info.value.searched == [nested.resolve(), tmp_path.resolve()]
    assert str(nested.resolve()) in str(info.value)


def test_walk_stops_at_git_root(tmp_path: Path) -> None:
    """A descriptor above the git root is not picked up.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    _write(tmp_path, site="outside")
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    with pytest.raises(WorkspaceNotFound) as info:
        load_workspace(repo)
    assert tmp_path.resolve() not in info.value.searched
    _write(repo, site="inside")
    assert load_workspace(repo).site == "inside"


def test_relative_paths_resolve_against_descriptor(tmp_path: Path, monkeypatch) -> None:
    """Relative config_dir, data_root and run_store resolve against the descriptor, not the cwd.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :return: None."""
    root = tmp_path / "lab"
    _write(root, config_dir="cfg", data_root="./data", run_store="bt")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    ws = load_workspace(root)
    assert ws.resolved_config_dir() == root.resolve() / "cfg"
    assert ws.resolved_data_root() == root.resolve() / "data"
    assert ws.resolved_run_store() == root.resolve() / "bt"
    assert ws.resolve("/abs/x") == Path("/abs/x")


def test_interpolation_default_and_env(tmp_path: Path, monkeypatch) -> None:
    """${MD_DATA_ROOT:-./data} takes the environment value, else the default.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :return: None."""
    _write(tmp_path, data_root="${MD_DATA_ROOT:-./data}")
    assert load_workspace(tmp_path).resolved_data_root() == tmp_path.resolve() / "data"
    monkeypatch.setenv("MD_DATA_ROOT", "/shared/data")
    assert load_workspace(tmp_path).resolved_data_root() == Path("/shared/data")
    monkeypatch.setenv("MD_DATA_ROOT", "")
    assert load_workspace(tmp_path).resolved_data_root() == tmp_path.resolve() / "data"


def test_unset_interpolation(tmp_path: Path, monkeypatch) -> None:
    """A variable with no value and no default is an error, never an empty path.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :return: None."""
    monkeypatch.delenv("MD_NOPE", raising=False)
    _write(tmp_path, data_root="${MD_NOPE}/data")
    with pytest.raises(WorkspaceError, match="MD_NOPE"):
        load_workspace(tmp_path)


def test_expand_is_not_a_shell(monkeypatch) -> None:
    """Only ${VAR} and ${VAR:-default} are interpreted; commands and bare $VAR are left alone.

    :param monkeypatch: Pytest monkeypatch fixture.
    :return: None."""
    monkeypatch.setenv("MD_X", "x")
    assert _expand("$(echo hi)/$MD_X/${MD_X}/${MD_Y:-}") == "$(echo hi)/$MD_X/x/"


def test_unknown_field_rejected(tmp_path: Path) -> None:
    """A misspelt descriptor field is an error, not silently ignored.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    _write(tmp_path, datadir="x")
    with pytest.raises(WorkspaceError, match="datadir"):
        load_workspace(tmp_path)


def test_site_profile(tmp_path: Path) -> None:
    """The site profile loads from sites/<site>.yaml with defaults filled in.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    _write(tmp_path, site="gpu04")
    (tmp_path / "sites").mkdir()
    (tmp_path / "sites" / "gpu04.yaml").write_text(
        yaml.safe_dump({"partitions": ["p1", "p2"], "modules": ["cuda"]})
    )
    site = load_workspace(tmp_path).load_site()
    assert site == Site(name="gpu04", partitions=["p1", "p2"], modules=["cuda"])
    assert site.venv_activate is None
    assert site.slurm_output == "./slurm/output"


def test_unknown_site(tmp_path: Path) -> None:
    """A site with no profile file names the sites/ directory that was searched.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    _write(tmp_path, site="nowhere")
    ws = load_workspace(tmp_path)
    with pytest.raises(WorkspaceError, match="nowhere") as info:
        ws.load_site()
    assert str(tmp_path.resolve() / "sites") in str(info.value)
    with pytest.raises(WorkspaceError, match="No site set"):
        Workspace(root=tmp_path).load_site()


def test_configs_listing(tmp_path: Path) -> None:
    """configs() is sorted and skips generated/.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    _write(tmp_path)
    cfg = tmp_path / "configs"
    (cfg / "generated").mkdir(parents=True)
    (cfg / "sub").mkdir()
    for name in ("b.yaml", "a.yaml", "sub/c.yaml", "generated/g.yaml", "notes.txt"):
        (cfg / name).write_text("x: 1\n")
    ws = load_workspace(tmp_path)
    assert [p.relative_to(ws.resolved_config_dir()).as_posix() for p in ws.configs()] == [
        "a.yaml",
        "b.yaml",
        "sub/c.yaml",
    ]


def test_root_not_serialised(tmp_path: Path) -> None:
    """The descriptor's directory is not part of its YAML form.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    assert "root" not in Workspace(root=tmp_path).model_dump()
