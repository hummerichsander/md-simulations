import json
from pathlib import Path

import pytest
import yaml

from md_simulations.cli import main
from md_simulations.preflight import Check, check_job, format_checks, parse_script
from md_simulations.workspace import Workspace

HEADER = """#!/bin/bash
#SBATCH --job-name=t
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --output=./slurm/output/%x-%j.out
#SBATCH --error=./slurm/output/%x-%j.err
set -euo pipefail
"""


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> None:
    """Keep the ambient workspace variables out of every test.

    :param monkeypatch: Pytest monkeypatch fixture.
    :return: None."""
    monkeypatch.delenv("MD_WORKSPACE", raising=False)
    monkeypatch.delenv("MD_DATA_ROOT", raising=False)


def _make(root: Path, *, site: dict | None = None, **workspace) -> Workspace:
    """Write a workspace, a site, a config and its input under ``root``.

    :param root: The lab directory.
    :param site: Site profile fields, or ``None`` for a workspace without a site.
    :param workspace: Extra descriptor fields.
    :return: The loaded workspace."""
    (root / "configs").mkdir(parents=True)
    (root / "data" / "pdbs").mkdir(parents=True)
    (root / "data" / "pdbs" / "x.pdb").write_text("END\n")
    (root / "slurm" / "output").mkdir(parents=True)
    (root / "configs" / "x.yaml").write_text(
        yaml.safe_dump(
            {"engine": "amber", "system": "x", "output_subdir": "out/x", "input_pdb": "pdbs/x.pdb"}
        )
    )
    fields = {"data_root": "data", **workspace}
    if site is not None:
        (root / "sites").mkdir()
        (root / "sites" / "s.yaml").write_text(yaml.safe_dump(site))
        fields["site"] = "s"
    (root / "md-workspace.yaml").write_text(yaml.safe_dump(fields))
    return Workspace(root=root, **fields)


def _script(root: Path, body: str, header: str = HEADER) -> Path:
    """Write a job script.

    :param root: The lab directory.
    :param body: Commands after the header.
    :param header: The ``#SBATCH`` header.
    :return: The script path."""
    path = root / "slurm" / "job.sh"
    path.write_text(header + body)
    return path


def _by_name(checks: list[Check]) -> dict[str, Check]:
    """Index checks by name.

    :param checks: The result of ``check_job``.
    :return: The checks by name."""
    return {c.name: c for c in checks}


def test_correct_script_all_ok(tmp_path: Path) -> None:
    """A correct script reports every check ok.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    venv = tmp_path / "activate"
    venv.write_text("")
    ws = _make(tmp_path, site={"partitions": ["a30", "a100"], "venv_activate": str(venv)})
    checks = check_job(_script(tmp_path, "md-sim run configs/x.yaml\n"), ws)
    assert [c.status for c in checks] == ["ok"] * 7, format_checks(checks)
    assert [c.name for c in checks] == [
        "directives",
        "config",
        "inputs",
        "partition",
        "gres",
        "output",
        "venv",
    ]


def test_partition_not_in_site(tmp_path: Path) -> None:
    """An unknown partition fails and names it and the allowed set.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["gpu", "cpu"]})
    check = _by_name(check_job(_script(tmp_path, "md-sim run configs/x.yaml\n"), ws))["partition"]
    assert check.status == "fail"
    assert "a30" in check.detail and "gpu, cpu" in check.detail


def test_partition_list_and_missing(tmp_path: Path) -> None:
    """Every member of a partition list must be known; a missing one warns.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30", "a100"]})
    ok = HEADER.replace("--partition=a30", "--partition=a30,a100")
    assert _by_name(check_job(_script(tmp_path, "", ok), ws))["partition"].status == "ok"
    bad = HEADER.replace("--partition=a30", "-p a30,zzz")
    assert _by_name(check_job(_script(tmp_path, "", bad), ws))["partition"].status == "fail"
    none = HEADER.replace("#SBATCH --partition=a30\n", "")
    assert _by_name(check_job(_script(tmp_path, "", none), ws))["partition"].status == "warn"


def test_unknown_partitions_skipped(tmp_path: Path) -> None:
    """A site with no partitions, or no site at all, skips the check instead of failing it.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": []})
    checks = _by_name(check_job(_script(tmp_path, "md-sim run configs/x.yaml\n"), ws))
    assert checks["partition"].status == "skipped"
    assert checks["venv"].status == "skipped"
    other = tmp_path / "other"
    other.mkdir()
    ws = _make(other / "lab")
    checks = _by_name(check_job(_script(other / "lab", "md-sim run configs/x.yaml\n"), ws))
    assert checks["partition"].status == "skipped"
    assert checks["venv"].status == "skipped"


def test_broken_site_fails(tmp_path: Path) -> None:
    """A site that is set but has no profile fails: it cannot be verified against.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"]})
    (tmp_path / "sites" / "s.yaml").unlink()
    checks = _by_name(check_job(_script(tmp_path, "md-sim run configs/x.yaml\n"), ws))
    assert checks["partition"].status == "fail"
    assert checks["venv"].status == "fail"


def test_venv_missing(tmp_path: Path) -> None:
    """A venv_activate that does not exist fails.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"], "venv_activate": ".venv/bin/activate"})
    check = _by_name(check_job(_script(tmp_path, "md-sim run configs/x.yaml\n"), ws))["venv"]
    assert check.status == "fail"
    assert str(tmp_path / ".venv" / "bin" / "activate") in check.detail


def test_gres(tmp_path: Path) -> None:
    """A GPU request is recognised in its spellings; none is skipped, never failed.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"]})
    for gres in ("--gres=gpu:1", "--gres=gpu:a30:2", "--gres=gpu", "--gpus=1"):
        header = HEADER.replace("--gres=gpu:1", gres)
        assert _by_name(check_job(_script(tmp_path, "", header), ws))["gres"].status == "ok", gres
    for gres in ("--gres=gpu:0", "--gres=shard:1", "--time=1"):
        header = HEADER.replace("--gres=gpu:1", gres)
        assert _by_name(check_job(_script(tmp_path, "", header), ws))["gres"].ok, gres
        assert _by_name(check_job(_script(tmp_path, "", header), ws))["gres"].status == "skipped"


def test_require_gpu(tmp_path: Path, monkeypatch, capsys) -> None:
    """--require-gpu fails a script with no GPU request and passes one that has it.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param capsys: Pytest capture fixture.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"]})
    cpu = HEADER.replace("#SBATCH --gres=gpu:1\n", "")
    gres = _by_name(check_job(_script(tmp_path, "", cpu), ws, require_gpu=True))["gres"]
    assert gres.status == "fail" and "--gres" in gres.detail
    assert _by_name(check_job(_script(tmp_path, "", HEADER), ws, require_gpu=True))["gres"].ok
    assert _by_name(check_job(_script(tmp_path, "", cpu), ws))["gres"].status == "skipped"
    monkeypatch.chdir(tmp_path)
    for header, code in ((cpu, 1), (HEADER, 0)):
        _script(tmp_path, "md-sim run configs/x.yaml\n", header)
        monkeypatch.setattr(
            "sys.argv", ["md-sim", "check-job", "slurm/job.sh", "--require-gpu", "--json"]
        )
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == code
        assert json.loads(capsys.readouterr().out)["ok"] is (code == 0)


def test_missing_config_path(tmp_path: Path) -> None:
    """A config path that does not exist fails and names it.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"]})
    checks = _by_name(check_job(_script(tmp_path, "md-sim run configs/nope.yaml\n"), ws))
    assert checks["config"].status == "fail"
    assert "configs/nope.yaml" in checks["config"].detail
    assert checks["inputs"].status == "skipped"


def test_invalid_config(tmp_path: Path) -> None:
    """A config that does not validate fails.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"]})
    (tmp_path / "configs" / "bad.yaml").write_text("engine: amber\nsystem: x\n")
    check = _by_name(check_job(_script(tmp_path, "md-sim run configs/bad.yaml\n"), ws))["config"]
    assert check.status == "fail" and "invalid config" in check.detail


@pytest.mark.parametrize(
    "body",
    [
        'md-sim run "$CONFIG"\n',
        'for n in 2 3; do\n  md-sim run "configs/ala${n}.yaml"\ndone\n',
        "python -m md_simulations.cli run ${args[@]}\n",
        "md-sim run configs/$(date +%F).yaml\n",
    ],
)
def test_dynamic_config_path(tmp_path: Path, body: str) -> None:
    """A config path built at run time is skipped, never passed.

    :param tmp_path: Pytest temporary directory.
    :param body: The commands after the header.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"]})
    checks = _by_name(check_job(_script(tmp_path, body), ws))
    assert checks["config"].status == "skipped"
    assert "not statically determinable" in checks["config"].detail
    assert checks["inputs"].status == "skipped"


def test_mixed_static_and_dynamic_warns(tmp_path: Path) -> None:
    """A static path is checked, and the dynamic one next to it keeps the check from reading ok.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"]})
    body = 'md-sim run configs/x.yaml\nmd-sim run "$OTHER"\n'
    assert _by_name(check_job(_script(tmp_path, body), ws))["config"].status == "warn"


def test_orchestration_script_not_crashed_on(tmp_path: Path) -> None:
    """A submit/job-mode script with heredocs and a ``--help`` probe is handled, not crashed on.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"]})
    body = """
if [[ -z ${CONFIG:-} ]]; then
    cmd=(sbatch --parsable "--job-name=${job}" "--export=ALL,CONFIG=${config}")
    "${cmd[@]}"
    exit 0
fi
# md-sim run configs/commented.yaml
if [[ ${RESUME:-0} == 1 ]] && ! python -m md_simulations.cli run --help 2>/dev/null | grep -q -- '--resume'; then
    echo "error: this node's md_simulations has no --resume." >&2
    python - >&2 <<'PY'
import md_simulations.cli as c
print("md-sim run nothing.yaml")
PY
    exit 1
fi
args=("$CONFIG")
python -m md_simulations.cli run "${args[@]}"
"""
    facts = parse_script(HEADER + body)
    assert [r.config for r in facts.runs] == [None]
    checks = _by_name(check_job(_script(tmp_path, body), ws))
    assert checks["config"].status == "skipped"


def test_sbatch_after_code(tmp_path: Path) -> None:
    """#SBATCH after the first command is ignored by slurm, so it warns and is not read.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["gpu"]})
    body = "md-sim run configs/x.yaml\n#SBATCH --partition=gpu\n"
    facts = parse_script(HEADER + body)
    assert len(facts.late_sbatch) == 1 and facts.directives["partition"] == "a30"
    checks = _by_name(check_job(_script(tmp_path, body), ws))
    assert checks["directives"].status == "warn"
    assert checks["partition"].status == "fail"


def test_absolute_input_path(tmp_path: Path) -> None:
    """An absolute input path resolves as-is, not under the data root.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"]})
    elsewhere = tmp_path / "elsewhere.pdb"
    elsewhere.write_text("END\n")
    cfg = yaml.safe_load((tmp_path / "configs" / "x.yaml").read_text())
    cfg["input_pdb"] = str(elsewhere)
    (tmp_path / "configs" / "x.yaml").write_text(yaml.safe_dump(cfg))
    assert _by_name(check_job(_script(tmp_path, "md-sim run configs/x.yaml\n"), ws))["inputs"].ok
    elsewhere.unlink()
    inputs = _by_name(check_job(_script(tmp_path, "md-sim run configs/x.yaml\n"), ws))["inputs"]
    assert inputs.status == "fail" and str(elsewhere) in inputs.detail


def test_missing_input_under_data_root(tmp_path: Path) -> None:
    """An input missing under the data root fails and names where it looked.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"]})
    (tmp_path / "data" / "pdbs" / "x.pdb").unlink()
    inputs = _by_name(check_job(_script(tmp_path, "md-sim run configs/x.yaml\n"), ws))["inputs"]
    assert inputs.status == "fail"
    assert str(tmp_path / "data" / "pdbs" / "x.pdb") in inputs.detail


def test_data_root_precedence(tmp_path: Path) -> None:
    """The run's --data-root beats the check's, which beats the script's export and the workspace.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"]})
    other = tmp_path / "other"
    (other / "pdbs").mkdir(parents=True)
    (other / "pdbs" / "x.pdb").write_text("END\n")
    run = "md-sim run configs/x.yaml"
    bad = _script(tmp_path, f"export MD_DATA_ROOT={tmp_path}/empty\n{run}\n")
    inputs = _by_name(check_job(bad, ws))["inputs"]
    assert inputs.status == "fail" and "empty" in inputs.detail
    assert _by_name(check_job(bad, ws, data_root=str(other)))["inputs"].status == "ok"
    flagged = _script(
        tmp_path, f"export MD_DATA_ROOT={tmp_path}/empty\n{run} --data-root {other}\n"
    )
    assert _by_name(check_job(flagged, ws))["inputs"].status == "ok"


def test_exported_root_differs_from_workspace_warns(tmp_path: Path) -> None:
    """A static export that points elsewhere than the workspace data root warns even if inputs exist.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"]})
    other = tmp_path / "other"
    (other / "pdbs").mkdir(parents=True)
    (other / "pdbs" / "x.pdb").write_text("END\n")
    script = _script(tmp_path, f"export MD_DATA_ROOT={other}\nmd-sim run configs/x.yaml\n")
    assert _by_name(check_job(script, ws))["inputs"].status == "warn"


def test_dynamic_data_root_export_skips_inputs(tmp_path: Path) -> None:
    """A data root exported from run-time values makes the inputs check skipped.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"]})
    body = 'export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"\nmd-sim run configs/x.yaml\n'
    assert _by_name(check_job(_script(tmp_path, body), ws))["inputs"].status == "skipped"


def test_output_directory(tmp_path: Path) -> None:
    """A missing --output directory warns (slurm does not create it); an uncreatable one fails.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"]})
    (tmp_path / "slurm" / "output").rmdir()
    check = _by_name(check_job(_script(tmp_path, ""), ws))["output"]
    assert check.status == "warn" and "mkdir" in check.detail
    blocked = HEADER.replace("./slurm/output/%x-%j", "/proc/nope/%x-%j")
    assert _by_name(check_job(_script(tmp_path, "", blocked), ws))["output"].status == "fail"
    dynamic = HEADER.replace("./slurm/output/%x-%j", "$HOME/out/%x-%j")
    assert _by_name(check_job(_script(tmp_path, "", dynamic), ws))["output"].status == "skipped"
    # Nothing was created by checking.
    assert not (tmp_path / "slurm" / "output").exists()


def test_chdir_changes_the_base(tmp_path: Path) -> None:
    """A static --chdir is where relative config paths resolve; a dynamic one skips them.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"]})
    (tmp_path / "sub").mkdir()
    header = HEADER.replace("#SBATCH --job-name=t", "#SBATCH --job-name=t\n#SBATCH --chdir=sub")
    body = "md-sim run ../configs/x.yaml\n"
    assert _by_name(check_job(_script(tmp_path, body, header), ws))["config"].status == "ok"
    dyn = HEADER.replace("#SBATCH --job-name=t", "#SBATCH --job-name=t\n#SBATCH -D $HOME/x")
    assert _by_name(check_job(_script(tmp_path, body, dyn), ws))["config"].status == "skipped"


def test_parse_directive_forms() -> None:
    """Long, short, joined and quoted options all land under their long name.

    :return: None."""
    text = (
        "#!/bin/bash\n"
        "#SBATCH -p a30 -J job\n"
        "#SBATCH --time 1:00:00   # comment\n"
        '#SBATCH --output="x/%j.out"\n'
        "#SBATCH --exclusive\n"
        "#SBATCH -t2\n"
        "\n"
        "# a comment\n"
        "echo hi\n"
    )
    d = parse_script(text).directives
    assert d["partition"] == "a30" and d["job-name"] == "job"
    assert d["time"] == "2" and d["output"] == "x/%j.out" and d["exclusive"] == ""


def test_parse_exports_and_run_options() -> None:
    """``md-sim run`` options are read; ``env`` and continuations do not hide the command.

    :return: None."""
    text = "FOO=1 md-sim run \\\n  configs/a.yaml --resume --data-root=/d\nexport MD_DATA_ROOT=/r\n"
    facts = parse_script(text)
    assert [(r.config, r.data_root) for r in facts.runs] == [("configs/a.yaml", "/d")]
    assert facts.data_root_export == "/r"


def test_cli_exit_codes_and_json(tmp_path: Path, monkeypatch, capsys) -> None:
    """check-job exits 0/1/2, and --json prints ok and one entry per check.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param capsys: Pytest capture fixture.
    :return: None."""
    ws = _make(tmp_path, site={"partitions": ["a30"]})
    good = _script(tmp_path, "md-sim run configs/x.yaml\n")
    monkeypatch.chdir(tmp_path)

    def run(*argv: str) -> int:
        monkeypatch.setattr("sys.argv", ["md-sim", "check-job", *argv])
        with pytest.raises(SystemExit) as exc:
            main()
        return int(exc.value.code)

    assert run(str(good)) == 0
    assert "OK" in capsys.readouterr().out
    assert run(str(good), "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert {"name", "ok", "detail"} <= set(payload["checks"][0])
    assert [c["name"] for c in payload["checks"]][:2] == ["directives", "config"]

    bad = tmp_path / "slurm" / "bad.sh"
    bad.write_text(HEADER + "md-sim run configs/nope.yaml\n")
    assert run(str(bad), "--json") == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert run(str(tmp_path / "missing.sh")) == 2
    # From elsewhere the workspace is found from the script's own directory.
    monkeypatch.chdir(Path("/"))
    assert run(str(good)) == 0
    assert ws.root == tmp_path
