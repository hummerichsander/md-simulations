import argparse
import json
import logging
import sys
from pathlib import Path

from md_simulations.config import load_config, resolve_data_root
from md_simulations.preflight import check_job, format_checks
from md_simulations.workspace import WorkspaceError, load_workspace


def _cmd_run(args: argparse.Namespace) -> None:
    """Load a config, resolve the data root, and run the simulation.

    :param args: Parsed CLI arguments.
    :return: None."""
    # Deferred: core and engines need the ``simulate`` extra, but ``md-sim check-job``
    # and the other config-only commands must run without it.
    try:
        from md_simulations.core import setup_logger
        from md_simulations.engines import build_engine
    except ModuleNotFoundError as exc:  # message is core/engines ``EXTRA_HINT`` on a slim install
        sys.exit(f"md-sim run: {exc}")

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
        logger.error("--resume needs a config with a replica_exchange block.")
        sys.exit(1)

    engine = build_engine(config, data_root, logger, resume=args.resume)
    try:
        engine.run()
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        sys.exit(1)


def _cmd_check_job(args: argparse.Namespace) -> None:
    """Preflight an sbatch script and exit 0 if no check failed, 1 if one did, 2 on a usage error.

    The workspace is found from the cwd (or ``$MD_WORKSPACE``), then from the script's directory.

    :param args: Parsed CLI arguments.
    :return: None."""
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
        if args.json:
            print(json.dumps({"ok": False, "error": str(err), "checks": []}))
        else:
            print(f"md-sim check-job: {err}", file=sys.stderr)
        sys.exit(2)
    ok = all(c.ok for c in checks)
    if args.json:
        payload = {
            "ok": ok,
            "checks": [
                {"name": c.name, "ok": c.ok, "status": c.status, "detail": c.detail} for c in checks
            ],
        }
        print(json.dumps(payload))
    else:
        print(format_checks(checks))
    sys.exit(0 if ok else 1)


def main() -> None:
    """Entry point for the ``md-sim`` command."""
    parser = argparse.ArgumentParser(
        prog="md-sim",
        description="Run config-driven MD simulations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a simulation from a YAML config.")
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
    p_run.set_defaults(func=_cmd_run)

    p_check = sub.add_parser(
        "check-job",
        help="Check an sbatch script against the workspace, site profile and configs.",
    )
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
