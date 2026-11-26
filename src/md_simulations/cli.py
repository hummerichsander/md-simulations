import argparse
import logging
import sys

from md_simulations.config import load_config, resolve_data_root
from md_simulations.core import setup_logger
from md_simulations.engines import build_engine


def _cmd_run(args: argparse.Namespace) -> None:
    """Load a config, resolve the data root, and run the simulation.

    :param args: Parsed CLI arguments.
    :return: None."""
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
