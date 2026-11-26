"""The ``md-sim`` command line: one noun/verb entry point over every command.

``md-sim run`` and ``md-sim check-job`` are parsed here, ``md-sim config`` is the config
introspection group, and ``prep``, ``analysis`` and ``sampling`` pass through to the
standalone ``md-sim-*`` scripts (which stay installed as aliases). Importing this package
imports no third-party stack: every verb resolves its module only when it is chosen."""

from md_simulations.cli.main import main

__all__ = ["main"]
