"""Lazy dispatch of ``md-sim <group> <verb>`` to the standalone ``md-sim-<verb>`` mains.

The table holds ``module:function`` strings only; a module is imported when its verb is
chosen, so ``md-sim --help`` and a group listing never load torch, openmm or mdtraj."""

import importlib
import sys

GROUPS: dict[str, dict[str, str]] = {
    "prep": {
        "extract-frames": "md_simulations.prep.extract_frames:main",
        "cgschnet-map": "md_simulations.prep.cgschnet_map:main",
        "cgschnet-embeddings": "md_simulations.prep.cgschnet_embeddings:main",
        "cgschnet-reexport": "md_simulations.prep.cgschnet_reexport:main",
        "cgschnet-dataset": "md_simulations.prep.cgschnet_dataset:main",
        "migrate-forces": "md_simulations.prep.migrate_forces:main",
        "cgschnet-finetune": "md_simulations.prep.cgschnet_finetune:main",
        "cgschnet-priors": "md_simulations.prep.cgschnet_priors:main",
    },
    "analysis": {
        "tica": "md_simulations.analysis.tica:main",
        "cvs": "md_simulations.analysis.cvs:main",
        "basins": "md_simulations.analysis.basins:main",
    },
    "sampling": {
        "umbrella": "md_simulations.sampling.umbrella:main",
        "equilibrate-hbonds": "md_simulations.sampling.equilibrate_h_bonds:main",
        "resample-hbonds": "md_simulations.sampling.resample_h_bonds:main",
    },
}

GROUP_SUMMARIES: dict[str, str] = {
    "prep": "Prepare inputs: frames, CGSchNet maps, embeddings, datasets, priors.",
    "analysis": "Analyse trajectories: TICA, collective variables, basins.",
    "sampling": "Enhanced-sampling tools: umbrella grids and constrained-bond restarts.",
}

SUMMARIES: dict[str, str] = {
    "extract-frames": "Write the frames of a trajectory out as PDB files.",
    "cgschnet-map": "Build the CG structure and embeddings a CGSchNet run needs from a PDB.",
    "cgschnet-embeddings": "Recover CGSchNet bead embeddings from a distributed artifact.",
    "cgschnet-reexport": "Re-export a legacy CGSchNet/mlcg checkpoint for the current stack.",
    "cgschnet-dataset": "Build a CGSchNet force-matching dataset from an all-atom run.",
    "migrate-forces": "Convert legacy forces.txt runs to forces.npy.",
    "cgschnet-finetune": "Fine-tuning helpers for the CGSchNet transferable checkpoint.",
    "cgschnet-priors": "Fit the priors of a CGSchNet trained from scratch.",
    "tica": "Time-lagged independent component analysis of a trajectory.",
    "cvs": "Collective-variable free-energy surfaces of a trajectory.",
    "basins": "Two-state basin populations, kinetics and convergence.",
    "umbrella": "Grid umbrella sampling of the alanine-dipeptide (phi, psi) plane.",
    "equilibrate-hbonds": "Per-frame equilibration of constrained bonds.",
    "resample-hbonds": "Give a trajectory's constrained bond lengths their thermal width.",
}


def alias(verb: str) -> str:
    """Return the standalone command name for a verb.

    :param verb: A verb of any group, e.g. ``"cgschnet-priors"``.
    :return: The ``md-sim-<verb>`` entry-point name, also the ``prog`` its usage line shows."""
    return f"md-sim-{verb}"


def dispatch(group: str, verb: str, argv: list[str]) -> int:
    """Run a verb's ``main()`` as its standalone command would.

    ``sys.argv`` is set to the alias name plus ``argv`` untouched, so the script's own argparse
    shows ``md-sim-<verb>`` in its usage and sees exactly the arguments the alias would.

    :param group: A key of :data:`GROUPS`.
    :param verb: A verb of that group.
    :param argv: Every argument after the verb, passed through unparsed.
    :return: The process exit code (0 when ``main()`` returns or exits with ``None``)."""
    module_name, _, func_name = GROUPS[group][verb].partition(":")
    saved = sys.argv
    sys.argv = [alias(verb), *argv]
    try:
        result = getattr(importlib.import_module(module_name), func_name)()
    except SystemExit as exc:
        return exit_code(exc.code)
    finally:
        sys.argv = saved
    return result if isinstance(result, int) else 0


def exit_code(code: object) -> int:
    """Translate a ``SystemExit`` code as the interpreter would.

    :param code: ``SystemExit.code``.
    :return: ``0`` for ``None``, the int itself, and ``1`` after printing any other value
        to stderr."""
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1
