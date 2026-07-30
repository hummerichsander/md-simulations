"""Bundled custom OpenMM force-field XML files and a name resolver."""

from pathlib import Path

_FF_DIR = Path(__file__).parent


def resolve_forcefield_files(files: list[str]) -> list[str]:
    """Resolve force-field entries, mapping bundled names to their real paths.

    OpenMM's :class:`~openmm.app.ForceField` resolves built-in names (e.g.
    ``amber14-all.xml``) against its own search path. Any entry that instead
    matches a file bundled in this package's ``forcefields/`` directory (e.g.
    ``charmm36_hexadecane.xml``) is rewritten to its absolute path; everything
    else passes through unchanged.

    :param files: Force-field entries from a config.
    :return: The resolved list, safe to pass to ``ForceField(*files)``."""
    resolved = []
    for f in files:
        bundled = _FF_DIR / f
        resolved.append(str(bundled) if bundled.is_file() else f)
    return resolved
