"""Convert legacy ``forces.txt`` runs to the float32 ``forces.npy`` the code now writes.

The text held six significant digits per component, which float32 carries exactly, so the
rewrite is lossless in the values that were actually stored while costing 2.1x less space.

Each run is converted to a temporary file, verified against the text it came from, and only
then swapped in; the original is removed only after that check passes. A run whose
``forces.npy`` already exists is skipped, so an interrupted migration can simply be rerun."""

import argparse
import logging
from pathlib import Path
from typing import Iterator

import numpy as np

from md_simulations.core.forces import FORCES_FILE, ForceWriter, read_forces

logger = logging.getLogger("md_simulations.prep.migrate_forces")

LEGACY_FILE = "forces.txt"


def parse_block(lines: list[str]) -> np.ndarray:
    """Parse a block of force lines into an array.

    :param lines: Whitespace-delimited lines, one per frame.
    :return: Array of shape (frames, components).
    :raises ValueError: If the lines do not all carry the same number of components."""

    rows = [np.fromstring(line, sep=" ") for line in lines]
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"ragged force lines: {sorted(widths)} components")

    return np.asarray(rows)


def iter_blocks(path: Path, chunk: int) -> Iterator[list[str]]:
    """Yield blocks of lines from a text file.

    :param path: The file.
    :param chunk: Lines per block.
    :return: Iterator over lists of lines."""

    with path.open() as handle:
        block: list[str] = []
        for line in handle:
            if line.strip():
                block.append(line)
            if len(block) == chunk:
                yield block
                block = []
        if block:
            yield block


def convert_run(run: Path, chunk: int = 2000, keep: bool = False) -> int:
    """Convert one run's ``forces.txt`` to ``forces.npy``, verify it, and drop the text.

    :param run: Directory holding the legacy file.
    :param chunk: Frames converted per block.
    :param keep: Leave the text file in place after a successful verification.
    :return: Bytes freed, or 0 if the run was skipped.
    :raises ValueError: If the written array does not reproduce the text."""

    legacy, target = run / LEGACY_FILE, run / FORCES_FILE
    if target.exists():
        logger.info(f"{run}: already converted, skipping")
        return 0

    tmp = run / (FORCES_FILE + ".partial")
    n_atoms, writer = None, None
    for block in iter_blocks(legacy, chunk):
        values = parse_block(block)
        if writer is None:
            n_atoms = values.shape[1] // 3
            writer = ForceWriter(tmp, n_atoms)
        writer.write(values.reshape(len(values), n_atoms, 3))
    if writer is None:
        logger.warning(f"{run}: {LEGACY_FILE} is empty, leaving it alone")
        return 0
    writer.close()

    verify(legacy, tmp, chunk)
    tmp.rename(target)
    freed = legacy.stat().st_size - target.stat().st_size
    if keep:
        logger.info(f"{run}: verified, keeping {LEGACY_FILE} (--keep)")
        return 0
    legacy.unlink()
    logger.info(
        f"{run}: {writer.n_frames:,} frames x {n_atoms} atoms, freed {freed / 2**30:.1f} GB"
    )

    return freed


def verify(legacy: Path, converted: Path, chunk: int) -> None:
    """Check that the converted array reproduces every value of the text file.

    :param legacy: The original text file.
    :param converted: The ``.npy`` written from it.
    :param chunk: Frames compared per block.
    :return: None.
    :raises ValueError: On any difference in frame count or values."""

    forces = read_forces(converted)
    seen = 0
    for block in iter_blocks(legacy, chunk):
        values = parse_block(block).reshape(len(block), *forces.shape[1:])
        written = np.asarray(forces[seen : seen + len(block)], dtype=np.float64)
        if written.shape != values.shape:
            raise ValueError(
                f"{converted}: shape {written.shape} at frame {seen}, expected {values.shape}"
            )
        if not np.array_equal(written, values.astype(np.float32).astype(np.float64)):
            raise ValueError(f"{converted}: values differ from {legacy} at frame {seen}")
        seen += len(block)
    if seen != len(forces):
        raise ValueError(f"{converted}: holds {len(forces)} frames, the text has {seen}")


def find_runs(root: Path) -> list[Path]:
    """Find every run directory still holding a legacy force file.

    :param root: Directory to search.
    :return: The run directories, largest legacy file first."""

    runs = [p.parent for p in root.rglob(LEGACY_FILE)]

    return sorted(runs, key=lambda r: (r / LEGACY_FILE).stat().st_size, reverse=True)


def main() -> None:
    """Command-line entry point.

    :return: None."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="data root to search for forces.txt")
    parser.add_argument("--chunk", type=int, default=2000, help="frames per block")
    parser.add_argument("--keep", action="store_true", help="do not delete the text files")
    parser.add_argument("--dry-run", action="store_true", help="list what would be converted")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runs = find_runs(args.root)
    total = sum((r / LEGACY_FILE).stat().st_size for r in runs)
    logger.info(f"{len(runs)} run(s), {total / 2**30:.1f} GB of {LEGACY_FILE}")

    if args.dry_run:
        for run in runs:
            size = (run / LEGACY_FILE).stat().st_size / 2**30
            logger.info(f"  {size:7.1f} GB  {run}")
        return

    freed = 0
    for k, run in enumerate(runs, start=1):
        logger.info(f"[{k}/{len(runs)}] {run}")
        freed += convert_run(run, chunk=args.chunk, keep=args.keep)
    logger.info(f"done, freed {freed / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
