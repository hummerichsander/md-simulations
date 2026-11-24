#!/usr/bin/env python3
"""Write individual frames of a trajectory out as PDB files, one per frame.

Turns selected frames of an existing trajectory into starting structures for new simulations,
which the engines take as ``input_pdb``. Provenance (source trajectory, topology, frame index)
goes into REMARK lines of each PDB, so a run's ``initial_structure.pdb`` can always be traced back
to the frame it came from.

Note that a PDB stores coordinates to three decimals in Angstrom, i.e. 0.001 A = 1e-4 nm. That is
far below the scale of any structural observable, but it does mean the written frame is not
bit-identical to the source frame.

Usage::

    md-sim-extract-frames --trajectory traj.trr --topology top.pdb \\
        --frames 0 1 2 3 4 5 -o data/pdbs/cln-rep
"""

import argparse
from pathlib import Path

import mdtraj as md


def extract_frames(
    trajectory: str | Path,
    topology: str | Path,
    frames: list[int],
    out_prefix: str | Path,
) -> list[Path]:
    """Write each of *frames* to ``<out_prefix><index>.pdb``.

    :param trajectory: Trajectory file, any format mdtraj reads.
    :param topology: Topology file the trajectory is read against.
    :param frames: Zero-based frame indices.
    :param out_prefix: Output path prefix; the frame's position in *frames* is appended.
    :return: The paths written, in the order of *frames*.
    :raises ValueError: If a requested frame is out of range."""
    traj = md.load(str(trajectory), top=str(topology))
    out_of_range = [f for f in frames if f >= traj.n_frames or f < -traj.n_frames]
    if out_of_range:
        raise ValueError(
            f"frames {out_of_range} are out of range for {trajectory} ({traj.n_frames} frames)"
        )

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    written = []
    for position, frame in enumerate(frames):
        path = out_prefix.with_name(f"{out_prefix.name}{position}.pdb")
        traj[frame].save_pdb(str(path))
        remarks = (
            f"REMARK   1 FRAME {frame} OF {Path(trajectory).name}\n"
            f"REMARK   1 TOPOLOGY {Path(topology).name}\n"
            f"REMARK   1 WRITTEN BY md-sim-extract-frames\n"
        )
        path.write_text(remarks + path.read_text())
        written.append(path)
    return written


def main() -> None:
    """CLI entry point.

    :return: None."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--trajectory", required=True, help="trajectory file")
    parser.add_argument("--topology", required=True, help="topology the trajectory is read against")
    parser.add_argument(
        "--frames", required=True, type=int, nargs="+", help="zero-based frame indices"
    )
    parser.add_argument(
        "-o", "--out-prefix", required=True, help="output prefix; 0, 1, ... .pdb are appended"
    )
    args = parser.parse_args()

    for path in extract_frames(args.trajectory, args.topology, args.frames, args.out_prefix):
        print(path)


if __name__ == "__main__":
    main()
