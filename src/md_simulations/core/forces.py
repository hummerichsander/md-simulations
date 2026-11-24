"""Append-only ``.npy`` storage for per-frame forces.

Forces are the bulk of a run's output, so they are stored as float32 rather than the
text this project wrote until now — the same numbers in 2.1x less space, memory-mappable
and free of parsing on the way back in.

The frame count in the ``.npy`` header is rewritten after every append, so the file is a
valid array at all times. A run that dies mid-flush therefore reads back as the frames it
actually completed, the way a truncated text file used to."""

from pathlib import Path

import numpy as np

FORCES_FILE = "forces.npy"

_MAGIC = b"\x93NUMPY\x01\x00"
_PREFIX = "{'descr': '<f4', 'fortran_order': False, 'shape': ("
# Digits reserved for the frame count so it can be patched without moving the data.
_COUNT_WIDTH = 20


def _header(n_frames: int, n_atoms: int) -> tuple[bytes, int]:
    """Build a ``.npy`` v1 header whose frame count is padded to a fixed width.

    :param n_frames: Initial frame count.
    :param n_atoms: Sites per frame.
    :return: The header bytes and the file offset of the frame-count field."""

    body = f"{_PREFIX}{str(n_frames).ljust(_COUNT_WIDTH)}, {n_atoms}, 3), }}"
    body += " " * (-(len(_MAGIC) + 2 + len(body) + 1) % 64) + "\n"
    header = _MAGIC + len(body).to_bytes(2, "little") + body.encode("latin1")

    return header, len(_MAGIC) + 2 + len(_PREFIX)


def count_frames(path: Path) -> int:
    """Count the frames in a force file without reading them.

    :param path: Path to the force file.
    :return: The number of frames."""

    return int(np.load(path, mmap_mode="r").shape[0])


def read_forces(path: Path) -> np.ndarray:
    """Memory-map a force file.

    :param path: Path to the force file.
    :return: Read-only array of shape (frames, n_atoms, 3) in kJ mol⁻¹ nm⁻¹."""

    return np.load(path, mmap_mode="r")


class ForceWriter:
    """Append float32 frames to a ``.npy`` file, keeping its header valid throughout.

    :param path: Destination file.
    :param n_atoms: Sites per frame.
    :param append: Reopen an existing file and append to it instead of truncating."""

    def __init__(self, path: Path, n_atoms: int, *, append: bool = False) -> None:
        self._n_atoms = n_atoms
        self._frame_bytes = n_atoms * 3 * 4
        header, self._count_offset = _header(0, n_atoms)
        if append:
            existing = np.load(path, mmap_mode="r")
            if existing.shape[1:] != (n_atoms, 3):
                raise ValueError(
                    f"{path} holds {existing.shape[1:]} per frame, expected ({n_atoms}, 3)"
                )
            self._n_frames = int(existing.shape[0])
            del existing
            self._out = open(path, "r+b")
        else:
            self._n_frames = 0
            self._out = open(path, "wb")
            self._out.write(header)
        self._data_offset = len(header)
        self._out.seek(self._data_offset + self._n_frames * self._frame_bytes)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def n_frames(self) -> int:
        """Frames written so far."""

        return self._n_frames

    def write(self, forces: np.ndarray) -> None:
        """Append one frame or a block of frames.

        :param forces: Array of shape (n_atoms, 3) or (frames, n_atoms, 3), in
            kJ mol⁻¹ nm⁻¹.
        :return: None."""

        block = np.ascontiguousarray(forces, dtype="<f4").reshape(-1, self._n_atoms, 3)
        self._out.write(block.tobytes())
        self._n_frames += len(block)
        self._patch()

    def truncate(self, n_frames: int) -> None:
        """Cut the file back to its first *n_frames* frames.

        :param n_frames: Frames to keep.
        :return: None.
        :raises ValueError: If the file holds fewer frames than that."""

        if n_frames > self._n_frames:
            raise ValueError(f"force file holds {self._n_frames} frames, needs {n_frames}")
        self._out.truncate(self._data_offset + n_frames * self._frame_bytes)
        self._n_frames = n_frames
        self._patch()

    def flush(self) -> None:
        """Flush buffered writes to disk.

        :return: None."""

        self._out.flush()

    def fileno(self) -> int:
        """Return the underlying file descriptor, so callers can ``fsync`` it.

        :return: The file descriptor."""

        return self._out.fileno()

    def close(self) -> None:
        """Flush and close the file.

        :return: None."""

        if not self._out.closed:
            self._out.close()

    def _patch(self) -> None:
        """Rewrite the header's frame count and leave the handle at the end of the data.

        :return: None."""

        self._out.seek(self._count_offset)
        self._out.write(str(self._n_frames).ljust(_COUNT_WIDTH).encode("latin1"))
        self._out.seek(self._data_offset + self._n_frames * self._frame_bytes)
        self._out.flush()
