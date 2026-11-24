from pathlib import Path

import numpy as np
import pytest

from md_simulations.core.forces import ForceWriter, count_frames, read_forces


@pytest.fixture
def reference() -> np.ndarray:
    """Forces spanning the magnitude range a real run produces.

    :return: Array of shape (37, 11, 3)."""
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 500.0, (37, 11, 3)).astype(np.float32)


def test_header_stays_valid_while_writing(tmp_path: Path, reference: np.ndarray) -> None:
    """Every append leaves a readable array holding exactly the frames written so far.

    :param tmp_path: Pytest temporary directory.
    :param reference: Reference forces.
    :return: None."""
    path = tmp_path / "forces.npy"
    writer = ForceWriter(path, 11)
    for k, frame in enumerate(reference, start=1):
        writer.write(frame)
        assert np.array_equal(read_forces(path), reference[:k])
        assert count_frames(path) == k
    writer.close()


def test_blocks_and_single_frames_agree(tmp_path: Path, reference: np.ndarray) -> None:
    """Appending a block gives the same file as appending its frames one by one.

    :param tmp_path: Pytest temporary directory.
    :param reference: Reference forces.
    :return: None."""
    block_path, frame_path = tmp_path / "block.npy", tmp_path / "frame.npy"
    writer = ForceWriter(block_path, 11)
    writer.write(reference)
    writer.close()
    writer = ForceWriter(frame_path, 11)
    for frame in reference:
        writer.write(frame)
    writer.close()

    assert block_path.read_bytes() == frame_path.read_bytes()
    assert np.array_equal(np.load(block_path), reference)


def test_reopen_appends(tmp_path: Path, reference: np.ndarray) -> None:
    """Reopening with append continues the file rather than restarting it.

    :param tmp_path: Pytest temporary directory.
    :param reference: Reference forces.
    :return: None."""
    path = tmp_path / "forces.npy"
    writer = ForceWriter(path, 11)
    writer.write(reference[:20])
    writer.close()

    writer = ForceWriter(path, 11, append=True)
    assert writer.n_frames == 20
    writer.write(reference[20:])
    writer.close()

    assert np.array_equal(np.load(path), reference)


def test_truncate_cuts_back_to_a_frame_boundary(tmp_path: Path, reference: np.ndarray) -> None:
    """Truncating drops the tail and lets the file be rewritten from there.

    :param tmp_path: Pytest temporary directory.
    :param reference: Reference forces.
    :return: None."""
    path = tmp_path / "forces.npy"
    writer = ForceWriter(path, 11)
    writer.write(reference)
    writer.truncate(12)
    writer.write(reference[12:25])
    writer.close()

    assert np.array_equal(np.load(path), reference[:25])


def test_truncate_past_the_end_is_rejected(tmp_path: Path, reference: np.ndarray) -> None:
    """A resume asking for more frames than the file holds fails loudly.

    :param tmp_path: Pytest temporary directory.
    :param reference: Reference forces.
    :return: None."""
    path = tmp_path / "forces.npy"
    writer = ForceWriter(path, 11)
    writer.write(reference[:5])
    with pytest.raises(ValueError, match="holds 5 frames"):
        writer.truncate(6)
    writer.close()


def test_a_partial_trailing_frame_is_ignored(tmp_path: Path, reference: np.ndarray) -> None:
    """A crash mid-write leaves bytes past the header count; they must not be read.

    :param tmp_path: Pytest temporary directory.
    :param reference: Reference forces.
    :return: None."""
    path = tmp_path / "forces.npy"
    writer = ForceWriter(path, 11)
    writer.write(reference)
    writer.close()
    path.write_bytes(path.read_bytes() + b"\x00" * 7)

    assert np.array_equal(np.load(path), reference)


def test_wrong_site_count_on_append_is_rejected(tmp_path: Path, reference: np.ndarray) -> None:
    """Appending with a different topology would silently reshape the run's forces.

    :param tmp_path: Pytest temporary directory.
    :param reference: Reference forces.
    :return: None."""
    path = tmp_path / "forces.npy"
    writer = ForceWriter(path, 11)
    writer.write(reference)
    writer.close()
    with pytest.raises(ValueError, match="expected"):
        ForceWriter(path, 12, append=True)
