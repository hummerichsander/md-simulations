import sys
import time
import logging
from pathlib import Path


class RelativeTimeFormatter(logging.Formatter):
    """Log formatter that prefixes each message with HH:MM:SS elapsed time.

    The clock starts when the formatter is instantiated (i.e. when
    :func:`setup_logger` is first called), so all timestamps are relative
    to the start of the simulation process.
    """

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        super().__init__(fmt, datefmt)
        self.start_time = time.time()

    def format(self, record: logging.LogRecord) -> str:
        elapsed = time.time() - self.start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        record.relativeTime = f"{h:02d}:{m:02d}:{s:02d}"
        return super().format(record)


def setup_logger(
    name: str,
    output_dir: Path,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """Create (or retrieve) a named logger with a relative-time console handler.

    Calling this function a second time with the same *name* is safe: existing
    handlers are cleared before the new one is attached, so log lines are never
    duplicated.

    :param name: Logger name, e.g. ``"OpenMM_MD"`` or ``"OpenAWSEM_MD"``.
    :param output_dir: Output directory (accepted for API symmetry; not used
        for a file handler here, but callers may extend this).
    :param log_level: Logging level (default :data:`logging.INFO`).
    :return: Configured :class:`logging.Logger`."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers.clear()

    fmt = RelativeTimeFormatter("[%(relativeTime)s] %(levelname)s: %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger
