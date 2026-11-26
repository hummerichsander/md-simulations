"""The JSON envelope shared by every ``md-sim`` command.

Success is ``{"ok": true, ...payload}`` and failure is
``{"ok": false, "error": {"type", "message"}}``, one object on stdout and nothing else.
Exit codes: 0 ok, 1 expected failure, 2 usage."""

import json
import sys
from typing import NoReturn

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


def emit(data: dict, *, json_mode: bool, human: str | None = None) -> None:
    """Print a command's result.

    :param data: The payload. In JSON mode it follows ``"ok": true``, and an ``"ok"`` key in it
        wins, so a command that reports a failed check can say ``"ok": false``.
    :param json_mode: Print one JSON object instead of the human text.
    :param human: The human rendering; ``None`` prints ``key: value`` lines, ``""`` prints nothing.
    :return: None."""
    if json_mode:
        print(json.dumps({"ok": True, **data}, sort_keys=False))
    elif human is None:
        for key, value in data.items():
            print(f"{key}: {value}")
    elif human:
        print(human)


def fail(
    exc: BaseException,
    *,
    json_mode: bool,
    code: int = EXIT_FAILURE,
    prog: str = "md-sim",
    error_type: str | None = None,
) -> NoReturn:
    """Report a failure and exit.

    :param exc: The error; its class name and message become the envelope's ``error``.
    :param json_mode: Print the envelope on stdout instead of ``prog: message`` on stderr.
    :param code: Process exit code (1 expected failure, 2 usage).
    :param prog: Command name for the human message.
    :param error_type: Override for ``error.type`` (default: the exception's class name).
    :return: Never returns."""
    message = str(exc)
    if json_mode:
        error = {"type": error_type or type(exc).__name__, "message": message}
        print(json.dumps({"ok": False, "error": error}, sort_keys=False))
    else:
        print(f"{prog}: {message}", file=sys.stderr)
    sys.exit(code)
