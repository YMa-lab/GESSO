import logging
import os
import sys
from contextlib import contextmanager
from typing import Literal

GESSO_LOGGER = "gesso"
INIT_LOGGER = "gesso.init"
COMPUTE_LOGGER = "gesso.compute"
GENESET_LOGGER = "gesso.compute.geneset"
DEFAULT_HANDLER_TAG = "_gesso_default_handler"
logging.getLogger(GESSO_LOGGER).addHandler(logging.NullHandler())


def bold_text(text: str) -> str:
    """Returns text in bold."""
    return "\033[1m" + text + "\033[0m"


def len_ignore_format(text: str) -> int:
    """Returns the length of a string without ANSI codes."""
    base_len = len(text)
    if "\033[1m" in text:
        base_len -= 4 * text.count("\033[1m")
    if "\033[0m" in text:
        base_len -= 4 * text.count("\033[0m")
    return base_len


def fill_ignore_format_single_line(
    text: str,
    width: int = 88,
    initial_indent: int = 0,
    subsequent_indent: int = 13,
) -> str:
    """Wraps a single line of text to a max width, ignoring ANSI codes."""
    if "\n" in text:
        raise ValueError("Text must not contain newline characters.")

    text_split = text.split(" ")
    newstr = " " * initial_indent
    line_length = initial_indent

    for word in text_split:
        if line_length + len_ignore_format(word) > width:
            newstr += "\n" + " " * subsequent_indent
            line_length = subsequent_indent
        newstr += word + " "
        line_length += len_ignore_format(word) + 1

    return newstr


def fill_ignore_format(
    text: str,
    width: int = 88,
    initial_indent: int = 0,
    subsequent_indent: int = 15,
) -> str:
    """Wraps text to a max width, ignoring ANSI codes. Preserves newlines."""
    return "\n".join(
        fill_ignore_format_single_line(line, width, initial_indent, subsequent_indent)
        for line in text.split("\n")
    )


class GessoFormatter(logging.Formatter):
    """Formatter that prefixes messages with bold 'GESSO (level): ' and wraps
    long lines. Used by the default handler installed via gesso.logging.enable().
    """

    def __init__(self, max_line_width: int = 88):
        super().__init__()
        self._max_line_width = max_line_width

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.WARNING:
            prefix = bold_text(f"GESSO ({record.levelname.lower()}): ")
        elif record.levelno == logging.DEBUG:
            prefix = bold_text("GESSO (debug): ")
        else:
            prefix = bold_text("GESSO (info): ")
        return fill_ignore_format(prefix + record.getMessage(), width=self._max_line_width)


def ensure_worker_handler(level: int = logging.INFO):
    """Called from joblib workers to ensure log records reach the parent
    process. Workers spawn fresh and do not inherit the parent's logger
    configuration, so we install a stderr handler on the 'gesso' logger if
    none is present. Idempotent across repeated calls; on each invocation
    the default handler's level is updated to reflect the parent's current
    config (loky workers persist across joblib calls, so a stale level
    would otherwise stick around).
    """
    logger = logging.getLogger(GESSO_LOGGER)
    default_handler = next(
        (h for h in logger.handlers if getattr(h, DEFAULT_HANDLER_TAG, False)),
        None,
    )
    if default_handler is None:
        default_handler = logging.StreamHandler(stream=sys.stderr)
        default_handler.setFormatter(GessoFormatter())
        setattr(default_handler, DEFAULT_HANDLER_TAG, True)
        logger.addHandler(default_handler)
    default_handler.setLevel(level)
    logger.setLevel(level)


@contextmanager
def silence_stdout():
    """Redirect stdout to os.devnull within the context. Kept for
    backwards compatibility; not used by the logging system."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        try:
            sys.stdout = devnull
            yield
        finally:
            sys.stdout = old_stdout
