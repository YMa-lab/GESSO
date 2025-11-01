import logging
import sys
from contextlib import contextmanager
import os
from typing import Literal
from pathlib import Path


def make_default_logger() -> logging.Logger:
    default_logger_level = logging.INFO
    logger = logging.Logger(name="Default GESSO Logger")
    logger.setLevel(default_logger_level)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(default_logger_level)
    logger.addHandler(stream_handler)
    return logger


class _PrintOptions:
    """Class for setting and tracking options for printing and logging."""

    def __init__(self):
        """Initializes the a _PrintOptions object with default settings."""
        self._logger = make_default_logger()
        self._muted = False

        self._n_decimals = 5
        self._max_line_width = 88  # consistent with Python Black

    def _log_info(self, msg: str):
        if not self._muted:
            self._logger.info(msg)

    def _log_debug(self, msg: str):
        if not self._muted:
            self._logger.debug(msg)

    def reset_logger(self, logger: logging.Logger | None = None):
        """Sets a new logger.

        Parameters
        ----------
        logger : logging.Logger | None
            Default : None. If None, resets the logger to the default.
        """
        if logger is None:
            self._logger = make_default_logger()
        else:
            self._logger = logger

    def add_log_file(self, path: Path | str, level: int = logging.DEBUG):
        """Adds a file handler to the logger.

        Parameters
        ----------
        path : Path | str
            Path to the .log or .txt file.

        level : int
            Default: logging.DEBUG.
        """
        file_handler = logging.FileHandler(str(path))
        file_handler.setLevel(level)
        self._logger.addHandler(file_handler)

    def mute(self):
        """Mutes. No messages will be printed."""
        self._muted = True
        self.reset_logger()
        self._logger.setLevel(logging.CRITICAL)

    def unmute(self):
        """Unmutes. Messages will be printed."""
        self._muted = False
        self.reset_logger()
        self._logger.setLevel(logging.INFO)


print_options = _PrintOptions()


def bold_text(text):
    """Returns text in bold."""
    return "\033[1m" + text + "\033[0m"


def print_wrapped(
    text: str, level: Literal["INFO", "DEBUG"] = "INFO", verbose: bool = True
):
    """Logs text.

    Parameters
    ----------
    text : str.

    level : Literal['INFO', 'DEBUG'].
        Default: 'INFO'.

    verbose : bool
        Default: True. If False, does not print the message.
    """
    base_message = text

    if not verbose:
        return

    if level == "DEBUG":
        base_message = bold_text("GESSO (debug): ") + base_message
        print_options._log_debug(
            fill_ignore_format(base_message, width=print_options._max_line_width)
        )
    elif level == "INFO":
        base_message = bold_text("GESSO (info): ") + base_message
        print_options._log_info(
            fill_ignore_format(base_message, width=print_options._max_line_width)
        )


def fill_ignore_format_single_line(
    text: str,
    width: int = print_options._max_line_width,
    initial_indent: int = 0,
    subsequent_indent: int = 13,
) -> str:
    """Wraps text to a max width of TOSTR_MAX_WIDTH. Text must NOT
    contain any newline characters.

    Parameters
    ----------
    text : str
        The text to be wrapped.

    width : int
        Default: print_options._max_line_width. The maximum width of the wrapped text.

    initial_indent : int
        Default: 0. The number of spaces to indent the first line.

    subsequent_indent : int
        Default: 6. The number of spaces to indent subsequent lines.

    Returns
    -------
    str
    """
    if "\n" in text:
        raise ValueError("Text must not contain newline characters.")

    text_split = text.split(" ")
    newstr = ""

    newstr += " " * initial_indent
    line_length = initial_indent

    for word in text_split:
        if line_length + len_ignore_format(word) > width:
            newstr += "\n"
            newstr += " " * subsequent_indent
            line_length = subsequent_indent
        newstr += word + " "
        line_length += len_ignore_format(word) + 1

    return newstr


def len_ignore_format(text: str) -> int:
    """Returns the length of a string without ANSI codes.

    Parameters
    ----------
    text : str

    Returns
    -------
    int
        The length without ANSI codes.
    """
    base_len = len(text)
    if "\033[1m" in text:
        count = text.count("\033[1m")
        base_len -= 4 * count
    if "\033[0m" in text:
        count = text.count("\033[0m")
        base_len -= 4 * count
    return base_len


def fill_ignore_format(
    text: str,
    width: int = print_options._max_line_width,
    initial_indent: int = 0,
    subsequent_indent: int = 15,
) -> str:
    """Wraps text to a max width of TOSTR_MAX_WIDTH.

    Parameters
    ----------
    test : str
        The text to be wrapped.

    width : int
        Default: print_options._max_line_width. The maximum width of the wrapped text.

    initial_indent : int
        Default: 0. The number of spaces to indent the first line.

    subsequent_indent : int
        Default: 6. The number of spaces to indent subsequent lines.

    Returns
    -------
    str
    """
    return "\n".join(
        [
            fill_ignore_format_single_line(
                line, width, initial_indent, subsequent_indent
            )
            for line in text.split("\n")
        ]
    )


@contextmanager
def silence_stdout():
    # Redirect stdout to /dev/null or NUL (Windows)
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        try:
            sys.stdout = devnull
            yield
        finally:
            sys.stdout = old_stdout
