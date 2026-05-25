from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Union

from ._src.console import (
    COMPUTE_LOGGER,
    DEFAULT_HANDLER_TAG,
    GENESET_LOGGER,
    GESSO_LOGGER,
    INIT_LOGGER,
    GessoFormatter,
)

__all__ = [
    "enable",
    "disable",
    "set_level",
    "silence_per_geneset",
    "unsilence_per_geneset",
    "add_file_handler",
    "remove_handler",
    "get_logger",
    "GESSO_LOGGER",
    "INIT_LOGGER",
    "COMPUTE_LOGGER",
    "GENESET_LOGGER",
]


LevelLike = Union[int, str]


def _to_level(level: LevelLike) -> int:
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if not isinstance(resolved, int):
            raise ValueError(f"unknown log level: {level!r}")
        return resolved
    return int(level)


def get_logger(name: str = GESSO_LOGGER) -> logging.Logger:
    """Return the underlying stdlib logger by name (escape hatch)."""
    return logging.getLogger(name)


def enable(
    level: LevelLike = logging.INFO,
    stream=None,
    fmt: logging.Formatter | None = None,
) -> logging.Handler:
    if stream is None:
        stream = sys.stderr
    logger = logging.getLogger(GESSO_LOGGER)
    for h in list(logger.handlers):
        if getattr(h, DEFAULT_HANDLER_TAG, False):
            logger.removeHandler(h)
    handler = logging.StreamHandler(stream=stream)
    handler.setLevel(_to_level(level))
    handler.setFormatter(fmt if fmt is not None else GessoFormatter())
    setattr(handler, DEFAULT_HANDLER_TAG, True)
    logger.addHandler(handler)
    logger.setLevel(_to_level(level))
    return handler


def disable() -> None:
    logger = logging.getLogger(GESSO_LOGGER)
    for h in list(logger.handlers):
        if getattr(h, DEFAULT_HANDLER_TAG, False):
            logger.removeHandler(h)
    logger.setLevel(logging.WARNING)


def set_level(level: LevelLike, logger: str = GESSO_LOGGER) -> None:
    logging.getLogger(logger).setLevel(_to_level(level))


def silence_per_geneset(silent: bool = True) -> None:
    level = logging.WARNING if silent else logging.NOTSET
    logging.getLogger(GENESET_LOGGER).setLevel(level)


def unsilence_per_geneset() -> None:
    silence_per_geneset(False)


def add_file_handler(
    path: Path | str,
    level: LevelLike = logging.DEBUG,
    logger: str = GESSO_LOGGER,
    fmt: logging.Formatter | None = None,
) -> logging.Handler:
    handler = logging.FileHandler(str(path))
    handler.setLevel(_to_level(level))
    if fmt is None:
        fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    handler.setFormatter(fmt)
    logging.getLogger(logger).addHandler(handler)
    return handler


def remove_handler(handler: logging.Handler, logger: str = GESSO_LOGGER) -> None:
    logging.getLogger(logger).removeHandler(handler)
