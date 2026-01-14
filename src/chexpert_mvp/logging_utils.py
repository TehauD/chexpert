from __future__ import annotations
import logging
import sys
from rich.console import Console
from rich.logging import RichHandler

console = Console()

def setup_logger(name: str = "chexpert") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Rich pretty logging
    rich_handler = RichHandler(console=console, rich_tracebacks=True, show_time=True, show_level=True)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    # stderr warnings/errors
    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(rich_handler)
    logger.addHandler(err_handler)
    logger.propagate = False
    return logger
