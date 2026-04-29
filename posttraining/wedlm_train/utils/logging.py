# coding=utf-8
"""Logging configuration utility."""

import logging
import sys


def setup_logger(name: str = "wedlm_train", level: int = logging.INFO) -> logging.Logger:
    """Set up a console logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    return logger
