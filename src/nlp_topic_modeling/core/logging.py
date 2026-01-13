"""Centralized logging configuration for the NLP Topic Modeling package."""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Logging level (defaults to INFO)

    Returns:
        Configured logger instance
    """
    if level is None:
        level = logging.INFO

    logger = logging.getLogger(name)

    # Only configure if no handlers exist (avoid duplicate handlers)
    if not logger.handlers:
        logger.setLevel(level)

        # Create console handler with formatting
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger
