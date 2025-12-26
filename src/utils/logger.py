"""
Logging Utilities
Configures logging for the fish health monitoring system
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    console_output: bool = True
) -> None:
    """
    Setup logging configuration

    Args:
        log_file: Path to log file
        level: Logging level
        console_output: Whether to output to console
    """
    # Remove default logger
    logger.remove()

    # Add console handler if enabled
    if console_output:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level=level,
            colorize=True
        )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation="10 MB",  # Rotate when file reaches 10 MB
            retention="1 week",  # Keep logs for 1 week
            compression="zip"  # Compress rotated logs
        )

    logger.info(f"Logging initialized at {level} level")


def get_logger(name: str):
    """
    Get logger instance

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logger.bind(name=name)
