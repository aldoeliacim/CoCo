"""
Simplified logging configuration for CoCo project.
Just logs everything to main.log with module names for easy filtering.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "main", level: str = None) -> logging.Logger:
    """
    Simple logger setup - everything goes to main.log with module names.

    Args:
        name: Logger/module name
        level: Log level ("DEBUG", "INFO", "WARNING", "ERROR")

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Don't setup if already configured
    if logger.handlers:
        return logger

    # Prevent propagation to avoid duplicate logs from parent loggers
    logger.propagate = False

    # Clear any existing handlers to be safe
    logger.handlers.clear()

    # Get config from settings with fallback
    try:
        from config.settings import LOGGING_CONFIG
        config = LOGGING_CONFIG
    except ImportError:
        config = {"level": "INFO", "console_output": True}

    # Determine log level
    if level:
        log_level = level
    else:
        config_level = config.get("level", "INFO")
        log_level = config_level() if callable(config_level) else config_level

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Console handler
    if config.get("console_output", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        console_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        console_formatter = logging.Formatter(console_format, datefmt="%H:%M:%S")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Single main.log file handler
    try:
        from config.settings import OUT_DIR
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        log_file = OUT_DIR / "main.log"

        # Backup large log files
        if log_file.exists() and log_file.stat().st_size > 50 * 1024 * 1024:  # 50MB
            backup_name = f"main_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            log_file.rename(OUT_DIR / backup_name)

        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(numeric_level)

        file_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    except ImportError:
        # Skip file logging if settings unavailable
        pass

    return logger


# Global logger for easy import
log = setup_logger()

# Simple helper functions
def info(msg): log.info(msg)
def warn(msg): log.warning(msg)
def error(msg): log.error(msg)
def debug(msg): log.debug(msg)