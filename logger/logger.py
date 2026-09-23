"""
logger/logger.py
-----------------
Centralized logging utility for the Driver Cognitive State Monitoring
System.

Every module in this project should obtain its logger through
`get_logger(__name__)` instead of using `print()`. This gives us:

- Consistent formatting across the whole application.
- Timestamps and severity levels on every message.
- Simultaneous logging to console (for development) and to a
  rotating log file (for later analysis / IEEE experiment records).
- Easy global control of verbosity via config.py.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from config import LOGGING_CONFIG


def _ensure_log_directory_exists(log_dir: str) -> None:
    """Create the log directory if it does not already exist.

    Parameters
    ----------
    log_dir : str
        Relative or absolute path to the directory where log files
        should be stored.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Create (or retrieve) a configured logger instance.

    Parameters
    ----------
    name : str
        Name of the logger, conventionally passed as `__name__` from
        the calling module. This makes every log line traceable back
        to the exact module that produced it.

    Returns
    -------
    logging.Logger
        A fully configured logger with console and/or file handlers
        attached, according to config.py.
    """
    logger = logging.getLogger(name)

    # Avoid attaching duplicate handlers if get_logger() is called
    # multiple times for the same module (e.g. on module re-import).
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOGGING_CONFIG.LOG_LEVEL.upper(), logging.DEBUG))

    formatter = logging.Formatter(
        fmt=LOGGING_CONFIG.LOG_FORMAT,
        datefmt=LOGGING_CONFIG.LOG_DATE_FORMAT,
    )

    if LOGGING_CONFIG.LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if LOGGING_CONFIG.LOG_TO_FILE:
        _ensure_log_directory_exists(LOGGING_CONFIG.LOG_DIR)
        log_file_path = os.path.join(LOGGING_CONFIG.LOG_DIR, LOGGING_CONFIG.LOG_FILE_NAME)

        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=5 * 1024 * 1024,  # 5 MB per file before rotating
            backupCount=3,             # keep 3 old log files
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent messages from propagating to the root logger and being
    # printed twice.
    logger.propagate = False

    return logger
