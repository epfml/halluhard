"""Logging configuration for the research_questions_refactored package.

This module is separate to avoid circular imports.
"""

import logging

# Package-level logger name
LOGGER_NAME = "research_questions_refactored"


def get_logger() -> logging.Logger:
    """Get the package-level logger.
    
    All modules in this package should use this function to get a logger.
    This ensures consistent logging across the entire pipeline.
    
    Usage:
        from research_questions_refactored.logging_config import get_logger
        logger = get_logger()
        logger.info("Message")
    """
    return logging.getLogger(LOGGER_NAME)


def configure_logging(
    level: int = logging.INFO,
    format: str = "%(asctime)s - %(levelname)s - %(message)s",
) -> None:
    """Configure logging for the entire package.
    
    Call this once at the start of your script to set up logging.
    
    Args:
        level: Logging level (default: INFO)
        format: Log message format
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    
    # Only add handler if none exist (avoid duplicate handlers)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(format))
        logger.addHandler(handler)

