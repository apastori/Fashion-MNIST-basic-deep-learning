"""Configure file logging for Fashion-MNIST training."""

import logging
from pathlib import Path


def configure_file_logging(
    logger: logging.Logger, append: bool = True
) -> Path:
    """Configure training INFO logging to append or start a fresh log file."""
    log_directory: Path = Path(__file__).resolve().parents[2] / 'log'
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path: Path = log_directory / 'mnist-training.log'
    resolved_log_path: str = str(log_path.resolve())
    existing_handler: logging.FileHandler | None = None

    for handler in logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and handler.baseFilename == resolved_log_path
        ):
            existing_handler = handler
            break

    if existing_handler is not None and not append:
        logger.removeHandler(existing_handler)
        existing_handler.close()
        existing_handler = None

    if existing_handler is None:
        file_handler: logging.FileHandler = logging.FileHandler(
            log_path, mode='a' if append else 'w', encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s %(levelname)s %(name)s: %(message)s'
            )
        )
        logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    return log_path
