"""Tests for Fashion-MNIST training log configuration."""

import logging
from pathlib import Path

import pytest

from mnistnn import log_config


def test_configure_file_logging_appends_or_clears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify append mode preserves entries and clean mode replaces them."""
    fake_module_path = tmp_path / 'src' / 'mnistnn' / 'log_config.py'
    monkeypatch.setattr(log_config, '__file__', str(fake_module_path))
    logger = logging.Logger('test_configure_file_logging')

    try:
        log_path = log_config.configure_file_logging(logger)
        logger.info('first entry')

        log_config.configure_file_logging(logger, append=True)
        logger.info('second entry')
        assert log_path.read_text(encoding='utf-8').count('entry') == 2

        log_config.configure_file_logging(logger, append=False)
        logger.info('fresh entry')
        contents = log_path.read_text(encoding='utf-8')
        assert 'first entry' not in contents
        assert 'second entry' not in contents
        assert 'fresh entry' in contents
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
