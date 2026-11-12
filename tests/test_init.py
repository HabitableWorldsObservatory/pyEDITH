import pyEDITH
import sys
import logging
import pytest
from unittest.mock import patch, MagicMock
import importlib


def test_version_is_not_unknown():
    """Check that version was read successfully."""
    assert pyEDITH.__version__ != "unknown"


def test_version_format():
    """Check that version follows semantic versioning."""
    import re

    pattern = r"^\d+\.\d+\.\d+.*$"
    assert re.match(
        pattern, pyEDITH.__version__
    ), f"Version {pyEDITH.__version__} doesn't match semantic versioning"


def test_version_matches_pyproject():
    """Check that version matches pyproject.toml."""
    import tomllib  # Python 3.11+, or use tomli for earlier versions

    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    expected_version = pyproject["project"]["version"]
    assert pyEDITH.__version__ == expected_version, (
        f"Package version {pyEDITH.__version__} doesn't match "
        f"pyproject.toml version {expected_version}"
    )


def test_version_fallback_when_package_not_found():
    """Test that __version__ is 'unknown' when package metadata not found."""
    # This test mocks the importlib.metadata.version function to raise PackageNotFoundError
    from unittest.mock import patch
    from importlib.metadata import PackageNotFoundError
    import sys

    # Remove pyEDITH from sys.modules to force reimport
    if "pyEDITH" in sys.modules:
        del sys.modules["pyEDITH"]

    with patch("importlib.metadata.version", side_effect=PackageNotFoundError("test")):
        # Now import pyEDITH (this will go through the __init__.py again)
        # Since we mocked version(), it will raise PackageNotFoundError
        # and __version__ should be "unknown"
        import pyEDITH


def test_color_codes_exist():
    """Check that color codes are defined."""
    from pyEDITH import ColorCodes

    assert hasattr(ColorCodes, "WARNING")
    assert hasattr(ColorCodes, "ERROR")
    assert hasattr(ColorCodes, "RESET")


def test_color_codes_are_strings():
    """Check that color codes are valid strings."""
    from pyEDITH import ColorCodes

    assert isinstance(ColorCodes.WARNING, str)
    assert isinstance(ColorCodes.ERROR, str)
    assert isinstance(ColorCodes.RESET, str)


def test_color_codes_contain_ansi_escape():
    """Check that color codes contain ANSI escape sequences."""
    from pyEDITH import ColorCodes

    assert "\033[" in ColorCodes.WARNING
    assert "\033[" in ColorCodes.ERROR
    assert ColorCodes.RESET == "\033[0m"


def test_formatter_exists():
    """Check that ColoredFormatter class exists."""
    from pyEDITH import ColoredFormatter

    assert issubclass(ColoredFormatter, logging.Formatter)


def test_formatter_format_method_exists():
    """Check that formatter has format method."""
    from pyEDITH import ColoredFormatter

    formatter = ColoredFormatter()
    assert hasattr(formatter, "format")
    assert callable(formatter.format)


def test_formatter_colors_warning():
    """Check that formatter applies color to WARNING level."""
    from pyEDITH import ColoredFormatter, ColorCodes

    formatter = ColoredFormatter("[%(name)s] %(levelname)s [%(asctime)s] %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=1,
        msg="test warning",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    assert ColorCodes.WARNING in formatted
    assert ColorCodes.RESET in formatted


def test_formatter_colors_error():
    """Check that formatter applies color to ERROR level."""
    from pyEDITH import ColoredFormatter, ColorCodes

    formatter = ColoredFormatter("[%(name)s] %(levelname)s [%(asctime)s] %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=1,
        msg="test error",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    assert ColorCodes.ERROR in formatted
    assert ColorCodes.RESET in formatted


def test_formatter_no_color_info():
    """Check that formatter doesn't color INFO level."""
    from pyEDITH import ColoredFormatter, ColorCodes

    formatter = ColoredFormatter("[%(name)s] %(levelname)s [%(asctime)s] %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="test info",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    assert ColorCodes.WARNING not in formatted
    assert ColorCodes.ERROR not in formatted


def test_adds_null_handler_when_no_handlers(monkeypatch):
    """
    When the root logger has no handlers, importing the package should
    attach a NullHandler (library best practice to suppress the
    'No handlers could be found' warning).
    """
    root = logging.getLogger()

    # monkeypatch.setattr records the original value and restores it
    # automatically at teardown—no try/finally required.
    monkeypatch.setattr(root, "handlers", [])

    # Re-import the package so __init__.py executes afresh against
    # our handler-less root logger. Replace 'pyEDITH' as appropriate.
    import pyEDITH

    importlib.reload(pyEDITH)

    # Assert the branch fired: a NullHandler must now be present.
    assert any(
        isinstance(h, logging.NullHandler) for h in root.handlers
    ), "Expected __init__ to attach a NullHandler when none existed"


def test_init_skips_null_handler_when_handlers_exist(monkeypatch):
    """
    When the root logger already has handlers, importing the package
    should NOT add another NullHandler (avoiding duplicate handlers).
    """
    root = logging.getLogger()

    # monkeypatch.setattr records the original value and restores it
    # automatically at teardown—no try/finally required.
    monkeypatch.setattr(root, "handlers", [])

    # Contrive a state with exactly one non-Null handler present.
    sentinel = logging.StreamHandler()
    root.handlers = [sentinel]

    import pyEDITH

    importlib.reload(pyEDITH)

    # The NullHandler must NOT have been added.
    null_handlers = [h for h in root.handlers if isinstance(h, logging.NullHandler)]
    assert not null_handlers, "NullHandler should not be added when handlers exist"
    assert sentinel in root.handlers


def test_pyedith_logger_exists():
    """Check that pyedith_logger is created."""
    from pyEDITH import pyedith_logger

    assert isinstance(pyedith_logger, logging.Logger)
    assert pyedith_logger.name == "pyEDITH"


def test_pyedith_logger_has_handler():
    """Check that pyedith_logger has at least one handler."""
    from pyEDITH import pyedith_logger

    assert len(pyedith_logger.handlers) > 0


def test_pyedith_logger_handler_is_stream_handler():
    """Check that pyedith_logger handler is StreamHandler."""
    from pyEDITH import pyedith_logger

    assert any(isinstance(h, logging.StreamHandler) for h in pyedith_logger.handlers)


def test_pyedith_logger_formatter_is_colored():
    """Check that pyedith_logger uses ColoredFormatter."""
    from pyEDITH import pyedith_logger, ColoredFormatter

    handlers = pyedith_logger.handlers
    assert any(isinstance(h.formatter, ColoredFormatter) for h in handlers)


def test_pyedith_logger_propagate_true():
    """Check that pyedith_logger propagate is True."""
    from pyEDITH import pyedith_logger

    assert pyedith_logger.propagate is True


def test_set_verbosity_error():
    """Test setting verbosity to error."""
    from pyEDITH import set_verbosity, pyedith_logger

    set_verbosity("error")
    assert pyedith_logger.level == logging.ERROR


def test_set_verbosity_quiet():
    """Test setting verbosity to quiet."""
    from pyEDITH import set_verbosity, pyedith_logger

    set_verbosity("quiet")
    assert pyedith_logger.level == logging.ERROR


def test_set_verbosity_warning():
    """Test setting verbosity to warning."""
    from pyEDITH import set_verbosity, pyedith_logger

    set_verbosity("warning")
    assert pyedith_logger.level == logging.WARNING


def test_set_verbosity_default():
    """Test setting verbosity to default."""
    from pyEDITH import set_verbosity, pyedith_logger

    set_verbosity("default")
    assert pyedith_logger.level == logging.WARNING


def test_set_verbosity_info():
    """Test setting verbosity to info."""
    from pyEDITH import set_verbosity, pyedith_logger

    set_verbosity("info")
    assert pyedith_logger.level == logging.INFO


def test_set_verbosity_verbose():
    """Test setting verbosity to verbose."""
    from pyEDITH import set_verbosity, pyedith_logger

    set_verbosity("verbose")
    assert pyedith_logger.level == logging.INFO


def test_set_verbosity_debug():
    """Test setting verbosity to debug."""
    from pyEDITH import set_verbosity, pyedith_logger

    set_verbosity("debug")
    assert pyedith_logger.level == logging.DEBUG


def test_set_verbosity_case_insensitive():
    """Test that set_verbosity is case insensitive."""
    from pyEDITH import set_verbosity, pyedith_logger

    set_verbosity("ERROR")
    assert pyedith_logger.level == logging.ERROR

    set_verbosity("DeBuG")
    assert pyedith_logger.level == logging.DEBUG


def test_set_verbosity_invalid_falls_back_to_warning():
    """Test that invalid level falls back to WARNING."""
    from pyEDITH import set_verbosity, pyedith_logger

    set_verbosity("invalid_level")
    assert pyedith_logger.level == logging.WARNING


def test_set_verbosity_sets_yippy_logger():
    """Test that set_verbosity also sets yippy logger level."""
    from pyEDITH import set_verbosity

    set_verbosity("debug")

    yippy_logger = logging.getLogger("yippy")
    assert yippy_logger.level == logging.DEBUG


def test_set_verbosity_yippy_propagate_false():
    """Test that yippy logger propagate is set to False."""
    from pyEDITH import set_verbosity

    set_verbosity("info")

    yippy_logger = logging.getLogger("yippy")
    assert yippy_logger.propagate is False


def test_formatter_warning_fallback_malformed_format():
    """Test WARNING formatter fallback when log format is malformed."""
    from pyEDITH import ColoredFormatter, ColorCodes

    # Create formatter with different format that won't split nicely
    formatter = ColoredFormatter("%(message)s")  # Minimal format
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=1,
        msg="test warning",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    # Should use fallback coloring (entire line)
    assert ColorCodes.WARNING in formatted
    assert ColorCodes.RESET in formatted
    assert "test warning" in formatted


def test_formatter_error_fallback_malformed_format():
    """Test ERROR formatter fallback when log format is malformed."""
    from pyEDITH import ColoredFormatter, ColorCodes

    # Create formatter with different format that won't split nicely
    formatter = ColoredFormatter("%(message)s")  # Minimal format
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=1,
        msg="test error",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    # Should use fallback coloring (entire line)
    assert ColorCodes.ERROR in formatted
    assert ColorCodes.RESET in formatted
    assert "test error" in formatted


def test_main_classes_importable():
    """Check that main classes are importable from pyEDITH."""
    from pyEDITH import (
        AstrophysicalScene,
        Observation,
        Observatory,
        Telescope,
        Coronagraph,
        Detector,
    )

    assert all(
        [
            AstrophysicalScene,
            Observation,
            Observatory,
            Telescope,
            Coronagraph,
            Detector,
        ]
    )


def test_main_functions_importable():
    """Check that main functions are importable from pyEDITH."""
    from pyEDITH import (
        calculate_exposure_time_or_snr,
        calculate_texp,
        calculate_snr,
    )

    assert all(
        [
            calculate_exposure_time_or_snr,
            calculate_texp,
            calculate_snr,
        ]
    )


def test_cli_main_importable():
    """Check that CLI main function is importable."""
    from pyEDITH import main

    assert callable(main)


def test_parse_input_importable():
    """Check that parse_input module is importable."""
    from pyEDITH import parse_input

    assert parse_input is not None


def test_all_is_defined():
    """Check that __all__ is defined."""
    import pyEDITH

    assert hasattr(pyEDITH, "__all__")
    assert isinstance(pyEDITH.__all__, list)
    assert len(pyEDITH.__all__) > 0


def test_all_contents():
    """Check that __all__ contains expected items."""
    import pyEDITH

    expected = [
        "AstrophysicalScene",
        "Observation",
        "Observatory",
        "Telescope",
        "Coronagraph",
        "Detector",
        "calculate_exposure_time_or_snr",
        "main",
        "calculate_texp",
        "calculate_snr",
        "parse_input",
    ]
    for item in expected:
        assert item in pyEDITH.__all__, f"{item} not in __all__"
