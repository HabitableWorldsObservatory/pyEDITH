# src/pyEDITH/__init__.py
import sys
import logging


# ANSI color codes for your palette
class ColorCodes:
    WARNING = "\033[38;2;226;147;0m"  # #E29300 - Orange for warnings
    ERROR = "\033[38;2;156;29;16m"  # #9C1D10 - Dark red for errors
    RESET = "\033[0m"


# Custom formatter with colors for WARNING and ERROR only
class ColoredFormatter(logging.Formatter):
    """Custom formatter that colors entire log line except message for WARNING and ERROR levels."""

    def format(self, record):
        # Format the record first
        formatted = super().format(record)

        # Add color to the entire line for WARNING and ERROR
        if record.levelno == logging.WARNING:
            # Split at the message to only color the prefix
            # Format is: [name] LEVEL [timestamp] message
            # Color everything before the message
            parts = formatted.split(
                "] ", 2
            )  # Split into [name, LEVEL [timestamp, message]
            if len(parts) == 3:
                colored_prefix = (
                    f"{ColorCodes.WARNING}{parts[0]}] {parts[1]}]{ColorCodes.RESET}"
                )
                formatted = f"{colored_prefix} {parts[2]}"
            else:
                # Fallback: color entire line
                formatted = f"{ColorCodes.WARNING}{formatted}{ColorCodes.RESET}"

        elif record.levelno >= logging.ERROR:  # ERROR and CRITICAL
            parts = formatted.split("] ", 2)
            if len(parts) == 3:
                colored_prefix = (
                    f"{ColorCodes.ERROR}{parts[0]}] {parts[1]}]{ColorCodes.RESET}"
                )
                formatted = f"{colored_prefix} {parts[2]}"
            else:
                # Fallback: color entire line
                formatted = f"{ColorCodes.ERROR}{formatted}{ColorCodes.RESET}"

        return formatted


# Create a handler ONLY for pyEDITH
handler = logging.StreamHandler()
formatter = ColoredFormatter(
    "[%(name)s] %(levelname)s [%(asctime)s] %(message)s",
)
handler.setFormatter(formatter)


# Create pyEDITH logger
pyedith_logger = logging.getLogger("pyEDITH")
pyedith_logger.addHandler(handler)
pyedith_logger.propagate = True


def set_verbosity(level="warning"):
    """
    Set the verbosity level for pyEDITH+yippy logging.

    Parameters
    ----------
    level : str, optional
        Verbosity level. Options are:
        - 'error' or 'quiet': Only show errors
        - 'warning' or 'default': Show warnings and errors (default)
        - 'info' or 'verbose': Show info, warnings, and errors
        - 'debug': Show all messages including debug
    """
    level_map = {
        "error": logging.ERROR,
        "quiet": logging.ERROR,
        "warning": logging.WARNING,
        "default": logging.WARNING,
        "info": logging.INFO,
        "verbose": logging.INFO,
        "debug": logging.DEBUG,
    }

    log_level = level_map.get(level.lower(), logging.WARNING)

    # Set pyEDITH logger level
    pyedith_logger.setLevel(log_level)
    pyedith_logger.info(f"Logging level set to: {logging.getLevelName(log_level)}")

    # Set yippy logger level AND its handlers
    yippy_logger = logging.getLogger("yippy")

    yippy_logger.setLevel(log_level)
    yippy_logger.info(f"Logging level set to: {logging.getLevelName(log_level)}")

    yippy_logger.propagate = False  # Don't send to root logger
    for h in yippy_logger.handlers:
        h.setLevel(log_level)


if not logging.getLogger().handlers:
    logging.getLogger().addHandler(logging.NullHandler())

from .astrophysical_scene import AstrophysicalScene
from .observation import Observation
from .observatory import Observatory
from .coronagraphs import Coronagraph
from .telescopes import Telescope
from .detectors import Detector
from .filters import Filter

# Import main functions
from .exposure_time_calculator import calculate_exposure_time_or_snr
from .coronagraphs import generate_radii
from . import parse_input
from .utils import *
from .units import *

# Import CLI functions
from .cli import main, calculate_texp, calculate_snr

# Set a __all__ variable to control what gets imported with "from pyEDITH import *"
__all__ = [
    "AstrophysicalScene",
    "Observation",
    "Observatory",
    "Telescope",
    "Coronagraph",
    "Detector",
    "Filter",
    "calculate_exposure_time_or_snr",
    "main",
    "calculate_texp",
    "calculate_snr",
    "parse_input",
    "generate_radii",
    "average_over_bandpass",
    "interpolate_over_bandpass",
    "validate_attributes",
    "set_verbosity",
]

set_verbosity(level="warning")

# __init__.py
# Read version from pyproject.toml dynamically
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyEDITH")
except PackageNotFoundError:
    __version__ = "unknown"
