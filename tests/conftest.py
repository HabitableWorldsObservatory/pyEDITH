import pytest
from yippy import Coronagraph, fetch_yip


@pytest.fixture(scope="session")
def coronagraph_path():
    """Session-scoped path to a real coronagraph YIP fetched from Zenodo."""
    return fetch_yip("eac1_aavc_2d")


@pytest.fixture(scope="session")
def yippy_coronagraph(coronagraph_path):
    """Session-scoped yippy Coronagraph built from the fetched YIP."""
    return Coronagraph(coronagraph_path)


import logging


@pytest.fixture(autouse=True)
def reset_logger_state():
    """Prevents tests from permanently mutating loggers."""

    # 1. Backup pyEDITH logger
    pyedith_logger = logging.getLogger("pyEDITH")
    pyedith_level = pyedith_logger.level
    pyedith_handlers = list(pyedith_logger.handlers)

    # 2. Backup yippy logger (since set_verbosity modifies it)
    yippy_logger = logging.getLogger("yippy")
    yippy_level = yippy_logger.level
    yippy_handlers = list(yippy_logger.handlers)

    # 3. Backup root logger
    root_logger = logging.getLogger()
    root_level = root_logger.level
    root_handlers = list(root_logger.handlers)

    yield

    # Restore pyEDITH logger
    pyedith_logger.setLevel(pyedith_level)
    pyedith_logger.handlers.clear()
    for h in pyedith_handlers:
        pyedith_logger.addHandler(h)

    # Restore yippy logger
    yippy_logger.setLevel(yippy_level)
    for h in yippy_logger.handlers:
        h.setLevel(yippy_level)

    # Restore root logger
    root_logger.setLevel(root_level)
    root_logger.handlers.clear()
    for h in root_handlers:
        root_logger.addHandler(h)
