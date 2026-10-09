import pytest
import numpy as np
from astropy import units as u
from pyEDITH.telescopes import ToyModelTelescope, EACTelescope
from pyEDITH.units import LENGTH, TIME, DIMENSIONLESS, TEMPERATURE, WAVELENGTH

# ============================================================================
# Mock Objects and Fixtures
# ============================================================================


class MockMediator:
    """Mock mediator for testing telescope configurations."""

    def __init__(self, observing_mode="IMAGER", eac_config=None, active_channel=None):
        self.observing_mode = observing_mode
        self.eac_config = eac_config
        self.active_channel = active_channel

    def get_observation_parameter(self, param):
        if param == "wavelength":
            if self.observing_mode == "IFS":
                return np.array([0.5, 0.7, 1.1]) * WAVELENGTH
            elif self.observing_mode == "IMAGER":
                return np.array([0.7]) * WAVELENGTH
        elif param == "wavelength_range":
            if self.observing_mode == "IFS":
                return np.array([0.5, 1.1]) * WAVELENGTH
            elif self.observing_mode == "IMAGER":
                return np.array([0.7 * (1 - 0.2 / 2), 0.7 * (1 + 0.2 / 2)]) * WAVELENGTH
        elif param == "delta_wavelength":
            if self.observing_mode == "IFS":
                return np.array([0.5 / 140, 0.7 / 140, 1.1 / 140]) * WAVELENGTH
            elif self.observing_mode == "IMAGER":
                return np.array([0.7 / 140]) * WAVELENGTH
        elif param == "observing_mode":
            return self.observing_mode
        return 1.0

    def get_eac_configuration(self):
        return self.eac_config

    def get_active_channel(self):
        return self.active_channel


@pytest.fixture
def mock_eac_config():
    """Fake unified EAC configuration dict, as produced by
    Observatory._load_eac_configuration(). EACTelescope only reads the
    top-level 'diameter' and 'temperature' keys.
    """
    return {
        "diameter": 8.0,
        "temperature": 290,
        "IMAGER": {},
        "IFS": {},
    }


@pytest.fixture
def full_telescope_parameters_imager():
    """Fixture providing complete parameters for ToyModelTelescope testing."""
    return {
        "diameter": 8.0,
        "unobscured_area": 0.9,
        "toverhead_fixed": 9000,
        "toverhead_multi": 1.2,
        "temperature": 280,
        "T_contamination": 0.98,
        "wavelength": 0.7,  # must be provided ALWAYS
    }


@pytest.fixture
def full_telescope_parameters_ifs():
    """Fixture providing complete parameters for ToyModelTelescope testing."""
    return {
        "diameter": 8.0,
        "unobscured_area": 0.9,
        "toverhead_fixed": 9000,
        "toverhead_multi": 1.2,
        "temperature": 280,
        "T_contamination": 0.98,
        "wavelength": [0.5, 0.7, 1.1],  # must be provided ALWAYS
    }


# ============================================================================
# Tests for ToyModelTelescope initialization
# ============================================================================


def test_toy_model_telescope_init():
    """Test that ToyModelTelescope initializes with expected defaults."""
    telescope = ToyModelTelescope()

    assert telescope.path is None
    assert telescope.keyword == "ToyModel"


def test_toy_model_telescope_init_explicit_args():
    """Test that explicit path/keyword args are stored as given."""
    telescope = ToyModelTelescope(path="/some/path", keyword="CustomToy")

    assert telescope.path == "/some/path"
    assert telescope.keyword == "CustomToy"


# ============================================================================
# Tests for ToyModelTelescope.load_configuration - IMAGER
# ============================================================================


def test_toy_model_telescope_load_configuration_user_params_imager(
    full_telescope_parameters_imager,
):
    """Test loading ToyModelTelescope configuration with user parameters."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    telescope.load_configuration(full_telescope_parameters_imager, mediator)

    assert telescope.diameter == 8.0 * LENGTH
    assert telescope.unobscured_area == 0.9
    assert telescope.toverhead_fixed == 9000 * TIME
    assert telescope.toverhead_multi == 1.2 * DIMENSIONLESS
    assert telescope.temperature == 280 * TEMPERATURE
    assert telescope.T_contamination == 0.98 * DIMENSIONLESS
    assert np.isclose(telescope.Area.value, 45.2389, rtol=1e-4)
    assert telescope.Area.unit == LENGTH**2


def test_toy_model_telescope_load_configuration_default_values_imager():
    """Test that defaults are used when parameters not provided (IMAGER)."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    parameters = {"wavelength": 0.5}

    telescope.load_configuration(parameters, mediator)

    assert telescope.diameter == 7.87 * LENGTH
    assert telescope.unobscured_area == 0.879
    assert telescope.toverhead_fixed == 8.25e3 * TIME
    assert telescope.toverhead_multi == 1.1 * DIMENSIONLESS
    assert telescope.temperature == 290 * TEMPERATURE
    assert telescope.T_contamination == 0.95 * DIMENSIONLESS
    assert np.isclose(
        telescope.Area.value, np.single(np.pi) / 4.0 * 7.87**2.0 * 0.879, rtol=1e-4
    )
    assert telescope.Area.unit == LENGTH**2


# ============================================================================
# Tests for ToyModelTelescope.load_configuration - IFS
# ============================================================================


def test_toy_model_telescope_load_configuration_user_params_ifs(
    full_telescope_parameters_ifs,
):
    """Test loading ToyModelTelescope configuration with user parameters."""
    telescope = ToyModelTelescope()
    mediator = MockMediator("IFS")

    telescope.load_configuration(full_telescope_parameters_ifs, mediator)

    assert telescope.diameter == 8.0 * LENGTH
    assert telescope.unobscured_area == 0.9
    assert telescope.toverhead_fixed == 9000 * TIME
    assert telescope.toverhead_multi == 1.2 * DIMENSIONLESS
    assert telescope.temperature == 280 * TEMPERATURE
    assert telescope.T_contamination == 0.98 * DIMENSIONLESS
    assert np.isclose(telescope.Area.value, 45.2389, rtol=1e-4)
    assert telescope.Area.unit == LENGTH**2


def test_toy_model_telescope_load_configuration_default_values_ifs():
    """Test that defaults are used when parameters not provided (IFS)."""
    telescope = ToyModelTelescope()
    mediator = MockMediator("IFS")

    parameters = {"wavelength": mediator.get_observation_parameter("wavelength")}

    telescope.load_configuration(parameters, mediator)

    assert telescope.diameter == 7.87 * LENGTH
    assert telescope.unobscured_area == 0.879
    assert telescope.toverhead_fixed == 8.25e3 * TIME
    assert telescope.toverhead_multi == 1.1 * DIMENSIONLESS
    assert telescope.temperature == 290 * TEMPERATURE
    assert telescope.T_contamination == 0.95 * DIMENSIONLESS
    assert np.isclose(
        telescope.Area.value, np.single(np.pi) / 4.0 * 7.87**2.0 * 0.879, rtol=1e-4
    )
    assert telescope.Area.unit == LENGTH**2


# ============================================================================
# Tests for EACTelescope.load_configuration - IMAGER mode
# ============================================================================


def test_eac_telescope_load_configuration_user_params_imager(
    mock_eac_config,
    full_telescope_parameters_imager,
    caplog,
):
    """Test loading EACTelescope configuration with user parameters (IMAGER)."""
    import logging

    telescope = EACTelescope(keyword="EAC1")
    mediator = MockMediator("IMAGER", eac_config=mock_eac_config)

    with caplog.at_level(logging.DEBUG):
        telescope.load_configuration(full_telescope_parameters_imager, mediator)

    # These values can be changed by the user
    assert telescope.toverhead_fixed == 9000 * TIME
    assert telescope.toverhead_multi == 1.2 * DIMENSIONLESS

    # LOCKED PARAMETERS: EVEN IF WE LOAD USER PARAMETERS, WE WANT THE
    # VALUES FROM THE UNIFIED EAC CONFIG, NOT THE USER-SUPPLIED ONES
    assert telescope.diameter == mock_eac_config["diameter"] * LENGTH
    assert telescope.unobscured_area == 1.0
    assert telescope.temperature == mock_eac_config["temperature"] * TEMPERATURE
    assert telescope.T_contamination == 1.0 * DIMENSIONLESS

    expected_area = np.single(np.pi) / 4.0 * mock_eac_config["diameter"] ** 2.0 * 1.0
    assert np.isclose(telescope.Area.value, expected_area, rtol=1e-4)
    assert telescope.Area.unit == LENGTH**2

    # Check that warning messages were logged for locked parameters using
    # the EAC-supplied values instead of the user's override attempt
    warning_messages = [
        record.message for record in caplog.records if record.levelname == "WARNING"
    ]
    locked_keys = {"diameter", "unobscured_area", "T_contamination", "temperature"}
    for key in locked_keys:
        assert any(
            key in msg and "is locked in this mode" in msg for msg in warning_messages
        ), f"Expected warning message for locked key '{key}' not found"


def test_eac_telescope_load_configuration_default_values_imager(mock_eac_config):
    """Test that defaults are used when optional parameters not provided (IMAGER)."""
    mediator = MockMediator("IMAGER", eac_config=mock_eac_config)

    telescope = EACTelescope(keyword="EAC1")
    parameters = {
        "wavelength": mediator.get_observation_parameter("wavelength"),
    }

    telescope.load_configuration(parameters, mediator)

    assert telescope.diameter == mock_eac_config["diameter"] * LENGTH
    assert telescope.unobscured_area == 1.0
    assert telescope.toverhead_fixed == 8.25e3 * TIME
    assert telescope.toverhead_multi == 1.1 * DIMENSIONLESS
    assert telescope.temperature == mock_eac_config["temperature"] * TEMPERATURE
    assert telescope.T_contamination == 1.0 * DIMENSIONLESS

    expected_area = np.single(np.pi) / 4.0 * mock_eac_config["diameter"] ** 2.0 * 1.0
    assert np.isclose(telescope.Area.value, expected_area, rtol=1e-4)
    assert telescope.Area.unit == LENGTH**2


# ============================================================================
# Tests for EACTelescope.load_configuration - IFS mode
# ============================================================================


def test_eac_telescope_load_configuration_user_params_ifs(
    mock_eac_config,
    full_telescope_parameters_ifs,
    caplog,
):
    """Test loading EACTelescope configuration with user parameters (IFS)."""
    import logging

    telescope = EACTelescope(keyword="EAC1")
    mediator = MockMediator("IFS", eac_config=mock_eac_config)

    with caplog.at_level(logging.DEBUG):
        telescope.load_configuration(full_telescope_parameters_ifs, mediator)

    assert telescope.toverhead_fixed == 9000 * TIME
    assert telescope.toverhead_multi == 1.2 * DIMENSIONLESS

    assert telescope.diameter == mock_eac_config["diameter"] * LENGTH
    assert telescope.unobscured_area == 1.0
    assert telescope.temperature == mock_eac_config["temperature"] * TEMPERATURE
    assert telescope.T_contamination == 1.0 * DIMENSIONLESS

    expected_area = np.single(np.pi) / 4.0 * mock_eac_config["diameter"] ** 2.0 * 1.0
    assert np.isclose(telescope.Area.value, expected_area, rtol=1e-4)
    assert telescope.Area.unit == LENGTH**2

    warning_messages = [
        record.message for record in caplog.records if record.levelname == "WARNING"
    ]
    locked_keys = {"diameter", "unobscured_area", "T_contamination", "temperature"}
    for key in locked_keys:
        assert any(
            key in msg and "is locked in this mode" in msg for msg in warning_messages
        ), f"Expected warning message for locked key '{key}' not found"


def test_eac_telescope_load_configuration_default_values_ifs(mock_eac_config):
    """Test that defaults are used when optional parameters not provided (IFS)."""
    mediator = MockMediator("IFS", eac_config=mock_eac_config)

    telescope = EACTelescope(keyword="EAC1")
    parameters = {
        "wavelength": mediator.get_observation_parameter("wavelength"),
    }

    telescope.load_configuration(parameters, mediator)

    assert telescope.diameter == mock_eac_config["diameter"] * LENGTH
    assert telescope.unobscured_area == 1.0
    assert telescope.toverhead_fixed == 8.25e3 * TIME
    assert telescope.toverhead_multi == 1.1 * DIMENSIONLESS
    assert telescope.temperature == mock_eac_config["temperature"] * TEMPERATURE
    assert telescope.T_contamination == 1.0 * DIMENSIONLESS

    expected_area = np.single(np.pi) / 4.0 * mock_eac_config["diameter"] ** 2.0 * 1.0
    assert np.isclose(telescope.Area.value, expected_area, rtol=1e-4)
    assert telescope.Area.unit == LENGTH**2


# ============================================================================
# Tests for the mediator-driven EAC-config fail-fast guard
# ============================================================================


def test_eac_telescope_missing_eac_configuration_raises_runtime_error():
    """EACTelescope with an empty configuration will fail with a RuntimeError."""
    telescope = EACTelescope(keyword="CustomModel")
    mediator = MockMediator("IMAGER", eac_config=None)
    parameters = {"wavelength": mediator.get_observation_parameter("wavelength")}

    with pytest.raises(RuntimeError, match="Failed to load EAC configuration"):
        telescope.load_configuration(parameters, mediator)


# ============================================================================
# Regression tests for the DEFAULT_CONFIG shared-mutable-class-attribute fix
# ============================================================================


def test_toy_model_telescope_default_config_not_shared_class_attribute():
    """self.DEFAULT_CONFIG must be a distinct object from the class-level
    dict immediately after __init__, and mutating one instance's copy must
    not affect the class attribute or any other instance."""
    telescope = ToyModelTelescope()

    assert telescope.DEFAULT_CONFIG is not ToyModelTelescope.DEFAULT_CONFIG
    assert telescope.DEFAULT_CONFIG == ToyModelTelescope.DEFAULT_CONFIG  # equal values

    telescope.DEFAULT_CONFIG["diameter"] = 999 * LENGTH

    other = ToyModelTelescope()
    assert ToyModelTelescope.DEFAULT_CONFIG["diameter"] == 7.87 * LENGTH
    assert other.DEFAULT_CONFIG["diameter"] == 7.87 * LENGTH


def test_eac_telescope_default_config_not_shared_class_attribute():
    """Same check for EACTelescope."""
    telescope = EACTelescope(keyword="EAC1")

    assert telescope.DEFAULT_CONFIG is not EACTelescope.DEFAULT_CONFIG
    assert telescope.DEFAULT_CONFIG == EACTelescope.DEFAULT_CONFIG

    telescope.DEFAULT_CONFIG["diameter"] = 999 * LENGTH

    other = EACTelescope(keyword="EAC2")
    assert EACTelescope.DEFAULT_CONFIG["diameter"] is None
    assert other.DEFAULT_CONFIG["diameter"] is None


# ============================================================================
# Tests for Telescope.validate_configuration
# ============================================================================
# Unaffected by the refactor -- Telescope.validate_configuration() itself
# did not change. Kept as-is.


def test_telescope_validate_configuration_all_valid(
    full_telescope_parameters_imager,
):
    """Test that validation passes with all correct attributes."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    telescope.load_configuration(full_telescope_parameters_imager, mediator)

    telescope.validate_configuration()  # should not raise


def test_telescope_validate_configuration_missing_diameter(
    full_telescope_parameters_imager,
):
    """Test that missing diameter attribute raises AttributeError."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    telescope.load_configuration(full_telescope_parameters_imager, mediator)
    delattr(telescope, "diameter")

    with pytest.raises(
        AttributeError, match="Telescope is missing attribute: diameter"
    ):
        telescope.validate_configuration()


def test_telescope_validate_configuration_diameter_not_quantity(
    full_telescope_parameters_imager,
):
    """Test that non-Quantity diameter raises TypeError."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    telescope.load_configuration(full_telescope_parameters_imager, mediator)
    telescope.diameter = 8.0  # Not a Quantity

    with pytest.raises(
        TypeError, match="Telescope attribute diameter should be a Quantity"
    ):
        telescope.validate_configuration()


def test_telescope_validate_configuration_incorrect_diameter_units(
    full_telescope_parameters_imager,
):
    """Test that diameter with incorrect units raises ValueError."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    telescope.load_configuration(full_telescope_parameters_imager, mediator)
    telescope.diameter = 8.0 * u.s  # Wrong unit

    with pytest.raises(
        ValueError, match="Telescope attribute diameter has incorrect units"
    ):
        telescope.validate_configuration()


# ============================================================================
# Tests for derived parameter calculations
# ============================================================================


def test_toy_model_telescope_area_with_no_obscuration():
    """Test area calculation with no obscuration (unobscured_area = 1.0)."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    parameters = {
        "diameter": 10.0,
        "unobscured_area": 1.0,
        "wavelength": mediator.get_observation_parameter("wavelength"),
    }

    telescope.load_configuration(parameters, mediator)

    expected_area = np.pi * (10.0**2) / 4.0
    assert np.isclose(telescope.Area.value, expected_area, rtol=1e-6)


def test_eac_telescope_default_contamination(mock_eac_config):
    """Test that EACTelescope uses default contamination factor."""
    mediator = MockMediator("IMAGER", eac_config=mock_eac_config)
    telescope = EACTelescope(keyword="EAC1")
    parameters = {
        "observing_mode": "IMAGER",
        "wavelength": mediator.get_observation_parameter("wavelength"),
    }

    telescope.load_configuration(parameters, mediator)

    assert telescope.T_contamination == 1.0 * DIMENSIONLESS


def test_eac_telescope_temperature_from_eac_config(mock_eac_config):
    """Test that EACTelescope's temperature comes from the mediator's unified
    eac_config, not a hardcoded class default (temperature default is None
    in DEFAULT_CONFIG and must be filled in from eac_config)."""
    mediator = MockMediator("IMAGER", eac_config=mock_eac_config)
    telescope = EACTelescope(keyword="EAC1")
    parameters = {
        "observing_mode": "IMAGER",
        "wavelength": mediator.get_observation_parameter("wavelength"),
    }

    telescope.load_configuration(parameters, mediator)

    assert telescope.temperature == mock_eac_config["temperature"] * TEMPERATURE
