import pytest
import numpy as np
from astropy import units as u
from unittest.mock import patch, MagicMock
from pyEDITH.telescopes import ToyModelTelescope, EACTelescope
from pyEDITH.units import LENGTH, TIME, DIMENSIONLESS, TEMPERATURE, WAVELENGTH
from pyEDITH.utils import average_over_bandpass, interpolate_over_bandpass
from pyEDITH.filters import Filter
from copy import deepcopy

# ============================================================================
# Mock Objects and Fixtures
# ============================================================================


class MockMediator:
    """Mock mediator for testing telescope configurations."""

    def __init__(self, observing_mode="IMAGER"):
        self.observing_mode = observing_mode

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
        # Return None - tests use ToyModel
        return None


@pytest.fixture
def mock_telescope_params():
    """Fixture providing mock telescope parameters from EAC."""

    class MockTelescope:
        def __init__(self):
            self.diam_circ = 8.0
            self.lam = u.Quantity([0.2, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5] * WAVELENGTH)
            self.total_tele_refl = np.array([0.9, 0.8, 0.9, 0.98, 0.75, 0.55, 0.34])

    return MockTelescope()


@pytest.fixture
def full_telescope_parameters_imager():
    """Fixture providing complete parameters for ToyModelTelescope testing."""
    return {
        "diameter": 8.0,
        "unobscured_area": 0.9,
        "toverhead_fixed": 9000,
        "toverhead_multi": 1.2,
        "telescope_optical_throughput": 0.85,
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
        "telescope_optical_throughput": 0.85,
        "temperature": 280,
        "T_contamination": 0.98,
        "wavelength": [0.5, 0.7, 1.1],  # must be provided ALWAYS
    }


# ============================================================================
# Tests for ToyModelTelescope initialization
# ============================================================================


def test_toy_model_telescope_init():
    """Test that ToyModelTelescope initializes with None values."""
    telescope = ToyModelTelescope()

    assert telescope.path is None
    assert telescope.keyword is None


# ============================================================================
# Tests for ToyModelTelescope.load_configuration - IMAGER
# ============================================================================


def test_toy_model_telescope_load_configuration_user_params(
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
    assert np.all(telescope.telescope_optical_throughput == [0.85] * DIMENSIONLESS)
    assert telescope.temperature == 280 * TEMPERATURE
    assert telescope.T_contamination == 0.98 * DIMENSIONLESS
    assert np.isclose(telescope.Area.value, 45.2389, rtol=1e-4)
    assert telescope.Area.unit == LENGTH**2


def test_toy_model_telescope_load_configuration_default_values(
    full_telescope_parameters_imager,
):
    """Test that defaults are used when parameters not provided."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    parameters = {"wavelength": 0.5}

    telescope.load_configuration(parameters, mediator)

    assert telescope.diameter == 7.87 * LENGTH
    assert telescope.unobscured_area == 0.879
    assert telescope.toverhead_fixed == 8.25e3 * TIME
    assert telescope.toverhead_multi == 1.1 * DIMENSIONLESS
    assert np.all(telescope.telescope_optical_throughput == [0.823] * DIMENSIONLESS)
    assert telescope.temperature == 290 * TEMPERATURE
    assert telescope.T_contamination == 0.95 * DIMENSIONLESS
    assert np.isclose(
        telescope.Area.value, np.single(np.pi) / 4.0 * 7.87**2.0 * 0.879, rtol=1e-4
    )
    assert telescope.Area.unit == LENGTH**2


# ============================================================================
# Tests for ToyModelTelescope.load_configuration - IFS
# ============================================================================


def test_toy_model_telescope_load_configuration_user_params(
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
    assert np.all(
        telescope.telescope_optical_throughput == [0.85, 0.85, 0.85] * DIMENSIONLESS
    )
    assert telescope.temperature == 280 * TEMPERATURE
    assert telescope.T_contamination == 0.98 * DIMENSIONLESS
    assert np.isclose(telescope.Area.value, 45.2389, rtol=1e-4)
    assert telescope.Area.unit == LENGTH**2


def test_toy_model_telescope_load_configuration_default_values():
    """Test that defaults are used when parameters not provided."""
    telescope = ToyModelTelescope()
    mediator = MockMediator("IFS")

    parameters = {"wavelength": mediator.get_observation_parameter("wavelength")}

    telescope.load_configuration(parameters, mediator)

    assert telescope.diameter == 7.87 * LENGTH
    assert telescope.unobscured_area == 0.879
    assert telescope.toverhead_fixed == 8.25e3 * TIME
    assert telescope.toverhead_multi == 1.1 * DIMENSIONLESS
    assert np.all(
        telescope.telescope_optical_throughput == [0.823, 0.823, 0.823] * DIMENSIONLESS
    )
    assert telescope.temperature == 290 * TEMPERATURE
    assert telescope.T_contamination == 0.95 * DIMENSIONLESS
    assert np.isclose(
        telescope.Area.value, np.single(np.pi) / 4.0 * 7.87**2.0 * 0.879, rtol=1e-4
    )
    assert telescope.Area.unit == LENGTH**2


# ============================================================================
# Tests for EACTelescope.load_configuration - IMAGER mode
# ============================================================================


@patch("eacy.load_telescope")
def test_eac_telescope_load_configuration_user_params(
    mock_load_telescope,
    mock_telescope_params,
    full_telescope_parameters_imager,
    caplog,
):
    """Test loading EACTelescope configuration with user parameters."""
    import logging

    mock_load_telescope.return_value = deepcopy(mock_telescope_params)
    telescope = EACTelescope(keyword="EAC1")
    mediator = MockMediator("IMAGER")

    with caplog.at_level(logging.DEBUG):
        telescope.load_configuration(full_telescope_parameters_imager, mediator)

    # These values can be changed by the user
    assert telescope.toverhead_fixed == 9000 * TIME
    assert telescope.toverhead_multi == 1.2 * DIMENSIONLESS

    # LOCKED PARAMETERS: EVEN IF WE LOAD USER PARAMETERS, WE WANT THE DEFAULTS
    assert telescope.diameter == 8.0 * LENGTH
    assert telescope.unobscured_area == 1.0

    assert telescope.temperature == 290 * TEMPERATURE
    assert telescope.T_contamination == 1.0 * DIMENSIONLESS

    expected_throughput = average_over_bandpass(
        {
            "lam": mock_telescope_params.lam,
            "total_tele_refl": mock_telescope_params.total_tele_refl.copy(),
        },
        mediator.get_observation_parameter("wavelength_range"),
    )["total_tele_refl"]

    assert np.isclose(
        telescope.telescope_optical_throughput[0].value,
        expected_throughput,
        rtol=1e-5,
    )
    assert np.isclose(telescope.Area.value, 50.2655, rtol=1e-4)
    assert telescope.Area.unit == LENGTH**2

    # Check that warning messages were logged for locked parameters using default values
    warning_messages = [
        record.message for record in caplog.records if record.levelname == "WARNING"
    ]
    locked_keys = {
        "diameter",
        "unobscured_area",
        "T_contamination",
        "temperature",
        "telescope_optical_throughput",
    }
    for key in locked_keys:
        assert any(
            key in msg and "is locked in this mode" in msg for msg in warning_messages
        ), f"Expected warning message for locked key '{key}' not found"


@patch("eacy.load_telescope")
def test_eac_telescope_load_configuration_default_values(
    mock_load_telescope, mock_telescope_params
):
    """Test that defaults are used when parameters not provided."""
    mock_load_telescope.return_value = deepcopy(mock_telescope_params)
    mediator = MockMediator("IMAGER")

    telescope = EACTelescope(keyword="EAC1")
    parameters = {
        "wavelength": mediator.get_observation_parameter("wavelength"),
    }

    telescope.load_configuration(parameters, mediator)

    # Defaults
    assert telescope.diameter == 8.0 * LENGTH
    assert telescope.unobscured_area == 1.0
    assert telescope.toverhead_fixed == 8.25e3 * TIME
    assert telescope.toverhead_multi == 1.1 * DIMENSIONLESS
    assert telescope.temperature == 290 * TEMPERATURE
    assert telescope.T_contamination == 1.0 * DIMENSIONLESS
    wavelength = mediator.get_observation_parameter("wavelength")
    wavelength_range = [min(wavelength), max(wavelength)]
    expected_throughput = average_over_bandpass(
        {
            "lam": mock_telescope_params.lam,
            "total_tele_refl": mock_telescope_params.total_tele_refl.copy(),
        },
        wavelength_range,
    )["total_tele_refl"]

    assert np.isclose(
        telescope.telescope_optical_throughput[0].value,
        expected_throughput,
        rtol=1e-5,
    )
    assert np.isclose(telescope.Area.value, 50.2655, rtol=1e-4)
    assert telescope.Area.unit == LENGTH**2


# # ============================================================================
# # Tests for EACTelescope.load_configuration - IFS mode
# # ============================================================================


@patch("eacy.load_telescope")
def test_eac_telescope_load_configuration_user_params(
    mock_load_telescope,
    mock_telescope_params,
    full_telescope_parameters_ifs,
    caplog,
):
    """Test loading EACTelescope configuration with user parameters."""
    import logging

    mock_load_telescope.return_value = deepcopy(mock_telescope_params)
    telescope = EACTelescope(keyword="EAC1")
    mediator = MockMediator("IFS")

    with caplog.at_level(logging.DEBUG):
        telescope.load_configuration(full_telescope_parameters_ifs, mediator)

    # These values can be changed by the user
    assert telescope.toverhead_fixed == 9000 * TIME
    assert telescope.toverhead_multi == 1.2 * DIMENSIONLESS

    # LOCKED PARAMETERS: EVEN IF WE LOAD USER PARAMETERS, WE WANT THE DEFAULTS
    assert telescope.diameter == 8.0 * LENGTH
    assert telescope.unobscured_area == 1.0

    assert telescope.temperature == 290 * TEMPERATURE
    assert telescope.T_contamination == 1.0 * DIMENSIONLESS
    wavelengths = mediator.get_observation_parameter("wavelength")
    expected_throughput = interpolate_over_bandpass(
        {
            "lam": mock_telescope_params.lam,
            "total_tele_refl": mock_telescope_params.total_tele_refl.copy(),
        },
        wavelengths,
    )["total_tele_refl"]
    print(expected_throughput, telescope.telescope_optical_throughput)
    assert np.allclose(
        telescope.telescope_optical_throughput.value,
        expected_throughput,
        rtol=1e-5,
    )
    assert np.isclose(telescope.Area.value, 50.2655, rtol=1e-4)
    assert telescope.Area.unit == LENGTH**2

    # Check that warning messages were logged for locked parameters using default values
    warning_messages = [
        record.message for record in caplog.records if record.levelname == "WARNING"
    ]
    locked_keys = {
        "diameter",
        "unobscured_area",
        "T_contamination",
        "temperature",
        "telescope_optical_throughput",
    }
    for key in locked_keys:
        assert any(
            key in msg and "is locked in this mode" in msg for msg in warning_messages
        ), f"Expected warning message for locked key '{key}' not found"


@patch("eacy.load_telescope")
def test_eac_telescope_load_configuration_default_values(
    mock_load_telescope, mock_telescope_params
):
    """Test that defaults are used when parameters not provided."""
    mock_load_telescope.return_value = deepcopy(mock_telescope_params)
    mediator = MockMediator("IFS")

    telescope = EACTelescope(keyword="EAC1")
    parameters = {
        "wavelength": mediator.get_observation_parameter("wavelength"),
    }

    telescope.load_configuration(parameters, mediator)

    # Defaults
    assert telescope.diameter == 8.0 * LENGTH
    assert telescope.unobscured_area == 1.0
    assert telescope.toverhead_fixed == 8.25e3 * TIME
    assert telescope.toverhead_multi == 1.1 * DIMENSIONLESS
    assert telescope.temperature == 290 * TEMPERATURE
    assert telescope.T_contamination == 1.0 * DIMENSIONLESS
    wavelengths = mediator.get_observation_parameter("wavelength")
    expected_throughput = interpolate_over_bandpass(
        {
            "lam": mock_telescope_params.lam,
            "total_tele_refl": mock_telescope_params.total_tele_refl.copy(),
        },
        wavelengths,
    )["total_tele_refl"]

    assert np.allclose(
        telescope.telescope_optical_throughput.value,
        expected_throughput,
        rtol=1e-5,
    )
    assert np.isclose(telescope.Area.value, 50.2655, rtol=1e-4)
    assert telescope.Area.unit == LENGTH**2


# # ============================================================================
# Tests for Telescope.validate_configuration
# ============================================================================


def test_telescope_validate_configuration_all_valid(
    full_telescope_parameters_imager,
):
    """Test that validation passes with all correct attributes."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    telescope.load_configuration(full_telescope_parameters_imager, mediator)

    # Should not raise
    telescope.validate_configuration()


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
# Tests for array parameter conversion
# ============================================================================


def test_toy_model_telescope_throughput_array_conversion():
    """Test that telescope throughput is converted to numpy array."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    parameters = {
        "diameter": 8.0,
        "telescope_optical_throughput": [0.85, 0.90],  # List input
        "wavelength": [0.5, 0.6],
    }

    telescope.load_configuration(parameters, mediator)

    assert isinstance(telescope.telescope_optical_throughput.value, np.ndarray)
    assert len(telescope.telescope_optical_throughput) == 2


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


@patch("eacy.load_telescope")
def test_eac_telescope_default_contamination(
    mock_load_telescope, mock_telescope_params
):
    """Test that EACTelescope uses default contamination factor."""
    mock_load_telescope.return_value = deepcopy(mock_telescope_params)

    telescope = EACTelescope(keyword="EAC1")
    mediator = MockMediator("IMAGER")
    parameters = {
        "observing_mode": "IMAGER",
        "wavelength": mediator.get_observation_parameter("wavelength"),
    }

    telescope.load_configuration(parameters, mediator)

    assert telescope.T_contamination == 1.0 * DIMENSIONLESS


@patch("eacy.load_telescope")
def test_eac_telescope_default_temperature(mock_load_telescope, mock_telescope_params):
    """Test that EACTelescope uses default temperature."""
    mock_load_telescope.return_value = deepcopy(mock_telescope_params)

    telescope = EACTelescope(keyword="EAC1")
    mediator = MockMediator("IMAGER")
    parameters = {
        "observing_mode": "IMAGER",
        "wavelength": mediator.get_observation_parameter("wavelength"),
    }

    telescope.load_configuration(parameters, mediator)

    assert telescope.temperature == 290 * TEMPERATURE


# ============================================================================
# Tests for throughput shape consistency
# ============================================================================


@patch("eacy.load_telescope")
def test_eac_telescope_throughput_shape_imager(
    mock_load_telescope, mock_telescope_params
):
    """Test that throughput has correct shape in IMAGER mode (scalar)."""
    mock_load_telescope.return_value = deepcopy(mock_telescope_params)

    telescope = EACTelescope(keyword="EAC1")
    mediator = MockMediator("IMAGER")
    parameters = {
        "observing_mode": "IMAGER",
        "wavelength": mediator.get_observation_parameter("wavelength"),
    }

    telescope.load_configuration(parameters, mediator)

    assert telescope.telescope_optical_throughput.shape == (1,)


@patch("eacy.load_telescope")
def test_eac_telescope_throughput_shape_ifs(mock_load_telescope, mock_telescope_params):
    """Test that throughput has correct shape in IFS mode (matches wavelength array)."""
    mock_load_telescope.return_value = deepcopy(mock_telescope_params)

    telescope = EACTelescope(keyword="EAC1")
    mediator = MockMediator("IFS")

    parameters = {
        "observing_mode": "IFS",
        "wavelength": mediator.get_observation_parameter("wavelength"),
    }

    telescope.load_configuration(parameters, mediator)

    wavelengths = mediator.get_observation_parameter("wavelength")
    assert telescope.telescope_optical_throughput.shape == wavelengths.shape
