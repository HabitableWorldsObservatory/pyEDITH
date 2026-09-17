import pytest
import numpy as np
from astropy import units as u
from astropy import constants as const
from unittest.mock import patch, MagicMock
from pyEDITH.detectors import ToyModelDetector, EACDetector
from pyEDITH.units import (
    MAS,
    DIMENSIONLESS,
    DARK_CURRENT,
    READ_NOISE,
    READ_TIME,
    CLOCK_INDUCED_CHARGE,
    QUANTUM_EFFICIENCY,
    WAVELENGTH,
    LENGTH,
    ARCSEC,
    SECOND,
    FRAME,
)

# ============================================================================
# Mock Objects and Fixtures
# ============================================================================


class MockMediator:
    """Mock mediator for testing detector configurations."""

    def __init__(self, observing_mode="IMAGER"):
        self.observing_mode = observing_mode

    def get_scene_parameter(self, param):
        if param == "stellar_radius":
            return 1 * const.R_sun
        return 1.0

    def get_telescope_parameter(self, param):
        if param == "diameter":
            return 8.0 * LENGTH
        return 1.0

    def get_observation_parameter(self, param):
        if param == "wavelength":
            if self.observing_mode == "IFS":
                return np.array([0.5, 0.7, 1.2]) * WAVELENGTH
            elif self.observing_mode == "IMAGER":
                return np.array([0.5]) * WAVELENGTH
        elif param == "wavelength_range":
            if self.observing_mode == "IFS":
                return np.array([0.5, 1.2]) * WAVELENGTH
            elif self.observing_mode == "IMAGER":
                return np.array([0.5 * (1 - 0.2 / 2), 0.5 * (1 + 0.2 / 2)]) * WAVELENGTH
        elif param == "observing_mode":
            return self.observing_mode
        return 1.0

    def get_coronagraph_parameter(self, param):
        if param == "bandwidth":
            return 0.2
        return 1.0

    def get_eac_configuration(self):
        # Return None - tests use ToyModel
        return None


@pytest.fixture
def mock_instrument():
    """Fixture providing a mock instrument object for EAC detector testing."""
    mock = MagicMock()
    mock.lam = [0.5, 1.5] * u.um

    array_length = 2
    default_array = np.linspace(0.359, 0.988, array_length)

    mock.__dict__.update(
        {
            "verbose": False,
            "lam": mock.lam,
            "OP_full": [
                "PM",
                "SM",
                "TCA",
                "wave_beamsplitter",
                "pol_beamsplitter",
                "FSM",
                "OAPs_forward",
                "DM1",
                "DM2",
                "Fold",
                "OAPs_back",
                "Apodizer",
                "Focal_Plane_Mask",
                "Lyot_Stop",
                "Field_Stop",
                "filters",
                "Detector",
            ],
            "OP_tele": ["PM", "SM"],
            "OP_inst": [
                "TCA",
                "wave_beamsplitter",
                "pol_beamsplitter",
                "FSM",
                "OAPs_forward",
                "DM1",
                "DM2",
                "Fold",
                "OAPs_back",
                "Apodizer",
                "Focal_Plane_Mask",
                "Lyot_Stop",
                "Field_Stop",
                "filters",
            ],
            "OP_det": ["Detector"],
            "TCA": default_array,
            "wb_tran": np.concatenate([np.zeros(5), np.ones(5)]),
            "wb_refl": np.concatenate([np.ones(5), np.zeros(5)]),
            "wave_beamsplitter": np.ones(array_length),
            "pol_beamsplitter": np.ones(array_length),
            "FSM": default_array,
            "OAPs_forward": default_array,
            "DM1": default_array,
            "DM2": default_array,
            "Fold": default_array,
            "OAPs_back": default_array,
            "Apodizer": np.full(array_length, 0.95),
            "Focal_Plane_Mask": np.linspace(0.91, 0.89, array_length),
            "Lyot_Stop": default_array,
            "Field_Stop": np.linspace(0.91, 0.89, array_length),
            "filters": np.ones(array_length),
            "total_inst_refl": np.full(array_length, 0.7),
        }
    )

    return mock


@pytest.fixture
def mock_detector():
    """Fixture factory for creating mock detector objects for different observing modes."""

    def _create_mock(detector_type):
        mock = MagicMock()

        mock.lam = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]) * u.um
        mock.verbose = False

        qe_vis = np.array([0.9, 0.9, 0.9, 0.9, 0.9, np.nan, np.nan, np.nan, np.nan])
        qe_nir = np.array(
            [np.nan, np.nan, np.nan, np.nan, np.nan, 0.85, 0.85, 0.85, 0.85]
        )

        common_dict = {
            "lam": mock.lam,
            "verbose": False,
            "qe_vis": qe_vis,
            "dc_vis": 3e-05,
            "cic_vis": None,
            "qe_nir": qe_nir,
            "dc_nir": 0.0001,
            "cic_nir": None,
        }

        if detector_type == "IMAGER":
            mock.__dict__.update(
                {
                    **common_dict,
                    "rn_vis": 0.1,
                    "rn_nir": 0.3,
                }
            )
        elif detector_type == "IFS":
            mock.__dict__.update(
                {
                    **common_dict,
                    "rn_vis": 0.0,
                    "rn_nir": 0.4,
                }
            )
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

        return mock

    return _create_mock


@pytest.fixture
def imager_toy_detector_parameters():
    """Fixture providing standard parameters for ToyModelDetector testing."""
    return {
        "pixscale_mas": 10,
        "npix_multiplier": 2,
        "DC": 4e-5,
        "RN": 1.0,
        "tread": 1100,
        "CIC": 1.5e-3,
        "wavelength": 0.5,
        "observing_mode": "IMAGER",
    }


@pytest.fixture
def ifs_toy_detector_parameters():
    """Fixture providing standard parameters for ToyModelDetector testing."""
    return {
        "pixscale_mas": 10,
        "npix_multiplier": 2,
        "DC": 4e-5,
        "RN": 1.0,
        "tread": 1100,
        "CIC": 1.5e-3,
        "wavelength": [0.5, 0.7, 1.2],
        "observing_mode": "IFS",
    }


@pytest.fixture
def imager_eac_detector_parameters():
    """Fixture providing standard parameters for EACDetector testing."""
    return {"wavelength": 0.5, "observing_mode": "IMAGER"}


@pytest.fixture
def ifs_eac_detector_parameters():
    """Fixture providing standard parameters for EACDetector testing."""
    return {"wavelength": [0.5, 0.7, 1.2], "observing_mode": "IFS"}


# ============================================================================
# Tests for ToyModelDetector initialization
# ============================================================================


def test_toy_model_detector_init():
    """Test that ToyModelDetector initializes with None values."""
    detector = ToyModelDetector()

    assert detector.path is None
    assert detector.keyword is None


# ============================================================================
# Tests for ToyModelDetector.load_configuration - IMAGER mode
# ============================================================================


def test_toy_model_detector_load_configuration_imager_user_params(
    imager_toy_detector_parameters,
):
    """Test loading ToyModelDetector configuration with user parameters in IMAGER mode."""
    detector = ToyModelDetector()
    mediator = MockMediator("IMAGER")
    parameters = imager_toy_detector_parameters.copy()

    detector.load_configuration(parameters, mediator)

    assert detector.pixscale_mas == 10 * MAS
    assert detector.npix_multiplier == 2 * DIMENSIONLESS
    assert np.all(detector.DC == [4e-5] * DARK_CURRENT)
    assert np.all(detector.RN == [1.0] * READ_NOISE)
    assert np.all(detector.tread == [1100] * READ_TIME)
    assert np.all(detector.CIC == [1.5e-3] * CLOCK_INDUCED_CHARGE)
    assert np.all(detector.QE == [0.9] * QUANTUM_EFFICIENCY)  # default
    assert np.all(detector.dQE == [0.75] * DIMENSIONLESS)  # default


def test_toy_model_detector_load_configuration_imager_defaults():
    """Test that default pixscale is calculated correctly in IMAGER mode."""
    detector = ToyModelDetector()
    mediator = MockMediator("IMAGER")

    detector.load_configuration({"wavelength": 0.5}, mediator)

    assert np.isclose(detector.pixscale_mas, 6.4457752 * MAS)
    assert detector.npix_multiplier == 1 * DIMENSIONLESS
    assert np.all(detector.DC == [3e-5] * DARK_CURRENT)
    assert np.all(detector.RN == [0.0] * READ_NOISE)
    assert np.all(detector.tread == [1000] * READ_TIME)
    assert np.all(detector.CIC == [1.3e-3] * CLOCK_INDUCED_CHARGE)
    assert np.all(detector.QE == [0.9] * QUANTUM_EFFICIENCY)  # defaults
    assert np.all(detector.dQE == [0.75] * DIMENSIONLESS)  # defaults


# ============================================================================
# Tests for ToyModelDetector.load_configuration - IFS mode
# ============================================================================


def test_toy_model_detector_load_configuration_ifs_user_params(
    ifs_toy_detector_parameters,
):
    """Test loading ToyModelDetector configuration with user parameters in IFS mode."""
    detector = ToyModelDetector()
    mediator = MockMediator("IFS")

    detector.load_configuration(ifs_toy_detector_parameters, mediator)

    assert detector.pixscale_mas == 10 * MAS
    assert detector.npix_multiplier == 2 * DIMENSIONLESS
    assert np.all(detector.DC == [4e-5, 4e-5, 4e-5] * DARK_CURRENT)
    assert np.all(detector.RN == [1.0, 1.0, 1.0] * READ_NOISE)
    assert np.all(detector.tread == [1100, 1100, 1100] * READ_TIME)
    assert np.all(detector.CIC == [1.5e-3, 1.5e-3, 1.5e-3] * CLOCK_INDUCED_CHARGE)
    assert np.all(detector.QE == [0.9, 0.9, 0.9] * QUANTUM_EFFICIENCY)  # defaults
    assert np.all(detector.dQE == [0.75, 0.75, 0.75] * DIMENSIONLESS)  # defaults


def test_toy_model_detector_load_configuration_ifs_defaults():
    """Test that default pixscale is calculated correctly in IFS mode."""
    detector = ToyModelDetector()
    mediator = MockMediator("IFS")

    detector.load_configuration({"wavelength": [0.5, 0.7, 1.2]}, mediator)

    assert np.isclose(detector.pixscale_mas, 6.4457752 * MAS)
    assert detector.npix_multiplier == 1 * DIMENSIONLESS
    assert np.all(detector.DC == [3e-5, 3e-5, 3e-5] * DARK_CURRENT)
    assert np.all(detector.RN == [0.0, 0.0, 0.0] * READ_NOISE)
    assert np.all(detector.tread == [1000, 1000, 1000] * READ_TIME)
    assert np.all(detector.CIC == [1.3e-3, 1.3e-3, 1.3e-3] * CLOCK_INDUCED_CHARGE)
    assert np.all(detector.QE == [0.9, 0.9, 0.9] * QUANTUM_EFFICIENCY)  # defaults
    assert np.all(detector.dQE == [0.75, 0.75, 0.75] * DIMENSIONLESS)  # defaults


# # ============================================================================
# # Tests for EACDetector.load_configuration - IMAGER mode
# # ============================================================================


@patch("eacy.load_detector")
@patch("eacy.load_instrument")
def test_eac_detector_load_configuration_imager_basic(
    mock_load_instrument,
    mock_load_detector,
    mock_instrument,
    mock_detector,
    imager_eac_detector_parameters,
):
    """Test basic EACDetector configuration loading in IMAGER mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_detector.return_value = mock_detector("IMAGER")
    parameters = imager_eac_detector_parameters.copy()
    detector = EACDetector()
    mediator = MockMediator("IMAGER")

    detector.load_configuration(parameters, mediator)

    assert detector.pixscale_mas is not None
    assert detector.npix_multiplier == 1 * DIMENSIONLESS

    assert detector.DC.unit == DARK_CURRENT
    assert detector.RN.unit == READ_NOISE
    assert detector.tread.unit == READ_TIME
    assert detector.CIC.unit == CLOCK_INDUCED_CHARGE
    assert detector.QE.unit == QUANTUM_EFFICIENCY
    assert detector.dQE.unit == DIMENSIONLESS
    expected_shape = (1,)
    assert detector.DC.shape == expected_shape
    assert detector.RN.shape == expected_shape
    assert detector.QE.shape == expected_shape

    assert np.allclose(detector.DC.value, 3e-05)  # vis channel
    assert np.allclose(detector.RN.value, 0.1)  # vis channel
    assert np.allclose(detector.QE.value, 0.9)  # vis channel
    assert np.allclose(detector.dQE.value, 0.75)  # hardcoded


# ============================================================================
# Tests for EACDetector.load_configuration - IFS mode
# ============================================================================


@patch("eacy.load_detector")
@patch("eacy.load_instrument")
def test_eac_detector_load_configuration_ifs_basic(
    mock_load_instrument,
    mock_load_detector,
    mock_instrument,
    mock_detector,
    ifs_eac_detector_parameters,
):
    """Test basic EACDetector configuration loading in IFS mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_detector.return_value = mock_detector("IFS")

    detector = EACDetector()
    parameters = ifs_eac_detector_parameters.copy()
    mediator = MockMediator("IFS")

    detector.load_configuration(parameters, mediator)

    assert detector.pixscale_mas is not None
    assert detector.npix_multiplier == 1 * DIMENSIONLESS
    assert detector.DC.unit == DARK_CURRENT
    assert detector.RN.unit == READ_NOISE
    assert detector.tread.unit == READ_TIME
    assert detector.CIC.unit == CLOCK_INDUCED_CHARGE
    assert detector.QE.unit == QUANTUM_EFFICIENCY
    assert detector.dQE.unit == DIMENSIONLESS

    expected_shape = (3,)
    assert detector.DC.shape == expected_shape
    assert detector.RN.shape == expected_shape
    assert detector.QE.shape == expected_shape
    assert detector.CIC.shape == expected_shape

    # VIS wavelengths (< 1 μm)
    assert np.allclose(detector.DC[:2].value, 3e-05)
    assert np.allclose(detector.RN[:2].value, 0.0)

    # NIR wavelengths (>= 1 μm)
    assert np.allclose(detector.DC[2:].value, 0.0001)
    assert np.allclose(detector.RN[2:].value, 0.4)


# # ============================================================================
# # Tests for EACDetector validation inputs
# # ============================================================================


@pytest.mark.parametrize("observing_mode", ["IMAGER", "IFS"])
def test_eac_detector_etc_validation_inputs(
    observing_mode, ifs_eac_detector_parameters, imager_eac_detector_parameters
):
    """Test that ETC validation inputs are correctly loaded."""
    detector = EACDetector()
    mediator = MockMediator(observing_mode)
    parameters = (
        imager_eac_detector_parameters
        if observing_mode == "IMAGER"
        else ifs_eac_detector_parameters
    )
    parameters["t_photon_count_input"] = 0.7
    parameters["det_npix_input"] = 200

    detector.load_configuration(parameters, mediator)

    assert hasattr(detector, "t_photon_count_input")
    assert hasattr(detector, "det_npix_input")
    assert detector.t_photon_count_input == 0.7 * SECOND / FRAME
    assert np.allclose(detector.det_npix_input, 200 * DIMENSIONLESS)


# ============================================================================
# Tests for Detector.validate_configuration
# ============================================================================


def test_detector_validate_configuration_all_valid(imager_toy_detector_parameters):
    """Test that validation passes with all correct attributes."""
    detector = ToyModelDetector()
    parameters = {
        **imager_toy_detector_parameters,
        "QE": [0.95],
        "dQE": [0.8],
    }
    mediator = MockMediator()

    detector.load_configuration(parameters, mediator)

    # Should not raise
    detector.validate_configuration()


def test_detector_validate_configuration_missing_pixscale(
    imager_toy_detector_parameters,
):
    """Test that missing pixscale_mas attribute raises AttributeError."""
    detector = ToyModelDetector()
    mediator = MockMediator()

    detector.load_configuration(imager_toy_detector_parameters, mediator)
    delattr(detector, "pixscale_mas")

    with pytest.raises(
        AttributeError, match="Detector is missing attribute: pixscale_mas"
    ):
        detector.validate_configuration()


def test_detector_validate_configuration_pixscale_not_quantity(
    imager_toy_detector_parameters,
):
    """Test that non-Quantity pixscale_mas raises TypeError."""
    detector = ToyModelDetector()
    mediator = MockMediator()

    detector.load_configuration(imager_toy_detector_parameters, mediator)
    detector.pixscale_mas = 10  # Not a Quantity

    with pytest.raises(
        TypeError, match="Detector attribute pixscale_mas should be a Quantity"
    ):
        detector.validate_configuration()


def test_detector_validate_configuration_incorrect_pixscale_units(
    imager_toy_detector_parameters,
):
    """Test that pixscale_mas with incorrect units raises ValueError."""
    detector = ToyModelDetector()
    mediator = MockMediator()

    detector.load_configuration(imager_toy_detector_parameters, mediator)
    detector.pixscale_mas = 10 * u.arcsec  # Wrong unit

    with pytest.raises(
        ValueError, match="Detector attribute pixscale_mas has incorrect units"
    ):
        detector.validate_configuration()


# # ============================================================================
# # Tests for parameter broadcasting in IFS mode
# # ============================================================================


def test_toy_model_detector_scalar_to_array_broadcasting():
    """Test that scalar detector parameters are correctly broadcast to arrays in IFS mode."""
    detector = ToyModelDetector()
    mediator = MockMediator("IFS")

    parameters = {
        "DC": [4e-5],  # Single value
        "RN": [1.0],
        "tread": [1100],
        "CIC": [1.5e-3],
        "wavelength": [0.5, 0.6, 0.7],
        "observing_mode": "IFS",
    }

    detector.load_configuration(parameters, mediator)

    # Should be broadcast to match wavelength array length (3)
    assert len(detector.DC) == 3
    assert len(detector.RN) == 3
    assert len(detector.tread) == 3
    assert len(detector.CIC) == 3
