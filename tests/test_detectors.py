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

    def __init__(
        self,
        observing_mode="IMAGER",
        eac_config=None,
        active_channel=None,
        delta_wavelength=None,
    ):
        self.observing_mode = observing_mode
        self._eac_config = eac_config
        self._active_channel = active_channel
        self._delta_wavelength = delta_wavelength

    def get_scene_parameter(self, param):
        if param == "stellar_radius":
            return 1 * const.R_sun
        return 1.0

    def get_telescope_parameter(self, param):
        if param == "diameter":
            return 8.0 * LENGTH
        return 1.0

    def get_observation_parameter(self, param):
        if param == "observing_mode":
            return self.observing_mode
        if param == "wavelength":
            if self.observing_mode == "IFS":
                return np.array([0.5, 0.7, 1.2]) * WAVELENGTH
            return np.array([0.5]) * WAVELENGTH
        if param == "wavelength_range":
            if self.observing_mode == "IFS":
                return np.array([0.5, 1.2]) * WAVELENGTH
            return np.array([0.45, 0.55]) * WAVELENGTH
        if param == "delta_wavelength":
            return self._delta_wavelength
        return 1.0

    def get_eac_configuration(self):
        return self._eac_config

    def get_active_channel(self):
        return self._active_channel


@pytest.fixture
def fake_eac_config():
    """Minimal unified EAC configuration dict, matching hwome/eacy output shape."""
    spectral_imager = {
        "wavelength": np.array([0.4, 0.5, 0.6]),
        "optics_throughput": np.array([0.8, 0.8, 0.8]),
        "qe": np.array([0.9, 0.9, 0.9]),
        "dqe": np.array([0.75, 0.75, 0.75]),
    }
    spectral_ifs = {
        "wavelength": np.array([0.5, 0.7, 0.9, 1.1, 1.2]),
        "optics_throughput": np.array([0.8, 0.8, 0.8, 0.8, 0.8]),
        "qe": np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
        "dqe": np.array([0.75, 0.75, 0.75, 0.75, 0.75]),
    }

    return {
        "diameter": 8.0,  # must match telescope diameter (in meters) to avoid pixscale recompute warning
        "temperature": 290.0,
        "IMAGER": {
            "VIS": {
                "pixscale_mas": 10.0,
                "dc": 3e-5,
                "rn": 0.1,
                "cic": 0.0,
                "wavelength_range": (0.45, 0.55),
                "spectral": spectral_imager,
            },
        },
        "IFS": {
            "VIS": {
                "pixscale_mas": 10.0,
                "dc": 3e-5,
                "rn": 0.1,
                "cic": 0.0,
                "wavelength_range": (0.5, 1.2),
                "spectral": spectral_ifs,
            },
        },
    }


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
    assert detector.keyword is "ToyModel"


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
def test_eac_detector_load_configuration_imager_basic(
    fake_eac_config,
    imager_eac_detector_parameters,
):
    """Test basic EACDetector configuration loading in IMAGER mode."""
    parameters = imager_eac_detector_parameters.copy()
    detector = EACDetector(keyword="EAC1")
    mediator = MockMediator(
        "IMAGER",
        eac_config=fake_eac_config,
        active_channel="VIS",
    )

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


# ============================================================================
# Tests for EACDetector.load_configuration - IFS mode
# ============================================================================


def test_eac_detector_load_configuration_ifs_basic(
    fake_eac_config,
    ifs_eac_detector_parameters,
):
    """Test basic EACDetector configuration loading in IFS mode."""

    detector = EACDetector(keyword="EAC1")
    parameters = ifs_eac_detector_parameters.copy()
    mediator = MockMediator(
        "IFS",
        eac_config=fake_eac_config,
        active_channel="VIS",
    )

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


# # ============================================================================
# # Tests for EACDetector validation inputs
# # ============================================================================


@pytest.mark.parametrize("observing_mode", ["IMAGER", "IFS"])
def test_eac_detector_etc_validation_inputs(
    observing_mode,
    fake_eac_config,
    ifs_eac_detector_parameters,
    imager_eac_detector_parameters,
):
    """Test that ETC validation inputs are correctly loaded."""
    detector = EACDetector(keyword="EAC1")
    mediator = MockMediator(
        observing_mode,
        eac_config=fake_eac_config,
        active_channel="VIS",
    )
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
# Regression tests for the DEFAULT_CONFIG shared-mutable-class-attribute fix
# ============================================================================


def test_toy_model_detector_default_config_not_shared_class_attribute():
    """Direct identity check: self.DEFAULT_CONFIG must be a distinct object
    from the class-level dict immediately after __init__, and mutating one
    instance's copy must not affect the class attribute or sibling
    instances. This works regardless of load_configuration() internals."""
    detector = ToyModelDetector()

    assert detector.DEFAULT_CONFIG is not ToyModelDetector.DEFAULT_CONFIG

    detector.DEFAULT_CONFIG["DC"] = [999.0] * DARK_CURRENT

    other = ToyModelDetector()
    assert ToyModelDetector.DEFAULT_CONFIG["DC"] == [3e-5] * DARK_CURRENT
    assert other.DEFAULT_CONFIG["DC"] == [3e-5] * DARK_CURRENT


def test_toy_model_detector_default_config_leak_ifs_then_imager():
    """True reproduction of the originally reported bug: an IFS-mode
    instance (nlambda=3) run BEFORE an IMAGER-mode instance (nlambda=1),
    where NEITHER instance supplies the array-valued keys explicitly --
    both rely on parse_input.normalize_list_shapes() resizing
    DEFAULT_CONFIG's hardcoded scalar defaults to the current wavelength
    grid length.

    Before the copy.deepcopy() fix, the IFS call would resize the *shared
    class-level* DEFAULT_CONFIG arrays up to length 3; the later IMAGER
    call would then find DEFAULT_CONFIG already at length 3 and fail to
    shrink it back to length 1 (matching the reported
    "DC has length 3 but the expected input size is 1" error).
    """
    mediator_ifs = MockMediator("IFS")
    detector_ifs = ToyModelDetector()
    detector_ifs.load_configuration(
        {
            "wavelength": mediator_ifs.get_observation_parameter("wavelength"),
            "observing_mode": "IFS",
        },  # array_params keys deliberately omitted
        mediator_ifs,
    )

    mediator_imager = MockMediator("IMAGER")
    detector_imager = ToyModelDetector()
    detector_imager.load_configuration(
        {
            "wavelength": mediator_imager.get_observation_parameter("wavelength"),
            "observing_mode": "IMAGER",
        },  # array_params keys deliberately omitted
        mediator_imager,
    )

    n_ifs = len(mediator_ifs.get_observation_parameter("wavelength"))
    n_imager = len(mediator_imager.get_observation_parameter("wavelength"))

    assert detector_ifs.DC.shape == (n_ifs,)
    assert detector_imager.DC.shape == (n_imager,)  # would be (n_ifs,) if leaking

    # class-level dict must remain at its original, un-resized hardcoded default
    assert len(ToyModelDetector.DEFAULT_CONFIG["DC"]) == 1


def test_eac_detector_default_config_not_shared_class_attribute():
    """Direct identity check for EACDetector. Unlike ToyModelDetector,
    EACDetector's load_configuration() never resizes a *stale* previous
    value -- every array key (QE, dQE, DC, RN, CIC, pixscale_mas) is
    unconditionally recomputed fresh from the current call's eac_config/
    active_channel/wavelength grid, and `tread`'s shape always comes from
    the current call's wavelength array too. So there is no length-leak
    symptom to reproduce via load_configuration() the way there is for
    ToyModelDetector -- this identity check is the only test capable of
    actually discriminating the fix for this class."""
    detector = EACDetector(keyword="EAC1")

    assert detector.DEFAULT_CONFIG is not EACDetector.DEFAULT_CONFIG

    detector.DEFAULT_CONFIG["DC"] = [999.0] * DARK_CURRENT

    other = EACDetector(keyword="EAC2")
    assert EACDetector.DEFAULT_CONFIG["DC"] is None
    assert other.DEFAULT_CONFIG["DC"] is None


# ============================================================================
# Correctness test (NOT a regression test for the shared-mutable-attribute
# bug -- see test_eac_detector_default_config_not_shared_class_attribute
# above for that). This just confirms two EACDetector instances loaded with
# different eac_configs/active_channels produce independent, correct
# attribute values. Since EACDetector always unconditionally recomputes its
# array-valued keys from fresh data each call, this test would pass even
# without the copy.deepcopy() fix -- it's kept for general correctness
# coverage, not bug-regression coverage.
# ============================================================================


def test_eac_detector_load_configuration_independent_per_call():
    """Two EACDetector instances loaded with different eac_configs produce
    independent, correct pixscale_mas/DC/RN/CIC/QE/dQE values."""

    eac_config_1 = {
        "diameter": 8.0,
        "temperature": 290.0,
        "IMAGER": {
            "vis": {
                "dc": 3e-5,
                "rn": 0.1,
                "cic": 0.0,
                "pixscale_mas": 10.0,
                "wavelength_range": (0.4, 0.6),
                "spectral": {
                    "wavelength": np.array([0.4, 0.5, 0.6]),
                    "optics_throughput": np.array([0.8, 0.8, 0.8]),
                    "qe": np.array([0.9, 0.9, 0.9]),
                    "dqe": np.array([0.75, 0.75, 0.75]),
                },
            }
        },
        "IFS": {},
    }

    eac_config_2 = {
        "diameter": 8.0,
        "temperature": 290.0,
        "IMAGER": {
            "vis": {
                "dc": 6e-5,
                "rn": 0.5,
                "cic": 1.0,
                "pixscale_mas": 25.0,
                "wavelength_range": (0.4, 0.6),
                "spectral": {
                    "wavelength": np.array([0.4, 0.5, 0.6]),
                    "optics_throughput": np.array([0.7, 0.7, 0.7]),
                    "qe": np.array([0.5, 0.5, 0.5]),
                    "dqe": np.array([0.6, 0.6, 0.6]),
                },
            }
        },
        "IFS": {},
    }

    mediator1 = MockMediator("IMAGER", eac_config=eac_config_1, active_channel="vis")
    detector1 = EACDetector(keyword="EAC1")
    detector1.load_configuration(
        {"wavelength": 0.5, "observing_mode": "IMAGER"}, mediator1
    )

    mediator2 = MockMediator("IMAGER", eac_config=eac_config_2, active_channel="vis")
    detector2 = EACDetector(keyword="EAC2")
    detector2.load_configuration(
        {"wavelength": 0.5, "observing_mode": "IMAGER"}, mediator2
    )

    # Instance-level values must reflect their own eac_config, not leak from the other
    assert np.isclose(detector1.pixscale_mas.value, 10.0)
    assert np.all(detector1.DC.value == 3e-5)
    assert np.all(detector1.RN.value == 0.1)
    assert np.all(detector1.CIC.value == 0.0)
    assert np.all(detector1.QE.value == 0.9)
    assert np.all(detector1.dQE.value == 0.75)

    assert np.isclose(detector2.pixscale_mas.value, 25.0)
    assert np.all(detector2.DC.value == 6e-5)
    assert np.all(detector2.RN.value == 0.5)
    assert np.all(detector2.CIC.value == 1.0)
    assert np.all(detector2.QE.value == 0.5)
    assert np.all(detector2.dQE.value == 0.6)


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
