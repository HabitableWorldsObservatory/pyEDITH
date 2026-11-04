import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock
from astropy import units as u
from pathlib import Path
from pyEDITH.observatory import Observatory, ObservatoryMediator
from pyEDITH.telescopes import Telescope, ToyModelTelescope, EACTelescope
from pyEDITH.detectors import Detector, ToyModelDetector, EACDetector
from pyEDITH.coronagraphs import (
    Coronagraph,
    ToyModelCoronagraph,
    CoronagraphYIP,
)
from pyEDITH.observation import Observation
from pyEDITH.astrophysical_scene import AstrophysicalScene
from pyEDITH.units import (
    PHOTON_FLUX_DENSITY,
    DIMENSIONLESS,
    LENGTH,
    LAMBDA_D,
    ARCSEC,
    WAVELENGTH,
    TEMPERATURE,
    DARK_CURRENT,
    READ_NOISE,
    READ_TIME,
    TIME,
    FRAME,
    CLOCK_INDUCED_CHARGE,
    QUANTUM_EFFICIENCY,
    ELECTRON,
    PHOTON_COUNT,
    INV_SQUARE_ARCSEC,
    PIXEL,
    ZODI,
)
from pyEDITH.filters import Filter
from yippy import Coronagraph as yippycoro
import logging
from pyEDITH.observatory import DEFAULT_OPTICS_THROUGHPUT

# ============================================================================
# Mock Helper Functions
# ============================================================================


def create_mock_telescope():
    """Create a mock telescope for unit testing in IMAGER mode."""
    mock_tel = MagicMock(spec=Telescope)

    # Basic attributes
    mock_tel.diameter = 7.87 * LENGTH
    mock_tel.Area = 42.75906827 * LENGTH**2
    mock_tel.unobscured_area = 0.879

    # Overhead times
    mock_tel.toverhead_fixed = 8381.3 * TIME
    mock_tel.toverhead_multi = 1.1 * DIMENSIONLESS

    # Optical properties (nlambda=1 for IMAGER mode)
    mock_tel.telescope_optical_throughput = np.array([0.823]) * DIMENSIONLESS

    # Thermal properties
    mock_tel.temperature = 290.0 * TEMPERATURE
    mock_tel.T_contamination = 0.95 * DIMENSIONLESS

    # Methods
    mock_tel.validate_configuration = MagicMock()

    return mock_tel


def create_mock_detector():
    """Create a mock detector for unit testing in IMAGER mode."""
    mock_det = MagicMock(spec=Detector)

    # Basic attributes (nlambda=1 for IMAGER mode)
    mock_det.pixscale_mas = 0.25 * LAMBDA_D
    mock_det.npix_multiplier = np.array([1.0]) * DIMENSIONLESS

    # Detector noise characteristics
    mock_det.DC = np.array([3e-5]) * DARK_CURRENT
    mock_det.RN = np.array([0.0]) * READ_NOISE
    mock_det.tread = np.array([1000.0]) * READ_TIME
    mock_det.CIC = np.array([1.3e-3]) * CLOCK_INDUCED_CHARGE

    # Quantum efficiency
    mock_det.QE = np.array([0.897]) * QUANTUM_EFFICIENCY
    mock_det.dQE = np.array([0.75]) * DIMENSIONLESS

    # Methods
    mock_det.validate_configuration = MagicMock()

    return mock_det


def create_mock_coronagraph():
    """Create a mock coronagraph for unit testing in IMAGER mode."""
    mock_coro = MagicMock(spec=Coronagraph)

    # Basic attributes
    mock_coro.pixscale = 30.0 * LAMBDA_D
    mock_coro.npix = 4
    mock_coro.xcenter = 2.0 * PIXEL
    mock_coro.ycenter = 2.0 * PIXEL
    mock_coro.coronagraph_bandwidth = 0.2
    mock_coro.nrolls = 1
    mock_coro.npsfratios = 1

    # Coronagraph optical properties
    mock_coro.coronagraph_optical_throughput = np.array([0.44]) * DIMENSIONLESS
    mock_coro.coronagraph_spectral_resolution = 1.0 * DIMENSIONLESS

    # Parameters
    mock_coro.contrast = 1.05e-13 * DIMENSIONLESS
    mock_coro.noisefloor_factor = 0.03 * DIMENSIONLESS
    mock_coro.Tcore = 0.2968371 * DIMENSIONLESS
    mock_coro.TLyot = 0.65 * DIMENSIONLESS
    mock_coro.PSFpeak = 0.01625 * DIMENSIONLESS

    # 2D arrays for spatial maps
    mock_coro.r = (
        np.array(
            [
                [63.63961031, 47.4341649, 47.4341649, 63.63961031],
                [47.4341649, 21.21320344, 21.21320344, 47.4341649],
                [47.4341649, 21.21320344, 21.21320344, 47.4341649],
                [63.63961031, 47.4341649, 47.4341649, 63.63961031],
            ]
        )
        * LAMBDA_D
    )

    mock_coro.skytrans = np.full((4, 4), 0.65) * DIMENSIONLESS
    mock_coro.Istar = np.full((4, 4), 1.70625e-15) * DIMENSIONLESS
    mock_coro.noisefloor = np.full((4, 4), 5.11875e-17) * DIMENSIONLESS

    # 3D arrays with psf_trunc_ratio dimension
    mock_coro.omega_lod = np.full((4, 4, 1), 2.26980069) * LAMBDA_D**2
    mock_coro.photometric_aperture_throughput = (
        np.array(
            [
                [[0.0], [0.2968371], [0.2968371], [0.0]],
                [[0.2968371], [0.2968371], [0.2968371], [0.2968371]],
                [[0.2968371], [0.2968371], [0.2968371], [0.2968371]],
                [[0.0], [0.2968371], [0.2968371], [0.0]],
            ]
        )
        * DIMENSIONLESS
    )

    # Photometric aperture (one or the other should be None)
    mock_coro.photometric_aperture_radius = 0.7 * LAMBDA_D
    mock_coro.psf_trunc_ratio = 0.3 * DIMENSIONLESS

    # Methods
    mock_coro.validate_configuration = MagicMock()

    return mock_coro


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_observatory():
    """Fixture providing a mock observatory with components."""
    obs = Observatory()
    obs.telescope = create_mock_telescope()
    obs.detector = create_mock_detector()
    obs.coronagraph = create_mock_coronagraph()
    return obs


@pytest.fixture
def mock_observation_imager():
    """Fixture providing a mock observation."""
    obs = Observation()
    obs.observing_mode = "IMAGER"
    obs.td_limit = 1.0e20 * u.s
    obs.wavelength = u.Quantity([0.5], u.micron)
    obs.SNR = u.Quantity([7], DIMENSIONLESS)
    obs.CRb_multiplier = 2
    obs.tp = 0.0 * u.s
    obs.exptime = u.Quantity([0.0], u.s)
    obs.fullsnr = u.Quantity([0.0], DIMENSIONLESS)
    return obs


@pytest.fixture
def mock_observation_ifs():
    """Fixture providing a mock observation."""
    obs = Observation()
    obs.observing_mode = "IFS"
    obs.td_limit = 1.0e20 * u.s
    obs.wavelength = u.Quantity([0.5, 0.6], u.micron)
    obs.SNR = u.Quantity([7, 7], DIMENSIONLESS)
    obs.CRb_multiplier = 2
    obs.exptime = u.Quantity([0.0, 0.0], u.s)
    obs.fullsnr = u.Quantity([0.0, 0.0], DIMENSIONLESS)
    return obs


@pytest.fixture
def mock_scene():
    """Fixture providing a mock astrophysical scene."""
    scene = AstrophysicalScene()
    scene.F0V = 10374.9964895 * u.photon / u.nm / u.s / u.cm**2
    scene.dist = 14.8 * u.pc
    scene.F0 = u.Quantity([12638.83670769], u.photon / u.nm / u.s / u.cm**2)
    scene.vmag = 5.84 * u.mag
    scene.mag = u.Quantity([6.189576], u.mag)
    scene.deltamag = u.Quantity([25.5], u.mag)
    scene.min_deltamag = 25 * u.mag
    scene.Fs_over_F0 = u.Quantity([0.00334326], DIMENSIONLESS)
    scene.Fp_over_Fs = u.Quantity([6.30957344e-11], DIMENSIONLESS)
    scene.Fp_min_over_Fs = 1.0e-10 * DIMENSIONLESS
    scene.stellar_angular_diameter_arcsec = 0.01 * ARCSEC
    scene.nzodis = 3 * ZODI
    scene.ra = 236.00757737 * u.deg
    scene.dec = 2.51516683 * u.deg
    scene.separation = 0.0628 * u.arcsec
    scene.xp = 0.0628 * u.arcsec
    scene.yp = 0.0 * u.arcsec
    scene.M_V = 4.98869142 * u.mag
    scene.Fzodi_list = (u.Quantity([6.11055505e-10], 1 / u.arcsec**2),)
    scene.Fexozodi_list = (u.Quantity([2.97724302e-09], 1 / u.arcsec**2),)
    scene.Fbinary_list = u.Quantity([0], DIMENSIONLESS)
    return scene


@pytest.fixture
def configured_mock_observatory(mock_observatory, mock_observation_imager, mock_scene):
    """Fixture providing a fully configured mock observatory."""
    # Mock observatory already has properly initialized components from create_mock_* functions
    # No need to call load_configuration with empty dicts

    # Set observatory-level parameters that would be set by load_configuration
    mock_observatory.observing_mode = "IMAGER"
    mock_observatory.optics_throughput = np.array([0.8]) * DIMENSIONLESS
    mock_observatory.total_throughput = np.array([0.6]) * QUANTUM_EFFICIENCY
    mock_observatory.epswarmTrcold = np.array([0.2]) * DIMENSIONLESS

    return mock_observatory


# ============================================================================
# Tests for Observatory Preset List
# ============================================================================


def test_observatory_presets_exist():
    """Test that Observatory has defined presets."""
    assert hasattr(Observatory, "PRESETS")
    assert isinstance(Observatory.PRESETS, dict)
    assert len(Observatory.PRESETS) > 0


def test_observatory_presets_contain_required_keys():
    """Test that each preset contains required component keys."""
    required_keys = {"telescope", "coronagraph", "detector"}

    for preset_name, preset_config in Observatory.PRESETS.items():
        assert required_keys.issubset(
            preset_config.keys()
        ), f"Preset {preset_name} missing required keys"


def test_observatory_toymodel_preset_definition():
    """Test ToyModel preset is properly defined."""
    assert "ToyModel" in Observatory.PRESETS
    preset = Observatory.PRESETS["ToyModel"]

    assert preset["telescope"] == "ToyModel"
    assert preset["coronagraph"] == "ToyModel"
    assert preset["detector"] == "ToyModel"


def test_observatory_eac1_preset_definition():
    """Test EAC1 preset is properly defined."""
    assert "EAC1" in Observatory.PRESETS
    preset = Observatory.PRESETS["EAC1"]

    assert preset["telescope"] == "EAC1"
    assert preset["detector"] == "EAC1"
    # Coronagraph should be defined but we don't enforce specific value


def test_observatory_eac5_preset_definition():
    """Test EAC5 preset is properly defined."""
    assert "EAC5" in Observatory.PRESETS
    preset = Observatory.PRESETS["EAC5"]

    assert preset["telescope"] == "EAC5"
    assert preset["detector"] == "EAC5"


# ============================================================================
# Tests for Observatory initialization
# ============================================================================


def test_observatory_init():
    """Test that Observatory initializes with None components."""
    obs = Observatory()

    assert obs.telescope is None
    assert obs.detector is None
    assert obs.coronagraph is None


# ============================================================================
# Tests for Observatory.create_observatory with Presets
# ============================================================================


def test_create_observatory_toymodel_preset():
    """Test creating observatory with ToyModel preset."""
    obs = Observatory()
    obs.create_observatory("ToyModel")

    assert hasattr(obs, "telescope")
    assert hasattr(obs, "coronagraph")
    assert hasattr(obs, "detector")
    assert obs.telescope is not None
    assert obs.coronagraph is not None
    assert obs.detector is not None
    assert isinstance(obs.telescope, ToyModelTelescope)
    assert isinstance(obs.coronagraph, ToyModelCoronagraph)
    assert isinstance(obs.detector, ToyModelDetector)


def test_create_observatory_eac1_preset():
    """Test creating observatory with EAC1 preset."""

    obs = Observatory()
    obs.create_observatory("EAC1")

    assert hasattr(obs, "telescope")
    assert hasattr(obs, "coronagraph")
    assert hasattr(obs, "detector")
    assert obs.telescope is not None
    assert obs.coronagraph is not None
    assert obs.detector is not None
    assert isinstance(obs.telescope, EACTelescope)
    assert isinstance(obs.coronagraph, CoronagraphYIP)
    assert isinstance(obs.detector, EACDetector)


def test_create_observatory_eac5_preset():
    """Test creating observatory with EAC5 preset."""

    obs = Observatory()
    obs.create_observatory("EAC5")

    assert hasattr(obs, "telescope")
    assert hasattr(obs, "coronagraph")
    assert hasattr(obs, "detector")
    assert obs.telescope is not None
    assert obs.coronagraph is not None
    assert obs.detector is not None
    assert isinstance(obs.telescope, EACTelescope)
    assert isinstance(obs.coronagraph, CoronagraphYIP)
    assert isinstance(obs.detector, EACDetector)


def test_create_observatory_invalid_preset():
    """Test that invalid preset raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Unknown preset: InvalidPreset\. Available presets: \['ToyModel', 'EAC1', 'EAC5'\]",
    ):
        obs = Observatory()
        obs.create_observatory("InvalidPreset")


# ============================================================================
# Tests for Observatory.create_observatory with Custom Config
# ============================================================================


def test_create_observatory_custom_config():
    """Test creating observatory with custom configuration dictionary."""
    config = {
        "telescope": "ToyModel",
        "coronagraph": "ToyModel",
        "detector": "ToyModel",
    }

    obs = Observatory()
    obs.create_observatory(config)

    assert obs.telescope is not None
    assert obs.coronagraph is not None
    assert obs.detector is not None
    assert isinstance(obs.telescope, ToyModelTelescope)
    assert isinstance(obs.coronagraph, ToyModelCoronagraph)
    assert isinstance(obs.detector, ToyModelDetector)


def test_create_observatory_mixed_custom_config():
    """Test creating observatory with mixed component types."""

    config = {
        "telescope": "EAC1",
        "coronagraph": "ToyModel",
        "detector": "EAC1",
    }

    obs = Observatory()
    obs.create_observatory(config)

    assert obs.telescope is not None
    assert obs.coronagraph is not None
    assert obs.detector is not None
    assert isinstance(obs.telescope, EACTelescope)
    assert isinstance(obs.coronagraph, ToyModelCoronagraph)
    assert isinstance(obs.detector, EACDetector)


def test_create_observatory_custom_config_missing_keys():
    """Test that missing required keys raises ValueError."""
    incomplete_config = {"telescope": "ToyModel"}

    with pytest.raises(
        ValueError, match="Config missing required keys\: \['coronagraph', 'detector'\]"
    ):
        obs = Observatory()
        obs.create_observatory(incomplete_config)


def test_create_observatory_invalid_config_type():
    """Test that invalid configuration type raises ValueError."""
    with pytest.raises(ValueError, match="Invalid configuration"):
        obs = Observatory()
        obs.create_observatory(123)


# ============================================================================
# Tests for Observatory._create_telescope
# ============================================================================


def test_create_telescope_toymodel():
    """Test creating ToyModel telescope."""
    telescope = Observatory._create_telescope("ToyModel")
    assert isinstance(telescope, ToyModelTelescope)
    assert telescope is not None


def test_create_telescope_eac():
    """Test creating EAC telescope."""
    telescope = Observatory._create_telescope("EAC1")
    assert telescope is not None
    assert isinstance(telescope, EACTelescope)


def test_create_telescope_invalid_keyword():
    """Test that invalid telescope keyword raises ValueError."""
    with pytest.raises(
        ValueError,
        match="Unknown telescope type: InvalidTelescope\. Expected 'ToyModel' or 'EAC\*'",
    ):
        Observatory._create_telescope("InvalidTelescope")


# ============================================================================
# Tests for Observatory._create_detector
# ============================================================================


def test_create_detector_toymodel():
    """Test creating ToyModel detector."""
    detector = Observatory._create_detector("ToyModel")

    assert detector is not None
    assert isinstance(detector, ToyModelDetector)


def test_create_detector_eac():
    """Test creating EAC detector."""
    detector = Observatory._create_detector("EAC1")

    assert detector is not None
    assert isinstance(detector, EACDetector)


def test_create_detector_invalid_keyword():
    """Test that invalid detector keyword raises ValueError."""
    with pytest.raises(
        ValueError,
        match="Unknown detector type: InvalidDetector. Expected 'ToyModel' or 'EAC\\*'",
    ):
        Observatory._create_detector("InvalidDetector")


# ============================================================================
# Tests for Observatory._create_coronagraph Priority System
# ============================================================================


def test_create_coronagraph_toymodel():
    """Test creating ToyModel coronagraph."""
    coro = Observatory._create_coronagraph("ToyModel")

    assert coro is not None
    assert isinstance(coro, ToyModelCoronagraph)


@patch("pyEDITH.observatory.coronagraphs.CoronagraphYIP")
def test_create_coronagraph_priority0_yippy_object(mock_coro_yip):
    """Test Priority 0: Pre-constructed yippy Coronagraph object."""
    # Use spec to make isinstance() work correctly
    mock_yippy_instance = MagicMock(spec=yippycoro)

    mock_coro_yip_instance = MagicMock()
    mock_coro_yip.return_value = mock_coro_yip_instance

    coro = Observatory._create_coronagraph(mock_yippy_instance)

    assert coro is not None
    mock_coro_yip.assert_called_once_with(yippy_coro=mock_yippy_instance)


@patch("os.path.exists")
@patch("pathlib.Path.exists")
@patch("pyEDITH.observatory.coronagraphs.CoronagraphYIP")
def test_create_coronagraph_priority1_explicit_path(
    mock_coro_yip, mock_path_exists, mock_os_exists
):
    """Test Priority 1: Direct path that exists."""
    mock_path_exists.return_value = True
    mock_os_exists.return_value = True

    mock_coro_yip_instance = MagicMock()
    mock_coro_yip.return_value = mock_coro_yip_instance

    test_path = "/explicit/path/to/coronagraph"
    coro = Observatory._create_coronagraph(test_path)

    assert coro is not None
    mock_coro_yip.assert_called_once_with(path=Path(test_path))


@patch("os.path.exists")
@patch("os.environ.get")
@patch("pyEDITH.observatory.coronagraphs.CoronagraphYIP")
def test_create_coronagraph_priority2_yip_coro_dir(
    mock_coro_yip, mock_env_get, mock_exists
):
    """Test Priority 2: Check YIP_CORO_DIR environment variable."""

    def exists_side_effect(path):
        return "/yip_dir/test_coro" in str(path)

    mock_exists.side_effect = exists_side_effect
    mock_env_get.return_value = "/yip_dir"

    mock_coro_yip_instance = MagicMock()
    mock_coro_yip.return_value = mock_coro_yip_instance

    coro = Observatory._create_coronagraph("test_coro")

    assert coro is not None

    mock_coro_yip.assert_called_once_with(path=Path("/yip_dir/test_coro"))


@patch("os.path.exists")
@patch("pathlib.Path.exists")
@patch("os.environ.get")
@patch("pyEDITH.observatory.fetch_yip")
@patch("pyEDITH.observatory.coronagraphs.CoronagraphYIP")
def test_create_coronagraph_priority3_remote_fetch(
    mock_coro_yip, mock_fetch, mock_env_get, mock_path_exists, mock_os_exists
):
    """Test Priority 3: Remote fetch from database."""

    mock_path_exists.return_value = False
    mock_os_exists.return_value = False
    mock_env_get.return_value = None
    mock_fetch.return_value = "/downloaded/path/coronagraph"

    mock_coro_yip_instance = MagicMock()
    mock_coro_yip.return_value = mock_coro_yip_instance
    coro = Observatory._create_coronagraph("remote_coronagraph")

    assert coro is not None

    mock_fetch.assert_called_once_with("remote_coronagraph")

    mock_coro_yip.assert_called_once_with(path=Path("/downloaded/path/coronagraph"))


@patch("os.path.exists")
@patch("pathlib.Path.exists")
@patch("os.environ.get")
@patch("pyEDITH.observatory.fetch_yip")
def test_create_coronagraph_not_found_raises_error(
    mock_fetch, mock_env_get, mock_path_exists, mock_os_exists
):
    """Test that nonexistent coronagraph raises FileNotFoundError."""

    mock_path_exists.return_value = False
    mock_os_exists.return_value = False
    mock_env_get.return_value = None
    mock_fetch.side_effect = Exception("Not found in database")

    with pytest.raises(FileNotFoundError, match="Could not find or fetch coronagraph"):
        Observatory._create_coronagraph("nonexistent_coronagraph")


@patch("os.path.exists")
@patch("pathlib.Path.exists")
@patch("os.environ.get")
@patch("pyEDITH.observatory.fetch_yip")
def test_create_coronagraph_error_provides_solutions(
    mock_fetch, mock_env_get, mock_path_exists, mock_os_exists
):
    """Test that error message provides helpful solutions."""

    mock_path_exists.return_value = False
    mock_os_exists.return_value = False
    mock_env_get.return_value = "/yip_dir"
    mock_fetch.side_effect = Exception("Network error")

    with pytest.raises(FileNotFoundError, match="Solutions:"):
        Observatory._create_coronagraph("test_coro")


# ============================================================================
# Tests for Observatory.validate_configuration
# ============================================================================


def test_observatory_validate_configuration_valid(configured_mock_observatory):
    """Test that validation passes with valid configuration."""
    configured_mock_observatory.optics_throughput = [0.8] * DIMENSIONLESS
    configured_mock_observatory.total_throughput = [0.6] * QUANTUM_EFFICIENCY
    configured_mock_observatory.epswarmTrcold = [0.2] * DIMENSIONLESS

    # Should not raise any exception
    configured_mock_observatory.validate_configuration()


def test_observatory_validate_configuration_missing_attribute(
    configured_mock_observatory,
):
    """Test that missing attribute raises AttributeError."""
    configured_mock_observatory.optics_throughput = [0.8] * DIMENSIONLESS
    configured_mock_observatory.total_throughput = [0.6] * QUANTUM_EFFICIENCY
    configured_mock_observatory.epswarmTrcold = [0.2] * DIMENSIONLESS
    delattr(configured_mock_observatory, "optics_throughput")

    with pytest.raises(
        AttributeError, match="Observatory is missing attribute: optics_throughput"
    ):
        configured_mock_observatory.validate_configuration()


def test_observatory_validate_configuration_not_quantity(configured_mock_observatory):
    """Test that non-Quantity attribute raises TypeError."""
    configured_mock_observatory.optics_throughput = 0.8
    configured_mock_observatory.total_throughput = [0.6] * QUANTUM_EFFICIENCY
    configured_mock_observatory.epswarmTrcold = [0.2] * DIMENSIONLESS

    with pytest.raises(
        TypeError, match="Observatory attribute optics_throughput should be a Quantity"
    ):
        configured_mock_observatory.validate_configuration()


def test_observatory_validate_configuration_incorrect_units(
    configured_mock_observatory,
):
    """Test that incorrect units raise ValueError."""
    configured_mock_observatory.optics_throughput = [0.8] * u.meter
    configured_mock_observatory.total_throughput = [0.6] * QUANTUM_EFFICIENCY
    configured_mock_observatory.epswarmTrcold = [0.2] * DIMENSIONLESS

    with pytest.raises(
        ValueError, match="Observatory attribute optics_throughput has incorrect units"
    ):
        configured_mock_observatory.validate_configuration()


# ============================================================================
# Tests for Observatory.calculate_optics_throughput
# ============================================================================


def test_calculate_optics_throughput_with_t_optical(
    configured_mock_observatory, mock_observation_imager, mock_scene
):
    """Test calculation of optics throughput with explicit T_optical parameter."""
    parameters = {"T_optical": 0.8, "observing_mode": "IMAGER", "wavelength": 0.5}
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation_imager, mock_scene
    )

    configured_mock_observatory.calculate_optics_throughput(parameters, mediator)

    assert configured_mock_observatory.optics_throughput.value == [0.8]


def test_calculate_optics_throughput_ifs_mode(
    configured_mock_observatory, mock_observation_ifs, mock_scene
):
    """Test calculation of optics throughput in IFS mode with IFS efficiency."""
    parameters = {
        "T_optical": [0.8, 0.84],
        "observing_mode": "IFS",
        "IFS_eff": [0.9, 0.92],
        "wavelength": [0.5, 0.6],
    }
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation_ifs, mock_scene
    )

    configured_mock_observatory.calculate_optics_throughput(parameters, mediator)

    assert np.allclose(
        configured_mock_observatory.optics_throughput.value,
        [
            0.8 * 0.9,
            0.84 * 0.92,
        ],
        rtol=1e-5,
    )


def test_calculate_optics_throughput_from_components(
    configured_mock_observatory, mock_observation_imager, mock_scene, caplog
):
    """Test calculation of optics throughput from component throughputs."""
    parameters = {"observing_mode": "IMAGER", "wavelength": 0.5}
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation_imager, mock_scene
    )

    configured_mock_observatory.calculate_optics_throughput(parameters, mediator)

    expected = [
        configured_mock_observatory.telescope.telescope_optical_throughput.value[0]
        * configured_mock_observatory.coronagraph.coronagraph_optical_throughput.value[
            0
        ]
    ]
    assert configured_mock_observatory.optics_throughput.value == expected


# ============================================================================
# Tests for Observatory.calculate_warmemissivity_coldtransmission
# ============================================================================


def test_calculate_warmemissivity_coldtransmission_explicit(
    configured_mock_observatory, mock_observation_imager, mock_scene
):
    """Test calculation with explicit epswarmTrcold parameter."""
    parameters = {"epswarmTrcold": 0.3, "wavelength": 0.5}
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation_imager, mock_scene
    )

    configured_mock_observatory.calculate_warmemissivity_coldtransmission(
        parameters, mediator
    )

    assert configured_mock_observatory.epswarmTrcold.value == 0.3


def test_calculate_warmemissivity_coldtransmission_calculated(
    configured_mock_observatory, mock_observation_imager, mock_scene
):
    """Test calculation derived from optics throughput."""
    parameters = {"wavelength": 0.5}
    configured_mock_observatory.optics_throughput = [0.8] * DIMENSIONLESS
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation_imager, mock_scene
    )

    configured_mock_observatory.calculate_warmemissivity_coldtransmission(
        parameters, mediator
    )

    assert configured_mock_observatory.epswarmTrcold.value == 1 - 0.8


# ============================================================================
# Tests for Observatory.calculate_total_throughput
# ============================================================================


def test_calculate_total_throughput(mock_observatory):
    """Test calculation of total system throughput."""
    mock_observatory.optics_throughput = [0.8] * DIMENSIONLESS
    mock_observatory.detector.dQE = [0.9] * DIMENSIONLESS
    mock_observatory.detector.QE = [0.9] * QUANTUM_EFFICIENCY
    mock_observatory.telescope.T_contamination = 0.95 * DIMENSIONLESS

    mock_observatory.calculate_total_throughput()

    expected = 0.8 * 0.9 * 0.9 * 0.95
    assert np.isclose(mock_observatory.total_throughput.value, expected)
    assert mock_observatory.total_throughput.unit == QUANTUM_EFFICIENCY


# ============================================================================
# Tests for Observatory.load_configuration
# ============================================================================


def test_observatory_load_configuration(
    mock_observatory, mock_observation_imager, mock_scene
):
    """Test loading complete observatory configuration."""
    parameters = {"observing_mode": "IMAGER", "T_optical": 0.8, "wavelength": 0.5}

    mock_observatory.load_configuration(parameters, mock_observation_imager, mock_scene)

    assert mock_observatory.observing_mode == "IMAGER"
    assert mock_observatory.optics_throughput.value == [0.8]
    assert hasattr(mock_observatory, "epswarmTrcold")
    assert hasattr(mock_observatory, "total_throughput")


# ============================================================================
# Tests for ObservatoryMediator.get_telescope_parameter
# ============================================================================


def test_observatory_mediator_get_telescope_parameter(
    configured_mock_observatory, mock_observation_imager, mock_scene
):
    """Test getting telescope parameter through mediator."""
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation_imager, mock_scene
    )

    result = mediator.get_telescope_parameter("diameter")

    assert result == configured_mock_observatory.telescope.diameter


def test_observatory_mediator_get_telescope_parameter_nonexistent(
    configured_mock_observatory, mock_observation_imager, mock_scene
):
    """Test getting non-existent telescope parameter returns None."""
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation_imager, mock_scene
    )

    result = mediator.get_telescope_parameter("nonexistent")

    assert result is None


# ============================================================================
# Tests for ObservatoryMediator.get_coronagraph_parameter
# ============================================================================


def test_observatory_mediator_get_coronagraph_parameter(
    configured_mock_observatory, mock_observation_imager, mock_scene
):
    """Test getting coronagraph parameter through mediator."""
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation_imager, mock_scene
    )

    result = mediator.get_coronagraph_parameter("contrast")

    assert result == configured_mock_observatory.coronagraph.contrast


# ============================================================================
# Tests for ObservatoryMediator.get_detector_parameter
# ============================================================================


def test_observatory_mediator_get_detector_parameter(
    configured_mock_observatory, mock_observation_imager, mock_scene
):
    """Test getting detector parameter through mediator."""
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation_imager, mock_scene
    )

    result = mediator.get_detector_parameter("pixscale_mas")

    assert result == configured_mock_observatory.detector.pixscale_mas


# ============================================================================
# Tests for ObservatoryMediator.get_observation_parameter
# ============================================================================


def test_observatory_mediator_get_observation_parameter(
    configured_mock_observatory, mock_observation_imager, mock_scene
):
    """Test getting observation parameter through mediator."""
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation_imager, mock_scene
    )

    result = mediator.get_observation_parameter("wavelength")

    assert result == mock_observation_imager.wavelength


# ============================================================================
# Tests for ObservatoryMediator.get_scene_parameter
# ============================================================================


def test_observatory_mediator_get_scene_parameter(
    configured_mock_observatory, mock_observation_imager, mock_scene
):
    """Test getting scene parameter through mediator."""
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation_imager, mock_scene
    )

    result = mediator.get_scene_parameter("vmag")

    assert result == mock_scene.vmag
