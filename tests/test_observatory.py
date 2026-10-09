import pytest
import numpy as np
import types, sys, copy
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


# Reset presets just in case a test changes them.
@pytest.fixture(autouse=True)
def _restore_observatory_class_state():
    """Snapshot and restore Observatory's mutable class-level dicts around every
    test, so a test that accidentally mutates PRESETS / TOY_MODEL_COMPONENTS
    cannot leak that mutation into subsequent tests (an order-dependent Heisenbug)."""
    saved_presets = copy.deepcopy(Observatory.PRESETS)
    saved_toy = copy.deepcopy(Observatory.TOY_MODEL_COMPONENTS)
    try:
        yield
    finally:
        Observatory.PRESETS = saved_presets
        Observatory.TOY_MODEL_COMPONENTS = saved_toy


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


import numpy.ma as ma


def _masked_scalar(value, masked=False, fill_value=1e20):
    """Mimic hwome's `.v` accessor: masked_array(data=value, mask=masked, fill_value=fill_value)."""
    return ma.masked_array(value, mask=masked, fill_value=fill_value)


def _make_fake_hwome_system():
    system = MagicMock()
    system.load_configuration = MagicMock(return_value=None)

    mask_obj = MagicMock()
    mask_obj.value = "baseline_mask"
    system.system.Mask.name = {"only_mask": mask_obj}

    system.system.Telescope.circumscribing_diameter.q = 6.0 * u.m

    optical_path = MagicMock()

    # temperature.v -> masked array (median must work over this)
    optical_path.temperature.v = ma.masked_array(
        [290.0, 290.0, 290.0], mask=[False, False, False], fill_value=1e20
    )

    tp = MagicMock()
    tp.v = ma.masked_array(
        [
            [0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90],
            [0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80],
            [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
        ],
        mask=False,
        fill_value=1e20,
    )
    tp.w = np.array([0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.70]) * u.um
    optical_path.throughput = MagicMock(return_value=tp)

    optical_path.Channel.band_min.q = 0.40 * u.um
    optical_path.Channel.band_max.q = 0.70 * u.um

    system.system.OpticalPath.select = MagicMock(return_value=optical_path)

    nav_chan = MagicMock()
    nav_chan.Detector.pixel_pitch.v = _masked_scalar(13.0e-6)
    nav_chan.focal_length.v = _masked_scalar(20.0)
    nav_chan.Detector.dark_current.v = _masked_scalar(1.0e-05)
    nav_chan.Detector.read_noise.v = _masked_scalar(0.1)
    nav_chan.Detector.cic.v = _masked_scalar(1.3e-3)
    system.resolve = MagicMock(return_value=nav_chan)

    return system


def _fake_search_configuration(channel_type, wavelength_range_nm, center_nm):
    if channel_type == "cg_di":
        return {
            "vis": {
                "vis_FULL": {
                    "path": "CI.CI_VIS_IFS.CI_4F874",
                    "center": 0.5 * u.um,
                    "width": 0.1 * u.um,
                }
            }
        }
    # No IFS channels defined in this fake dataset
    return {}


def install_fake_hwome(monkeypatch, system_instance, search_configuration_fn):
    """Install fake hwome.roam.analyzer.Analyzer / hwome.core.navigator.search_configuration
    into sys.modules so the local imports inside _ingest_from_hwome succeed."""

    hwome_mod = types.ModuleType("hwome")
    roam_mod = types.ModuleType("hwome.roam")
    roam_analyzer_mod = types.ModuleType("hwome.roam.analyzer")
    core_mod = types.ModuleType("hwome.core")
    core_navigator_mod = types.ModuleType("hwome.core.navigator")

    hwome_mod.roam = roam_mod
    hwome_mod.core = core_mod
    roam_mod.analyzer = roam_analyzer_mod
    core_mod.navigator = core_navigator_mod

    roam_analyzer_mod.Analyzer = MagicMock(return_value=system_instance)
    core_navigator_mod.search_configuration = MagicMock(
        side_effect=search_configuration_fn
    )

    for name, mod in [
        ("hwome", hwome_mod),
        ("hwome.roam", roam_mod),
        ("hwome.roam.analyzer", roam_analyzer_mod),
        ("hwome.core", core_mod),
        ("hwome.core.navigator", core_navigator_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


def _eacy_ns(**attrs):
    """eacy's load_* return objects whose .__dict__ carries the data;
    SimpleNamespace reproduces that faithfully."""
    return types.SimpleNamespace(**attrs)


def install_fake_eacy(monkeypatch, load_detector_fn=None):
    """Install a fake `eacy` module so _convert_eacy_to_unified_format runs
    with no real package or data files.

    Shared wavelength grid: vis finite for the first 3 samples, nir for the
    last 3, NaN elsewhere, so the finite-qe intersection cleanly splits the
    two channels into distinct 3-point domains.
    """
    shared_lam = np.array([0.40, 0.50, 0.60, 0.90, 1.20, 1.50]) * u.um
    n = shared_lam.size

    def load_telescope(keyword):
        return _eacy_ns(
            diam_circ=6.0,
            lam=shared_lam,
            total_tele_refl=np.full(n, 0.90),
        )

    def load_instrument(name):
        # source hardcodes "CI"
        return _eacy_ns(total_inst_refl=np.full(n, 0.80))

    def default_load_detector(obs_mode):
        return _eacy_ns(
            qe_vis=np.array([0.90, 0.90, 0.90, np.nan, np.nan, np.nan]),
            qe_nir=np.array([np.nan, np.nan, np.nan, 0.80, 0.80, 0.80]),
            dc_vis=1.0e-5,
            dc_nir=2.0e-5,
            rn_vis=0.10,
            rn_nir=0.20,
        )

    eacy_mod = types.ModuleType("eacy")
    eacy_mod.load_telescope = MagicMock(side_effect=load_telescope)
    eacy_mod.load_instrument = MagicMock(side_effect=load_instrument)
    eacy_mod.load_detector = MagicMock(
        side_effect=load_detector_fn or default_load_detector
    )
    monkeypatch.setitem(sys.modules, "eacy", eacy_mod)
    return eacy_mod


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
    obs.configuration = None
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
    assert obs.configuration is None
    assert isinstance(obs.telescope, ToyModelTelescope)
    assert isinstance(obs.coronagraph, ToyModelCoronagraph)
    assert isinstance(obs.detector, ToyModelDetector)


@patch("pyEDITH.observatory.Observatory.validate_engineering_config")
@patch("pyEDITH.observatory.Observatory._load_eac_configuration")
@patch("pyEDITH.observatory.Observatory._create_detector")
@patch("pyEDITH.observatory.Observatory._create_coronagraph")
@patch("pyEDITH.observatory.Observatory._create_telescope")
def test_create_observatory_eac1_preset(
    mock_tel, mock_coro, mock_det, mock_load, mock_validate
):
    """EAC1 preset: verify the correct keywords are dispatched to each factory,
    and that EAC config loading + validation are invoked. No real eacy/network."""
    mock_load.return_value = {"sentinel": "eac1_config"}

    obs = Observatory()
    obs.create_observatory("EAC1")

    mock_tel.assert_called_once_with("EAC1")
    mock_coro.assert_called_once_with("eac1_aavc_2d")
    mock_det.assert_called_once_with("EAC1")
    mock_load.assert_called_once_with("EAC1")
    mock_validate.assert_called_once_with({"sentinel": "eac1_config"}, "EAC1")
    assert obs.configuration == {"sentinel": "eac1_config"}


@patch("pyEDITH.observatory.Observatory.validate_engineering_config")
@patch("pyEDITH.observatory.Observatory._load_eac_configuration")
@patch("pyEDITH.observatory.Observatory._create_detector")
@patch("pyEDITH.observatory.Observatory._create_coronagraph")
@patch("pyEDITH.observatory.Observatory._create_telescope")
def test_create_observatory_eac5_preset(
    mock_tel, mock_coro, mock_det, mock_load, mock_validate
):
    """EAC5 preset: same as EAC1 but with EAC5 keywords."""
    mock_load.return_value = {"sentinel": "eac5_config"}

    obs = Observatory()
    obs.create_observatory("EAC5")

    mock_tel.assert_called_once_with("EAC5")
    mock_coro.assert_called_once_with("eac1_aavc_2d")
    mock_det.assert_called_once_with("EAC5")
    mock_load.assert_called_once_with("EAC5")
    mock_validate.assert_called_once_with({"sentinel": "eac5_config"}, "EAC5")
    assert obs.configuration == {"sentinel": "eac5_config"}


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


@patch("pyEDITH.observatory.telescopes.EACTelescope")
def test_create_telescope_eac(mock_eac_telescope):
    """EAC telescope: verify EACTelescope is constructed with the keyword."""
    result = Observatory._create_telescope("EAC1")
    mock_eac_telescope.assert_called_once_with(keyword="EAC1")
    assert result is mock_eac_telescope.return_value


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


@patch("pyEDITH.observatory.detectors.EACDetector")
def test_create_detector_eac(mock_eac_detector):
    """EAC detector: verify EACDetector is constructed with the keyword."""
    result = Observatory._create_detector("EAC1")
    mock_eac_detector.assert_called_once_with(keyword="EAC1")
    assert result is mock_eac_detector.return_value


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
# Tests for Observatory._select_active_channel
# ============================================================================


def test_select_active_channel_single_match():
    """Test that the one channel fully containing the wavelength range is selected."""
    mode_config = {
        "vis": {"wavelength_range": (0.4, 0.6)},
        "nir": {"wavelength_range": (0.9, 1.8)},
    }
    wavelength_range = u.Quantity([0.45, 0.55], u.micron)

    result = Observatory._select_active_channel(mode_config, wavelength_range)

    assert result == "vis"


def test_select_active_channel_no_match_raises():
    """Test that ValueError is raised when no channel fully contains the range."""
    mode_config = {
        "vis": {"wavelength_range": (0.4, 0.6)},
        "nir": {"wavelength_range": (0.9, 1.8)},
    }
    wavelength_range = u.Quantity([0.7, 0.8], u.micron)

    with pytest.raises(ValueError, match="does not fall entirely within"):
        Observatory._select_active_channel(mode_config, wavelength_range)


def test_select_active_channel_multiple_matches_raises():
    """Test that ValueError is raised when more than one channel contains the range."""
    mode_config = {
        "a": {"wavelength_range": (0.4, 1.0)},
        "b": {"wavelength_range": (0.3, 1.2)},
    }
    wavelength_range = u.Quantity([0.45, 0.55], u.micron)

    with pytest.raises(ValueError, match="spans multiple"):
        Observatory._select_active_channel(mode_config, wavelength_range)


# ============================================================================
# Tests for Observatory._convert_eacy_to_unified_format
# ============================================================================


def test_convert_eacy_builds_both_modes_and_channels(monkeypatch):
    """Happy path: both modes, both channels, split by finite-qe masking."""
    install_fake_eacy(monkeypatch)

    config = Observatory._convert_eacy_to_unified_format("EAC1")

    assert config["diameter"] == pytest.approx(6.0)
    assert config["temperature"] == pytest.approx(290.0)  # eacy hardcoded default

    for mode in ("IMAGER", "IFS"):
        assert set(config[mode].keys()) == {"vis", "nir"}

        vis = config[mode]["vis"]
        assert vis["dc"] == pytest.approx(1.0e-5)
        assert vis["rn"] == pytest.approx(0.10)
        assert vis["cic"] == 0.0  # eacy cic always 0.0
        assert vis["pixscale_mas"] > 0

        # combined throughput = 0.90 * 0.80 = 0.72 on the surviving samples
        assert np.allclose(vis["spectral"]["optics_throughput"], 0.72)
        # DEFAULT_DQE broadcast
        assert np.allclose(vis["spectral"]["dqe"], 0.75)

        # only the 3 finite-qe samples survive
        assert len(vis["spectral"]["wavelength"]) == 3
        assert len(vis["spectral"]["qe"]) == 3
        lo, hi = vis["wavelength_range"]
        assert lo == pytest.approx(0.40)
        assert hi == pytest.approx(0.60)

        nir = config[mode]["nir"]
        assert nir["dc"] == pytest.approx(2.0e-5)
        assert len(nir["spectral"]["wavelength"]) == 3
        assert nir["wavelength_range"][0] == pytest.approx(0.90)
        assert nir["wavelength_range"][1] == pytest.approx(1.50)


def test_convert_eacy_no_intersection_raises(monkeypatch):
    """The 'Could not determine distinct ... domain' guard fires when a channel's
    qe is never finite where throughput is finite."""

    def bad_detector(obs_mode):
        return _eacy_ns(
            qe_vis=np.full(6, np.nan),  # never finite -> empty mask
            qe_nir=np.array([np.nan, np.nan, np.nan, 0.8, 0.8, 0.8]),
            dc_vis=1e-5,
            dc_nir=2e-5,
            rn_vis=0.1,
            rn_nir=0.2,
        )

    install_fake_eacy(monkeypatch, load_detector_fn=bad_detector)

    with pytest.raises(ValueError, match="Could not determine distinct vis"):
        Observatory._convert_eacy_to_unified_format("EAC1")


def test_convert_eacy_calls_load_detector_once_per_mode(monkeypatch):
    """The source states load_detector is called once per mode (QE differs by
    mode). Verify that contract, since it's a documented design intent."""
    eacy_mod = install_fake_eacy(monkeypatch)

    Observatory._convert_eacy_to_unified_format("EAC1")

    # Called exactly twice: once for IMAGER, once for IFS
    assert eacy_mod.load_detector.call_count == 2
    modes_requested = {c.args[0] for c in eacy_mod.load_detector.call_args_list}
    assert modes_requested == {"IMAGER", "IFS"}
    # telescope/instrument consulted once each (mode-independent)
    eacy_mod.load_telescope.assert_called_once_with("EAC1")
    eacy_mod.load_instrument.assert_called_once_with("CI")


def test_load_eac_configuration_dispatches_to_eacy_for_eac1(monkeypatch):
    """EAC1 must route through eacy (the else branch), NOT hwome."""
    eacy_mod = install_fake_eacy(monkeypatch)

    config = Observatory._load_eac_configuration("EAC1")

    assert config["diameter"] == pytest.approx(6.0)
    eacy_mod.load_telescope.assert_called_once_with("EAC1")


def test_load_eac_configuration_dispatches_to_eacy_for_eac3(monkeypatch):
    """EAC2/EAC3 also route through eacy (they're not in the hwome list)."""
    eacy_mod = install_fake_eacy(monkeypatch)

    Observatory._load_eac_configuration("EAC3")

    eacy_mod.load_telescope.assert_called_once_with("EAC3")


# ============================================================================
# Tests for Observatory._ingest_from_hwome
# ============================================================================


def test_load_eac_configuration_dispatches_to_hwome_for_eac5(monkeypatch):
    fake_system = _make_fake_hwome_system()
    install_fake_hwome(monkeypatch, fake_system, _fake_search_configuration)

    config = Observatory._load_eac_configuration("EAC5")

    assert config["diameter"] == pytest.approx(6.0)
    fake_system.load_configuration.assert_called_once_with("eac5.yaml")


def test_ingest_from_hwome_builds_unified_config(monkeypatch):
    fake_system = _make_fake_hwome_system()
    install_fake_hwome(monkeypatch, fake_system, _fake_search_configuration)

    result = Observatory._ingest_from_hwome(eac_name="eac5")

    # top-level scalars
    assert result["diameter"] == pytest.approx(6.0)
    assert result["temperature"] == pytest.approx(290.0)

    # channel made it through
    assert "vis" in result["IMAGER"]
    chan = result["IMAGER"]["vis"]

    assert chan["dc"] == pytest.approx(1.0e-5)
    assert chan["rn"] == pytest.approx(0.1)
    assert chan["cic"] == pytest.approx(1.3e-3)
    assert chan["pixscale_mas"] > 0

    lo, hi = chan["wavelength_range"]
    assert lo == pytest.approx(0.45)
    assert hi == pytest.approx(0.55)

    spectral = chan["spectral"]
    n = len(spectral["wavelength"])
    assert (
        n
        == len(spectral["qe"])
        == len(spectral["optics_throughput"])
        == len(spectral["dqe"])
    )
    assert n == 3  # only wavelengths strictly inside (0.45, 0.55) survive the mask

    # sanity: threading of mask_name / chan_name into the API calls
    fake_system.resolve.assert_called_once_with("CI.vis")
    fake_system.system.OpticalPath.select.assert_any_call(
        Instrument="CI", Channel="CI_VIS_IFS", Filter="CI_4F874", Mask="baseline_mask"
    )


def test_ingest_from_hwome_masked_scalar_is_unmasked_to_float(monkeypatch):
    """hwome returns scalar detector parameters as single-element masked arrays
    (via its `.v` accessor) regardless of validity -- the mask is hwome's data
    representation, not a signal about the data. We convert them with float(),
    which correctly yields the underlying scalar value.

    This test pins that behaviour: a masked single-element scalar from hwome
    must survive as its plain numeric value.
    """
    fake_system = _make_fake_hwome_system()
    # hwome hands us dc as a masked single-element array. We do not control this
    # representation; float() must extract the underlying value regardless of mask.
    fake_system.resolve.return_value.Detector.dark_current.v = _masked_scalar(
        1.0e-05, masked=True
    )
    install_fake_hwome(monkeypatch, fake_system, _fake_search_configuration)

    result = Observatory._ingest_from_hwome(eac_name="eac5")

    dc = result["IMAGER"]["vis"]["dc"]
    assert isinstance(dc, float)  # extracted to a plain Python float
    assert dc == pytest.approx(1.0e-5)  # underlying value, mask disregarded


def test_scalar_from_hwome_rejects_multi_element(monkeypatch):
    """If hwome ever sends a multi-element array where a scalar is expected,
    ingestion must fail loudly rather than silently truncate."""
    fake_system = _make_fake_hwome_system()
    fake_system.resolve.return_value.Detector.dark_current.v = ma.masked_array(
        [1e-5, 2e-5], mask=[False, False]
    )
    install_fake_hwome(monkeypatch, fake_system, _fake_search_configuration)

    with pytest.raises(ValueError, match=r"expected a scalar for"):
        Observatory._ingest_from_hwome(eac_name="eac5")


def test_scalar_from_hwome_never_returns_fill_value(monkeypatch):
    """A masked scalar must yield its true underlying value, never the fill-value."""
    fake_system = _make_fake_hwome_system()
    fake_system.resolve.return_value.Detector.dark_current.v = _masked_scalar(
        1.0e-05, masked=True, fill_value=1e20
    )
    install_fake_hwome(monkeypatch, fake_system, _fake_search_configuration)

    result = Observatory._ingest_from_hwome(eac_name="eac5")
    assert result["IMAGER"]["vis"]["dc"] == pytest.approx(1.0e-5)  # not 1e20


def test_ingest_from_hwome_non_finite_scalar_raises(monkeypatch):
    """A finite-typed but NaN-valued hwome scalar trips _scalar_from_hwome's
    finiteness guard (line 530->531), failing loudly rather than propagating NaN."""
    fake_system = _make_fake_hwome_system()
    # A single-element (passes size guard) but NaN-valued (fails finiteness guard)
    fake_system.resolve.return_value.Detector.dark_current.v = _masked_scalar(
        np.nan, masked=False
    )
    install_fake_hwome(monkeypatch, fake_system, _fake_search_configuration)

    with pytest.raises(ValueError, match="is not finite"):
        Observatory._ingest_from_hwome(eac_name="eac5")


# ============================================================================
# Tests for Observatory.validate_engineering_config
# ============================================================================


def _minimal_valid_channel():
    return {
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


def _minimal_valid_config():
    return {
        "diameter": 8.0,
        "temperature": 290.0,
        "IMAGER": {"vis": _minimal_valid_channel()},
        "IFS": {"vis": _minimal_valid_channel()},
    }


def test_validate_engineering_config_valid():
    """Test that a well-formed config passes without raising."""
    Observatory.validate_engineering_config(_minimal_valid_config(), "EAC1")


def test_validate_engineering_config_none_raises():
    """Test that a None config raises ValueError."""
    with pytest.raises(ValueError, match="failed to load"):
        Observatory.validate_engineering_config(None, "EAC1")


def test_validate_engineering_config_missing_top_level_key():
    """Test that a missing top-level key raises ValueError."""
    config = _minimal_valid_config()
    del config["temperature"]

    with pytest.raises(ValueError, match="missing required key 'temperature'"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_wrong_type_top_level_key():
    """Test that a top-level key with the wrong type raises TypeError."""
    config = _minimal_valid_config()
    config["IMAGER"] = "not_a_dict"

    with pytest.raises(TypeError, match="expected"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_bad_diameter():
    """Test that an unphysical diameter raises ValueError."""
    config = _minimal_valid_config()
    config["diameter"] = -5.0

    with pytest.raises(ValueError, match="looks unphysical"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_bad_temperature():
    """Test that an unphysical temperature raises ValueError."""
    config = _minimal_valid_config()
    config["temperature"] = 5000.0

    with pytest.raises(ValueError, match="looks unphysical"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_missing_channels():
    """Test that an empty mode dict (no channels) raises ValueError."""
    config = _minimal_valid_config()
    config["IFS"] = {}

    with pytest.raises(ValueError, match="must define at least one channel"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_missing_channel_scalar_key():
    """Test that a channel missing 'dc'/'rn'/'cic' raises ValueError."""
    config = _minimal_valid_config()
    del config["IMAGER"]["vis"]["rn"]

    with pytest.raises(ValueError, match="missing required key 'rn'"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_negative_scalar_key():
    """Test that a negative dc/rn/cic raises ValueError."""
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["dc"] = -1.0

    with pytest.raises(ValueError, match="must be a finite, non-negative number"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_bad_pixscale():
    """Test that a non-positive pixscale_mas raises ValueError."""
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["pixscale_mas"] = 0.0

    with pytest.raises(ValueError, match="must be a finite, positive number"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_missing_spectral():
    """Test that a missing 'spectral' key raises ValueError."""
    config = _minimal_valid_config()
    del config["IMAGER"]["vis"]["spectral"]

    with pytest.raises(ValueError, match="missing required key 'spectral'"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_missing_wavelength_range():
    """Test that a missing 'wavelength_range' key raises ValueError."""
    config = _minimal_valid_config()
    del config["IMAGER"]["vis"]["wavelength_range"]

    with pytest.raises(ValueError, match="missing required key 'wavelength_range'"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_spectral_missing_wavelength():
    """Test that a spectral dict missing 'wavelength' raises ValueError."""
    config = _minimal_valid_config()
    del config["IMAGER"]["vis"]["spectral"]["wavelength"]

    with pytest.raises(ValueError, match="missing 'wavelength'"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_spectral_length_mismatch():
    """Test that mismatched array lengths in spectral data raise ValueError."""
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["spectral"]["qe"] = np.array(
        [0.9, 0.9]
    )  # len 2 vs wavelength len 3

    with pytest.raises(ValueError, match="length mismatch"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_wavelength_not_increasing():
    """Test that a non-monotonic wavelength array raises ValueError."""
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["spectral"]["wavelength"] = np.array([0.5, 0.4, 0.6])

    with pytest.raises(ValueError, match="not strictly increasing"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_qe_out_of_bounds():
    """Test that qe values outside [0, 1] raise ValueError."""
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["spectral"]["qe"] = np.array([0.9, 1.5, 0.9])

    with pytest.raises(ValueError, match=r"must lie within \[0, 1\]"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_wavelength_range_too_narrow():
    """Test that wavelength_range not covering the spectral domain raises ValueError."""
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["wavelength_range"] = (0.45, 0.5)

    with pytest.raises(ValueError, match="does not"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_wavelength_range_wrong_shape():
    """Test that a malformed wavelength_range raises ValueError."""
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["wavelength_range"] = (0.4, 0.5, 0.6)  # 3 elements, not 2

    with pytest.raises(ValueError, match="2-element"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_wavelength_range_min_gte_max():
    """Test that wavelength_range with min >= max raises ValueError."""
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["wavelength_range"] = (0.6, 0.4)

    with pytest.raises(ValueError, match="must be strictly"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_nan_scalar_key_raises():
    """Test that a NaN dc/rn/cic value raises ValueError (not silently accepted)."""
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["dc"] = float("nan")

    with pytest.raises(ValueError, match="must be a finite"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_engineering_config_nan_pixscale_raises():
    """Test that a NaN pixscale_mas value raises ValueError."""
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["pixscale_mas"] = float("nan")

    with pytest.raises(ValueError, match="must be a finite"):
        Observatory.validate_engineering_config(config, "EAC1")


# ============================================================================
# validate_engineering_config -- interior raises (nested helper coverage)
# ============================================================================


def test_validate_config_channel_not_a_dict_raises():
    """A channel whose value isn't a dict raises TypeError."""
    config = _minimal_valid_config()
    config["IMAGER"]["vis"] = "not_a_dict"
    with pytest.raises(TypeError, match="must be a dict"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_config_diameter_wrong_type_raises():
    """A non-numeric diameter raises TypeError (top-level type check)."""
    config = _minimal_valid_config()
    config["diameter"] = "8.0"  # string, not numeric
    with pytest.raises(TypeError, match=r"'diameter' expected"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_config_empty_wavelength_array_raises():
    """A present-but-empty wavelength array raises ValueError."""
    config = _minimal_valid_config()
    ch = config["IMAGER"]["vis"]
    ch["spectral"]["wavelength"] = np.array([])
    # keep the value arrays empty too, else we'd trip the length-mismatch first
    ch["spectral"]["optics_throughput"] = np.array([])
    ch["spectral"]["qe"] = np.array([])
    ch["spectral"]["dqe"] = np.array([])
    with pytest.raises(ValueError, match="empty 'wavelength' array"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_config_wavelength_contains_nan_raises():
    """A NaN in the wavelength array raises ValueError."""
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["spectral"]["wavelength"] = np.array([0.4, np.nan, 0.6])
    with pytest.raises(ValueError, match=r"'wavelength' contains NaN/Inf"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_config_missing_value_key_raises():
    """A spectral dict missing a required value key (e.g. dqe) raises ValueError."""
    config = _minimal_valid_config()
    del config["IMAGER"]["vis"]["spectral"]["dqe"]
    with pytest.raises(ValueError, match="missing 'dqe'"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_config_empty_value_array_raises():
    """A value array of length 0 (when wavelength is non-empty) raises ValueError.

    NOTE: this trips the empty-array check inside the value loop. Because the
    length-mismatch check would ALSO fire, we must reach 'empty' first: the
    source checks `val.size == 0` before the shape comparison, so an empty
    qe against a length-3 wavelength hits the empty branch.
    """
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["spectral"]["qe"] = np.array([])
    with pytest.raises(ValueError, match=r"'qe' array is empty"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_config_value_array_contains_nan_raises():
    """A NaN in a value array (matching length, in-bounds otherwise) raises ValueError.

    Must be same length as wavelength (3) to pass the shape check and reach the
    finiteness check -- and NOT a unity-bounded violation, so use throughput
    with a NaN rather than an out-of-range number.
    """
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["spectral"]["optics_throughput"] = np.array(
        [0.8, np.nan, 0.8]
    )
    with pytest.raises(ValueError, match=r"'optics_throughput' contains NaN/Inf"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_config_wavelength_range_non_numeric_raises():
    """Non-numeric wavelength_range elements raise TypeError."""
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["wavelength_range"] = ("a", "b")
    with pytest.raises(TypeError, match="elements must be numeric"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_config_wavelength_range_nan_raises():
    """A NaN in wavelength_range raises ValueError."""
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["wavelength_range"] = (float("nan"), 0.6)
    with pytest.raises(ValueError, match=r"'wavelength_range' contains NaN/Inf"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_config_qe_above_one_raises():
    """A qe value > 1 trips the unity-bound upper check (line 751)."""
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["spectral"]["qe"] = np.array([0.9, 1.5, 0.9])
    with pytest.raises(ValueError, match=r"must lie within \[0, 1\]"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_config_throughput_below_zero_raises():
    """A negative throughput trips the unity-bound LOWER check (line 751)."""
    config = _minimal_valid_config()
    config["IMAGER"]["vis"]["spectral"]["optics_throughput"] = np.array(
        [0.8, -0.1, 0.8]
    )
    with pytest.raises(ValueError, match=r"must lie within \[0, 1\]"):
        Observatory.validate_engineering_config(config, "EAC1")


def test_validate_config_missing_pixscale_key_raises():
    """A channel missing 'pixscale_mas' entirely trips line 882."""
    config = _minimal_valid_config()
    del config["IMAGER"]["vis"]["pixscale_mas"]
    with pytest.raises(ValueError, match="missing required key 'pixscale_mas'"):
        Observatory.validate_engineering_config(config, "EAC1")


# ============================================================================
# Tests for Observatory.validate_configuration
# ============================================================================


def test_observatory_validate_configuration_valid(configured_mock_observatory):
    """Validation passes with correctly-typed, correctly-unitted attributes."""
    configured_mock_observatory.optics_throughput = [0.8] * DIMENSIONLESS
    configured_mock_observatory.total_throughput = [0.6] * QUANTUM_EFFICIENCY
    configured_mock_observatory.epswarmTrcold = [0.2] * DIMENSIONLESS

    # Should not raise
    configured_mock_observatory.validate_configuration()

    # And each sub-component's validation was invoked
    configured_mock_observatory.telescope.validate_configuration.assert_called_once()
    configured_mock_observatory.detector.validate_configuration.assert_called_once()
    configured_mock_observatory.coronagraph.validate_configuration.assert_called_once()


def test_observatory_validate_configuration_missing_attribute(
    configured_mock_observatory,
):
    """A missing observatory-level attribute raises AttributeError."""
    configured_mock_observatory.optics_throughput = [0.8] * DIMENSIONLESS
    configured_mock_observatory.total_throughput = [0.6] * QUANTUM_EFFICIENCY
    configured_mock_observatory.epswarmTrcold = [0.2] * DIMENSIONLESS
    delattr(configured_mock_observatory, "optics_throughput")

    with pytest.raises(
        AttributeError,
        match=r"Observatory is missing attribute: optics_throughput",
    ):
        configured_mock_observatory.validate_configuration()


def test_observatory_validate_configuration_not_quantity(configured_mock_observatory):
    """A non-Quantity observatory attribute raises TypeError."""
    configured_mock_observatory.optics_throughput = 0.8  # bare float, no unit
    configured_mock_observatory.total_throughput = [0.6] * QUANTUM_EFFICIENCY
    configured_mock_observatory.epswarmTrcold = [0.2] * DIMENSIONLESS

    with pytest.raises(
        TypeError,
        match=r"Observatory attribute optics_throughput should be a Quantity",
    ):
        configured_mock_observatory.validate_configuration()


def test_observatory_validate_configuration_incorrect_units(
    configured_mock_observatory,
):
    """A Quantity with wrong units raises ValueError."""
    configured_mock_observatory.optics_throughput = [0.8] * u.meter  # wrong unit
    configured_mock_observatory.total_throughput = [0.6] * QUANTUM_EFFICIENCY
    configured_mock_observatory.epswarmTrcold = [0.2] * DIMENSIONLESS

    with pytest.raises(
        ValueError,
        match=r"Observatory attribute optics_throughput has incorrect units",
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


@patch("pyEDITH.observatory.utils.rebin_channel_curves_to_grid")
def test_calculate_optics_throughput_from_eac_config(
    mock_rebin, configured_mock_observatory, mock_observation_imager, mock_scene
):
    """Test optics throughput calculation from EAC config, mocking the rebin step."""
    mock_rebin.return_value = {"optics_throughput": np.array([0.75])}

    spectral = {
        "wavelength": np.array([0.45, 0.5, 0.55]),
        "optics_throughput": np.array([0.7, 0.8, 0.9]),
        "qe": np.array([0.9, 0.9, 0.9]),
        "dqe": np.array([0.75, 0.75, 0.75]),
    }
    configured_mock_observatory.configuration = {
        "diameter": 8.0,
        "temperature": 290.0,
        "IMAGER": {
            "vis": {
                "dc": 3e-5,
                "rn": 0.1,
                "cic": 0.0,
                "pixscale_mas": 10.0,
                "wavelength_range": (0.45, 0.55),
                "spectral": spectral,
            }
        },
        "IFS": {},
    }
    configured_mock_observatory.active_channel = "vis"
    mock_observation_imager.wavelength_range = u.Quantity([0.45, 0.55], u.micron)

    parameters = {"observing_mode": "IMAGER", "wavelength": 0.5}
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation_imager, mock_scene
    )

    configured_mock_observatory.calculate_optics_throughput(parameters, mediator)

    assert configured_mock_observatory.optics_throughput.unit == DIMENSIONLESS
    assert np.allclose(configured_mock_observatory.optics_throughput.value, [0.75])

    mock_rebin.assert_called_once()
    called_spectral = mock_rebin.call_args.args[0]
    assert called_spectral is spectral


def test_calculate_optics_throughput_no_config_raises(
    configured_mock_observatory, mock_observation_imager, mock_scene
):
    """Test that ValueError is raised when no T_optical and no EAC config are available."""
    configured_mock_observatory.configuration = None
    parameters = {"observing_mode": "IMAGER", "wavelength": 0.5}
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation_imager, mock_scene
    )

    with pytest.raises(
        ValueError, match="Could not calculate optics throughput from the YAML files"
    ):
        configured_mock_observatory.calculate_optics_throughput(parameters, mediator)


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


def test_observatory_load_configuration_toymodel_active_channel_none(
    mock_observatory, mock_observation_imager, mock_scene
):
    """Test that active_channel is None when observatory has no EAC configuration."""
    mock_observatory.configuration = None
    parameters = {"observing_mode": "IMAGER", "T_optical": 0.8, "wavelength": 0.5}

    mock_observatory.load_configuration(parameters, mock_observation_imager, mock_scene)

    assert mock_observatory.active_channel is None


def test_observatory_load_configuration_eac_selects_active_channel(
    mock_observatory, mock_observation_imager, mock_scene
):
    """Test that load_configuration selects the active channel from EAC configuration."""
    mock_observatory.configuration = {
        "IMAGER": {
            "vis": {"wavelength_range": (0.4, 0.6)},
            "nir": {"wavelength_range": (0.9, 1.8)},
        },
        "IFS": {},
    }
    mock_observation_imager.wavelength_range = u.Quantity([0.45, 0.55], u.micron)

    parameters = {"observing_mode": "IMAGER", "T_optical": 0.8, "wavelength": 0.5}

    mock_observatory.load_configuration(parameters, mock_observation_imager, mock_scene)

    assert mock_observatory.active_channel == "vis"


def test_load_configuration_delegates_to_all_components(
    mock_observatory, mock_observation_imager, mock_scene
):
    """load_configuration must invoke each sub-component's load_configuration
    exactly once -- coverage alone won't catch a dropped delegation."""
    parameters = {"observing_mode": "IMAGER", "T_optical": 0.8, "wavelength": 0.5}

    mock_observatory.load_configuration(parameters, mock_observation_imager, mock_scene)

    mock_observatory.telescope.load_configuration.assert_called_once()
    mock_observatory.coronagraph.load_configuration.assert_called_once()
    mock_observatory.detector.load_configuration.assert_called_once()

    # And each received the SAME mediator instance (contract, not accident)
    tel_args = mock_observatory.telescope.load_configuration.call_args.args
    coro_args = mock_observatory.coronagraph.load_configuration.call_args.args
    assert tel_args[1] is coro_args[1]  # same mediator threaded through


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
