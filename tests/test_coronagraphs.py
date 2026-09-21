import pytest
import numpy as np
from astropy import units as u
from pyEDITH.coronagraphs import (
    ToyModelCoronagraph,
    CoronagraphYIP,
    generate_radii,
)
from pyEDITH.units import (
    LAMBDA_D,
    DIMENSIONLESS,
    LENGTH,
    WAVELENGTH,
    PHOTON_FLUX_DENSITY,
    PIXEL,
    ARCSEC,
)
from unittest.mock import patch, MagicMock
import logging

# ============================================================================
# Mock Objects and Fixtures
# ============================================================================


class MockMediator_IMAGER:
    def get_observation_parameter(self, param):
        if param == "wavelength":
            return np.array([0.7]) * WAVELENGTH
        elif param == "wavelength_range":
            return np.array([0.7 * (1 - 0.2 / 2), 0.7 * (1 + 0.2 / 2)]) * WAVELENGTH
        elif param == "observing_mode":
            return "IMAGER"
        else:
            return 1.0

    def get_scene_parameter(self, param):
        if param == "stellar_angular_diameter_arcsec":
            return 1e-3 * ARCSEC
        else:
            return 1.0

    def get_telescope_parameter(self, param):
        if param == "diameter":
            return 8.0 * LENGTH
        return 1.0

    def get_eac_configuration(self):
        # Return None - tests use ToyModel or direct YIP files
        return None


class MockMediator_IFS:
    def get_observation_parameter(self, param):
        if param == "wavelength":
            return np.array([0.5, 0.6, 0.7]) * WAVELENGTH
        elif param == "wavelength_range":
            return np.array([0.5, 0.7]) * WAVELENGTH
        elif param == "observing_mode":
            return "IFS"
        else:
            return 1.0

    def get_scene_parameter(self, param):
        if param == "stellar_angular_diameter_arcsec":
            return 1e-3 * ARCSEC
        else:
            return 1.0

    def get_telescope_parameter(self, param):
        if param == "diameter":
            return 8.0 * LENGTH
        return 1.0

    def get_eac_configuration(self):
        # Return None - tests use ToyModel or direct YIP files
        return None


@pytest.fixture
def mock_instrument():
    mock = MagicMock()
    mock.lam = np.linspace(0.3, 1.6, 10) * WAVELENGTH
    mock.total_inst_refl = np.array(
        [
            4.80759739e-29,
            4.05330898e-01,
            4.40641747e-01,
            3.94770896e-01,
            4.12956241e-01,
            5.15044124e-01,
            5.76293823e-01,
            5.38605236e-01,
            6.27117118e-01,
            6.63022075e-01,
        ]
    )
    return mock


@pytest.fixture
def mock_telescope():
    mock = MagicMock()
    mock.diam_circ = 8.0
    return mock


@pytest.fixture
def imager_toymodel_basic_params():
    """Fixture providing IMAGER observation parameters for the ToyModel."""
    return {
        "pixscale": 0.3,
        "contrast": 1e-10,
        "noisefloor_factor": 0.05,
        "coronagraph_bandwidth": 0.1,
        "photometric_aperture_radius": 0.6,
        "Tcore": 0.3,
        "TLyot": 0.7,
        "nrolls": 1,
        "wavelength": 0.7,
    }


@pytest.fixture
def ifs_toymodel_basic_params():
    """Fixture providing IFS observation parameters for the Toy Model coronagraph."""
    return {
        "pixscale": 0.3,
        "contrast": 1e-10,
        "noisefloor_factor": 0.05,
        "photometric_aperture_radius": 0.6,
        "Tcore": 0.3,
        "TLyot": 0.7,
        "nrolls": 1,
        "wavelength": [0.5, 0.6, 0.7],
    }


@pytest.fixture
def imager_yipcoronagraph_basic_params():
    """Fixture providing IMAGER observation parameters for the YIP coronagraph."""
    return {
        "observing_mode": "IMAGER",
        "psf_trunc_ratio": 0.3,
        "az_avg": True,
        "wavelength": 0.7,
    }


@pytest.fixture
def ifs_yipcoronagraph_basic_params():
    """Fixture providing IFS mode observation parameters."""
    return {
        "observing_mode": "IFS",
        "psf_trunc_ratio": 0.3,
        "az_avg": True,
        "wavelength": [0.5, 0.6, 0.7],
    }


@pytest.fixture
def single_wavelength_params():
    """Fixture providing single wavelength observation parameters."""
    return {
        "wavelength": [0.5],
        "snr": [7.0],
        "CRb_multiplier": 2.0,
        "observing_mode": "IMAGER",
    }


# ============================================================================
# Tests for generate_radii
# ============================================================================


def test_generate_radii_even_dimensions():
    """Test radii generation with even dimensions."""
    radii = generate_radii(10, 10)
    assert radii.shape == (10, 10)


def test_generate_radii_odd_dimensions():
    """Test radii generation with odd dimensions has zero at center."""
    radii = generate_radii(5, 5)

    assert np.isclose(radii[2, 2], 0.0)
    assert np.isclose(radii[0, 0], np.sqrt(radii[0, 2] ** 2 + radii[2, 0] ** 2))


def test_generate_radii_default_square():
    """Test radii generation defaults to square when y dimension not provided."""
    radii = generate_radii(5)

    assert radii.shape == (5, 5)
    assert np.isclose(radii[2, 2], 0.0)
    assert np.isclose(radii[0, 0], np.sqrt(radii[0, 2] ** 2 + radii[2, 0] ** 2))


# ============================================================================
# Tests for ToyModelCoronagraph initialization
# ============================================================================


def test_toy_model_coronagraph_init():
    """Test ToyModelCoronagraph initializes with None values and Locked Keys are empty."""
    coronagraph = ToyModelCoronagraph()

    assert coronagraph.path is None

    assert coronagraph.LOCKED_KEYS == set()


# ============================================================================
# Tests for ToyModelCoronagraph.load_configuration (IMAGER mode)
# ============================================================================


def test_toy_model_load_configuration_basic_parameters(
    caplog, imager_toymodel_basic_params
):
    """Test that basic parameters are loaded correctly."""
    with caplog.at_level(logging.DEBUG, logger="pyEDITH"):
        coronagraph = ToyModelCoronagraph()
        parameters = imager_toymodel_basic_params.copy()
        mediator = MockMediator_IMAGER()

        coronagraph.load_configuration(parameters, mediator)

        assert coronagraph.pixscale == 0.3 * LAMBDA_D
        assert coronagraph.contrast == 1e-10 * DIMENSIONLESS
        assert coronagraph.noisefloor_factor == 0.05 * DIMENSIONLESS
        assert coronagraph.coronagraph_bandwidth == 0.1
        assert coronagraph.photometric_aperture_radius == 0.6 * LAMBDA_D
        assert coronagraph.Tcore == 0.3 * DIMENSIONLESS
        assert coronagraph.TLyot == 0.7 * DIMENSIONLESS
        assert coronagraph.nrolls == 1
        assert coronagraph.coronagraph_spectral_resolution == 1 * DIMENSIONLESS
        assert hasattr(coronagraph, "npsfratios")
        assert hasattr(coronagraph, "npix")
        assert hasattr(coronagraph, "xcenter")
        assert hasattr(coronagraph, "ycenter")
        assert hasattr(coronagraph, "r")
        assert hasattr(coronagraph, "omega_lod")
        assert hasattr(coronagraph, "skytrans")
        assert hasattr(coronagraph, "photometric_aperture_radius")
        assert hasattr(coronagraph, "photometric_aperture_throughput")
        assert hasattr(coronagraph, "PSFpeak")
        assert hasattr(coronagraph, "Istar")
        assert hasattr(coronagraph, "noisefloor")
        assert coronagraph.npix == 400
        assert coronagraph.xcenter == 200 * PIXEL
        assert coronagraph.ycenter == 200 * PIXEL
        assert coronagraph.r.shape == (coronagraph.npix, coronagraph.npix)
        assert np.isclose(coronagraph.r[0, 0], 84.641)
        assert coronagraph.omega_lod.shape == (
            coronagraph.npix,
            coronagraph.npix,
            coronagraph.npsfratios,
        )
        assert np.all(
            coronagraph.omega_lod
            == np.pi * parameters["photometric_aperture_radius"] ** 2 * LAMBDA_D**2
        )
        assert coronagraph.skytrans.shape == (coronagraph.npix, coronagraph.npix)
        assert np.all(coronagraph.skytrans == 0.7 * DIMENSIONLESS)
        assert coronagraph.photometric_aperture_throughput.shape == (
            coronagraph.npix,
            coronagraph.npix,
            coronagraph.npsfratios,
        )
        assert np.all(
            (coronagraph.photometric_aperture_throughput == 0.3 * DIMENSIONLESS)
            | (coronagraph.photometric_aperture_throughput == 0.0 * DIMENSIONLESS)
        )
        assert np.isclose(coronagraph.PSFpeak, 0.025 * 0.7 * DIMENSIONLESS)
        assert coronagraph.Istar.shape == (coronagraph.npix, coronagraph.npix)
        assert np.allclose(coronagraph.Istar.value, 1e-10 * 0.025 * 0.7, rtol=1e-6)
        assert coronagraph.Istar.unit == DIMENSIONLESS
        assert any(
            "Calculating noisefloor by multiplying noisefloor_factor=0.05, contrast=1e-10, PSFpeak="
            + str(0.025 * 0.7)
            in record.message
            for record in caplog.records
        )

        assert coronagraph.noisefloor.shape == (coronagraph.npix, coronagraph.npix)
        assert np.allclose(
            coronagraph.noisefloor.value, 0.05 * 1e-10 * 0.025 * 0.7, rtol=1e-6
        )
        assert coronagraph.noisefloor.unit == DIMENSIONLESS


def test_toy_model_load_configuration_noisefloor_ppf_raises_error(
    imager_toymodel_basic_params,
):
    """Test that providing noisefloor_PPF raises appropriate error."""
    coronagraph = ToyModelCoronagraph()
    parameters = imager_toymodel_basic_params.copy()

    del parameters["noisefloor_factor"]
    parameters["noisefloor_PPF"] = 30
    mediator = MockMediator_IMAGER()

    with pytest.raises(
        KeyError,
        match="Noisefloor_PPF mode not implemented in ToyModel coronagraph",
    ):
        coronagraph.load_configuration(parameters, mediator)


def test_toy_model_load_configuration_default_noisefloor_factor(
    caplog, imager_toymodel_basic_params
):
    """Test that default noisefloor_factor is used when not provided."""
    coronagraph = ToyModelCoronagraph()
    parameters = imager_toymodel_basic_params.copy()

    del parameters["noisefloor_factor"]
    mediator = MockMediator_IMAGER()

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator)

    assert any(
        "noisefloor_factor value not provided. Using the default value: 0.03"
        in record.message
        for record in caplog.records
    )
    assert any(
        "Calculating noisefloor by multiplying noisefloor_factor=0.03, contrast=1e-10, PSFpeak="
        + str(0.025 * 0.7)
        in record.message
        for record in caplog.records
    )

    assert coronagraph.noisefloor.unit == DIMENSIONLESS


# ============================================================================
# Tests for ToyModelCoronagraph.load_configuration (IFS mode)
# ============================================================================


def test_toy_model_load_configuration_ifs_basic_parameters(
    caplog, ifs_toymodel_basic_params
):
    """Test that basic parameters are loaded correctly in IFS mode.

    This is the IFS analogue of
    ``test_toy_model_load_configuration_basic_parameters``. The key point is
    that IFS mode does NOT change the shape of the geometry arrays: per-position
    quantities keep their trailing ``npsfratios`` axis, and the 2D products keep
    their ``(npix, npix)`` shape. The number of wavelengths only affects the
    spectral iteration, not these arrays.
    """
    with caplog.at_level(logging.DEBUG, logger="pyEDITH"):
        coronagraph = ToyModelCoronagraph()
        parameters = ifs_toymodel_basic_params.copy()
        mediator = MockMediator_IFS()

        coronagraph.load_configuration(parameters, mediator)

        # --- Scalar parameters ---
        assert coronagraph.pixscale == 0.3 * LAMBDA_D
        assert coronagraph.contrast == 1e-10 * DIMENSIONLESS
        assert coronagraph.noisefloor_factor == 0.05 * DIMENSIONLESS
        assert coronagraph.coronagraph_bandwidth == 0.2  # default value, locked
        assert coronagraph.photometric_aperture_radius == 0.6 * LAMBDA_D
        assert coronagraph.Tcore == 0.3 * DIMENSIONLESS
        assert coronagraph.TLyot == 0.7 * DIMENSIONLESS
        assert coronagraph.nrolls == 1
        assert coronagraph.coronagraph_spectral_resolution == 1 * DIMENSIONLESS

        # --- Attributes exist ---
        assert hasattr(coronagraph, "npsfratios")
        assert hasattr(coronagraph, "npix")
        assert hasattr(coronagraph, "xcenter")
        assert hasattr(coronagraph, "ycenter")
        assert hasattr(coronagraph, "r")
        assert hasattr(coronagraph, "omega_lod")
        assert hasattr(coronagraph, "skytrans")
        assert hasattr(coronagraph, "photometric_aperture_radius")
        assert hasattr(coronagraph, "photometric_aperture_throughput")
        assert hasattr(coronagraph, "PSFpeak")
        assert hasattr(coronagraph, "Istar")
        assert hasattr(coronagraph, "noisefloor")

        # --- Grid geometry (independent of number of wavelengths) ---
        assert coronagraph.npix == 400
        assert coronagraph.xcenter == 200 * PIXEL
        assert coronagraph.ycenter == 200 * PIXEL
        assert coronagraph.r.shape == (coronagraph.npix, coronagraph.npix)
        assert np.isclose(coronagraph.r[0, 0], 84.641)

        # --- Per-position arrays carry a trailing axis of length npsfratios ---
        # IFS does NOT change this shape (it is not sized by nwave).
        npsfratios = coronagraph.npsfratios
        assert coronagraph.omega_lod.shape == (
            coronagraph.npix,
            coronagraph.npix,
            npsfratios,
        )
        assert np.all(
            coronagraph.omega_lod
            == np.pi * parameters["photometric_aperture_radius"] ** 2 * LAMBDA_D**2
        )

        assert coronagraph.skytrans.shape == (coronagraph.npix, coronagraph.npix)
        assert np.all(coronagraph.skytrans == 0.7 * DIMENSIONLESS)

        assert coronagraph.photometric_aperture_throughput.shape == (
            coronagraph.npix,
            coronagraph.npix,
            npsfratios,
        )
        assert np.all(
            (coronagraph.photometric_aperture_throughput == 0.3 * DIMENSIONLESS)
            | (coronagraph.photometric_aperture_throughput == 0.0 * DIMENSIONLESS)
        )

        # --- PSF peak / stellar intensity / noisefloor (2D, wavelength-agnostic) ---
        assert np.isclose(coronagraph.PSFpeak, 0.025 * 0.7 * DIMENSIONLESS)

        assert coronagraph.Istar.shape == (coronagraph.npix, coronagraph.npix)
        assert np.allclose(coronagraph.Istar.value, 1e-10 * 0.025 * 0.7, rtol=1e-6)
        assert coronagraph.Istar.unit == DIMENSIONLESS

        assert any(
            "Calculating noisefloor by multiplying noisefloor_factor=0.05, contrast=1e-10, PSFpeak="
            + str(0.025 * 0.7)
            in record.message
            for record in caplog.records
        )

        assert coronagraph.noisefloor.shape == (coronagraph.npix, coronagraph.npix)
        assert np.allclose(
            coronagraph.noisefloor.value, 0.05 * 1e-10 * 0.025 * 0.7, rtol=1e-6
        )
        assert coronagraph.noisefloor.unit == DIMENSIONLESS


def test_toy_model_load_configuration_ifs_array_shapes_match_npsfratios(
    ifs_toymodel_basic_params,
):
    """Test that IFS mode sizes per-position arrays by npsfratios, not nwave.

    This guards the primary shape contract: even though the mediator provides
    three wavelengths, the geometry arrays (``omega_lod``,
    ``photometric_aperture_throughput``) carry a trailing axis of length
    ``npsfratios`` and are therefore identical in shape to IMAGER mode. The
    trailing axis must NOT equal the number of wavelengths (unless they happen
    to coincide, which we explicitly rule out here since nwave==3).
    """
    mediator = MockMediator_IFS()
    nwave = len(mediator.get_observation_parameter("wavelength"))
    assert nwave == 3  # guard against fixture drift

    coronagraph = ToyModelCoronagraph()
    parameters = ifs_toymodel_basic_params.copy()

    coronagraph.load_configuration(parameters, mediator)

    npsfratios = coronagraph.npsfratios

    # Per-position arrays are sized by npsfratios (see original implementation).
    assert coronagraph.omega_lod.shape == (
        coronagraph.npix,
        coronagraph.npix,
        npsfratios,
    )
    assert coronagraph.photometric_aperture_throughput.shape == (
        coronagraph.npix,
        coronagraph.npix,
        npsfratios,
    )

    # The trailing (npsfratios) axis is NOT the wavelength axis.
    assert coronagraph.omega_lod.shape[-1] == npsfratios
    assert coronagraph.photometric_aperture_throughput.shape[-1] == npsfratios

    # Sanity check against the IMAGER path: the 2D (non-spectral) products keep
    # their shape regardless of the number of wavelengths.
    assert coronagraph.Istar.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.noisefloor.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.skytrans.shape == (coronagraph.npix, coronagraph.npix)


def test_toy_model_load_configuration_ifs_noisefloor_ppf_raises_error(
    ifs_toymodel_basic_params,
):
    """Test that noisefloor_PPF is rejected in IFS mode too.

    The ToyModel does not implement the PPF noise-floor path in either mode, so
    supplying ``noisefloor_PPF`` must raise regardless of observing_mode. This
    is the IFS counterpart of
    ``test_toy_model_load_configuration_noisefloor_ppf_raises_error``.
    """
    coronagraph = ToyModelCoronagraph()
    parameters = ifs_toymodel_basic_params.copy()

    del parameters["noisefloor_factor"]
    parameters["noisefloor_PPF"] = 30
    mediator = MockMediator_IFS()

    with pytest.raises(
        KeyError,
        match="Noisefloor_PPF mode not implemented in ToyModel coronagraph",
    ):
        coronagraph.load_configuration(parameters, mediator)


def test_toy_model_load_configuration_ifs_default_noisefloor_factor(
    caplog, ifs_toymodel_basic_params
):
    """Test that default noisefloor_factor is used when not provided in IFS mode.

    IFS counterpart of
    ``test_toy_model_load_configuration_default_noisefloor_factor``.
    """
    coronagraph = ToyModelCoronagraph()
    parameters = ifs_toymodel_basic_params.copy()

    del parameters["noisefloor_factor"]
    mediator = MockMediator_IFS()

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator)

    assert any(
        "noisefloor_factor value not provided. Using the default value: 0.03"
        in record.message
        for record in caplog.records
    )
    assert any(
        "Calculating noisefloor by multiplying noisefloor_factor=0.03, contrast=1e-10, PSFpeak="
        + str(0.025 * 0.7)
        in record.message
        for record in caplog.records
    )

    assert coronagraph.noisefloor.unit == DIMENSIONLESS


# ============================================================================
# Tests for CoronagraphYIP initialization
# ============================================================================


def test_coronagraph_yip_init_with_path():
    """Test CoronagraphYIP initialization with path."""
    coronagraph = CoronagraphYIP(path="test_path")

    assert coronagraph.path == "test_path"
    assert coronagraph.yippy_coro is None
    assert coronagraph.LOCKED_KEYS == {
        "pixscale",
        "npix",
        "xcenter",
        "ycenter",
        "skytrans",
        "r",
        "npsfratios",
        "nrolls",
        "omega_lod",
        "photometric_aperture_throughput",
        "Istar",
        "noisefloor",
        "coronagraph_bandwidth",
    }


def test_coronagraph_yip_init_with_yippy_coro(yippy_coronagraph):
    """Test CoronagraphYIP initialization with pre-constructed yippy object."""
    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)

    assert coronagraph.yippy_coro is yippy_coronagraph
    assert coronagraph.path is None


def test_coronagraph_yip_init_requires_path_or_yippy():
    """Test that initialization requires either path or yippy_coro."""
    with pytest.raises(
        ValueError, match="Either a path or a yippy_coro must be provided"
    ):
        CoronagraphYIP()


def test_coronagraph_yip_init_not_both_path_and_yippy(yippy_coronagraph):
    """Test that initialization rejects both path and yippy_coro."""
    with pytest.raises(
        ValueError, match="Only one of path or yippy_coro can be provided"
    ):
        CoronagraphYIP(path="some_path", yippy_coro=yippy_coronagraph)


# ============================================================================
# Tests for CoronagraphYIP.load_configuration - IMAGER mode
# ============================================================================


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_warns_when_user_overrides_locked_key(
    mock_load_telescope,
    mock_load_instrument,
    caplog,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    imager_yipcoronagraph_basic_params,
):
    """
    A user-supplied value for a LOCKED (YIP-owned) key must:
      1. emit a warning that names the locked key, and
      2. be ignored in favour of the YIP/model value.

    We use ``nrolls`` here: the user asks for 2, but the YIP forces 1.
    ``nrolls`` is in ``CoronagraphYIP.LOCKED_KEYS``, so the override is
    rejected with a warning.
    """
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    # Sanity check that the key we are testing really is locked, so this test
    # stays meaningful if LOCKED_KEYS is ever refactored.
    assert "nrolls" in CoronagraphYIP.LOCKED_KEYS

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = imager_yipcoronagraph_basic_params.copy()
    parameters["nrolls"] = 2  # <-- attempt to override a locked key

    mediator = MockMediator_IMAGER()

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator)

    # 1) The locked value from the YIP wins, not the user's 2.
    assert coronagraph.nrolls == 1
    # 2) A warning was emitted that mentions the locked key and that it is locked.
    locked_warnings = [
        rec.message for rec in caplog.records if rec.levelno == logging.WARNING
    ]
    expected_warning = (
        "Parameter 'nrolls' is locked in this mode and cannot be user-overridden; "
        "using the model-provided value instead of the supplied value 2."
    )
    assert expected_warning in locked_warnings


# @patch("eacy.load_instrument")
# @patch("eacy.load_telescope")
# def test_coronagraph_yip_no_warning_when_user_sets_unlocked_key(
#     mock_load_telescope,
#     mock_load_instrument,
#     caplog,
#     yippy_coronagraph,
#     mock_instrument,
#     mock_telescope,
#     imager_yipcoronagraph_basic_params,
# ):
#     """
#     The opposite case: a user-supplied value for an UNLOCKED key (``nois``)
#     must be applied and must NOT trigger a lock warning.
#     """
#     mock_load_instrument.return_value = mock_instrument
#     mock_load_telescope.return_value = mock_telescope

#     assert "psf_truncation_ratio" not in CoronagraphYIP.LOCKED_KEYS

#     coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
#     parameters = imager_yipcoronagraph_basic_params.copy()

#     mediator = MockMediator_IMAGER()

#     with caplog.at_level(logging.WARNING, logger="pyEDITH"):
#         coronagraph.load_configuration(parameters, mediator)

#     # User value is applied.
#     assert coronagraph.psf_truncation_ratio == 0.1

#     # No "locked" warning about bandwidth.
#     bandwidth_lock_warnings = [
#         rec.message
#         for rec in caplog.records
#         if rec.levelno == logging.WARNING
#         and "bandwidth" in rec.message
#         and "locked" in rec.message.lower()
#     ]
#     assert bandwidth_lock_warnings == []


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_basic_parameters(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    imager_yipcoronagraph_basic_params,
):
    """Test that basic YIP parameters are loaded correctly in IMAGER mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = imager_yipcoronagraph_basic_params.copy()

    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.pixscale == yippy_coronagraph.header.pixscale.value * LAMBDA_D
    bw = (
        yippy_coronagraph.header.maxlam - yippy_coronagraph.header.minlam
    ) / yippy_coronagraph.header.lambda0
    assert np.isclose(coronagraph.coronagraph_bandwidth, bw.value)
    assert coronagraph.nrolls == 1
    assert coronagraph.az_avg == True
    assert coronagraph.npix == yippy_coronagraph.header.naxis1
    assert coronagraph.xcenter == yippy_coronagraph.header.xcenter * PIXEL
    assert coronagraph.ycenter == yippy_coronagraph.header.ycenter * PIXEL

    assert hasattr(coronagraph, "npix")
    assert hasattr(coronagraph, "xcenter")
    assert hasattr(coronagraph, "ycenter")
    assert hasattr(coronagraph, "r")
    assert hasattr(coronagraph, "omega_lod")
    assert hasattr(coronagraph, "skytrans")
    assert hasattr(coronagraph, "photometric_aperture_throughput")
    assert hasattr(coronagraph, "Istar")
    assert hasattr(coronagraph, "noisefloor")

    assert coronagraph.r.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.omega_lod.shape == (coronagraph.npix, coronagraph.npix, 1)
    assert coronagraph.skytrans.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.photometric_aperture_throughput.shape == (
        coronagraph.npix,
        coronagraph.npix,
        1,
    )
    assert coronagraph.Istar.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.noisefloor.shape == (coronagraph.npix, coronagraph.npix)
    assert not np.all(coronagraph.omega_lod == 0)
    assert not np.all(coronagraph.skytrans == 0)
    assert not np.all(coronagraph.photometric_aperture_throughput == 0)
    assert not np.all(coronagraph.Istar == 0)
    assert coronagraph.omega_lod.unit == LAMBDA_D**2
    assert coronagraph.noisefloor.unit == DIMENSIONLESS
    assert np.all(coronagraph.skytrans == yippy_coronagraph.sky_trans() * DIMENSIONLESS)


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_default_noisefloor_ppf(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    caplog,
    imager_yipcoronagraph_basic_params,
):
    """Test that default noisefloor_PPF is used when not provided."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = imager_yipcoronagraph_basic_params.copy()

    mediator = MockMediator_IMAGER()

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator)

    assert any(
        "noisefloor_PPF value not provided. Using the default value: 30"
        in record.message
        for record in caplog.records
    )

    assert np.allclose(
        coronagraph.noisefloor,
        coronagraph.Istar / 30,
        rtol=1e-6,
        atol=1e-9,
        equal_nan=True,
    )


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_custom_noisefloor_ppf(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    caplog,
    imager_yipcoronagraph_basic_params,
):
    """Test that custom noisefloor_PPF is used correctly."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = imager_yipcoronagraph_basic_params.copy()
    parameters["noisefloor_PPF"] = 35
    mediator = MockMediator_IMAGER()

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator)

    assert any(
        "Setting the noise floor via user-supplied noisefloor_PPF..." in record.message
        for record in caplog.records
    )

    assert coronagraph.noisefloor.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.noisefloor.unit == DIMENSIONLESS
    assert np.allclose(
        coronagraph.noisefloor,
        coronagraph.Istar / 35,
        rtol=1e-6,
        atol=1e-9,
        equal_nan=True,
    )


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_noisefloor_factor_raises_error(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    imager_yipcoronagraph_basic_params,
):
    """Test that noisefloor_factor raises appropriate error in YIP mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = imager_yipcoronagraph_basic_params.copy()
    parameters["noisefloor_factor"] = 1e-10
    mediator = MockMediator_IMAGER()

    with pytest.raises(
        ValueError,
        match="Noisefloor_factor mode not implemented in CoronagraphYIP",
    ):
        coronagraph.load_configuration(parameters, mediator)


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_missing_aperture_raises_error(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    imager_yipcoronagraph_basic_params,
):
    """Test that missing both aperture parameters raises error."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = imager_yipcoronagraph_basic_params.copy()
    del parameters["psf_trunc_ratio"]
    mediator = MockMediator_IMAGER()

    with pytest.raises(
        KeyError,
        match="Either 'photometric_aperture_radius' or 'psf_trunc_ratio' must be provided",
    ):
        coronagraph.load_configuration(parameters, mediator)


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_az_avg_false(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    imager_yipcoronagraph_basic_params,
):
    """Test that az_avg=False uses full 2D stellar intensity map."""
    from unittest.mock import MagicMock

    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    # Wrap the methods as mocks while preserving their return values
    original_stellar_intens = yippy_coronagraph.stellar_intens
    original_core_mean = yippy_coronagraph.core_mean_intensity_map

    yippy_coronagraph.stellar_intens = MagicMock(side_effect=original_stellar_intens)
    yippy_coronagraph.core_mean_intensity_map = MagicMock(
        side_effect=original_core_mean
    )

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = imager_yipcoronagraph_basic_params.copy()
    parameters["az_avg"] = False
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    # Verify stellar_intens was called (else branch), not core_mean_intensity_map
    yippy_coronagraph.stellar_intens.assert_called_once()
    yippy_coronagraph.core_mean_intensity_map.assert_not_called()

    assert coronagraph.Istar.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.Istar.unit == DIMENSIONLESS
    assert not np.all(coronagraph.Istar == 0)
    assert coronagraph.Istar.ndim == 2


# ============================================================================
# Tests for CoronagraphYIP.load_configuration - IFS mode
# ============================================================================


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_ifs_basic_parameters(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    caplog,
    ifs_yipcoronagraph_basic_params,
):
    """Test that IFS mode correctly handles multiple wavelengths."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = ifs_yipcoronagraph_basic_params.copy()
    mediator = MockMediator_IFS()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.pixscale == yippy_coronagraph.header.pixscale.value * LAMBDA_D
    bw = (
        yippy_coronagraph.header.maxlam - yippy_coronagraph.header.minlam
    ) / yippy_coronagraph.header.lambda0
    assert np.isclose(coronagraph.coronagraph_bandwidth, bw.value)
    assert coronagraph.nrolls == 1
    assert coronagraph.az_avg == True
    assert coronagraph.npix == yippy_coronagraph.header.naxis1
    assert coronagraph.xcenter == yippy_coronagraph.header.xcenter * PIXEL
    assert coronagraph.ycenter == yippy_coronagraph.header.ycenter * PIXEL

    assert hasattr(coronagraph, "npix")
    assert hasattr(coronagraph, "xcenter")
    assert hasattr(coronagraph, "ycenter")
    assert hasattr(coronagraph, "r")
    assert hasattr(coronagraph, "omega_lod")
    assert hasattr(coronagraph, "skytrans")
    assert hasattr(coronagraph, "photometric_aperture_throughput")
    assert hasattr(coronagraph, "Istar")
    assert hasattr(coronagraph, "noisefloor")

    assert coronagraph.r.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.omega_lod.shape == (coronagraph.npix, coronagraph.npix, 1)
    assert coronagraph.skytrans.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.photometric_aperture_throughput.shape == (
        coronagraph.npix,
        coronagraph.npix,
        1,
    )
    assert coronagraph.Istar.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.noisefloor.shape == (coronagraph.npix, coronagraph.npix)
    assert not np.all(coronagraph.omega_lod == 0)
    assert not np.all(coronagraph.skytrans == 0)
    assert not np.all(coronagraph.photometric_aperture_throughput == 0)
    assert not np.all(coronagraph.Istar == 0)
    assert coronagraph.omega_lod.unit == LAMBDA_D**2
    assert coronagraph.noisefloor.unit == DIMENSIONLESS
    assert np.all(coronagraph.skytrans == yippy_coronagraph.sky_trans() * DIMENSIONLESS)


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_ifs_prioritize_psf_trunc_ratio(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    caplog,
    ifs_yipcoronagraph_basic_params,
):
    """Test that IFS mode correctly handles multiple wavelengths."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = ifs_yipcoronagraph_basic_params.copy()
    parameters["photometric_aperture_radius"] = 0.3
    mediator_ifs = MockMediator_IFS()

    with caplog.at_level(logging.INFO, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator_ifs)

    assert any(
        "Both 'photometric_aperture_radius' and 'psf_trunc_ratio' provided"
        in record.message
        for record in caplog.records
    )

    assert any(
        "Using psf_trunc_ratio to calculate Omega..." in record.message
        for record in caplog.records
    )


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_photometric_aperture_tcore_calculations(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    caplog,
    ifs_yipcoronagraph_basic_params,
):
    """Test photometric aperture calculation with user-defined Tcore."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = ifs_yipcoronagraph_basic_params.copy()
    del parameters["psf_trunc_ratio"]
    parameters["photometric_aperture_radius"] = 0.3
    parameters["Tcore"] = 0.5
    mediator = MockMediator_IMAGER()

    with caplog.at_level(logging.INFO, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator)

    assert any(
        "Using photometric_aperture_radius to calculate Omega..." in record.message
        for record in caplog.records
    )
    assert any(
        "Using user-defined Tcore..." in record.message for record in caplog.records
    )
    assert coronagraph.omega_lod.shape == (coronagraph.npix, coronagraph.npix, 1)
    assert np.all(
        coronagraph.omega_lod
        == np.pi * parameters["photometric_aperture_radius"] ** 2 * LAMBDA_D**2
    )
    assert coronagraph.photometric_aperture_throughput.shape == (
        coronagraph.npix,
        coronagraph.npix,
        1,
    )
    assert np.all(
        (
            coronagraph.photometric_aperture_throughput
            == parameters["Tcore"] * DIMENSIONLESS
        )
        | (coronagraph.photometric_aperture_throughput == 0.0 * DIMENSIONLESS)
    )


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_photometric_aperture_default_tcore(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    caplog,
    ifs_yipcoronagraph_basic_params,
):
    """Test that default Tcore is used when not provided."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = ifs_yipcoronagraph_basic_params.copy()
    del parameters["psf_trunc_ratio"]
    parameters["photometric_aperture_radius"] = 0.3
    mediator = MockMediator_IMAGER()

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator)

    assert any(
        "Using photometric_aperture_radius to calculate Omega..." in record.message
        for record in caplog.records
    )
    assert any("Using default Tcore..." in record.message for record in caplog.records)

    assert coronagraph.omega_lod.shape == (coronagraph.npix, coronagraph.npix, 1)
    assert np.all(
        coronagraph.omega_lod
        == np.pi * parameters["photometric_aperture_radius"] ** 2 * LAMBDA_D**2
    )

    assert coronagraph.photometric_aperture_throughput.shape == (
        coronagraph.npix,
        coronagraph.npix,
        1,
    )
    assert np.all(
        (coronagraph.photometric_aperture_throughput == 0.2968371 * DIMENSIONLESS)
        | (coronagraph.photometric_aperture_throughput == 0.0 * DIMENSIONLESS)
    )


# ============================================================================
# Tests for CoronagraphYIP with path
# ============================================================================


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_from_path(
    mock_load_telescope,
    mock_load_instrument,
    mock_instrument,
    mock_telescope,
    coronagraph_path,
    ifs_yipcoronagraph_basic_params,
):
    """Test that CoronagraphYIP can be constructed from a path."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(path=coronagraph_path)
    parameters = ifs_yipcoronagraph_basic_params.copy()

    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.npix > 0
    assert hasattr(coronagraph, "Istar")


# # ============================================================================
# # Tests for CoronagraphYIP with pre-constructed yippy_coro
# # ============================================================================


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_preconstruced_yippy_coro(
    mock_load_telescope,
    mock_load_instrument,
    mock_instrument,
    mock_telescope,
    yippy_coronagraph,
    ifs_yipcoronagraph_basic_params,
):
    """Test that pre-constructed yippy_coro is used directly."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = ifs_yipcoronagraph_basic_params.copy()
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.npix == yippy_coronagraph.header.naxis1
    assert hasattr(coronagraph, "Istar")
    assert hasattr(coronagraph, "noisefloor")


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_preconstruced_yippy_trunc_ratio_mismatch_warning(
    mock_load_telescope,
    mock_load_instrument,
    mock_instrument,
    mock_telescope,
    yippy_coronagraph,
    caplog,
    ifs_yipcoronagraph_basic_params,
):
    """Test warning when yippy_coro psf_trunc_ratio differs from parameters."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = ifs_yipcoronagraph_basic_params.copy()

    mediator = MockMediator_IMAGER()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator)

    assert any(
        "Pre-constructed yippy_coro has psf_trunc_ratio=" in record.message
        for record in caplog.records
    )


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_nrolls_from_yippy_object(
    mock_load_telescope,
    mock_load_instrument,
    mock_instrument,
    mock_telescope,
    yippy_coronagraph,
    monkeypatch,
    ifs_yipcoronagraph_basic_params,
):
    """Test that nrolls is read from yippy object when available."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    monkeypatch.setattr(yippy_coronagraph, "nrolls", 4, raising=False)

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = ifs_yipcoronagraph_basic_params.copy()

    coronagraph.load_configuration(parameters, MockMediator_IMAGER())

    assert coronagraph.DEFAULT_CONFIG["nrolls"] == 4


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_missing_bandwidth_warning(
    mock_load_telescope,
    mock_load_instrument,
    mock_instrument,
    mock_telescope,
    caplog,
    ifs_yipcoronagraph_basic_params,
):
    """Test warning when bandwidth info is missing from YIP."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    # Create a mock yippy object without bandwidth info
    mock_yippy = MagicMock()

    # Create a mock header that explicitly lacks bandwidth attributes
    mock_header = MagicMock(spec=["pixscale", "naxis1", "xcenter", "ycenter"])
    mock_header.pixscale = MagicMock(value=0.1)
    mock_header.naxis1 = 100
    mock_header.xcenter = 50
    mock_header.ycenter = 50

    mock_yippy.header = mock_header
    mock_yippy.nrolls = 1
    mock_yippy.psf_trunc_ratio = 0.3
    # Deliberately NOT setting lambda0, minlam, maxlam to trigger the warning

    # Mock the methods that return arrays
    mock_yippy.sky_trans.return_value = np.ones((100, 100))
    mock_yippy.separation_map.return_value = np.ones((100, 100))
    mock_yippy.core_area_map.return_value = np.ones((100, 100))
    mock_yippy.throughput_map.return_value = np.ones((100, 100))
    mock_yippy.core_mean_intensity_map.return_value = np.ones((100, 100))

    coronagraph = CoronagraphYIP(yippy_coro=mock_yippy)
    parameters = ifs_yipcoronagraph_basic_params.copy()
    mediator = MockMediator_IFS()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator)

    print(caplog.records)
    # Check that the warning was logged
    assert any(
        "No bandwidth info in YIPs, setting coronagraph_bandwidth = 0.2"
        in record.message
        for record in caplog.records
    )

    # Verify the default value was set
    assert coronagraph.coronagraph_bandwidth == 0.2


# ============================================================================
# Tests for Coronagraph.validate_configuration
# ============================================================================


@pytest.fixture
def valid_coronagraph():
    """Fixture providing a coronagraph with valid configuration."""
    coronagraph = ToyModelCoronagraph()
    coronagraph.Istar = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.noisefloor = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.psf_trunc_ratio = 0.3 * DIMENSIONLESS
    coronagraph.photometric_aperture_throughput = np.ones((100, 100, 1)) * DIMENSIONLESS
    coronagraph.omega_lod = np.ones((100, 100, 1)) * LAMBDA_D**2
    coronagraph.skytrans = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.pixscale = 0.1 * LAMBDA_D
    coronagraph.npix = 100
    coronagraph.xcenter = 50 * PIXEL
    coronagraph.ycenter = 50 * PIXEL
    coronagraph.coronagraph_bandwidth = 0.1
    coronagraph.npsfratios = 1
    coronagraph.nrolls = 1
    coronagraph.coronagraph_spectral_resolution = 1 * DIMENSIONLESS
    return coronagraph


def test_validate_configuration_valid_setup(valid_coronagraph):
    """Test that validate_configuration passes with valid configuration."""
    # Should not raise any exception
    valid_coronagraph.validate_configuration()


def test_validate_configuration_missing_istar(valid_coronagraph):
    """Test that missing Istar attribute raises AttributeError."""
    delattr(valid_coronagraph, "Istar")
    with pytest.raises(AttributeError, match="Coronagraph is missing attribute: Istar"):
        valid_coronagraph.validate_configuration()


def test_validate_configuration_incorrect_npix_type(valid_coronagraph):
    """Test that non-integer npix raises TypeError."""
    valid_coronagraph.npix = 100.0  # Should be int
    with pytest.raises(
        TypeError, match="Coronagraph attribute npix should be an integer"
    ):
        valid_coronagraph.validate_configuration()


def test_validate_configuration_incorrect_bandwidth_type(valid_coronagraph):
    """Test that non-float bandwidth raises TypeError."""
    valid_coronagraph.coronagraph_bandwidth = "0.1"  # Should be float
    with pytest.raises(
        TypeError, match="Coronagraph attribute coronagraph_bandwidth should be a float"
    ):
        valid_coronagraph.validate_configuration()


def test_validate_configuration_incorrect_pixscale_units(valid_coronagraph):
    """Test that incorrect pixscale units raise ValueError."""
    valid_coronagraph.pixscale = 0.1 * u.m  # Incorrect unit
    with pytest.raises(
        ValueError, match="Coronagraph attribute pixscale has incorrect units"
    ):
        valid_coronagraph.validate_configuration()


def test_validate_configuration_pixscale_not_quantity(valid_coronagraph):
    """Test that pixscale without units raises TypeError."""
    valid_coronagraph.pixscale = 0.1  # Missing unit
    with pytest.raises(
        TypeError, match="Coronagraph attribute pixscale should be a Quantity"
    ):
        valid_coronagraph.validate_configuration()


def test_validate_configuration_with_psf_trunc_ratio(valid_coronagraph):
    """Test that validation passes with psf_trunc_ratio instead of aperture radius."""
    # valid_coronagraph already has psf_trunc_ratio set
    # Should not raise
    valid_coronagraph.validate_configuration()


def test_validate_configuration_missing_both_aperture_params(valid_coronagraph):
    """Test that missing both aperture parameters raises AttributeError."""
    delattr(valid_coronagraph, "psf_trunc_ratio")
    with pytest.raises(AttributeError, match="photometric_aperture_radius"):
        valid_coronagraph.validate_configuration()


def test_validate_configuration_photometric_aperture_radius_units(valid_coronagraph):
    """Test that missing both aperture parameters raises AttributeError."""
    delattr(valid_coronagraph, "psf_trunc_ratio")
    valid_coronagraph.photometric_aperture_radius = 0.2 * LAMBDA_D
    # should validate correctly
    valid_coronagraph.validate_configuration()
