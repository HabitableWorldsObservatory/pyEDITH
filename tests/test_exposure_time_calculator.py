import pytest
import numpy as np
from astropy import units as u
from unittest.mock import patch, MagicMock
import logging

from pyEDITH.exposure_time_calculator import *
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
from pyEDITH.telescopes import ToyModelTelescope
from pyEDITH.coronagraphs import ToyModelCoronagraph
from pyEDITH.detectors import ToyModelDetector

# ============================================================================
# Fixtures - Mock Objects
# ============================================================================


@pytest.fixture
def mock_observation():
    """Fixture providing a mock observation object."""
    observation = MagicMock()
    observation.td_limit = 1.0e20 * u.s
    observation.wavelength = u.Quantity([0.5], u.micron)
    observation.delta_wavelength = u.Quantity([0.5 * 0.2], u.micron)
    observation.filter_bandwidth = 0.2
    observation.SNR = u.Quantity([7], DIMENSIONLESS)
    observation.CRb_multiplier = 2
    observation.tp = 0.0 * u.s
    # Initialize some arrays needed for outputs...
    observation.exptime = np.full(1, 0.0) * TIME
    observation.fullsnr = np.full(1, 0.0) * DIMENSIONLESS

    return observation


@pytest.fixture
def mock_scene():
    """Fixture providing a mock scene object."""
    scene = MagicMock()
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
    scene.Fzodi_list = u.Quantity([6.11055505e-10], 1 / u.arcsec**2)
    scene.Fexozodi_list = u.Quantity([2.97724302e-09], 1 / u.arcsec**2)
    scene.ez_PPF = u.Quantity([np.inf], DIMENSIONLESS)
    scene.Fbinary_list = u.Quantity([0], DIMENSIONLESS)
    return scene


@pytest.fixture
def mock_observatory():
    """Fixture providing a mock observatory object with all components."""
    observatory = MagicMock()
    observatory.observing_mode = "IMAGER"
    observatory.optics_throughput = u.Quantity([0.362], DIMENSIONLESS)
    observatory.epswarmTrcold = u.Quantity([0.638], DIMENSIONLESS)
    observatory.total_throughput = u.Quantity([0.23135872], u.electron / u.photon)

    # Configure telescope
    observatory.telescope = ToyModelTelescope()
    observatory.telescope.path = None
    observatory.telescope.keyword = "ToyModel"
    observatory.telescope.diameter = 7.87 * u.m
    observatory.telescope.unobscured_area = 0.879
    observatory.telescope.toverhead_fixed = 8381.3 * u.s
    observatory.telescope.toverhead_multi = 1.1 * DIMENSIONLESS

    observatory.telescope.temperature = 290.0 * u.K
    observatory.telescope.T_contamination = 0.95 * DIMENSIONLESS
    observatory.telescope.Area = 42.75906827 * u.m**2

    # Configure detector
    observatory.detector = ToyModelDetector()
    observatory.detector.path = None
    observatory.detector.keyword = "ToyModel"
    observatory.detector.pixscale_mas = 6.55224925 * u.mas
    observatory.detector.npix_multiplier = u.Quantity(1.0, DIMENSIONLESS)
    observatory.detector.DC = u.Quantity([3.0e-05], DARK_CURRENT)
    observatory.detector.RN = u.Quantity([0.0], READ_NOISE)
    observatory.detector.tread = u.Quantity([1000.0], READ_TIME)
    observatory.detector.CIC = u.Quantity([0.0013], CLOCK_INDUCED_CHARGE)
    observatory.detector.QE = u.Quantity([0.897], QUANTUM_EFFICIENCY)
    observatory.detector.dQE = u.Quantity([0.75], DIMENSIONLESS)

    # Configure coronagraph
    observatory.coronagraph = ToyModelCoronagraph()
    observatory.coronagraph.path = None
    observatory.coronagraph.keyword = "ToyModel"
    observatory.coronagraph.pixscale = 30.0 * LAMBDA_D
    observatory.coronagraph.contrast = 1.05e-13 * DIMENSIONLESS
    observatory.coronagraph.noisefloor_factor = 0.03 * DIMENSIONLESS
    observatory.coronagraph.coronagraph_bandwidth = 0.2
    observatory.coronagraph.photometric_aperture_radius = 0.85 * LAMBDA_D
    observatory.coronagraph.psf_trunc_ratio = 0.3 * DIMENSIONLESS
    observatory.coronagraph.Tcore = 0.2968371 * DIMENSIONLESS
    observatory.coronagraph.TLyot = 0.65 * DIMENSIONLESS
    observatory.coronagraph.nrolls = 2

    observatory.coronagraph.coronagraph_spectral_resolution = 1.0 * DIMENSIONLESS
    observatory.coronagraph.npsfratios = 1
    observatory.coronagraph.npix = 4
    observatory.coronagraph.xcenter = 2.0 * PIXEL
    observatory.coronagraph.ycenter = 2.0 * PIXEL
    observatory.coronagraph.r = u.Quantity(
        [
            [63.63961031, 47.4341649, 47.4341649, 63.63961031],
            [47.4341649, 21.21320344, 21.21320344, 47.4341649],
            [47.4341649, 21.21320344, 21.21320344, 47.4341649],
            [63.63961031, 47.4341649, 47.4341649, 63.63961031],
        ],
        LAMBDA_D,
    )
    observatory.coronagraph.omega_lod = u.Quantity(
        [
            [[2.26980069], [2.26980069], [2.26980069], [2.26980069]],
            [[2.26980069], [2.26980069], [2.26980069], [2.26980069]],
            [[2.26980069], [2.26980069], [2.26980069], [2.26980069]],
            [[2.26980069], [2.26980069], [2.26980069], [2.26980069]],
        ],
        LAMBDA_D**2,
    )
    observatory.coronagraph.skytrans = u.Quantity(
        [
            [0.65, 0.65, 0.65, 0.65],
            [0.65, 0.65, 0.65, 0.65],
            [0.65, 0.65, 0.65, 0.65],
            [0.65, 0.65, 0.65, 0.65],
        ],
        DIMENSIONLESS,
    )
    observatory.coronagraph.photometric_aperture_throughput = u.Quantity(
        [
            [[0.0], [0.2968371], [0.2968371], [0.0]],
            [[0.2968371], [0.2968371], [0.2968371], [0.2968371]],
            [[0.2968371], [0.2968371], [0.2968371], [0.2968371]],
            [[0.0], [0.2968371], [0.2968371], [0.0]],
        ],
        DIMENSIONLESS,
    )
    observatory.coronagraph.PSFpeak = u.Quantity(0.01625, DIMENSIONLESS)
    observatory.coronagraph.Istar = u.Quantity(
        [
            [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
            [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
            [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
            [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
        ],
        DIMENSIONLESS,
    )
    observatory.coronagraph.noisefloor = u.Quantity(
        [
            [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
            [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
            [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
            [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
        ],
        DIMENSIONLESS,
    )

    return observatory


@pytest.fixture
def mock_coronagraph():
    """Fixture providing a mock coronagraph for IWA performance testing."""
    coronagraph = MagicMock()
    coronagraph.photometric_aperture_throughput = (
        np.ones((100, 100, 1)) * 0.5 * DIMENSIONLESS
    )
    coronagraph.Istar = np.ones((100, 100)) * 1e-10 * DIMENSIONLESS
    coronagraph.skytrans = np.ones((100, 100)) * 0.9 * DIMENSIONLESS
    coronagraph.omega_lod = np.ones((100, 100, 1)) * 2 * LAMBDA_D**2
    coronagraph.npix = 100
    coronagraph.xcenter = 50 * PIXEL
    coronagraph.ycenter = 50 * PIXEL
    return coronagraph


# ============================================================================
# Tests for count rate calculation functions
# ============================================================================


def test_calculate_CRp():
    """Test planet count rate calculation."""
    F0 = 13400.0 * PHOTON_FLUX_DENSITY
    Fs_over_F0 = 0.005311289818550127 * DIMENSIONLESS
    Fp_over_Fs = 1e-9 * DIMENSIONLESS
    area = 427590.68268120557 * u.cm**2
    Upsilon = 0.2968371 * DIMENSIONLESS
    throughput = 0.35910000000000003 * ELECTRON / PHOTON_COUNT
    dlambda = 100 * u.nm

    result = calculate_CRp(
        F0, Fs_over_F0, Fp_over_Fs, area, Upsilon, throughput, dlambda
    )

    assert result.unit == (u.electron / u.s)
    assert np.isclose(result.value, 0.32438937)


def test_calculate_CRbs():
    """Test stellar speckle count rate calculation."""
    F0 = 13400.0 * PHOTON_FLUX_DENSITY
    Fs_over_F0 = 0.005311289818550127 * DIMENSIONLESS
    Istar = 2.3272595994978797e-14 * DIMENSIONLESS
    area = 427590.68268120557 * u.cm**2
    pixscale = 0.25 * LAMBDA_D
    throughput = 0.35910000000000003 * ELECTRON / PHOTON_COUNT
    dlambda = 100 * u.nm

    result = calculate_CRbs(F0, Fs_over_F0, Istar, area, pixscale, throughput, dlambda)

    assert result.unit == (u.electron / u.s)
    assert np.isclose(result.value, 0.00040692395)


def test_calculate_CRbz():
    """Test zodiacal light count rate calculation."""
    F0 = 13400.0 * PHOTON_FLUX_DENSITY
    Fzodi = 3.5213620474344346e-10 * INV_SQUARE_ARCSEC
    skytrans = 0.4006394155914143 * DIMENSIONLESS
    area = 427590.68268120557 * u.cm**2
    throughput = 0.35910000000000003 * ELECTRON / PHOTON_COUNT
    dlambda = 100 * u.nm
    lod_arcsec = 0.013104498490920989 * ARCSEC

    result = calculate_CRbz(F0, Fzodi, lod_arcsec, skytrans, area, throughput, dlambda)

    assert result.unit == (u.electron / u.s)
    assert np.isclose(result.value, 0.0049848673)


def test_calculate_CRbez():
    """Test exozodiacal light count rate calculation."""
    F0 = 13400.0 * PHOTON_FLUX_DENSITY
    Fexozodi = 7.1490465158365465e-09 * INV_SQUARE_ARCSEC
    skytrans = 0.6161309232588068 * DIMENSIONLESS
    sp = 0.02784705929320709 * ARCSEC
    dist = 18.195476531982425 * u.pc
    area = 427590.68268120557 * u.cm**2
    throughput = 0.35910000000000003 * ELECTRON / PHOTON_COUNT
    dlambda = 100 * u.nm
    lod_arcsec = 0.013104498490920989 * ARCSEC

    result = calculate_CRbez(
        F0,
        Fexozodi,
        lod_arcsec,
        skytrans,
        area,
        throughput,
        dlambda,
        dist,
        sp,
    )

    assert result.unit == (u.electron / u.s)
    assert np.isclose(result.value, 0.6062124)


def test_calculate_CRbbin():
    """Test binary companion count rate calculation."""
    F0 = 13400.0 * PHOTON_FLUX_DENSITY
    Fbinary = 0.0 * INV_SQUARE_ARCSEC
    skytrans = 0.6161309232588068 * DIMENSIONLESS
    area = 427590.68268120557 * u.cm**2
    throughput = 0.35910000000000003 * ELECTRON / PHOTON_COUNT
    dlambda = 100 * u.nm

    result = calculate_CRbbin(F0, Fbinary, skytrans, area, throughput, dlambda)

    assert result.unit == (u.electron / u.s)
    assert result.value == 0


def test_calculate_CRbth():
    """Test thermal background count rate calculation."""
    lam = 0.5 * WAVELENGTH
    area = 427590.68268120557 * u.cm**2
    dlambda = 100 * u.nm
    temp = 290 * u.K
    lod_rad = 6.353240152477764e-08 * u.rad
    emis = 0.468 * DIMENSIONLESS
    QE = 0.675 * QUANTUM_EFFICIENCY
    dQE = 1.0 * DIMENSIONLESS

    result = calculate_CRbth(lam, area, dlambda, temp, lod_rad, emis, QE, dQE)

    assert result.unit == (u.electron / u.s)
    assert np.isclose(result.value, 2.848015e-30)


def test_calculate_CRbd():
    """Test detector noise count rate calculation."""
    det_npix = 9.054697 * PIXEL
    det_DC = 3e-05 * DARK_CURRENT
    det_RN = 2 * READ_NOISE
    det_tread = 1000.0 * READ_TIME
    det_CIC = 1e-3 * CLOCK_INDUCED_CHARGE
    t_photon_count = 13.79303 * TIME / FRAME

    result = calculate_CRbd(
        det_npix, det_DC, det_RN, det_tread, det_CIC, t_photon_count
    )

    assert result.unit == (u.electron / u.s)
    assert np.isclose(result.value, 0.037146898)


def test_calculate_CRnf():
    """Test noise floor count rate calculation."""
    F0 = 13400.0 * PHOTON_FLUX_DENSITY
    Fs_over_F0 = 0.005311289818550127 * DIMENSIONLESS
    area = 427590.68268120557 * u.cm**2
    pixscale = 0.25 * LAMBDA_D
    throughput = 0.35910000000000003 * ELECTRON / PHOTON_COUNT
    dlambda = 100 * u.nm
    noisefloor = 7.25659425003725e-18 * DIMENSIONLESS

    result = calculate_CRnf(
        F0, Fs_over_F0, area, pixscale, throughput, dlambda, noisefloor
    )

    assert result.unit == (u.electron / u.s)
    assert np.isclose(
        result.value, 2.5376479000e-07 / 2
    )  # Value changed because of Update 1.7.1: SNR now outside CRnf


def test_calculate_t_photon_count():
    """Test photon counting time calculation."""
    det_npix = 9.054697 * PIXEL
    det_CR = 0.723971066592388 * ELECTRON / TIME

    result = calculate_t_photon_count(det_npix, det_CR)

    assert result.unit == (u.s / FRAME)
    assert np.isclose(result.value, 1.8583934)


# ============================================================================
# Tests for calculate_exposure_time_or_snr - Basic modes
# ============================================================================


def test_calculate_exposure_time_mode(mock_observation, mock_scene, mock_observatory):
    """Test exposure time calculation mode."""
    calculate_exposure_time_or_snr(mock_observation, mock_scene, mock_observatory)

    assert hasattr(mock_observation, "exptime")
    assert mock_observation.exptime.unit == u.s
    assert np.isclose(mock_observation.exptime.value, 487839.70631343)


def test_calculate_snr_mode(mock_observation, mock_scene, mock_observatory):
    """Test SNR calculation mode."""
    mock_observation.obstime = 10 * u.hr

    calculate_exposure_time_or_snr(
        mock_observation, mock_scene, mock_observatory, mode="signal_to_noise"
    )

    assert hasattr(mock_observation, "fullsnr")
    assert mock_observation.fullsnr.unit == u.dimensionless_unscaled
    assert np.isclose(mock_observation.fullsnr.value, 1.41457256)


def test_calculate_exposure_time_etc_validation(
    mock_observation, mock_scene, mock_observatory, caplog
):
    """Test ETC validation with fixed detector parameters."""
    mock_observatory.detector.det_npix_input = 100 * DIMENSIONLESS
    mock_observatory.detector.t_photon_count_input = 1.0 * TIME / FRAME

    with caplog.at_level(logging.DEBUG, logger="pyEDITH"):
        calculate_exposure_time_or_snr(
            mock_observation, mock_scene, mock_observatory, ETC_validation=True
        )

    assert np.isclose(mock_observation.validation_variables["det_npix"].value, 100)
    assert np.isclose(
        mock_observation.validation_variables["t_photon_count"].value, 1.0
    )


def test_calculate_exposure_time_invalid_mode(
    mock_observation, mock_scene, mock_observatory
):
    """Test that invalid mode raises ValueError."""
    with pytest.raises(
        ValueError, match="Invalid mode. Use 'exposure_time' or 'signal_to_noise'."
    ):
        calculate_exposure_time_or_snr(
            mock_observation, mock_scene, mock_observatory, mode="invalid"
        )


def test_calculate_exposure_time_verbose_output(
    mock_observation, mock_scene, mock_observatory, caplog
):
    """Test that verbose mode prints debug information."""
    with patch(
        "pyEDITH.exposure_time_calculator.utils.print_all_variables"
    ) as mock_print:
        with caplog.at_level(logging.DEBUG, logger="pyEDITH"):
            calculate_exposure_time_or_snr(
                mock_observation, mock_scene, mock_observatory
            )

        mock_print.assert_called()
        args, kwargs = mock_print.call_args
        assert args[0] == mock_observation
        assert args[1] == mock_scene
        assert args[2] == mock_observatory


def test_calculate_exposure_time_photometric_aperture_too_small(
    mock_observation, mock_scene, mock_observatory, caplog
):
    """Test that insufficient photometric aperture returns infinity."""
    original_omega_lod = mock_observatory.coronagraph.omega_lod.copy()
    mock_observatory.coronagraph.omega_lod = u.Quantity(
        np.zeros_like(original_omega_lod), LAMBDA_D**2
    )
    mock_observation.obstime = 10 * u.hr
    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        calculate_exposure_time_or_snr(mock_observation, mock_scene, mock_observatory)

    assert np.isinf(mock_observation.exptime[0])
    assert any(
        "Photometric aperture is not large enough" in record.message
        for record in caplog.records
        if record.levelno == logging.ERROR
    )


def test_calculate_snr_photometric_aperture_too_small(
    mock_observation, mock_scene, mock_observatory, caplog
):
    """Test that insufficient photometric aperture returns infinity in SNR mode."""

    mock_observation.obstime = 10 * u.hr

    original_omega_lod = mock_observatory.coronagraph.omega_lod.copy()
    mock_observatory.coronagraph.omega_lod = u.Quantity(
        np.zeros_like(original_omega_lod), LAMBDA_D**2
    )

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        calculate_exposure_time_or_snr(
            mock_observation, mock_scene, mock_observatory, mode="signal_to_noise"
        )

    assert np.isinf(mock_observation.fullsnr[0])
    assert any(
        "Photometric aperture is not large enough" in record.message
        for record in caplog.records
        if record.levelno == logging.ERROR
    )


def test_calculate_exposure_time_planet_below_noisefloor(
    mock_observation, mock_scene, mock_observatory, caplog
):
    """Test that planet signal below noise floor returns infinity."""
    original_noisefloor = mock_observatory.coronagraph.noisefloor.copy()
    mock_observatory.coronagraph.noisefloor = u.Quantity(
        np.ones_like(original_noisefloor) * 1e10, DIMENSIONLESS
    )

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        calculate_exposure_time_or_snr(mock_observation, mock_scene, mock_observatory)

    assert np.isinf(mock_observation.exptime[0])
    assert any(
        "Count rate of the planet smaller than the noise floor" in record.message
        for record in caplog.records
        if record.levelno == logging.ERROR
    )


def test_calculate_exposure_time_negative_result(
    mock_observation, mock_scene, mock_observatory
):
    """Test that negative calculated exposure time returns infinity."""
    mock_observatory.telescope.toverhead_fixed = -1000000 * u.s

    calculate_exposure_time_or_snr(mock_observation, mock_scene, mock_observatory)

    assert np.isinf(mock_observation.exptime[0])


def test_calculate_exposure_time_beyond_td_limit(
    mock_observation, mock_scene, mock_observatory
):
    """Test that exposure time beyond limit returns infinity."""
    mock_observation.td_limit = 1 * u.s

    calculate_exposure_time_or_snr(mock_observation, mock_scene, mock_observatory)

    assert np.isinf(mock_observation.exptime[0])


def test_calculate_exposure_time_planet_outside_yip_image(
    mock_observation, mock_scene, mock_observatory, caplog
):
    """Test that planet outside coronagraph YIP image returns infinity."""
    mock_scene.xp = 10 * u.arcsec

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        calculate_exposure_time_or_snr(mock_observation, mock_scene, mock_observatory)

    assert np.isinf(mock_observation.exptime[0])
    assert any(
        "Planet outside coronagraph YIP image" in record.message
        for record in caplog.records
        if record.levelno == logging.ERROR
    )


# ============================================================================
# Tests for calculate_exposure_time_or_snr - Bandwidth handling
# ============================================================================


def test_calculate_exposure_time_bandwidth_restriction_warning(
    mock_observation, mock_scene, mock_observatory, caplog
):
    """Test that exceeding coronagraph bandwidth triggers warning."""
    mock_observation.wavelength = [0.5] * u.um
    mock_observation.filter_bandwidth = 0.5
    mock_observatory.observing_mode = "IMAGER"
    mock_observatory.coronagraph.coronagraph_bandwidth = 0.1

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        calculate_exposure_time_or_snr(mock_observation, mock_scene, mock_observatory)

    assert any(
        "Bandwidth larger than what the coronagraph allows" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )
    assert np.isclose(
        mock_observation.validation_variables["deltalambda_nm"].value,
        mock_observatory.coronagraph.coronagraph_bandwidth
        * mock_observation.wavelength.to(NM).value,
    )


def test_calculate_exposure_time_ifs_mode(
    mock_observation, mock_scene, mock_observatory
):
    """Test that IFS mode handles multiple wavelengths correctly."""
    mock_observatory.observing_mode = "IFS"
    mock_observation.wavelength = [0.5, 0.6, 0.7] * u.um

    calculate_exposure_time_or_snr(mock_observation, mock_scene, mock_observatory)

    assert hasattr(mock_observation, "exptime")


def test_calculate_exposure_time_invalid_observing_mode(
    mock_observation, mock_scene, mock_observatory
):
    """Test that invalid observing mode raises ValueError."""
    mock_observatory.observing_mode = "INVALID"

    with pytest.raises(
        ValueError, match="Invalid observation mode. Choose 'IMAGER' or 'IFS'."
    ):
        calculate_exposure_time_or_snr(mock_observation, mock_scene, mock_observatory)


# ============================================================================
# Tests for measure_coronagraph_performance_at_IWA
# ============================================================================


def test_measure_coronagraph_performance_at_iwa(mock_coronagraph):
    """Test coronagraph performance measurement returns correct structure."""
    oneopixscale_arcsec = 10 / u.arcsec

    result = measure_coronagraph_performance_at_IWA(
        mock_coronagraph.photometric_aperture_throughput,
        mock_coronagraph.Istar,
        mock_coronagraph.skytrans,
        mock_coronagraph.omega_lod,
        mock_coronagraph.npix,
        mock_coronagraph.xcenter,
        mock_coronagraph.ycenter,
        oneopixscale_arcsec,
    )

    assert len(result) == 6
    assert all(isinstance(r, u.Quantity) for r in result)
