import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from pyEDITH.units import (
    LAMBDA_D,
    DIMENSIONLESS,
    WAVELENGTH,
    FRAME,
    INV_SQUARE_ARCSEC,
    QUANTUM_EFFICIENCY,
    LENGTH,
    PHOTON_FLUX_DENSITY,
)
import pytest
import os
from io import StringIO
import tempfile
from pyEDITH.utils import *
from pyEDITH import (
    Observation,
    AstrophysicalScene,
    Observatory,
)
from pyEDITH.telescopes import ToyModelTelescope
from pyEDITH.coronagraphs import ToyModelCoronagraph
from pyEDITH.detectors import ToyModelDetector

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_params_with_wavelength():
    """Fixture providing sample parameters with wavelength array for testing."""
    return {
        "lam": np.array([0.4, 0.5, 0.6, 0.7, 0.8]) * WAVELENGTH,
        "value": np.array([1, 2, 3, 4, 5]) * DIMENSIONLESS,
    }


@pytest.fixture
def test_object():
    """Fixture providing a fresh test object for each test."""

    class TestObject:
        pass

    return TestObject()


@pytest.fixture
def mock_observation():
    """Fixture providing a configured mock Observation object."""
    obs = Observation()
    obs.wavelength = [500, 600, 700] * u.nm
    obs.SNR = [10, 10, 10] * DIMENSIONLESS
    obs.td_limit = 24 * u.hour
    obs.CRb_multiplier = 1.0
    obs.fullsnr = [5, 6, 7] * DIMENSIONLESS
    obs.exptime = [1000, 1200, 1400] * u.s
    return obs


@pytest.fixture
def mock_scene():
    """Fixture providing a configured mock AstrophysicalScene object."""
    scene = AstrophysicalScene()
    scene.mag = 5.0
    scene.stellar_angular_diameter_arcsec = 0.1 * u.arcsec
    scene.F0 = [1e-8, 1e-8, 1e-8] * u.photon / (u.s * u.cm**2 * u.nm)
    scene.Fp_over_Fs = [1e-10, 1e-10, 1e-10] * DIMENSIONLESS
    scene.dist = 10 * u.pc
    scene.Fs_over_F0 = [1e-5, 1e-5, 1e-5]
    scene.Fzodi_list = [1e-7, 1e-7, 1e-7] * INV_SQUARE_ARCSEC
    scene.Fexozodi_list = [1e-8, 1e-8, 1e-8] * INV_SQUARE_ARCSEC
    scene.Fbinary_list = [1e-9, 1e-9, 1e-9] * DIMENSIONLESS
    scene.xp = 0.5 * u.arcsec
    scene.yp = 0.3 * u.arcsec
    scene.separation = 0.6 * u.arcsec
    return scene


@pytest.fixture
def mock_observatory():
    """Fixture providing a configured mock Observatory object."""
    observatory = Observatory()
    observatory.telescope = ToyModelTelescope()
    observatory.coronagraph = ToyModelCoronagraph()
    observatory.detector = ToyModelDetector()

    observatory.telescope.diameter = 2.4 * u.m
    observatory.telescope.temperature = 270 * u.K
    observatory.telescope.toverhead_multi = 1.1
    observatory.telescope.toverhead_fixed = 300 * u.s
    observatory.total_throughput = [0.3, 0.3, 0.3]
    observatory.epswarmTrcold = [0.1, 0.1, 0.1]

    observatory.coronagraph.coronagraph_bandwidth = 0.2
    observatory.coronagraph.Istar = np.ones((10, 10)) * 1e-10 * DIMENSIONLESS
    observatory.coronagraph.noisefloor = np.ones((10, 10)) * 1e-11 * DIMENSIONLESS
    observatory.coronagraph.npix = 100
    observatory.coronagraph.psf_trunc_ratio = 0.3
    observatory.coronagraph.pixscale = 0.1 * u.arcsec / u.pix
    observatory.coronagraph.photometric_aperture_throughput = (
        np.ones((10, 10, 1)) * 0.5 * DIMENSIONLESS
    )
    observatory.coronagraph.skytrans = np.ones((10, 10)) * 0.9 * DIMENSIONLESS
    observatory.coronagraph.omega_lod = np.ones((10, 10, 1)) * 0.1 * LAMBDA_D**2
    observatory.coronagraph.xcenter = 50 * u.pix
    observatory.coronagraph.ycenter = 50 * u.pix
    observatory.coronagraph.npsfratios = 1
    observatory.coronagraph.nrolls = 1

    observatory.detector.pixscale_mas = 100 * u.mas
    observatory.detector.QE = np.array([0.8, 0.8, 0.8]) * QUANTUM_EFFICIENCY
    observatory.detector.dQE = np.array([1.0, 1.0, 1.0]) * DIMENSIONLESS
    observatory.detector.npix_multiplier = 1.0 * DIMENSIONLESS
    observatory.detector.DC = [1e-3, 1e-3, 1e-3] * u.electron / u.s / u.pix
    observatory.detector.RN = [3, 3, 3] * u.electron / u.pix
    observatory.detector.tread = [100, 100, 100] * u.s
    observatory.detector.CIC = [1e-3, 1e-3, 1e-3] * u.electron / u.pix / FRAME

    return observatory


@pytest.fixture
def validation_kwargs():
    """Fixture providing keyword arguments for validation testing."""
    return {
        "deltalambda_nm": 1 * u.nm,
        "lod": 1 * u.dimensionless_unscaled,
        "lod_rad": 1 * u.rad,
        "lod_arcsec": 1 * u.arcsec,
        "area_cm2": 1 * u.cm**2,
        "detpixscale_lod": 1 * LAMBDA_D,
        "stellar_diam_lod": 1 * LAMBDA_D,
        "pixscale_rad": 1 * u.rad,
        "oneopixscale_arcsec": 1 / u.arcsec,
        "det_sep_pix": 1 * u.pix,
        "det_sep": 1 * u.arcsec,
        "det_Istar": 1 * u.dimensionless_unscaled,
        "det_skytrans": 1 * u.dimensionless_unscaled,
        "det_photometric_aperture_throughput": 1 * u.dimensionless_unscaled,
        "det_omega_lod": 1 * LAMBDA_D**2,
        "det_CRp": 1 * u.electron / u.s,
        "det_CRbs": 1 * u.electron / u.s,
        "det_CRbz": 1 * u.electron / u.s,
        "det_CRbez": 1 * u.electron / u.s,
        "det_CRbbin": 1 * u.electron / u.s,
        "det_CRbth": 1 * u.electron / u.s,
        "det_CR": 1 * u.electron / u.s,
        "ix": 1,
        "iy": 1,
        "sp_lod": 1 * LAMBDA_D,
        "CRp": 1 * u.electron / u.s,
        "CRnf": 1 * u.electron / u.s,
        "CRbs": 1 * u.electron / u.s,
        "CRbz": 1 * u.electron / u.s,
        "CRbez": 1 * u.electron / u.s,
        "CRbbin": 1 * u.electron / u.s,
        "t_photon_count": 1 * u.s,
        "CRbd": 1 * u.electron / u.s,
        "CRbth": 1 * u.electron / u.s,
        "CRb": 1 * u.electron / u.s,
    }


# ============================================================================
# Tests for average_over_bandpass
# ============================================================================


def test_average_over_bandpass_basic(sample_params_with_wavelength):
    """Test basic averaging over bandpass with simple values."""
    wavelength_range = [0.45 * WAVELENGTH, 0.75 * WAVELENGTH]

    result = average_over_bandpass(sample_params_with_wavelength, wavelength_range)

    assert np.isclose(result["value"].value, 3)


def test_average_over_bandpass_preserves_non_array_params(
    sample_params_with_wavelength,
):
    """Test that non-array parameters are preserved unchanged."""
    sample_params_with_wavelength["scalar_param"] = 42
    wavelength_range = [0.45 * WAVELENGTH, 0.75 * WAVELENGTH]

    result = average_over_bandpass(sample_params_with_wavelength, wavelength_range)

    assert result["scalar_param"] == 42


# ============================================================================
# Tests for interpolate_over_bandpass
# ============================================================================


def test_interpolate_over_bandpass_basic(sample_params_with_wavelength):
    """Test basic interpolation over bandpass with simple values."""
    wavelengths = u.Quantity([0.45, 0.55, 0.65, 0.75], WAVELENGTH)

    result = interpolate_over_bandpass(sample_params_with_wavelength, wavelengths)

    assert np.allclose(result["value"], np.array([1.5, 2.5, 3.5, 4.5]))


def test_interpolate_over_bandpass_preserves_non_array_params(
    sample_params_with_wavelength,
):
    """Test that non-array parameters are preserved unchanged."""
    sample_params_with_wavelength["scalar_param"] = 42
    wavelengths = u.Quantity([0.45, 0.55, 0.65, 0.75], WAVELENGTH)

    result = interpolate_over_bandpass(sample_params_with_wavelength, wavelengths)

    assert result["scalar_param"] == 42


# ============================================================================
# Tests for convert_to_numpy_array
# ============================================================================


def test_convert_to_numpy_array_with_quantity(test_object):
    """Test conversion of Quantity list to numpy array preserves units."""
    test_object.quantity_array = [1, 2, 3] * u.m
    array_params = ["quantity_array"]

    convert_to_numpy_array(test_object, array_params)

    assert isinstance(test_object.quantity_array, u.Quantity)
    assert isinstance(test_object.quantity_array.value, np.ndarray)
    assert test_object.quantity_array.unit == u.m
    assert np.array_equal(test_object.quantity_array.value, np.array([1, 2, 3]))
    assert test_object.quantity_array.dtype == np.float64


def test_convert_to_numpy_array_without_quantity(test_object):
    """Test conversion of regular list to numpy array."""
    test_object.regular_array = [4, 5, 6]
    array_params = ["regular_array"]

    convert_to_numpy_array(test_object, array_params)

    assert isinstance(test_object.regular_array, np.ndarray)
    assert np.array_equal(test_object.regular_array, np.array([4, 5, 6]))
    assert test_object.regular_array.dtype == np.float64


def test_convert_to_numpy_array_with_empty_list(test_object):
    """Test conversion of empty list produces empty numpy array."""
    test_object.empty_list = []
    array_params = ["empty_list"]

    convert_to_numpy_array(test_object, array_params)

    assert isinstance(test_object.empty_list, np.ndarray)
    assert test_object.empty_list.size == 0
    assert test_object.empty_list.dtype == np.float64


# ============================================================================
# Tests for validate_attributes - Basic validation
# ============================================================================


def test_validate_attributes_all_valid(test_object):
    """Test that validation passes with all correct attributes."""
    test_object.int_attr = 1
    test_object.float_attr = 1.0
    test_object.quantity_attr = 1.0 * u.m
    test_object.array_attr = np.array([1, 2, 3]) * u.m

    expected_args = {
        "int_attr": int,
        "float_attr": float,
        "quantity_attr": u.m,
        "array_attr": u.m,
    }

    # Should not raise
    validate_attributes(test_object, expected_args)


def test_validate_attributes_quantity_arrays(test_object):
    """Test validation of Quantity arrays."""
    test_object.array_attr = np.array([1, 2, 3]) * u.m

    expected_args = {"array_attr": u.m}

    # Should not raise
    validate_attributes(test_object, expected_args)


# ============================================================================
# Tests for validate_attributes - Missing attributes
# ============================================================================


def test_validate_attributes_missing_attribute(test_object):
    """Test that missing attribute raises AttributeError."""
    test_object.int_attr = 1

    expected_args = {"int_attr": int, "missing_attr": int}

    with pytest.raises(
        AttributeError, match="TestObject is missing attribute: missing_attr"
    ):
        validate_attributes(test_object, expected_args)


# ============================================================================
# Tests for validate_attributes - Incorrect types
# ============================================================================


def test_validate_attributes_incorrect_int_type(test_object):
    """Test that float instead of int raises TypeError."""
    test_object.int_attr = 1.0

    expected_args = {"int_attr": int}

    with pytest.raises(
        TypeError, match="TestObject attribute int_attr should be an integer"
    ):
        validate_attributes(test_object, expected_args)


def test_validate_attributes_incorrect_float_type(test_object):
    """Test that int instead of float raises TypeError."""
    test_object.float_attr = 1

    expected_args = {"float_attr": float}

    with pytest.raises(
        TypeError, match="TestObject attribute float_attr should be a float"
    ):
        validate_attributes(test_object, expected_args)


def test_validate_attributes_non_quantity_for_quantity_attr(test_object):
    """Test that non-Quantity value for Quantity attribute raises TypeError."""
    test_object.quantity_attr = 1.0

    expected_args = {"quantity_attr": u.m}

    with pytest.raises(
        TypeError, match="TestObject attribute quantity_attr should be a Quantity"
    ):
        validate_attributes(test_object, expected_args)


def test_validate_attributes_non_quantity_array(test_object):
    """Test that non-Quantity array raises TypeError when Quantity expected."""
    test_object.array_attr = np.array([1, 2, 3])

    expected_args = {"array_attr": u.m}

    with pytest.raises(
        TypeError, match="TestObject attribute array_attr should be a Quantity"
    ):
        validate_attributes(test_object, expected_args)


# ============================================================================
# Tests for validate_attributes - Incorrect units
# ============================================================================


def test_validate_attributes_incorrect_units(test_object):
    """Test that Quantity with incorrect units raises ValueError."""
    test_object.quantity_attr = 1.0 * u.s

    expected_args = {"quantity_attr": u.m}

    with pytest.raises(
        ValueError, match="TestObject attribute quantity_attr has incorrect units"
    ):
        validate_attributes(test_object, expected_args)


# ============================================================================
# Tests for validate_attributes - Unexpected specifications
# ============================================================================


def test_validate_attributes_unexpected_type_specification(test_object):
    """Test that unexpected type specification raises ValueError."""
    test_object.unexpected_attr = 10

    expected_args = {"unexpected_attr": "unexpected"}

    with pytest.raises(
        ValueError, match="Unexpected type specification for unexpected_attr"
    ):
        validate_attributes(test_object, expected_args)


# ============================================================================
# Tests for print_array_info - Full info mode
# ============================================================================


def test_print_array_info_with_units():
    """Test printing array info with units in full_info mode."""
    mock_file = StringIO()
    test_array = np.array([1, 2, 3]) * u.m

    print_array_info(mock_file, "test_array", test_array, mode="full_info")

    output = mock_file.getvalue()
    assert "test_array:" in output
    assert "Unit: m" in output
    assert "Shape: (3,)" in output
    assert "Max value: 3" in output
    assert "Min value: 1" in output


def test_print_array_info_none_input():
    """Test that function returns early when None is passed."""
    file = StringIO()
    print_array_info(file, "none_input", None)
    output = file.getvalue()
    # Should produce no output when arr is None
    assert output == ""


def test_print_array_info_empty_numpy_array():
    """Test printing info for empty numpy array."""
    empty_array = np.array([])
    file = StringIO()

    print_array_info(file, "empty_numpy_array", empty_array, mode="full_info")

    output = file.getvalue()
    assert "empty_numpy_array:" in output
    assert "Shape: (0,)" in output
    assert "Array is empty" in output


def test_print_array_info_empty_list():
    """Test printing info for empty list."""
    empty_list = []
    file = StringIO()

    print_array_info(file, "empty_list", empty_list, mode="full_info")

    output = file.getvalue()
    assert "empty_list:" in output
    assert "Shape: (0,)" in output
    assert "Array is empty" in output


def test_print_array_info_empty_quantity():
    """Test printing info for empty Quantity array."""
    empty_quantity = u.Quantity([], unit=u.m)
    file = StringIO()

    print_array_info(file, "empty_quantity", empty_quantity, mode="full_info")

    output = file.getvalue()
    assert "empty_quantity:" in output
    assert "Unit: m" in output
    assert "Shape: (0,)" in output
    assert "Array is empty" in output


def test_print_array_info_integer_scalar():
    """Test printing info for integer scalar in full_info mode."""
    file = StringIO()
    int_array = np.array(42)

    print_array_info(file, "int_scalar", int_array, mode="full_info")

    output = file.getvalue()
    assert "int_scalar:" in output
    assert "Shape: scalar" in output
    assert "Value: 42" in output


def test_print_array_info_float_scalar_full_info():
    """Test printing info for float scalar in full_info mode."""
    file = StringIO()
    test_array = np.array(3.14)

    print_array_info(file, "test_scalar", test_array, mode="full_info")

    output = file.getvalue()
    assert "test_scalar:" in output
    assert "Shape: scalar" in output


def test_print_array_info_none_scalar_full_info():
    """Test printing info for None scalar in full_info mode."""
    file = StringIO()
    none_array = np.array(None)

    print_array_info(file, "none_scalar", none_array, mode="full_info")

    output = file.getvalue()
    assert "none_scalar:" in output
    assert "Value: None" in output


# ============================================================================
# Tests for print_array_info - Validation mode
# ============================================================================


def test_print_array_info_float_scalar_validation():
    """Test printing info for float scalar in validation mode."""
    file = StringIO()
    test_array = np.array(3.14)

    print_array_info(file, "test_scalar", test_array, mode="validation")

    output = file.getvalue()
    assert "test_scalar:" in output
    assert "value: 3.14" in output


def test_print_array_info_none_scalar_validation():
    """Test printing info for None scalar in validation mode."""
    file = StringIO()
    none_array = np.array(None)

    print_array_info(file, "none_scalar_val", none_array, mode="validation")

    output = file.getvalue()
    assert "none_scalar_val:" in output
    assert "value: None" in output


# ============================================================================
# Tests for print_all_variables
# ============================================================================


def test_print_all_variables_creates_files(
    mock_observation, mock_scene, mock_observatory
):
    """Test that print_all_variables creates both output files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        original_dir = os.getcwd()
        os.chdir(tmpdirname)
        try:
            mock_observation.validation_variables = {
                "deltalambda_nm": np.array([10.0]),
                "lod": np.array([0.05]),
                "CRp": np.array([1.0]),
                "CRbs": np.array([0.5]),
                "CRb": np.array([2.0]),
            }

            print_all_variables(mock_observation, mock_scene, mock_observatory)

            # A single call writes BOTH files via the internal `for mode` loop.
            assert os.path.exists("pyedith_validation.txt")
            assert os.path.exists("pyedith_full_info.txt")

        finally:
            os.chdir(original_dir)


@pytest.mark.parametrize("mode", ["validation", "full_info"])
def test_print_all_variables_file_structure(
    mode, mock_observation, mock_scene, mock_observatory
):
    """Test that output file contains all expected sections."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        original_dir = os.getcwd()
        os.chdir(tmpdirname)
        try:
            mock_observation.validation_variables = {
                "deltalambda_nm": np.array([10.0]),
                "lod": np.array([0.05]),
                "CRp": np.array([1.0]),
                "CRbs": np.array([0.5]),
                "CRb": np.array([2.0]),
            }

            print_all_variables(mock_observation, mock_scene, mock_observatory)

            with open(f"pyedith_{mode}.txt", "r") as file:
                content = file.read()

            # Check for main sections
            assert "Input Objects and Their Relevant Properties:" in content
            assert "1. Observation:" in content
            assert "2. Scene:" in content
            assert "3. Observatory:" in content
            assert "Telescope:" in content
            assert "Coronagraph:" in content
            assert "Detector:" in content
            assert "Calculated Variables:" in content

        finally:
            os.chdir(original_dir)


@pytest.mark.parametrize("mode", ["validation", "full_info"])
def test_print_all_variables_includes_attributes(
    mode, mock_observation, mock_scene, mock_observatory
):
    """Test that output file includes specific object attributes."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        original_dir = os.getcwd()
        os.chdir(tmpdirname)
        try:
            mock_observation.validation_variables = {
                "deltalambda_nm": np.array([10.0]),
                "lod": np.array([0.05]),
                "CRp": np.array([1.0]),
                "CRbs": np.array([0.5]),
                "CRb": np.array([2.0]),
            }

            print_all_variables(mock_observation, mock_scene, mock_observatory)

            with open(f"pyedith_{mode}.txt", "r") as file:
                content = file.read()

            # Check for some explicitly set attributes
            assert "observation.wavelength" in content
            assert "scene.mag" in content
            assert "observatory.telescope.diameter" in content
            assert "observatory.coronagraph.coronagraph_bandwidth" in content
            assert "observatory.detector.pixscale_mas" in content

        finally:
            os.chdir(original_dir)


@pytest.mark.parametrize("mode", ["validation", "full_info"])
def test_print_all_variables_includes_calculated_vars(
    mode, mock_observation, mock_scene, mock_observatory
):
    """Test that output file includes calculated variables read from
    observation.validation_variables."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        original_dir = os.getcwd()
        os.chdir(tmpdirname)
        try:
            mock_observation.validation_variables = {
                "deltalambda_nm": np.array([10.0]),
                "lod": np.array([0.05]),
                "CRp": np.array([1.0]),
                "CRbs": np.array([0.5]),
                "CRb": np.array([2.0]),
            }

            print_all_variables(mock_observation, mock_scene, mock_observatory)

            with open(f"pyedith_{mode}.txt", "r") as file:
                content = file.read()

            # Check for calculated variables
            assert "deltalambda_nm" in content
            assert "lod" in content
            assert "CRp" in content
            assert "CRb" in content

        finally:
            os.chdir(original_dir)


def test_print_all_variables_mode_specific_output_full_info(
    mock_observation, mock_scene, mock_observatory
):
    """Test that full_info mode includes shape and unit information."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        original_dir = os.getcwd()
        os.chdir(tmpdirname)
        try:
            mock_observation.validation_variables = {
                "deltalambda_nm": np.array([10.0]),
                "lod": np.array([0.05]),
                "CRp": np.array([1.0]),
                "CRbs": np.array([0.5]),
                "CRb": np.array([2.0]),
            }

            print_all_variables(mock_observation, mock_scene, mock_observatory)

            with open("pyedith_full_info.txt", "r") as file:
                content = file.read()

            assert "Shape:" in content
            assert "Unit:" in content

        finally:
            os.chdir(original_dir)


def test_print_all_variables_mode_specific_output_validation(
    mock_observation, mock_scene, mock_observatory
):
    """Test that validation mode includes value information."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        original_dir = os.getcwd()
        os.chdir(tmpdirname)
        try:
            mock_observation.validation_variables = {
                "deltalambda_nm": np.array([10.0]),
                "lod": np.array([0.05]),
                "CRp": np.array([1.0]),
                "CRbs": np.array([0.5]),
                "CRb": np.array([2.0]),
            }

            print_all_variables(mock_observation, mock_scene, mock_observatory)

            with open("pyedith_validation.txt", "r") as file:
                content = file.read()

            assert "value:" in content
            assert "max value:" in content or "min value:" in content

        finally:
            os.chdir(original_dir)


# ============================================================================
# Tests for synthesize_observation
# ============================================================================


def test_synthesize_observation_basic():
    """Test basic observation synthesis returns expected shapes."""
    mock_scene = AstrophysicalScene()
    snr_arr = np.array([10, 15, 20])
    mock_scene.Fp_over_Fs = np.array([1e-6, 1.5e-6, 2e-6])

    obs, noise = synthesize_observation(snr_arr, mock_scene)

    assert obs.shape == (3,)
    assert noise.shape == (3,)
    assert np.all(np.isfinite(obs))
    assert np.all(np.isfinite(noise))


def test_synthesize_observation_reproducible_with_seed():
    """Test that random seed produces reproducible results."""
    mock_scene = AstrophysicalScene()
    snr_arr = np.array([10, 15, 20])
    mock_scene.Fp_over_Fs = np.array([1e-6, 1.5e-6, 2e-6])

    obs1, noise1 = synthesize_observation(snr_arr, mock_scene, random_seed=42)
    obs2, noise2 = synthesize_observation(snr_arr, mock_scene, random_seed=42)

    np.testing.assert_array_equal(obs1, obs2)
    np.testing.assert_array_equal(noise1, noise2)


def test_synthesize_observation_below_zero_handling():
    """Test that negative observations are handled with set_below_zero parameter."""
    mock_scene = AstrophysicalScene()
    snr_arr = np.array([10, 15, 20])
    mock_scene.Fp_over_Fs = np.array([1e-6, 1.5e-6, 2e-6])

    obs, noise = synthesize_observation(snr_arr, mock_scene, set_below_zero=-999)

    assert np.all(obs[obs < 0] == -999)


# ============================================================================
# Tests for fixed_resolution_grid
# ============================================================================


# def test_fixed_resolution_grid_basic():
#     """Test basic wavelength grid generation at fixed resolution."""
#     x_min, x_max, res = 0.5, 1.0, 100

#     x, Dx = fixed_resolution_grid(x_min, x_max, res)

#     assert x[0] == x_min
#     assert x[-1] < x_max
#     assert len(x) == len(Dx)
#     assert np.all(np.diff(x) > 0)


# def test_fixed_resolution_grid_maintains_resolution():
#     """Test that wavelength grid maintains constant resolution."""
#     x_min, x_max, res = 0.5, 1.0, 100

#     x, Dx = fixed_resolution_grid(x_min, x_max, res)

#     np.testing.assert_allclose(x / Dx, res, rtol=1e-5)


# ============================================================================
# Tests for generate_wavelength_grid
# ============================================================================


def test_generate_wavelength_grid_with_boundaries():
    """Test wavelength regridding with specified channel boundaries."""
    res = 50
    lam_low = 0.3
    lam_high = 0.5

    lam, dlam = generate_wavelength_grid(res, lam_low, lam_high)

    assert np.all(np.diff(lam) > 0)
    assert len(lam) == len(dlam)


# ============================================================================
# Tests for regrid_spec_gaussconv
# ============================================================================


def test_regrid_spec_gaussconv_basic():
    """Test Gaussian convolution regridding produces correct output length."""
    input_wls = np.linspace(0.4, 2.0, 100)
    input_spec = np.random.rand(100)
    new_lam = np.linspace(0.5, 1.9, 50)
    new_dlam = np.gradient(new_lam)

    spec_regrid = regrid_spec_gaussconv(input_wls, input_spec, new_lam, new_dlam)

    assert len(spec_regrid) == len(new_lam)


# ============================================================================
# Tests for resample_to_wavelength_grid
# ============================================================================


def test_resample_to_wavelength_grid_broadcast_scalar():
    """Test that a single value is broadcast to the entire grid."""
    values = np.array([1.5])
    from_wavelength = np.array([1.0])
    to_wavelength = np.linspace(0.5, 2.0, 100)

    result = resample_to_wavelength_grid(values, from_wavelength, to_wavelength)

    assert len(result) == len(to_wavelength)
    assert np.all(result == 1.5)


def test_resample_to_wavelength_grid_passthrough():
    """Test that values already on the target grid pass through unchanged."""
    to_wavelength = np.linspace(0.5, 2.0, 50)
    values = np.random.rand(50)
    from_wavelength = to_wavelength.copy()

    result = resample_to_wavelength_grid(values, from_wavelength, to_wavelength)

    assert len(result) == len(to_wavelength)
    np.testing.assert_array_equal(result, values)


def test_resample_to_wavelength_grid_interpolation_1d():
    """Test regridding using 1D interpolation."""
    from_wavelength = np.linspace(0.4, 2.0, 100)
    values = np.random.rand(100)
    to_wavelength = np.linspace(0.5, 1.9, 50)

    result = resample_to_wavelength_grid(
        values, from_wavelength, to_wavelength, interpolation="1d"
    )

    assert len(result) == len(to_wavelength)
    assert result.dtype == np.float64


def test_resample_to_wavelength_grid_interpolation_gaussian():
    """Test regridding using Gaussian convolution."""
    from_wavelength = np.linspace(0.4, 2.0, 100)
    values = np.random.rand(100)
    to_wavelength = np.linspace(0.5, 1.9, 50)
    to_delta_wavelength = np.gradient(to_wavelength)

    result = resample_to_wavelength_grid(
        values,
        from_wavelength,
        to_wavelength,
        to_delta_wavelength,
        interpolation="Gaussian",
    )

    assert len(result) == len(to_wavelength)
    assert result.dtype == np.float64


def test_resample_to_wavelength_grid_gaussian_missing_delta_wavelength():
    """Test that Gaussian interpolation raises error without delta_wavelength."""
    from_wavelength = np.linspace(0.4, 2.0, 100)
    values = np.random.rand(100)
    to_wavelength = np.linspace(0.5, 1.9, 50)

    with pytest.raises(ValueError, match="to_delta_wavelength.*not provided"):
        resample_to_wavelength_grid(
            values, from_wavelength, to_wavelength, interpolation="Gaussian"
        )


def test_resample_to_wavelength_grid_invalid_interpolation():
    """Test that invalid interpolation method raises error."""
    values = np.array([1.0, 2.0, 3.0])
    from_wavelength = np.array([0.5, 1.0, 1.5])
    to_wavelength = np.linspace(0.5, 1.5, 10)

    with pytest.raises(ValueError, match="Unknown interpolation type"):
        resample_to_wavelength_grid(
            values, from_wavelength, to_wavelength, interpolation="invalid_method"
        )


def test_resample_to_wavelength_grid_with_quantity():
    """Test that astropy Quantity units are preserved."""
    values = np.array([1.0, 2.0, 3.0]) * u.Jy
    from_wavelength = np.array([0.5, 1.0, 1.5])
    to_wavelength = np.linspace(0.5, 1.5, 10)

    result = resample_to_wavelength_grid(
        values, from_wavelength, to_wavelength, interpolation="1d"
    )

    assert isinstance(result, u.Quantity)
    assert result.unit == u.Jy
    assert len(result) == len(to_wavelength)


def test_resample_to_wavelength_grid_broadcast_quantity():
    """Test that a single Quantity value is broadcast correctly."""
    values = np.array([2.5]) * u.erg / u.s / u.cm**2
    from_wavelength = np.array([1.0])
    to_wavelength = np.linspace(0.5, 2.0, 50)

    result = resample_to_wavelength_grid(values, from_wavelength, to_wavelength)

    assert isinstance(result, u.Quantity)
    assert result.unit == u.erg / u.s / u.cm**2
    assert len(result) == len(to_wavelength)
    assert np.all(result.value == 2.5)


def test_resample_to_wavelength_grid_name_parameter():
    """Test that custom name appears in log messages (requires log capture)."""
    # This test ensures the name parameter is accepted
    values = np.random.rand(10)
    from_wavelength = np.linspace(0.5, 1.5, 10)
    to_wavelength = np.linspace(0.5, 1.5, 20)

    result = resample_to_wavelength_grid(
        values, from_wavelength, to_wavelength, name="custom_flux", interpolation="1d"
    )

    assert len(result) == len(to_wavelength)


def test_resample_to_wavelength_grid_dtype_conversion():
    """Test that output is always float64."""
    values = np.array([1, 2, 3], dtype=np.int32)
    from_wavelength = np.array([0.5, 1.0, 1.5])
    to_wavelength = np.linspace(0.5, 1.5, 10)

    result = resample_to_wavelength_grid(
        values, from_wavelength, to_wavelength, interpolation="1d"
    )

    assert result.dtype == np.float64


# ============================================================================
# Tests for fill_parameters
# ============================================================================


def test_fill_parameters_basic_default_values(test_object):
    """Test that default parameters are correctly assigned when no user params provided."""
    parameters = {}
    default_parameters = {
        "param1": 10,
        "param2": 20.5,
        "param3": "test_string",
    }

    fill_parameters(test_object, parameters, default_parameters)

    assert test_object.param1 == 10
    assert test_object.param2 == 20.5
    assert test_object.param3 == "test_string"


def test_fill_parameters_user_override(test_object):
    """Test that user-provided parameters override defaults."""
    parameters = {
        "param1": 100,
        "param2": 200.5,
    }
    default_parameters = {
        "param1": 10,
        "param2": 20.5,
        "param3": "default_string",
    }

    fill_parameters(test_object, parameters, default_parameters)

    assert test_object.param1 == 100
    assert test_object.param2 == 200.5
    assert test_object.param3 == "default_string"


def test_fill_parameters_locked_keys_prevent_override(test_object, caplog):
    """Test that locked keys cannot be overridden by user parameters."""
    parameters = {
        "locked_param": 999,
        "unlocked_param": 50,
    }
    default_parameters = {
        "locked_param": 42,
        "unlocked_param": 25,
    }
    locked_keys = {"locked_param"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(test_object, parameters, default_parameters, locked_keys)

    # Locked parameter should retain default value
    assert test_object.locked_param == 42
    # Unlocked parameter should use user value
    assert test_object.unlocked_param == 50
    # Warning should be logged
    assert "locked_param" in caplog.text
    assert "locked in this mode" in caplog.text


def test_fill_parameters_quantity_unit_conversion(test_object):
    """Test that Quantity parameters with matching units are converted correctly."""
    parameters = {
        "distance": 5.0 * u.pc,  # parsecs
    }
    default_parameters = {
        "distance": 10.0 * u.m,  # meters (different unit)
    }

    fill_parameters(test_object, parameters, default_parameters)

    # Should be converted to meters (the default unit)
    assert isinstance(test_object.distance, u.Quantity)
    assert test_object.distance.unit == u.m
    # 5 pc = 1.54285714e+17 m
    assert np.isclose(test_object.distance.value, 1.54285714e17, rtol=1e-4)


def test_fill_parameters_quantity_without_units(test_object):
    """Test that unitless user value gets assigned default units from Quantity default."""
    parameters = {
        "length": 100.0,  # No units
    }
    default_parameters = {
        "length": 1.0 * u.m,  # Has units
    }

    fill_parameters(test_object, parameters, default_parameters)

    assert isinstance(test_object.length, u.Quantity)
    assert test_object.length.unit == u.m
    assert test_object.length.value == 100.0


def test_fill_parameters_quantity_user_override_with_units(test_object):
    """Test that user Quantity with units correctly overrides default."""
    parameters = {
        "wavelength": 500 * u.nm,
    }
    default_parameters = {
        "wavelength": 1.0 * u.um,  # Different value and unit
    }

    fill_parameters(test_object, parameters, default_parameters)

    assert isinstance(test_object.wavelength, u.Quantity)
    # Should be converted to microns (the default unit)
    assert test_object.wavelength.unit == u.um
    assert np.isclose(test_object.wavelength.value, 0.5)


def test_fill_parameters_mixed_locked_and_unlocked_quantities(test_object, caplog):
    """Test handling of both locked and unlocked Quantity parameters."""
    parameters = {
        "locked_distance": 100 * u.m,
        "unlocked_distance": 200 * u.m,
    }
    default_parameters = {
        "locked_distance": 10 * u.m,
        "unlocked_distance": 20 * u.m,
    }
    locked_keys = {"locked_distance"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(test_object, parameters, default_parameters, locked_keys)

    # Locked should use default
    assert test_object.locked_distance == 10 * u.m
    # Unlocked should use user value
    assert test_object.unlocked_distance == 200 * u.m
    # Warning should be logged only for locked parameter
    assert "locked_distance" in caplog.text


def test_fill_parameters_empty_locked_keys(test_object):
    """Test that passing None for locked_keys works correctly."""
    parameters = {
        "param1": 100,
    }
    default_parameters = {
        "param1": 10,
        "param2": 20,
    }

    # Should not raise any errors with locked_keys=None (default)
    fill_parameters(test_object, parameters, default_parameters, locked_keys=None)

    assert test_object.param1 == 100
    assert test_object.param2 == 20


def test_fill_parameters_all_locked_keys(test_object, caplog):
    """Test behavior when all parameters are locked."""
    parameters = {
        "param1": 999,
        "param2": 888,
    }
    default_parameters = {
        "param1": 10,
        "param2": 20,
    }
    locked_keys = {"param1", "param2"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(test_object, parameters, default_parameters, locked_keys)

    # All should use defaults
    assert test_object.param1 == 10
    assert test_object.param2 == 20
    # Should have warnings for both
    assert "param1" in caplog.text
    assert "param2" in caplog.text


# ============================================================================
# Tests for fill_parameters - Override functionality
# ============================================================================


def test_fill_parameters_override_locked_key(test_object, caplog):
    """Test that locked keys can be overridden when explicitly allowed."""
    parameters = {
        "locked_param": 999,
    }
    default_parameters = {
        "locked_param": 42,
    }
    locked_keys = {"locked_param"}
    allow_override = {"locked_param"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(
            test_object, parameters, default_parameters, locked_keys, allow_override
        )

    # Locked parameter should now use user value
    assert test_object.locked_param == 999
    # Warning should indicate explicit override
    assert "locked_param" in caplog.text
    assert "explicitly overridden" in caplog.text


def test_fill_parameters_override_locked_quantity(test_object, caplog):
    """Test that locked Quantity parameters can be overridden."""
    parameters = {
        "locked_distance": 100 * u.m,
    }
    default_parameters = {
        "locked_distance": 10 * u.m,
    }
    locked_keys = {"locked_distance"}
    allow_override = {"locked_distance"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(
            test_object, parameters, default_parameters, locked_keys, allow_override
        )

    assert test_object.locked_distance == 100 * u.m
    assert "locked_distance" in caplog.text
    assert "explicitly overridden" in caplog.text


def test_fill_parameters_override_with_unit_conversion(test_object, caplog):
    """Test that overridden Quantity is converted to default units."""
    parameters = {
        "locked_distance": 5.0 * u.pc,
    }
    default_parameters = {
        "locked_distance": 10.0 * u.m,
    }
    locked_keys = {"locked_distance"}
    allow_override = {"locked_distance"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(
            test_object, parameters, default_parameters, locked_keys, allow_override
        )

    # Should be converted to meters (the default unit)
    assert isinstance(test_object.locked_distance, u.Quantity)
    assert test_object.locked_distance.unit == u.m
    assert np.isclose(test_object.locked_distance.value, 1.54285714e17, rtol=1e-4)


def test_fill_parameters_override_without_unit(test_object, caplog):
    """Test that overridden unitless value gets default units."""
    parameters = {
        "locked_length": 100.0,
    }
    default_parameters = {
        "locked_length": 1.0 * u.m,
    }
    locked_keys = {"locked_length"}
    allow_override = {"locked_length"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(
            test_object, parameters, default_parameters, locked_keys, allow_override
        )

    assert isinstance(test_object.locked_length, u.Quantity)
    assert test_object.locked_length.unit == u.m
    assert test_object.locked_length.value == 100.0


def test_fill_parameters_override_unrecognized_parameter():
    """Test that unknown override keys raise ValueError."""
    test_object = type("TestObject", (), {})()
    parameters = {}
    default_parameters = {
        "param1": 10,
        "param2": 20,
    }
    locked_keys = {"param1"}
    allow_override = {"param1", "nonexistent_param"}

    with pytest.raises(ValueError, match="unrecognized parameter name"):
        fill_parameters(
            test_object, parameters, default_parameters, locked_keys, allow_override
        )


def test_fill_parameters_override_multiple_unrecognized():
    """Test that multiple unknown override keys are reported."""
    test_object = type("TestObject", (), {})()
    parameters = {}
    default_parameters = {
        "param1": 10,
    }
    locked_keys = {"param1"}
    allow_override = {"nonexistent1", "nonexistent2"}

    with pytest.raises(ValueError) as excinfo:
        fill_parameters(
            test_object, parameters, default_parameters, locked_keys, allow_override
        )

    # Both names should appear in the error message
    assert "nonexistent1" in str(excinfo.value)
    assert "nonexistent2" in str(excinfo.value)


def test_fill_parameters_override_non_locked_key(test_object, caplog):
    """Test that override of non-locked key has no special effect."""
    parameters = {
        "unlocked_param": 100,
    }
    default_parameters = {
        "unlocked_param": 10,
    }
    locked_keys = set()
    allow_override = {"unlocked_param"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(
            test_object, parameters, default_parameters, locked_keys, allow_override
        )

    # Should use user value (not locked anyway)
    assert test_object.unlocked_param == 100
    # Should not log override warning since it wasn't locked
    assert "explicitly overridden" not in caplog.text


def test_fill_parameters_override_mixed_locked_unlocked(test_object, caplog):
    """Test overriding some locked parameters but not others."""
    parameters = {
        "locked_param1": 100,
        "locked_param2": 200,
        "unlocked_param": 300,
    }
    default_parameters = {
        "locked_param1": 10,
        "locked_param2": 20,
        "unlocked_param": 30,
    }
    locked_keys = {"locked_param1", "locked_param2"}
    allow_override = {"locked_param1"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(
            test_object, parameters, default_parameters, locked_keys, allow_override
        )

    # First locked param should use user value (explicitly overridden)
    assert test_object.locked_param1 == 100
    # Second locked param should use default (not overridden)
    assert test_object.locked_param2 == 20
    # Unlocked param should use user value
    assert test_object.unlocked_param == 300

    # Check log messages
    assert "locked_param1" in caplog.text
    assert "explicitly overridden" in caplog.text
    assert "locked_param2" in caplog.text
    assert "locked in this mode" in caplog.text


def test_fill_parameters_override_empty_set(test_object):
    """Test that empty allow_override set behaves correctly."""
    parameters = {
        "locked_param": 999,
    }
    default_parameters = {
        "locked_param": 42,
    }
    locked_keys = {"locked_param"}
    allow_override = set()

    fill_parameters(
        test_object, parameters, default_parameters, locked_keys, allow_override
    )

    # Should use default (not overridden)
    assert test_object.locked_param == 42


def test_fill_parameters_override_none_value(test_object, caplog):
    """Test that None for allow_override is handled correctly."""
    parameters = {
        "locked_param": 999,
    }
    default_parameters = {
        "locked_param": 42,
    }
    locked_keys = {"locked_param"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(
            test_object,
            parameters,
            default_parameters,
            locked_keys,
            allow_override=None,
        )

    # Should use default (not overridden)
    assert test_object.locked_param == 42
    # Should have standard locked warning
    assert "locked_param" in caplog.text
    assert "locked in this mode" in caplog.text


def test_fill_parameters_override_without_user_parameter(test_object, caplog):
    """Test override specification when user didn't provide the parameter."""
    parameters = {}
    default_parameters = {
        "locked_param": 42,
    }
    locked_keys = {"locked_param"}
    allow_override = {"locked_param"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(
            test_object, parameters, default_parameters, locked_keys, allow_override
        )

    # Should use default (user didn't override it)
    assert test_object.locked_param == 42
    # Should not log any override warning
    assert "explicitly overridden" not in caplog.text


def test_fill_parameters_override_all_locked_keys(test_object, caplog):
    """Test overriding all locked parameters."""
    parameters = {
        "locked_param1": 100,
        "locked_param2": 200,
        "locked_param3": 300,
    }
    default_parameters = {
        "locked_param1": 10,
        "locked_param2": 20,
        "locked_param3": 30,
    }
    locked_keys = {"locked_param1", "locked_param2", "locked_param3"}
    allow_override = {"locked_param1", "locked_param2", "locked_param3"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(
            test_object, parameters, default_parameters, locked_keys, allow_override
        )

    # All should use user values
    assert test_object.locked_param1 == 100
    assert test_object.locked_param2 == 200
    assert test_object.locked_param3 == 300

    # All should have override warnings
    assert caplog.text.count("explicitly overridden") == 3


def test_fill_parameters_override_logs_both_values(test_object, caplog):
    """Test that override warning includes both model and user values."""
    parameters = {
        "locked_param": 999,
    }
    default_parameters = {
        "locked_param": 42,
    }
    locked_keys = {"locked_param"}
    allow_override = {"locked_param"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(
            test_object, parameters, default_parameters, locked_keys, allow_override
        )

    # Warning should mention both values
    assert "42" in caplog.text
    assert "999" in caplog.text
    assert "Model-provided value" in caplog.text
    assert "user-supplied value" in caplog.text


def test_fill_parameters_override_quantity_logs_units(test_object, caplog):
    """Test that override warning for Quantity shows units."""
    parameters = {
        "locked_distance": 100 * u.m,
    }
    default_parameters = {
        "locked_distance": 10 * u.m,
    }
    locked_keys = {"locked_distance"}
    allow_override = {"locked_distance"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(
            test_object, parameters, default_parameters, locked_keys, allow_override
        )

    # Warning should mention units
    assert "10.0 m" in caplog.text or "10. m" in caplog.text
    assert "100.0 m" in caplog.text or "100. m" in caplog.text


# ============================================================================
# Tests for fill_parameters - Edge cases with overrides
# ============================================================================


def test_fill_parameters_override_superset_of_locked():
    """Test that allow_override can be a superset of locked_keys."""
    test_object = type("TestObject", (), {})()
    parameters = {
        "locked_param": 100,
        "unlocked_param": 200,
    }
    default_parameters = {
        "locked_param": 10,
        "unlocked_param": 20,
    }
    locked_keys = {"locked_param"}
    # Override set includes both locked and unlocked
    allow_override = {"locked_param", "unlocked_param"}

    fill_parameters(
        test_object, parameters, default_parameters, locked_keys, allow_override
    )

    # Both should use user values
    assert test_object.locked_param == 100
    assert test_object.unlocked_param == 200


def test_fill_parameters_override_disjoint_sets():
    """Test error when allow_override contains keys not in defaults."""
    test_object = type("TestObject", (), {})()
    parameters = {}
    default_parameters = {
        "param1": 10,
    }
    locked_keys = {"param1"}
    allow_override = {"completely_different_param"}

    with pytest.raises(ValueError, match="unrecognized parameter name"):
        fill_parameters(
            test_object, parameters, default_parameters, locked_keys, allow_override
        )


# ============================================================================
# Tests for fill_parameters - STEP 0: Rebinning override parameters
# ============================================================================


def test_fill_parameters_rebin_array_override(test_object, caplog):
    """Test that array override parameters (like DC) are rebinned to match resolved wavelength."""
    parameters = {
        "wavelength": np.array([0.5, 0.550, 0.60, 0.650, 0.700]) * WAVELENGTH,
        "DC": np.array([1e-3, 1.5e-3, 2e-3, 2.5e-3, 1.5e-3]) * DARK_CURRENT,
    }
    rebinned_wavelength = np.array([0.5, 0.6, 0.7]) * WAVELENGTH
    default_parameters = {
        "wavelength": np.array([0.5, 0.6, 0.7]) * WAVELENGTH,
        "DC": np.array([1e-3, 1e-3, 1e-3]) * DARK_CURRENT,
    }
    locked_keys = {"DC"}
    allow_override = {"DC"}

    with caplog.at_level(logging.INFO):
        fill_parameters(
            test_object,
            parameters,
            default_parameters,
            locked_keys,
            allow_override,
            rebinned_wavelength,
        )

    # Should be rebinned to match rebinned_wavelength length
    assert len(test_object.DC) == len(rebinned_wavelength)
    assert np.array_equal(test_object.DC, [1e-3, 2e-3, 1.5e-3] * DARK_CURRENT)
    assert isinstance(test_object.DC, u.Quantity)
    assert test_object.DC.unit == DARK_CURRENT
    assert "Before overriding: rebinning DC array" in caplog.text
    assert "explicitly overridden" in caplog.text


def test_fill_parameters_skip_scalar_override(test_object, caplog):
    """Test that scalar override parameters (like diameter) are not rebinned."""
    parameters = {
        "wavelength": np.array([0.5, 0.550, 0.60, 0.650, 0.700]) * WAVELENGTH,
        "diameter": 4.0 * LENGTH,
    }
    rebinned_wavelength = np.array([0.5, 0.6, 0.7]) * WAVELENGTH
    default_parameters = {
        "wavelength": np.array([0.5, 0.6, 0.7]) * WAVELENGTH,
        "diameter": 2.4 * LENGTH,
    }
    locked_keys = {"diameter"}
    allow_override = {"diameter"}

    with caplog.at_level(logging.INFO):
        fill_parameters(
            test_object,
            parameters,
            default_parameters,
            locked_keys,
            allow_override,
            rebinned_wavelength,
        )

    # Scalar should remain scalar and use user value
    assert test_object.diameter == 4.0 * LENGTH
    assert "diameter is a scalar and won't be rebinned" in caplog.text
    assert "explicitly overridden" in caplog.text


def test_fill_parameters_mixed_overrides(test_object, caplog):
    """Test rebinning with both array (DC) and scalar (diameter) overrides."""
    parameters = {
        "wavelength": np.array([0.5, 0.550, 0.60, 0.650, 0.700]) * WAVELENGTH,
        "DC": np.array([1e-3, 1.5e-3, 2e-3, 2.5e-3, 1.5e-3]) * DARK_CURRENT,
        "diameter": 4.0 * u.m,
    }
    rebinned_wavelength = np.array([0.5, 0.6, 0.7]) * WAVELENGTH
    default_parameters = {
        "wavelength": np.array([0.5, 0.6, 0.7]) * WAVELENGTH,
        "DC": np.array([1e-3, 1e-3, 1e-3]) * DARK_CURRENT,
        "diameter": 2.4 * u.m,
    }
    locked_keys = {"DC", "diameter"}
    allow_override = {"DC", "diameter"}

    with caplog.at_level(logging.INFO):
        fill_parameters(
            test_object,
            parameters,
            default_parameters,
            locked_keys,
            allow_override,
            rebinned_wavelength,
        )

    # DC should be rebinned
    assert len(test_object.DC) == len(rebinned_wavelength)
    assert np.array_equal(test_object.DC, [1e-3, 2e-3, 1.5e-3] * DARK_CURRENT)
    assert isinstance(test_object.DC, u.Quantity)
    assert test_object.DC.unit == DARK_CURRENT
    assert "Before overriding: rebinning DC array" in caplog.text

    # diameter should remain scalar
    assert test_object.diameter == 4.0 * LENGTH
    assert "diameter is a scalar and won't be rebinned" in caplog.text


def test_fill_parameters_mixed_overrides_non_quantities(test_object, caplog):
    """Test rebinning with both array (DC) and scalar (diameter) overrides."""
    parameters = {
        "wavelength": np.array([0.5, 0.550, 0.60, 0.650, 0.700]),
        "DC": np.array([1e-3, 1.5e-3, 2e-3, 2.5e-3, 1.5e-3]),
        "diameter": 4.0,
    }
    rebinned_wavelength = np.array([0.5, 0.6, 0.7]) * WAVELENGTH
    default_parameters = {
        "wavelength": np.array([0.5, 0.6, 0.7]) * WAVELENGTH,
        "DC": np.array([1e-3, 1e-3, 1e-3]) * DARK_CURRENT,
        "diameter": 2.4 * u.m,
    }
    locked_keys = {"DC", "diameter"}
    allow_override = {"DC", "diameter"}

    with caplog.at_level(logging.INFO):
        fill_parameters(
            test_object,
            parameters,
            default_parameters,
            locked_keys,
            allow_override,
            rebinned_wavelength,
        )

    # DC should be rebinned
    assert len(test_object.DC) == len(rebinned_wavelength)
    assert np.array_equal(test_object.DC, [1e-3, 2e-3, 1.5e-3] * DARK_CURRENT)
    assert isinstance(test_object.DC, u.Quantity)
    assert test_object.DC.unit == DARK_CURRENT
    assert "Before overriding: rebinning DC array" in caplog.text

    # diameter should remain scalar
    assert test_object.diameter == 4.0 * LENGTH
    assert "diameter is a scalar and won't be rebinned" in caplog.text


def test_fill_parameters_error_wrong_length_array(test_object, caplog):
    """Test that arrays with length != wavelength fail."""
    parameters = {
        "wavelength": np.array([0.5, 0.550, 0.60, 0.650, 0.700]) * WAVELENGTH,
        "DC": np.array([1e-3, 1.5e-3, 1.5e-3]) * DARK_CURRENT,
    }
    rebinned_wavelength = np.array([0.5, 0.6, 0.7]) * WAVELENGTH
    default_parameters = {
        "wavelength": np.array([0.5, 0.6, 0.7]) * WAVELENGTH,
        "DC": np.array([1e-3, 1e-3, 1e-3]) * DARK_CURRENT,
    }
    locked_keys = {"DC"}
    allow_override = {"DC"}

    with pytest.raises(
        ValueError,
        match="Before overriding: DC does not have length equal to input wavelength. Please check your inputs.",
    ):
        fill_parameters(
            test_object,
            parameters,
            default_parameters,
            locked_keys,
            allow_override,
            rebinned_wavelength,
        )


def test_fill_parameters_override_not_provided(test_object, caplog):
    """Test that override parameters not provided by user are not processed in STEP 0."""
    parameters = {
        "wavelength": np.array([0.5, 0.550, 0.60, 0.650, 0.700]) * WAVELENGTH,
        # DC intentionally not provided
    }
    rebinned_wavelength = np.array([0.5, 0.6, 0.7]) * WAVELENGTH
    default_parameters = {
        "wavelength": np.array([0.5, 0.6, 0.7]) * WAVELENGTH,
        "DC": np.array([1e-3, 1e-3, 1e-3]) * DARK_CURRENT,
    }
    locked_keys = {"DC"}
    allow_override = {"DC"}

    with caplog.at_level(logging.INFO):
        fill_parameters(
            test_object,
            parameters,
            default_parameters,
            locked_keys,
            allow_override,
            rebinned_wavelength,
        )

    # Should use default value (not rebinned since user didn't provide it)
    assert np.array_equal(test_object.DC, default_parameters["DC"])
    # No rebinning message for DC
    assert "Before overriding: rebinning DC" not in caplog.text


from unittest.mock import patch

# ============================================================================
# Fixtures for rebin_channel_curves_to_grid
# ============================================================================


@pytest.fixture
def sample_spectral_dict():
    """Fixture providing a simple channel-native spectral dict for testing."""
    return {
        "wavelength": np.array([0.40, 0.45, 0.50, 0.55, 0.60]),
        "qe": np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
        "dqe": np.array([0.90, 0.91, 0.92, 0.93, 0.94]),
    }


@pytest.fixture
def sample_to_wavelength():
    """Fixture providing a target wavelength grid."""
    return np.linspace(0.40, 0.60, 10)


# ============================================================================
# Tests for rebin_channel_curves_to_grid - Missing "wavelength" key
# ============================================================================


def test_rebin_channel_curves_to_grid_no_wavelength_key(sample_to_wavelength, caplog):
    """Test that missing 'wavelength' key returns an empty dict and logs debug."""
    spectral = {"qe": np.array([0.5, 0.6, 0.7])}

    with caplog.at_level(logging.DEBUG, logger="pyEDITH"):
        result = rebin_channel_curves_to_grid(spectral, ["qe"], sample_to_wavelength)

    assert result == {}
    assert "No 'wavelength' array found" in caplog.text


# ============================================================================
# Tests for rebin_channel_curves_to_grid - Missing curve keys
# ============================================================================


def test_rebin_channel_curves_to_grid_key_not_in_spectral(
    sample_spectral_dict, sample_to_wavelength, caplog
):
    """Test that curve keys not present in spectral are skipped and logged."""
    with caplog.at_level(logging.DEBUG, logger="pyEDITH"):
        result = rebin_channel_curves_to_grid(
            sample_spectral_dict,
            ["qe", "missing_curve"],
            sample_to_wavelength,
            obs_mode="IFS",
        )

    assert "qe" in result
    assert "missing_curve" not in result
    assert "missing_curve was not found in YAML files" in caplog.text


def test_rebin_channel_curves_to_grid_empty_curve_keys(
    sample_spectral_dict, sample_to_wavelength
):
    """Test that an empty curve_keys list returns an empty dict."""
    result = rebin_channel_curves_to_grid(
        sample_spectral_dict, [], sample_to_wavelength, obs_mode="IFS"
    )

    assert result == {}


# ============================================================================
# Tests for rebin_channel_curves_to_grid - IMAGER (temporary) path
# ============================================================================


def test_rebin_channel_curves_to_grid_imager_mode_basic(
    sample_spectral_dict, sample_to_wavelength
):
    """Test that IMAGER mode returns a bandpass-averaged single value."""
    wavelength_range = [0.45 * WAVELENGTH, 0.55 * WAVELENGTH]

    result = rebin_channel_curves_to_grid(
        sample_spectral_dict,
        ["qe"],
        sample_to_wavelength,
        obs_mode="IMAGER",
        wavelength_range=wavelength_range,
    )

    assert "qe" in result
    # Result should be a single-element list (bandpass average, not resampled)
    assert len(result["qe"]) == 1


def test_rebin_channel_curves_to_grid_imager_mode_multiple_keys(
    sample_spectral_dict, sample_to_wavelength
):
    """Test that IMAGER mode processes multiple curve keys independently."""
    wavelength_range = [0.40 * WAVELENGTH, 0.60 * WAVELENGTH]

    result = rebin_channel_curves_to_grid(
        sample_spectral_dict,
        ["qe", "dqe"],
        sample_to_wavelength,
        obs_mode="IMAGER",
        wavelength_range=wavelength_range,
    )

    assert set(result.keys()) == {"qe", "dqe"}
    assert len(result["qe"]) == 1
    assert len(result["dqe"]) == 1


def test_rebin_channel_curves_to_grid_imager_mode_logs_debug(
    sample_spectral_dict, sample_to_wavelength, caplog
):
    """Test that IMAGER mode logs a bandpass-averaging debug message."""
    wavelength_range = [0.45 * WAVELENGTH, 0.55 * WAVELENGTH]

    with caplog.at_level(logging.DEBUG, logger="pyEDITH"):
        rebin_channel_curves_to_grid(
            sample_spectral_dict,
            ["qe"],
            sample_to_wavelength,
            obs_mode="IMAGER",
            wavelength_range=wavelength_range,
        )

    assert "Bandpass-averaged 'qe'" in caplog.text
    assert "IMAGER temporary path" in caplog.text


def test_rebin_channel_curves_to_grid_imager_mode_calls_average_over_bandpass(
    sample_spectral_dict, sample_to_wavelength
):
    """Test that IMAGER mode delegates to average_over_bandpass with correct args."""
    wavelength_range = [0.45 * WAVELENGTH, 0.55 * WAVELENGTH]

    with patch(
        "pyEDITH.utils.average_over_bandpass",
        wraps=average_over_bandpass,
    ) as mock_avg:
        rebin_channel_curves_to_grid(
            sample_spectral_dict,
            ["qe"],
            sample_to_wavelength,
            obs_mode="IMAGER",
            wavelength_range=wavelength_range,
        )

    mock_avg.assert_called_once()
    called_params, called_range = mock_avg.call_args[0]
    assert "lam" in called_params
    assert "value" in called_params
    assert called_range == wavelength_range


# ============================================================================
# Tests for rebin_channel_curves_to_grid - IFS (resampling) path
# ============================================================================


def test_rebin_channel_curves_to_grid_non_imager_basic(
    sample_spectral_dict, sample_to_wavelength
):
    """Test that non-IMAGER mode resamples curve onto to_wavelength grid."""
    result = rebin_channel_curves_to_grid(
        sample_spectral_dict,
        ["qe"],
        sample_to_wavelength,
        obs_mode="IFS",
    )

    assert "qe" in result
    assert len(result["qe"]) == len(sample_to_wavelength)


def test_rebin_channel_curves_to_grid_non_imager_logs_debug(
    sample_spectral_dict, sample_to_wavelength, caplog
):
    """Test that non-IMAGER mode logs a rebinning debug message."""
    with caplog.at_level(logging.DEBUG, logger="pyEDITH"):
        rebin_channel_curves_to_grid(
            sample_spectral_dict,
            ["qe"],
            sample_to_wavelength,
            obs_mode="IFS",
        )

    assert "Rebinned 'qe'" in caplog.text
    assert "native points" in caplog.text


def test_rebin_channel_curves_to_grid_sorts_and_deduplicates_wavelengths(
    sample_to_wavelength,
):
    """Test that unsorted/duplicate native wavelengths are sorted and deduped
    before resampling."""
    spectral = {
        "wavelength": np.array([0.50, 0.40, 0.50, 0.60]),
        "qe": np.array([2.0, 1.0, 2.0, 3.0]),
    }

    # Should not raise despite unsorted/duplicate input, and should produce
    # a monotonic result consistent with sorted+deduped values [1, 2, 3]
    result = rebin_channel_curves_to_grid(
        spectral,
        ["qe"],
        sample_to_wavelength,
        obs_mode="IFS",
    )

    assert "qe" in result
    assert len(result["qe"]) == len(sample_to_wavelength)
    values = np.asarray(result["qe"])
    assert np.all(np.diff(values) >= -1e-8)  # roughly increasing


def test_rebin_channel_curves_to_grid_single_wavelength_point(
    sample_to_wavelength,
):
    """Test that a single native wavelength point skips sort/dedup and still resamples."""
    spectral = {
        "wavelength": np.array([0.50]),
        "qe": np.array([0.75]),
    }

    result = rebin_channel_curves_to_grid(
        spectral,
        ["qe"],
        sample_to_wavelength,
        obs_mode="IFS",
    )

    assert "qe" in result
    assert len(result["qe"]) == len(sample_to_wavelength)
    assert np.allclose(np.asarray(result["qe"]), 0.75)


def test_rebin_channel_curves_to_grid_gaussian_interpolation_with_delta(
    sample_spectral_dict, sample_to_wavelength
):
    """Test that Gaussian interpolation runs successfully when to_delta_wavelength
    is provided."""
    to_delta_wavelength = np.full_like(sample_to_wavelength, 0.02)

    result = rebin_channel_curves_to_grid(
        sample_spectral_dict,
        ["qe"],
        sample_to_wavelength,
        to_delta_wavelength=to_delta_wavelength,
        interpolation="Gaussian",
        obs_mode="IFS",
    )

    assert "qe" in result
    assert len(result["qe"]) == len(sample_to_wavelength)


def test_rebin_channel_curves_to_grid_non_imager_calls_resample(
    sample_spectral_dict, sample_to_wavelength
):
    """Test that non-IMAGER mode delegates to resample_to_wavelength_grid with
    correct arguments."""
    with patch(
        "pyEDITH.utils.resample_to_wavelength_grid",
        wraps=resample_to_wavelength_grid,
    ) as mock_resample:
        rebin_channel_curves_to_grid(
            sample_spectral_dict,
            ["qe"],
            sample_to_wavelength,
            interpolation="1d",
            obs_mode="IFS",
        )

    mock_resample.assert_called_once()
    _, kwargs = mock_resample.call_args
    assert kwargs["name"] == "qe"
    assert kwargs["interpolation"] == "1d"
    assert np.array_equal(kwargs["to_wavelength"], sample_to_wavelength)


# ============================================================================
# Tests for rebin_channel_curves_to_grid - Mixed IMAGER/non-IMAGER key skipping
# ============================================================================


def test_rebin_channel_curves_to_grid_returns_only_requested_present_keys(
    sample_spectral_dict, sample_to_wavelength
):
    """Test that only curve keys present in spectral and requested are returned."""
    result = rebin_channel_curves_to_grid(
        sample_spectral_dict,
        ["qe", "nonexistent"],
        sample_to_wavelength,
        obs_mode="IFS",
    )

    assert list(result.keys()) == ["qe"]


def test_rebin_channel_curves_to_grid_gaussian_without_delta_wavelength_raises(
    sample_spectral_dict, sample_to_wavelength
):
    """Test that Gaussian interpolation without to_delta_wavelength raises ValueError."""
    with pytest.raises(ValueError, match="to_delta_wavelength must be provided"):
        rebin_channel_curves_to_grid(
            sample_spectral_dict,
            ["qe"],
            sample_to_wavelength,
            to_delta_wavelength=None,
            interpolation="Gaussian",
            obs_mode="IFS",
        )


def test_rebin_channel_curves_to_grid_gaussian_without_delta_wavelength_raises_even_for_imager(
    sample_spectral_dict, sample_to_wavelength
):
    """Test that the Gaussian/to_delta_wavelength validation fires before the
    IMAGER early-return path (i.e., is checked regardless of obs_mode)."""
    with pytest.raises(ValueError, match="to_delta_wavelength must be provided"):
        rebin_channel_curves_to_grid(
            sample_spectral_dict,
            ["qe"],
            sample_to_wavelength,
            to_delta_wavelength=None,
            interpolation="Gaussian",
            obs_mode="IMAGER",
            wavelength_range=[0.1, 0.2] * WAVELENGTH,
        )
