import pytest
import numpy as np
from astropy import units as u
from pathlib import Path
import tempfile
import os
import logging

from pyEDITH.parse_input import *
from pyEDITH.units import WAVELENGTH, DIMENSIONLESS, LENGTH

# ============================================================================
# Fixtures - Temporary input files
# ============================================================================


@pytest.fixture
def sample_input_file():
    """Fixture providing a sample input file with valid parameters."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        ; This is a comment
        wavelength = 0.5
        bandwidth = 0.2
        distance = 10
        magV = 5.0
        nzodis = 3.0
        observing_mode = IMAGER
        secondary_observing_mode = IMAGER
        secondary_wavelength = 1.0
        secondary_bandwidth = 0.2
        """)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_input_file_imager_multi_wavelength():
    """Fixture providing an IMAGER mode input file with multiple wavelengths (invalid)."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        ; This is a comment
        wavelength = [0.5, 0.6]
        distance = 10
        magV = 5.0
        nzodis = 3.0
        observing_mode = IMAGER
        bandwidth = 0.2
        """)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def ifs_input_file_valid():
    """Fixture providing a valid IFS mode input file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = 'IFS'
        wavelength = [0.5, 0.6, 0.7]
        Fstar_10pc = [1e-8, 1e-8, 1e-8]
        Fp/Fs = [1e-10, 1e-10, 1e-10]
        """)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def ifs_input_file_no_observing_mode():
    """Fixture providing an input file without observing mode."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        wavelength = [0.5, 0.6, 0.7]
        Fstar_10pc = [1e-8, 1e-8, 1e-8]
        Fp/Fs = [1e-10, 1e-10, 1e-10]
        """)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def ifs_input_file_wrong_observing_mode():
    """Fixture providing an input file with wrong observing mode."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = 'Invalid'
        wavelength = [0.5, 0.6, 0.7]
        Fstar_10pc = [1e-8, 1e-8, 1e-8]
        Fp/Fs = [1e-10, 1e-10, 1e-10]
        """)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def ifs_input_file_missing_keys():
    """Fixture providing an IFS mode input file with missing required keys."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = 'IFS'
        """)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def ifs_input_file_mismatched_lengths():
    """Fixture providing an IFS mode input file with mismatched column lengths."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = 'IFS'
        wavelength = [0.5, 0.6, 0.7]
        Fstar_10pc = [1e-8, 1e-8]
        Fp/Fs = [1e-10, 1e-10, 1e-10]
        """)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def valid_spectrum_file():
    """Fixture providing a valid spectrum CSV file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
        tmp.write(
            "wavelength,Fstar_10pc,Fp/Fs\n0.5,1e-9,1e-11\n0.6,1e-9,1e-11\n0.7,1e-8,1e-10"
        )
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def spectrum_file_invalid_columns():
    """Fixture providing a spectrum file with incorrect number of columns."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
        tmp.write("wavelength,Fstar_10pc\n0.5,1e-8")
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def spectrum_file_non_numeric():
    """Fixture providing a spectrum file with non-numeric values."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
        tmp.write("wavelength,Fstar_10pc,Fp/Fs\n0.5,1e-8,invalid")
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def filter_input_file_center_bandwidth():
    """Fixture providing an input file with filters using center/bandwidth specification."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = 'IMAGER'
        wavelength = 0.6
        filter_name = ['F1']
        filter_center = [0.6]
        filter_bandwidth = [0.1]
        """)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def filter_input_file_low_high():
    """Fixture providing an input file with filters using low/high specification."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = 'IMAGER'
        wavelength = 0.6
        filter_name = ['F1']
        filter_low = [0.5]
        filter_high = [0.7]
        """)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def filter_input_file_ifs_with_resolution():
    """Fixture providing an IFS mode input file with filters including resolution."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = 'IFS'
        spectrum_file = 'inputs/spectrum.txt' 
        filter_name = ['Channel1', 'Channel2']
        filter_low = [0.5, 1.0]
        filter_high = [1.0, 1.7]
        filter_resolution = [140, 40]
        """)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def filter_input_file_single_filter():
    """Fixture providing an input file with a single filter (scalar values)."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = 'IMAGER'
        wavelength = 0.6
        filter_name = 'F1'
        filter_center = 0.6
        filter_bandwidth = 0.1
        """)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


# ============================================================================
# Tests for parse_input_file - Basic functionality
# ============================================================================


def test_parse_input_file_basic(sample_input_file):
    """Test parsing a basic input file with valid parameters."""
    variables, secondary_variables = parse_input_file(
        sample_input_file, secondary_flag=True
    )

    assert variables["wavelength"] == 0.5
    assert variables["distance"] == 10
    assert variables["magV"] == 5.0
    assert variables["nzodis"] == 3.0
    assert variables["observing_mode"] == "IMAGER"
    assert secondary_variables["wavelength"] == 1.0


def test_parse_input_file_secondary_prefix_removed(sample_input_file):
    """Test that 'secondary_' prefix is removed from secondary variable keys."""
    variables, secondary_variables = parse_input_file(
        sample_input_file, secondary_flag=True
    )

    # Should have 'wavelength', not 'secondary_wavelength'
    assert "wavelength" in secondary_variables
    assert "secondary_wavelength" not in secondary_variables


def test_parse_input_file_ifs_valid(ifs_input_file_valid):
    """Test parsing a valid IFS mode input file."""
    variables, secondary_variables = parse_input_file(
        ifs_input_file_valid, secondary_flag=False
    )

    assert np.all(variables["wavelength"] == [0.5, 0.6, 0.7])
    assert np.all(variables["Fstar_10pc"] == [1e-8, 1e-8, 1e-8])
    assert np.all(variables["Fp/Fs"] == [1e-10, 1e-10, 1e-10])


# ============================================================================
# Tests for parse_input_file - Error handling
# ============================================================================


def test_parse_input_file_no_observing_mode(ifs_input_file_no_observing_mode):
    """Test parsing an input file without observing_mode."""
    with pytest.raises(
        KeyError,
        match="Required parameter 'observing_mode' is not provided in the input file.",
    ):
        parse_input_file(ifs_input_file_no_observing_mode, secondary_flag=False)


def test_parse_input_file_wrong_observing_mode(ifs_input_file_wrong_observing_mode):
    """Test parsing an input file without observing_mode."""
    with pytest.raises(
        ValueError,
        match="Invalid observing mode 'Invalid'. Must be 'IMAGER' or 'IFS'.",
    ):
        parse_input_file(ifs_input_file_wrong_observing_mode, secondary_flag=False)


def test_parse_input_file_imager_multi_wavelength_error(
    sample_input_file_imager_multi_wavelength,
):
    """Test that IMAGER mode with multiple wavelengths raises KeyError."""
    with pytest.raises(
        KeyError, match="In IMAGER mode you can only use one wavelength at a time"
    ):
        parse_input_file(
            sample_input_file_imager_multi_wavelength, secondary_flag=False
        )


def test_parse_input_file_secondary_flag_no_secondary_vars(
    sample_input_file_imager_multi_wavelength,
):
    """Test that secondary_flag=True without secondary variables raises KeyError."""
    with pytest.raises(
        KeyError,
        match="Secondary flag is True but no secondary variables found in the input file",
    ):
        parse_input_file(sample_input_file_imager_multi_wavelength, secondary_flag=True)


def test_parse_input_file_ifs_missing_keys(ifs_input_file_missing_keys):
    """Test that IFS mode with missing required keys raises ValueError."""
    with pytest.raises(
        KeyError,
        match="Required parameters 'wavelength', 'Fstar_10pc', and 'Fp/Fs' are not provided",
    ):
        parse_input_file(ifs_input_file_missing_keys, secondary_flag=False)


def test_parse_input_file_ifs_mismatched_lengths(ifs_input_file_mismatched_lengths):
    """Test that IFS mode with mismatched column lengths raises ValueError."""
    with pytest.raises(
        ValueError,
        match="All of wavelength, Fstar_10pc, Fp/Fs must have the same length",
    ):
        parse_input_file(ifs_input_file_mismatched_lengths, secondary_flag=False)


def test_parse_input_file_invalid_spectrum_file():
    """Test that non-existent spectrum file raises FileNotFoundError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = "IFS"
        spectrum_file = 'nonexistent_file.csv'
        """)
        tmp.flush()

        with pytest.raises(
            FileNotFoundError, match="Spectrum file not found: nonexistent_file.csv"
        ):
            parse_input_file(tmp.name, secondary_flag=False)

        os.unlink(tmp.name)


def test_parse_input_file_spectrum_file_invalid_columns(spectrum_file_invalid_columns):
    """Test that spectrum file with incorrect number of columns raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write(f"""
        observing_mode = 'IFS'
        spectrum_file = '{spectrum_file_invalid_columns}'
        """)
        tmp.flush()

        with pytest.raises(
            ValueError, match="Spectrum file must contain exactly 3 columns"
        ):
            parse_input_file(tmp.name, secondary_flag=False)

        os.unlink(tmp.name)


def test_parse_input_file_spectrum_file_non_numeric(spectrum_file_non_numeric):
    """Test that spectrum file with non-numeric values raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write(f"""
        observing_mode = 'IFS'
        spectrum_file = '{spectrum_file_non_numeric}'
        """)
        tmp.flush()

        with pytest.raises(
            ValueError, match="Column 'Fp/Fs' contains non-numeric values"
        ):
            parse_input_file(tmp.name, secondary_flag=False)

        os.unlink(tmp.name)


# ============================================================================
# Tests for parse_input_file - Spectrum file integration
# ============================================================================


def test_parse_input_file_with_valid_spectrum_file(valid_spectrum_file):
    """Test parsing input file with valid spectrum file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write(f"""
        observing_mode = 'IFS'
        spectrum_file = '{valid_spectrum_file}'
        """)
        tmp.flush()

        variables, _ = parse_input_file(tmp.name, secondary_flag=False)

        assert variables["observing_mode"] == "IFS"
        assert "wavelength" in variables
        assert "Fstar_10pc" in variables
        assert "Fp/Fs" in variables
        assert len(variables["wavelength"]) == 3
        np.testing.assert_almost_equal(variables["wavelength"], [0.5, 0.6, 0.7])
        np.testing.assert_almost_equal(variables["Fstar_10pc"], [1e-9, 1e-9, 1e-8])
        np.testing.assert_almost_equal(variables["Fp/Fs"], [1e-11, 1e-11, 1e-10])

        os.unlink(tmp.name)


# ============================================================================
# Tests for _parse_array_value
# ============================================================================


def test_parse_array_value_unquoted_string(tmp_path):
    """A non-numeric array item with no surrounding quotes is
    appended as-is (the quote-stripping branch is skipped)."""
    file_path = tmp_path / "input.txt"
    file_path.write_text(
        "observing_mode = 'IMAGER'\n"
        "wavelength = 0.5\n"
        "modes = [photometry, spectroscopy]\n"  # unquoted, non-numeric items
    )
    variables, _ = parse_input_file(str(file_path), secondary_flag=False)
    assert variables["modes"] == ["photometry", "spectroscopy"]


def test_parse_line_without_equals_is_skipped(tmp_path):
    """A non-comment, non-empty line that contains no '=' is
    ignored, and parsing continues with subsequent lines."""
    file_path = tmp_path / "input.txt"
    file_path.write_text(
        "observing_mode = 'IMAGER'\n"
        "this line has no equals sign\n"  # no '=' -> skipped
        "wavelength = 0.5\n"
    )
    variables, _ = parse_input_file(str(file_path), secondary_flag=False)
    assert variables["observing_mode"] == "IMAGER"
    assert variables["wavelength"] == 0.5
    assert "this line has no equals sign" not in variables


# ============================================================================
# Tests for normalize_list_shapes
# ============================================================================


def test_normalize_list_shapes_scalar_single_wavelength():
    """Test scalar value with single wavelength (default_len=1)."""
    parameters = {"snr": 10.0}
    result = normalize_list_shapes(parameters, "snr", default_len=1)

    np.testing.assert_array_equal(result, np.array([10.0]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_normalize_list_shapes_scalar_multiple_wavelengths(caplog):
    """Test scalar value broadcast to multiple wavelengths (default_len>1)."""
    parameters = {"snr": 10.0}

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        result = normalize_list_shapes(parameters, "snr", default_len=3)

    np.testing.assert_array_equal(result, np.array([10.0, 10.0, 10.0]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert any(
        "snr should be a list of length 3" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_normalize_list_shapes_single_element_list_broadcast(caplog):
    """Test single-element list broadcast to multiple wavelengths."""
    parameters = {"snr": [10.0]}

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        result = normalize_list_shapes(parameters, "snr", default_len=3)

    np.testing.assert_array_equal(result, np.array([10.0, 10.0, 10.0]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert any(
        "snr should be a list of length 3" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_normalize_list_shapes_matching_length_list():
    """Test list with correct length passes through."""
    parameters = {"snr": [10.0, 20.0, 30.0]}
    result = normalize_list_shapes(parameters, "snr", default_len=3)

    np.testing.assert_array_equal(result, np.array([10.0, 20.0, 30.0]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_normalize_list_shapes_mismatched_length_error():
    """Test list with wrong length raises ValueError."""
    parameters = {"snr": [10.0, 20.0]}

    with pytest.raises(
        ValueError, match="snr should be a list of length 3, but it has length 2"
    ):
        normalize_list_shapes(parameters, "snr", default_len=3)


def test_normalize_list_shapes_quantity_scalar_single_wavelength():
    """Test Quantity scalar with single wavelength."""
    parameters = {"DC": 10.0 * DARK_CURRENT}
    result = normalize_list_shapes(parameters, "DC", default_len=1)

    assert isinstance(result, u.Quantity)
    np.testing.assert_array_equal(result.value, np.array([10.0]))
    assert result.unit == DARK_CURRENT


def test_normalize_list_shapes_quantity_scalar_broadcast(caplog):
    """Test Quantity scalar broadcast to multiple wavelengths."""
    parameters = {"DC": 10.0 * DARK_CURRENT}

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        result = normalize_list_shapes(parameters, "DC", default_len=3)

    assert isinstance(result, u.Quantity)
    np.testing.assert_array_equal(result.value, np.array([10.0, 10.0, 10.0]))
    assert result.unit == DARK_CURRENT
    assert any(
        "DC should be a list of length 3" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_normalize_list_shapes_quantity_single_element_broadcast(caplog):
    """Test single-element Quantity array broadcast to multiple wavelengths."""
    parameters = {"DC": u.Quantity([10.0], DARK_CURRENT)}

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        result = normalize_list_shapes(parameters, "DC", default_len=3)

    assert isinstance(result, u.Quantity)
    np.testing.assert_array_equal(result.value, np.array([10.0, 10.0, 10.0]))
    assert result.unit == DARK_CURRENT
    assert any(
        "DC should be a list of length 3" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_normalize_list_shapes_quantity_matching_length():
    """Test Quantity array with correct length passes through."""
    parameters = {"DC": u.Quantity([10.0, 20.0, 30.0], DARK_CURRENT)}
    result = normalize_list_shapes(parameters, "DC", default_len=3)

    assert isinstance(result, u.Quantity)
    np.testing.assert_array_equal(result.value, np.array([10.0, 20.0, 30.0]))
    assert result.unit == DARK_CURRENT


def test_normalize_list_shapes_quantity_mismatched_length():
    """Test Quantity array with wrong length raises ValueError."""
    parameters = {"DC": u.Quantity([10.0, 20.0], DARK_CURRENT)}

    with pytest.raises(
        ValueError, match="DC should be a list of length 3, but it has length 2"
    ):
        normalize_list_shapes(parameters, "DC", default_len=3)


def test_normalize_list_shapes_excess_length_single_wavelength(caplog):
    """Test multi-element array preserved when default_len=1 for downstream regridding."""
    parameters = {"snr": [10.0, 20.0, 30.0]}

    with pytest.raises(
        ValueError, match="snr has length 3 but the expected input size is 1"
    ):
        normalize_list_shapes(parameters, "snr", default_len=1)


def test_normalize_list_shapes_excess_length_quantity_single_wavelength(caplog):
    """Test multi-element Quantity preserved when default_len=1 for downstream regridding."""
    parameters = {"DC": u.Quantity([10.0, 20.0, 30.0], DARK_CURRENT)}

    with pytest.raises(
        ValueError, match="DC has length 3 but the expected input size is 1"
    ):
        normalize_list_shapes(parameters, "DC", default_len=1)


def test_normalize_list_shapes_numpy_array():
    """Test that numpy arrays are properly converted."""
    parameters = {"snr": np.array([10.0, 20.0, 30.0])}
    result = normalize_list_shapes(parameters, "snr", default_len=3)

    np.testing.assert_array_equal(result, np.array([10.0, 20.0, 30.0]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_normalize_list_shapes_tuple_input():
    """Test that tuples are properly converted to arrays."""
    parameters = {"snr": (10.0, 20.0, 30.0)}
    result = normalize_list_shapes(parameters, "snr", default_len=3)

    np.testing.assert_array_equal(result, np.array([10.0, 20.0, 30.0]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_normalize_list_shapes_integer_values():
    """Test that integer values are converted to float64."""
    parameters = {"snr": [10, 20, 30]}
    result = normalize_list_shapes(parameters, "snr", default_len=3)

    np.testing.assert_array_equal(result, np.array([10.0, 20.0, 30.0]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_normalize_list_shapes_mixed_numeric_types():
    """Test that mixed int/float values are converted to float64."""
    parameters = {"snr": [10, 20.5, 30]}
    result = normalize_list_shapes(parameters, "snr", default_len=3)

    np.testing.assert_array_equal(result, np.array([10.0, 20.5, 30.0]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_normalize_list_shapes_preserves_quantity_unit():
    """Test that Quantity units are preserved through conversion."""
    parameters = {"wavelength": u.Quantity([0.5, 0.6, 0.7], u.um)}
    result = normalize_list_shapes(parameters, "wavelength", default_len=3)

    assert isinstance(result, u.Quantity)
    np.testing.assert_array_equal(result.value, np.array([0.5, 0.6, 0.7]))
    assert result.unit == u.um


# ============================================================================
# Tests for parse_parameters - Wavelength handling
# ============================================================================


def test_parse_parameters_wavelength_scalar():
    """Test parsing wavelength as a scalar."""
    parsed = parse_parameters({"wavelength": 0.5})

    assert isinstance(parsed["wavelength"], np.ndarray)
    assert len(parsed["wavelength"]) == 1
    assert parsed["wavelength"][0] == 0.5


def test_parse_parameters_wavelength_list():
    """Test parsing wavelength as a list."""
    parsed = parse_parameters({"wavelength": [0.5, 0.6, 0.7]})

    assert isinstance(parsed["wavelength"], np.ndarray)
    assert len(parsed["wavelength"]) == 3
    assert np.allclose(parsed["wavelength"], [0.5, 0.6, 0.7])


def test_parse_parameters_wavelength_scalar_quantity():
    """Test parsing wavelength as a scalar Quantity."""
    parsed = parse_parameters({"wavelength": 0.5 * u.um})

    assert isinstance(parsed["wavelength"], (np.ndarray, u.Quantity))
    assert len(parsed["wavelength"]) == 1
    # Check value regardless of whether it's a Quantity or array
    value = (
        parsed["wavelength"][0].value
        if isinstance(parsed["wavelength"], u.Quantity)
        else parsed["wavelength"][0]
    )
    assert value == 0.5


def test_parse_parameters_wavelength_list_quantity():
    """Test parsing wavelength as a list Quantity."""
    parsed = parse_parameters(
        {
            "wavelength": [0.5, 0.6, 0.7] * u.um,
        }
    )

    assert isinstance(parsed["wavelength"], (np.ndarray, u.Quantity))
    assert len(parsed["wavelength"]) == 3
    # Check values
    if isinstance(parsed["wavelength"], u.Quantity):
        assert np.allclose(parsed["wavelength"].value, [0.5, 0.6, 0.7])
        assert parsed["wavelength"].unit == u.um
    else:
        assert np.allclose(parsed["wavelength"], [0.5, 0.6, 0.7])


def test_parse_parameters_wavelength_array():
    """Test parsing wavelength as a numpy array."""
    parsed = parse_parameters(
        {
            "wavelength": np.array([0.5, 0.6, 0.7]),
        }
    )

    assert isinstance(parsed["wavelength"], np.ndarray)
    assert np.allclose(parsed["wavelength"], [0.5, 0.6, 0.7])


def test_parse_parameters_wavelength_missing():
    """Test that missing wavelength raises ValueError."""
    with pytest.raises(ValueError, match="pyEDITH does not have access to wavelength"):
        parse_parameters({})


def test_parse_parameters_wavelength_empty_list():
    """Test parsing empty wavelength list."""
    with pytest.raises(
        ValueError, match="'wavelength' parameter cannot be an empty list."
    ):
        parse_parameters(
            {
                "wavelength": [],
            }
        )


# ============================================================================
# Tests for parse_parameters - Wavelength-dependent parameters
# ============================================================================


def test_parse_parameters_wavelength_dependent_single():
    """Test wavelength-dependent parameters with single wavelength."""
    wavelength_params = [
        "snr",
        "T_optical",
        "epswarmTrcold",
        "DC",
        "RN",
        "tread",
        "CIC",
        "QE",
        "dQE",
        "mag",
        "Fstar_10pc",
        "Fp/Fs",
        "delta_mag",
        "F0",
        "det_npix_input",
    ]

    for param in wavelength_params:
        parsed = parse_parameters({"wavelength": 0.5, param: 1.5})

        assert np.all(parsed[param] == np.array([1.5]))
        assert isinstance(parsed[param], np.ndarray)


def test_parse_parameters_wavelength_dependent_multiple():
    """Test wavelength-dependent parameters with multiple wavelengths."""
    wavelengths = [0.5, 0.6, 0.7]
    wavelength_params = ["snr", "T_optical", "QE"]

    for param in wavelength_params:
        parsed = parse_parameters({"wavelength": wavelengths, param: [1.5, 2.5, 3.5]})

        assert np.all(parsed[param] == np.array([1.5, 2.5, 3.5]))
        assert isinstance(parsed[param], np.ndarray)


def test_parse_parameters_wavelength_dependent_scalar_broadcast(caplog):
    """Test wavelength-dependent parameters broadcast from scalar."""
    wavelengths = [0.5, 0.6, 0.7]

    with caplog.at_level(logging.DEBUG, logger="pyEDITH"):
        parsed = parse_parameters({"wavelength": wavelengths, "snr": 10})

    assert np.all(parsed["snr"] == np.array([10, 10, 10]))
    assert any(
        "snr should be a list of length 3" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_parse_parameters_wavelength_dependent_quantity_broadcast(caplog):
    """Test wavelength-dependent parameters broadcast from Quantity scalar."""
    wavelengths = [0.5, 0.6, 0.7]

    with caplog.at_level(logging.DEBUG, logger="pyEDITH"):
        parsed = parse_parameters(
            {"wavelength": wavelengths, "snr": 10 * u.dimensionless_unscaled}
        )

    assert np.all(parsed["snr"] == np.array([10, 10, 10]) * u.dimensionless_unscaled)
    assert any(
        "snr should be a list of length 3" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_parse_parameters_wavelength_dependent_mismatched_length():
    """Test that mismatched lengths raise ValueError."""
    wavelengths = [0.5, 0.6, 0.7]

    with pytest.raises(
        ValueError, match="snr should be a list of length 3, but it has length 2"
    ):
        parse_parameters({"wavelength": wavelengths, "snr": [10, 20]})


def test_parse_parameters_wavelength_dependent_excess_length_scalar(caplog):
    """Test that excess length triggers error for single wavelength."""
    with pytest.raises(
        ValueError, match="snr has length 3 but the expected input size is 1"
    ):
        parse_parameters({"wavelength": 0.5, "snr": [10, 20, 30]})


def test_parse_parameters_wavelength_dependent_excess_length_quantity(caplog):
    """Test that excess length triggers warning and uses first value."""

    with pytest.raises(
        ValueError, match="DC has length 3 but the expected input size is 1."
    ):
        parse_parameters({"wavelength": 0.5, "DC": [10, 20, 30] * DARK_CURRENT})


def test_parse_parameters_wavelength_dependent_with_quantity():
    """Test wavelength-dependent parameters with Quantity input."""
    wavelengths = [0.5, 0.6, 0.7]

    parsed = parse_parameters(
        {"wavelength": wavelengths, "snr": 1.5 * u.dimensionless_unscaled}
    )

    assert np.all(parsed["snr"] == np.array([1.5, 1.5, 1.5]) * u.dimensionless_unscaled)
    assert isinstance(parsed["snr"], u.Quantity)


# ============================================================================
# Tests for parse_parameters - nlambda handling
# ============================================================================


def test_parse_parameters_wavelength_not_provided():
    """Test that missing wavelength raises ValueError."""
    with pytest.raises(
        ValueError, match="pyEDITH does not have access to wavelength here"
    ):
        parse_parameters({"snr": 10})


# ============================================================================
# Tests for parse_parameters - Target parameters
# ============================================================================


def test_parse_parameters_target_params():
    """Test parsing target-specific parameters (scalars)."""
    target_params = [
        "distance",
        "magV",
        "FstarV_10pc",
        "stellar_radius",
        "nzodis",
        "ra",
        "dec",
        "delta_mag_min",
        "Fp_min/Fs",
        "separation",
        "semimajor_axis",
        "npix_multiplier",
    ]

    for param in target_params:
        parsed = parse_parameters({"wavelength": 0.5, param: 1.5})

        assert parsed[param] == 1.5
        assert isinstance(parsed[param], float)


# ============================================================================
# Tests for parse_parameters - Scalar parameters
# ============================================================================


def test_parse_parameters_scalar_params():
    """Test parsing scalar observatory parameters."""
    scalar_params = [
        "photometric_aperture_radius",
        "psf_trunc_ratio",
        "diameter",
        "toverhead_fixed",
        "toverhead_multi",
        "contrast",
        "noisefloor_factor",
        "noisefloor_PPF",
        "Tcore",
        "TLyot",
        "temperature",
        "T_contamination",
        "CRb_multiplier",
        "t_photon_count_input",
    ]

    for param in scalar_params:
        parsed = parse_parameters({"wavelength": 0.5, param: 1.5})

        assert parsed[param] == 1.5
        assert isinstance(parsed[param], float)


# ============================================================================
# Tests for parse_parameters - Integer parameters
# ============================================================================


def test_parse_parameters_integer_param():
    """Test parsing integer parameter (nrolls)."""
    parsed = parse_parameters({"wavelength": 0.5, "nrolls": 3})

    assert parsed["nrolls"] == 3
    assert isinstance(parsed["nrolls"], int)


# ============================================================================
# Tests for parse_parameters - Observatory specifications
# ============================================================================


def test_parse_parameters_observatory_specs():
    """Test parsing observatory specification strings."""
    observatory_specs = [
        "observatory_preset",
        "telescope_type",
        "coronagraph_type",
        "detector_type",
    ]

    for spec in observatory_specs:
        parsed = parse_parameters({"wavelength": 0.5, spec: "TestSpec"})

        assert parsed[spec] == "TestSpec"
        assert isinstance(parsed[spec], str)
    parsed = parse_parameters({"wavelength": 0.5, "observing_mode": "IMAGER"})

    assert parsed["observing_mode"] == "IMAGER"
    assert isinstance(parsed["observing_mode"], str)


# ============================================================================
# Tests for parse_parameters - Boolean parameters
# ============================================================================


def test_parse_parameters_boolean_true():
    """Test parsing boolean parameter as True."""
    parsed = parse_parameters(
        {
            "wavelength": np.linspace(0.2, 1.1, 20),
            "az_avg": True,
        }
    )
    assert parsed["az_avg"] is True
    assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_false():
    """Test parsing boolean parameter as False."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": False,
        }
    )
    assert parsed["az_avg"] is False
    assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_string_true():
    """Test parsing boolean from 'true' string (case-insensitive)."""
    for true_string in ["true", "True", "TRUE"]:
        parsed = parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": true_string,
            }
        )
        assert parsed["az_avg"] is True
        assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_string_false():
    """Test parsing boolean from 'false' string (case-insensitive)."""
    for false_string in ["false", "False", "FALSE"]:
        parsed = parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": false_string,
            }
        )
        assert parsed["az_avg"] is False
        assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_string_one():
    """Test parsing boolean from '1' string."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": "1",
        }
    )
    assert parsed["az_avg"] is True
    assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_string_zero():
    """Test parsing boolean from '0' string."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": "0",
        }
    )
    assert parsed["az_avg"] is False
    assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_string_yes():
    """Test parsing boolean from 'yes' string (case-insensitive)."""
    for yes_string in ["yes", "Yes", "YES"]:
        parsed = parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": yes_string,
            }
        )
        assert parsed["az_avg"] is True
        assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_string_no():
    """Test parsing boolean from 'no' string (case-insensitive)."""
    for no_string in ["no", "No", "NO"]:
        parsed = parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": no_string,
            }
        )
        assert parsed["az_avg"] is False
        assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_int_one():
    """Test parsing boolean from integer 1."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": 1,
        }
    )
    assert parsed["az_avg"] is True
    assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_int_zero():
    """Test parsing boolean from integer 0."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": 0,
        }
    )
    assert parsed["az_avg"] is False
    assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_float_one():
    """Test parsing boolean from float 1.0."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": 1.0,
        }
    )
    assert parsed["az_avg"] is True
    assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_float_zero():
    """Test parsing boolean from float 0.0."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": 0.0,
        }
    )
    assert parsed["az_avg"] is False
    assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_invalid_string():
    """Test that invalid boolean string raises ValueError with helpful message."""
    with pytest.raises(
        ValueError,
        match=r"Invalid value 'maybe' for parameter 'az_avg'\. "
        r"Expected boolean or one of: 'true', 'false', '1', '0', 'yes', 'no' "
        r"\(case-insensitive\)\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": "maybe",
            }
        )


def test_parse_parameters_boolean_invalid_numeric():
    """Test that invalid numeric boolean value raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Invalid numeric value '2' for parameter 'az_avg'\. "
        r"Expected 0 or 1 for boolean parameters\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": 2,
            }
        )


def test_parse_parameters_boolean_invalid_numeric_float():
    """Test that invalid float boolean value raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Invalid numeric value '0\.5' for parameter 'az_avg'\. "
        r"Expected 0 or 1 for boolean parameters\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": 0.5,
            }
        )


def test_parse_parameters_boolean_invalid_type():
    """Test that invalid type for boolean raises TypeError."""
    with pytest.raises(
        TypeError,
        match=r"Invalid type list for parameter 'az_avg'\. "
        r"Expected boolean, string, or numeric \(0/1\)\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": [True],
            }
        )


def test_parse_parameters_boolean_invalid_type_dict():
    """Test that dict type for boolean raises TypeError."""
    with pytest.raises(
        TypeError,
        match=r"Invalid type dict for parameter 'az_avg'\. "
        r"Expected boolean, string, or numeric \(0/1\)\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": {"value": True},
            }
        )


def test_parse_parameters_boolean_empty_string():
    """Test that empty string for boolean raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Invalid value '' for parameter 'az_avg'\. "
        r"Expected boolean or one of: 'true', 'false', '1', '0', 'yes', 'no' "
        r"\(case-insensitive\)\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": "",
            }
        )


def test_parse_parameters_boolean_whitespace_string():
    """Test that whitespace-only string for boolean raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Invalid value '   ' for parameter 'az_avg'\. "
        r"Expected boolean or one of: 'true', 'false', '1', '0', 'yes', 'no' "
        r"\(case-insensitive\)\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": "   ",
            }
        )


def test_parse_parameters_boolean_none_type():
    """Test that None type for boolean raises TypeError."""
    with pytest.raises(
        TypeError,
        match=r"Invalid type NoneType for parameter 'az_avg'\. "
        r"Expected boolean, string, or numeric \(0/1\)\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": None,
            }
        )


# ============================================================================
# Tests for parse_parameters - Complete parameter set
# ============================================================================


def test_parse_parameters_complete():
    """Test parsing complete parameter set."""
    parameters = {
        "wavelength": [0.5, 0.6, 0.7],
        "distance": 10,
        "magV": 5.0,
        "nzodis": 3.0,
        "observing_mode": "IFS",
        "snr": [10, 20, 30],
        "T_optical": 0.8,
        "diameter": 2.4,
        "toverhead_fixed": 300,
        "contrast": 1e-10,
        "nrolls": 3,
        "observatory_preset": "EAC1",
    }

    parsed = parse_parameters(parameters)

    assert np.all(parsed["wavelength"] == np.array([0.5, 0.6, 0.7]))
    assert parsed["distance"] == 10
    assert parsed["magV"] == 5.0
    assert parsed["nzodis"] == 3.0
    assert parsed["observing_mode"] == "IFS"
    assert np.all(parsed["snr"] == np.array([10, 20, 30]))
    assert np.all(parsed["T_optical"] == np.array([0.8, 0.8, 0.8]))
    assert parsed["diameter"] == 2.4
    assert parsed["toverhead_fixed"] == 300
    assert parsed["contrast"] == 1e-10
    assert parsed["nrolls"] == 3
    assert parsed["observatory_preset"] == "EAC1"


# ============================================================================
# Tests for parse_parameters - IFS and IMAGER modes
# ============================================================================


def test_parse_parameters_ifs_mode():
    """Test parsing IFS mode parameters."""
    parameters = {
        "observing_mode": "IFS",
        "wavelength": [0.5, 0.6, 0.7],
        "Fstar_10pc": [1e-8, 1e-8, 1e-8],
        "Fp/Fs": [1e-10, 1e-10, 1e-10],
    }

    parsed = parse_parameters(parameters)

    assert parsed["observing_mode"] == "IFS"
    assert np.all(parsed["wavelength"] == np.array([0.5, 0.6, 0.7]))
    assert np.all(parsed["Fstar_10pc"] == np.array([1e-8, 1e-8, 1e-8]))
    assert np.all(parsed["Fp/Fs"] == np.array([1e-10, 1e-10, 1e-10]))


def test_parse_parameters_imager_mode():
    """Test parsing IMAGER mode parameters."""
    parameters = {
        "observing_mode": "IMAGER",
        "wavelength": [0.5],
    }

    parsed = parse_parameters(parameters)

    assert parsed["observing_mode"] == "IMAGER"
    assert np.all(parsed["wavelength"] == np.array([0.5]))


# ============================================================================
# Tests for read_configuration
# ============================================================================


def test_read_configuration_with_secondary(sample_input_file):
    """Test reading configuration with secondary parameters."""
    parsed_parameters, parsed_secondary_parameters = read_configuration(
        sample_input_file, secondary_flag=True
    )

    assert np.all(parsed_parameters["wavelength"] == np.array([0.5]))
    assert parsed_parameters["distance"] == 10
    assert parsed_parameters["magV"] == 5.0
    assert parsed_parameters["nzodis"] == 3.0
    assert parsed_parameters["observing_mode"] == "IMAGER"
    assert parsed_secondary_parameters["wavelength"] == np.array([1.0])


def test_read_configuration_without_secondary(sample_input_file):
    """Test reading configuration without secondary parameters."""
    parsed_parameters, parsed_secondary_parameters = read_configuration(
        sample_input_file, secondary_flag=False
    )

    assert parsed_secondary_parameters == {}


# ============================================================================
# Tests for get_observatory_config
# ============================================================================


def test_get_observatory_config_with_preset():
    """Test getting observatory config from preset."""
    parameters = {"observatory_preset": "EAC1"}

    config = get_observatory_config(parameters)

    assert config == "EAC1"


def test_get_observatory_config_with_custom_components():
    """Test getting observatory config from custom component specifications."""
    parameters = {
        "telescope_type": "EAC1",
        "coronagraph_type": "AAVC",
        "detector_type": "EAC1",
    }

    config = get_observatory_config(parameters)

    assert config == {"telescope": "EAC1", "coronagraph": "AAVC", "detector": "EAC1"}


def test_get_observatory_config_missing():
    """Test that missing observatory configuration raises ValueError."""
    with pytest.raises(ValueError):
        get_observatory_config({})


def test_get_observatory_config_missing_component():
    """Test that missing individual component raises ValueError with specific message."""
    parameters = {
        "telescope_type": "EAC1",
        "coronagraph_type": "AAVC",
        # detector_type missing
    }

    with pytest.raises(ValueError, match="Detector type not specified"):
        get_observatory_config(parameters)


def test_parsed_flag_prevents_reprocessing():
    """Test that _parsed flag prevents reprocessing of already-parsed parameters."""

    # Create a minimal set of parameters
    raw_params = {
        "wavelength": [1.0, 2.0, 3.0],
        "snr": [10.0, 15.0, 20.0],
        "distance": 10.0,
    }

    # Parse once
    parsed_once = parse_parameters(raw_params)

    # Verify _parsed flag is set
    assert "_parsed" in parsed_once
    assert parsed_once["_parsed"] is True

    # Store a reference to verify identity
    parsed_once_id = id(parsed_once)

    # Parse again - should return immediately without reprocessing
    parsed_twice = parse_parameters(parsed_once)

    # Should return the same object (not a copy)
    assert id(parsed_twice) == parsed_once_id
    assert parsed_twice is parsed_once

    # Verify all original parsed values are unchanged
    assert "_parsed" in parsed_twice


def test_parsed_flag_not_present_initially():
    """Test that _parsed flag is not present in raw parameters."""

    raw_params = {"wavelength": [1.0], "distance": 10.0}

    # Verify _parsed is not in raw params
    assert "_parsed" not in raw_params

    # Parse
    parsed = parse_parameters(raw_params)

    # Now it should be present
    assert "_parsed" in parsed
    assert parsed["_parsed"] is True


def test_parsed_flag_set_after_single_parse():
    """Test that _parsed flag is set after a single parsing operation."""

    raw_params = {"wavelength": 1.5, "snr": 25.0, "distance": 5.0}

    parsed = parse_parameters(raw_params)

    # Check the flag exists and is True
    assert parsed.get("_parsed") is True


# ============================================================================
# Tests for parse_parameters - overrides parameter
# ============================================================================


def test_parse_parameters_overrides_string():
    """Test parsing overrides parameter as a comma-separated string."""
    parsed = parse_parameters({"wavelength": 0.5, "overrides": "DC,RN,QE"})

    assert "overrides" in parsed
    assert parsed["overrides"] == ["DC", "RN", "QE"]
    assert isinstance(parsed["overrides"], list)


def test_parse_parameters_overrides_string_with_spaces():
    """Test parsing overrides parameter with spaces around commas."""
    parsed = parse_parameters({"wavelength": 0.5, "overrides": "DC, RN, QE"})

    assert parsed["overrides"] == ["DC", "RN", "QE"]
    assert isinstance(parsed["overrides"], list)


def test_parse_parameters_overrides_string_single_value():
    """Test parsing overrides parameter with single value string."""
    parsed = parse_parameters({"wavelength": 0.5, "overrides": "DC"})

    assert parsed["overrides"] == ["DC"]
    assert isinstance(parsed["overrides"], list)


def test_parse_parameters_overrides_list():
    """Test parsing overrides parameter as a list."""
    parsed = parse_parameters({"wavelength": 0.5, "overrides": ["DC", "RN", "QE"]})

    assert parsed["overrides"] == ["DC", "RN", "QE"]
    assert isinstance(parsed["overrides"], list)


def test_parse_parameters_overrides_tuple():
    """Test parsing overrides parameter as a tuple."""
    parsed = parse_parameters({"wavelength": 0.5, "overrides": ("DC", "RN")})

    assert parsed["overrides"] == ["DC", "RN"]
    assert isinstance(parsed["overrides"], list)


def test_parse_parameters_overrides_empty_string():
    """Test parsing overrides parameter as an empty string."""
    parsed = parse_parameters({"wavelength": 0.5, "overrides": ""})

    assert "overrides" not in parsed


def test_parse_parameters_overrides_empty_list():
    """Test parsing overrides parameter as an empty list."""
    parsed = parse_parameters({"wavelength": 0.5, "overrides": []})

    assert "overrides" not in parsed


def test_parse_parameters_overrides_not_present():
    """Test that absence of overrides parameter doesn't create key."""
    parsed = parse_parameters({"wavelength": 0.5})

    assert "overrides" not in parsed


def test_parse_parameters_overrides_string_extra_whitespace():
    """Test parsing overrides with extra whitespace."""
    parsed = parse_parameters({"wavelength": 0.5, "overrides": "  DC  ,  RN  ,  QE  "})

    assert parsed["overrides"] == ["DC", "RN", "QE"]
    assert isinstance(parsed["overrides"], list)


# ============================================================================
# Tests for parse_parameters - Deprecations
# ============================================================================


def test_parse_parameters_npix_multiplier_scalar(caplog):
    """Test that scalar npix_multiplier works without warnings."""
    parameters = {"wavelength": [1.0], "npix_multiplier": 2.5}

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        parsed = parse_parameters(parameters)
    assert not any(record.levelno == logging.WARNING for record in caplog.records)

    assert parsed["npix_multiplier"] == 2.5
    assert isinstance(parsed["npix_multiplier"], float)


def test_parse_parameters_npix_multiplier_list_raises_deprecation_warning(caplog):
    """Test that list npix_multiplier raises DeprecationWarning and uses first element."""
    parameters = {"wavelength": [1.0, 1.2], "npix_multiplier": [2.5, 3.0]}

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        parsed = parse_parameters(parameters)
    assert any(
        "Passing 'npix_multiplier' as an array is deprecated" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )

    assert parsed["npix_multiplier"] == 2.5
    assert isinstance(parsed["npix_multiplier"], float)


def test_parse_parameters_npix_multiplier_numpy_array_raises_deprecation_warning(
    caplog,
):
    """Test that numpy array npix_multiplier raises DeprecationWarning and uses first element."""
    parameters = {"wavelength": [1.0, 1.2], "npix_multiplier": np.array([2.5, 3.0])}

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        parsed = parse_parameters(parameters)
    assert any(
        "Passing 'npix_multiplier' as an array is deprecated" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )
    assert parsed["npix_multiplier"] == 2.5
    assert isinstance(parsed["npix_multiplier"], float)


def test_parse_parameters_npix_multiplier_single_element_array(caplog):
    """Test that single-element array npix_multiplier raises warning and uses first element."""
    parameters = {"wavelength": [1.0], "npix_multiplier": [2.5]}

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        parsed = parse_parameters(parameters)
    assert any(
        "Passing 'npix_multiplier' as an array is deprecated" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )

    assert parsed["npix_multiplier"] == 2.5
    assert isinstance(parsed["npix_multiplier"], float)


def test_parse_parameters_npix_multiplier_string_converts_to_float():
    """Test that string npix_multiplier is converted to float without warning."""
    parameters = {"wavelength": [1.0], "npix_multiplier": "2.5"}

    parsed = parse_parameters(parameters)

    assert parsed["npix_multiplier"] == 2.5
    assert isinstance(parsed["npix_multiplier"], float)


def test_parse_parameters_npix_multiplier_integer_array():
    """Test that integer arrays are also deprecated and converted to float."""
    parameters = {"wavelength": [1.0, 1.2], "npix_multiplier": [3, 4]}

    parsed = parse_parameters(parameters)

    assert parsed["npix_multiplier"] == 3.0
    assert isinstance(parsed["npix_multiplier"], float)


def test_parse_parameters_nchannels_deprecated(caplog):
    """Test that scalar npix_multiplier works without warnings."""
    parameters = {"wavelength": [1.0], "nchannels": 1}

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        parsed = parse_parameters(parameters)
    assert any(
        "DeprecationWarning: 'nchannels' is deprecated, disregarding. Results will be comparable to setting nchannels to 1."
        in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )
    assert "nchannels" not in parsed.keys()


def test_parse_parameters_no_nchannels(caplog):
    """Test that scalar npix_multiplier works without warnings."""
    parameters = {
        "wavelength": [1.0],
    }

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        parsed = parse_parameters(parameters)
    assert not any(record.levelno == logging.WARNING for record in caplog.records)

    assert "nchannels" not in parsed.keys()


# ============================================================================
# Tests for parse_input_file - Filter handling
# ============================================================================


def test_parse_input_file_filter_center_bandwidth(filter_input_file_center_bandwidth):
    """Test parsing filters with center/bandwidth specification."""
    variables, _ = parse_input_file(
        filter_input_file_center_bandwidth, secondary_flag=False
    )

    assert "filter_list" in variables
    assert len(variables["filter_list"]) == 1

    # Check first filter
    assert variables["filter_list"][0].name == "F1"
    assert hasattr(variables["filter_list"][0], "wavelength")
    assert hasattr(variables["filter_list"][0], "delta_wavelength")


def test_parse_input_file_filter_low_high(filter_input_file_low_high):
    """Test parsing filters with low/high specification."""
    variables, _ = parse_input_file(filter_input_file_low_high, secondary_flag=False)

    assert "filter_list" in variables
    assert len(variables["filter_list"]) == 1

    # Check first filter bounds
    filter1 = variables["filter_list"][0]
    assert filter1.name == "F1"
    assert filter1.low.value == 0.5
    assert filter1.high.value == 0.7


def test_parse_input_file_filter_ifs_with_resolution(
    filter_input_file_ifs_with_resolution,
):
    """Test parsing IFS mode filters with resolution."""
    variables, _ = parse_input_file(
        filter_input_file_ifs_with_resolution, secondary_flag=False
    )

    assert "filter_list" in variables
    assert len(variables["filter_list"]) == 2

    # Check that filters have resolution
    assert variables["filter_list"][0].resolution == 140
    assert variables["filter_list"][1].resolution == 40

    # Check that type is set correctly
    assert variables["filter_list"][0].type == "IFS"
    assert variables["filter_list"][1].type == "IFS"


def test_parse_input_file_filter_single_filter(filter_input_file_single_filter):
    """Test parsing a single filter (scalar values converted to list)."""
    variables, _ = parse_input_file(
        filter_input_file_single_filter, secondary_flag=False
    )

    assert "filter_list" in variables
    assert len(variables["filter_list"]) == 1
    assert variables["filter_list"][0].name == "F1"
    assert variables["filter_list"][0].type == "IMAGER"


def test_parse_input_file_filter_missing_name():
    """Test that missing filter_name raises KeyError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = IMAGER
        wavelength = 0.6
        filter_center = [0.6]
        filter_bandwidth = [0.1]
        """)
        tmp.flush()

        with pytest.raises(
            KeyError,
            match="'filter_name' is required whenever any filter_\\* parameter is provided",
        ):
            parse_input_file(tmp.name, secondary_flag=False)

        os.unlink(tmp.name)


def test_parse_input_file_filter_both_specs_provided():
    """Test that providing both center/bandwidth and low/high raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = IMAGER
        wavelength = 0.6
        filter_name = ['F1']
        filter_center = [0.6]
        filter_bandwidth = [0.1]
        filter_low = [0.5]
        filter_high = [0.7]
        """)
        tmp.flush()

        with pytest.raises(
            ValueError,
            match="Specify filters using either \\(filter_center, filter_bandwidth\\) or \\(filter_low, filter_high\\), not both",
        ):
            parse_input_file(tmp.name, secondary_flag=False)

        os.unlink(tmp.name)


def test_parse_input_file_filter_no_bounds_specified():
    """Test that providing filter_name without bounds raises KeyError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = IMAGER
        wavelength = 0.6
        filter_name = ['F1']
        """)
        tmp.flush()

        with pytest.raises(
            KeyError,
            match="Filters require either \\(filter_center, filter_bandwidth\\) or \\(filter_low, filter_high\\)",
        ):
            parse_input_file(tmp.name, secondary_flag=False)

        os.unlink(tmp.name)


def test_parse_input_file_filter_missing_center():
    """Test that missing filter_center when filter_bandwidth is provided raises KeyError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = IMAGER
        wavelength = 0.6
        filter_name = ['F1']
        filter_bandwidth = [0.1]
        """)
        tmp.flush()

        with pytest.raises(
            KeyError, match="'filter_center' is required to fully specify filter bounds"
        ):
            parse_input_file(tmp.name, secondary_flag=False)

        os.unlink(tmp.name)


def test_parse_input_file_filter_missing_bandwidth():
    """Test that missing filter_bandwidth when filter_center is provided raises KeyError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = IMAGER
        filter_name = ['F1']
        wavelength = 0.6
        filter_center = [0.6]
        """)
        tmp.flush()

        with pytest.raises(
            KeyError,
            match="'filter_bandwidth' is required to fully specify filter bounds",
        ):
            parse_input_file(tmp.name, secondary_flag=False)

        os.unlink(tmp.name)


def test_parse_input_file_filter_missing_low():
    """Test that missing filter_low when filter_high is provided raises KeyError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = IMAGER
        wavelength = 0.7
        filter_name = ['F1']
        filter_high = [0.7]
        """)
        tmp.flush()

        with pytest.raises(
            KeyError, match="'filter_low' is required to fully specify filter bounds"
        ):
            parse_input_file(tmp.name, secondary_flag=False)

        os.unlink(tmp.name)


def test_parse_input_file_filter_missing_high():
    """Test that missing filter_high when filter_low is provided raises KeyError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = IMAGER
        wavelength = 0.5
        filter_name = ['F1']
        filter_low = [0.5]
        """)
        tmp.flush()

        with pytest.raises(
            KeyError, match="'filter_high' is required to fully specify filter bounds"
        ):
            parse_input_file(tmp.name, secondary_flag=False)

        os.unlink(tmp.name)


def test_parse_input_file_filter_ifs_missing_resolution(valid_spectrum_file):
    """Test that IFS mode without filter_resolution raises KeyError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write(f"""
        observing_mode = IFS
        spectrum_file = '{valid_spectrum_file}'

        filter_name = ['Channel1']
        filter_low = [0.5]
        filter_high = [1.0]
        """)
        tmp.flush()

        with pytest.raises(
            KeyError,
            match="'filter_resolution' is required for filters when observing_mode is 'IFS'",
        ):
            parse_input_file(tmp.name, secondary_flag=False)

        os.unlink(tmp.name)


def test_parse_input_file_filter_mismatched_lengths():
    """Test that mismatched filter parameter lengths raise ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = IMAGER
        wavelength = 0.6
        filter_name = ['F1', 'F2']
        filter_center = [0.6]
        filter_bandwidth = [0.1]
        """)
        tmp.flush()

        with pytest.raises(
            ValueError, match="All filter_\\* parameters .* must have the same length"
        ):
            parse_input_file(tmp.name, secondary_flag=False)

        os.unlink(tmp.name)


def test_parse_input_file_filter_mismatched_lengths_with_resolution(
    valid_spectrum_file,
):
    """Test that mismatched filter parameter lengths including resolution raise ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write(f"""
        observing_mode = IFS
        spectrum_file = '{valid_spectrum_file}'
        filter_name = ['F1', 'F2']
        filter_low = [0.5, 0.7]
        filter_high = [0.7, 0.9]
        filter_resolution = [140]
        """)
        tmp.flush()

        with pytest.raises(
            ValueError, match="All filter_\\* parameters .* must have the same length"
        ):
            parse_input_file(tmp.name, secondary_flag=False)

        os.unlink(tmp.name)


def test_parse_input_file_filter_imager_no_resolution():
    """Test that IMAGER mode filters work without resolution (optional)."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = IMAGER
        wavelength = 0.6

        filter_name = ['F1']
        filter_center = [0.6]
        filter_bandwidth = [0.1]
        """)
        tmp.flush()

        variables, _ = parse_input_file(tmp.name, secondary_flag=False)

        # Should work without resolution for IMAGER
        assert "filter_list" in variables
        assert len(variables["filter_list"]) == 1

        os.unlink(tmp.name)


def test_parse_input_file_filter_keys_removed_after_processing(
    filter_input_file_low_high,
):
    """Test that filter_* keys are removed after creating filter_list."""
    variables, _ = parse_input_file(filter_input_file_low_high, secondary_flag=False)

    # Original filter_* keys should be removed
    assert "filter_name" not in variables
    assert "filter_low" not in variables
    assert "filter_high" not in variables

    # Only filter_list should remain
    assert "filter_list" in variables


def test_parse_input_file_filter_type_set_from_observing_mode(valid_spectrum_file):
    """Test that filter type is correctly set from observing_mode."""
    # Test IMAGER
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = IMAGER
        wavelength = 0.6
        Fstar_10pc = 122.9279 
        Fp/Fs = 1e-10
        filter_name = 'F1'
        filter_center = 0.6
        filter_bandwidth = 0.1
        """)
        tmp.flush()

        variables, _ = parse_input_file(tmp.name, secondary_flag=False)
        assert variables["filter_list"][0].type == "IMAGER"

        os.unlink(tmp.name)

    # Test IFS
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write(f"""
        observing_mode = IFS
        spectrum_file = '{valid_spectrum_file}'

        filter_name = 'Channel1'
        filter_low = 0.5
        filter_high = 1.0
        filter_resolution = 140
        """)
        tmp.flush()

        variables, _ = parse_input_file(tmp.name, secondary_flag=False)
        assert variables["filter_list"][0].type == "IFS"

        os.unlink(tmp.name)


# ============================================================================
# Tests for parse_filters - Filter handling
# ============================================================================


def test_parse_filters_single_filter():
    """Test parsing filter_list with a single filter object."""
    from pyEDITH.filters import Filter

    filter_obj = Filter("TestFilter", center=0.5 * u.um, bandwidth=0.1, type="IMAGER")

    active_filters = parse_filters(
        {"wavelength": 0.5, "filter_list": filter_obj, "observing_mode": "IMAGER"}
    )

    assert isinstance(active_filters, list)
    assert len(active_filters) == 1
    assert active_filters[0].name == "TestFilter"


def test_parse_filters_list_multiple_filters():
    """Test parsing filter_list with multiple filter objects."""
    from pyEDITH.filters import Filter

    filter1 = Filter("F1", center=0.6 * u.um, bandwidth=0.1, type="IMAGER")
    filter2 = Filter("F2", center=0.8 * u.um, bandwidth=0.15, type="IMAGER")

    active_filters = parse_filters(
        {
            "wavelength": np.linspace(0.4, 1, 100),
            "filter_list": [filter1, filter2],
            "observing_mode": "IMAGER",
        }
    )

    assert len(active_filters) == 2
    assert active_filters[0].name == "F1"
    assert active_filters[1].name == "F2"


def test_parse_filters_list_invalid_type():
    """Test that non-Filter object in filter_list raises TypeError."""

    with pytest.raises(
        TypeError, match="Filter at index 0: must be a Filter object, but got str"
    ):
        parse_filters(
            {
                "wavelength": 0.5,
                "filter_list": ["not_a_filter"],
                "observing_mode": "IMAGER",
            }
        )


def test_parse_filters_list_mixed_invalid():
    """Test that invalid object in list of filters raises TypeError."""
    from pyEDITH.filters import Filter

    filter1 = Filter("F1", center=0.6 * u.um, bandwidth=0.1, type="IMAGER")

    with pytest.raises(
        TypeError, match="Filter at index 1: must be a Filter object, but got int"
    ):
        parse_filters(
            {
                "wavelength": 0.5,
                "filter_list": [filter1, 42],
                "observing_mode": "IMAGER",
            }
        )


def test_parse_filters_list_empty():
    """Test parsing empty filter_list."""

    with pytest.raises(
        ValueError,
        match="No filters can be used. Specify different filters or change spectrum.",
    ):
        parse_filters(
            {"wavelength": 0.5, "filter_list": [], "observing_mode": "IMAGER"}
        )


def test_parse_filters_list_dict_not_filter():
    """Test that dict (non-Filter object with __dict__) raises TypeError."""

    with pytest.raises(
        TypeError, match="Filter at index 0: must be a Filter object, but got dict"
    ):
        parse_filters(
            {
                "wavelength": 0.5,
                "filter_list": [{"name": "not_a_filter"}],
                "observing_mode": "IMAGER",
            }
        )


def test_parse_filters_ifs_with_only_one_wavelength():
    """Check that in IFS mode wavelength is an array of length > 1."""
    f = Filter("VIS", low=0.4 * u.um, high=0.8 * u.um, resolution=100, type="IFS")
    parameters = {
        "wavelength": [0.55] * WAVELENGTH,
        "observing_mode": "IFS",
        "filter_list": [f],
    }
    with pytest.raises(
        ValueError, match="Assigned an IFS filter but only one wavelength datapoint"
    ):
        result = parse_filters(parameters)


def test_parse_filters_wavelength_already_quantity():
    """When wavelength arrives already as a Quantity, the unit-attach step is skipped (629 -> 632)."""
    f = Filter("VIS", low=0.4 * u.um, high=0.8 * u.um, resolution=100, type="IFS")
    parameters = {
        "wavelength": np.array([0.4, 0.6, 0.8]) * WAVELENGTH,  # already a Quantity
        "observing_mode": "IFS",
        "filter_list": [f],
    }
    result = parse_filters(parameters)
    assert result == [f]


def test_parse_filters_invalid_mode_raises():
    """Unrecognised observing_mode raises before any dispatch."""
    parameters = {
        "wavelength": np.array([0.4, 0.6]) * WAVELENGTH,
        "observing_mode": "SOMETHING_ELSE",
    }
    with pytest.raises(ValueError, match="unrecognised observing_mode"):
        parse_filters(parameters)


def test_parse_filters_ifs_empty_overlap_skips_resolution_check():
    """IFS filter is covered (input straddles it) but no input
    sample point lands inside [f.low, f.high], so overlap is empty and the loop
    continues without running the resolution diagnostic."""
    # Filter band [0.5, 0.6]; input samples at 0.4 and 0.7 straddle it
    # (0.4 <= 0.5 and 0.7 >= 0.6 -> covered), but neither lies within [0.5, 0.6].
    f = Filter("VIS", low=0.5 * u.um, high=0.6 * u.um, resolution=100, type="IFS")
    parameters = {
        "wavelength": np.array([0.4, 0.7]) * WAVELENGTH,
        "observing_mode": "IFS",
        "filter_list": [f],
    }
    result = parse_filters(parameters)
    assert result == [f]


# ============================================================================
# Tests for parse_filters - Legacy Filter Helper (IMAGER mode)
# ============================================================================


def test_parse_filters_legacy_imager_scalar_wavelength():
    """Test legacy filter creation with scalar wavelength."""
    parameters = {
        "wavelength": 0.6,
        "bandwidth": 0.2,
        "observing_mode": "IMAGER",
    }

    filters = parse_filters(parameters)

    assert len(filters) == 1
    assert filters[0].center == 0.6 * u.um
    assert filters[0].bandwidth == 0.2
    assert filters[0].type == "IMAGER"
    assert "0.6" in filters[0].name
    assert "0.2" in filters[0].name


def test_parse_filters_legacy_imager_array_wavelength_single_element():
    """Test legacy filter with single-element wavelength array."""
    parameters = {
        "wavelength": np.array([0.55]),
        "bandwidth": 0.15,
        "observing_mode": "IMAGER",
    }

    filters = parse_filters(parameters)

    assert len(filters) == 1
    assert filters[0].center.value == pytest.approx(0.55)
    assert filters[0].bandwidth == 0.15


def test_parse_filters_legacy_imager_missing_bandwidth():
    """Test error when IMAGER mode lacks bandwidth parameter."""
    parameters = {
        "wavelength": 0.6,
        "observing_mode": "IMAGER",
    }

    with pytest.raises(KeyError):
        parse_filters(parameters)


# ============================================================================
# Tests for parse_filters - Legacy Filter Helper (IFS mode)
# ============================================================================


def test_parse_filters_legacy_ifs_bounds_already_quantity():
    """Line 711: _as_wavelength_quantity receives an existing Quantity (not a
    list), taking the isinstance branch. Also covers the legacy IFS build path."""
    parameters = {
        "wavelength": np.array([0.3, 0.6, 1.0]) * WAVELENGTH,
        "observing_mode": "IFS",
        "spectral_resolution": [100],
        "lam_low": np.array([0.4]) * u.um,  # a Quantity, not a list
        "lam_high": np.array([0.8]) * u.um,  # a Quantity, not a list
    }
    result = parse_filters(parameters)
    assert len(result) == 1
    assert result[0].type == "IFS"
    assert np.isclose(result[0].low.to(u.um).value, 0.4)
    assert np.isclose(result[0].high.to(u.um).value, 0.8)


def test_parse_filters_legacy_ifs_creates_multiple_filters(caplog):
    """Test that multiple filters are created from legacy arrays."""
    parameters = {
        "wavelength": np.linspace(0.5, 2.0, 1000),
        "observing_mode": "IFS",
        "spectral_resolution": [100, 200],
        "lam_low": [0.6, 1.0],
        "lam_high": [0.9, 1.5],
    }

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        filters = parse_filters(parameters)

    assert len(filters) == 2

    # Check first filter
    assert filters[0].low.value == pytest.approx(0.6)
    assert filters[0].high.value == pytest.approx(0.9)
    assert filters[0].resolution == 100
    assert filters[0].type == "IFS"

    # Check second filter
    assert filters[1].low.value == pytest.approx(1.0)
    assert filters[1].high.value == pytest.approx(1.5)
    assert filters[1].resolution == 200
    assert filters[1].type == "IFS"
    # Check deprecation warning
    assert any("DeprecationWarning" in record.message for record in caplog.records)

    assert "0.6" in filters[0].name and "0.9" in filters[0].name
    assert "1.0" in filters[1].name and "1.5" in filters[1].name
    assert "IFS" in filters[0].name


def test_parse_filters_legacy_ifs_wavelength_boundaries_valid():
    """Test that input wavelength covers filter ranges."""
    parameters = {
        "wavelength": np.linspace(0.5, 2.0, 1000),
        "observing_mode": "IFS",
        "spectral_resolution": [100, 200],
        "lam_low": [0.6, 1.0],
        "lam_high": [0.9, 1.5],
    }

    filters = parse_filters(parameters)
    assert len(filters) == 2


def test_parse_filters_legacy_ifs_minimum_wavelength_too_high():
    """Test error when minimum input wavelength is above first filter."""
    parameters = {
        "wavelength": np.linspace(0.7, 2.0, 1000),
        "observing_mode": "IFS",
        "spectral_resolution": [100, 200],
        "lam_low": [0.6, 1.0],
        "lam_high": [0.9, 1.5],
    }

    with pytest.raises(AssertionError, match="minimum input wavelength is greater"):
        parse_filters(parameters)


def test_parse_filters_legacy_ifs_maximum_wavelength_too_low():
    """Test error when maximum input wavelength is below last filter."""
    parameters = {
        "wavelength": np.linspace(0.5, 1.4, 1000),
        "observing_mode": "IFS",
        "spectral_resolution": [100, 200],
        "lam_low": [0.6, 1.0],
        "lam_high": [0.9, 1.5],
    }

    with pytest.raises(AssertionError, match="maximum input wavelength is less"):
        parse_filters(parameters)


def test_parse_filters_legacy_ifs_mismatched_array_lengths():
    """Test error when legacy parameter arrays have different lengths."""
    parameters = {
        "wavelength": np.linspace(0.5, 2.0, 1000),
        "observing_mode": "IFS",
        "spectral_resolution": [100, 200, 300],  # 3 elements
        "lam_low": [0.6, 1.0],  # 2 elements
        "lam_high": [0.9, 1.5],  # 2 elements
    }

    with pytest.raises(AssertionError, match="have different lengths"):
        parse_filters(parameters)


def test_parse_filters_legacy_ifs_non_array_parameters():
    """Test error when legacy parameters are not arrays."""
    parameters = {
        "wavelength": np.linspace(0.5, 2.0, 1000),
        "observing_mode": "IFS",
        "spectral_resolution": 100,  # Should be array
        "lam_low": [0.6],
        "lam_high": [0.9],
    }

    with pytest.raises(ValueError, match="is not an array"):
        parse_filters(parameters)


def test_parse_filters_legacy_ifs_missing_parameters():
    """Test error when some legacy parameters are missing."""
    parameters = {
        "wavelength": np.linspace(0.5, 2.0, 1000),
        "observing_mode": "IFS",
        "spectral_resolution": [100],
        "lam_low": [0.6],
        # Missing lam_high
    }

    with pytest.raises(ValueError, match="some of the following keys are missing"):
        parse_filters(parameters)


def test_parse_filters_legacy_ifs_without_units_adds_units():
    """Test that legacy IFS adds WAVELENGTH units when missing."""
    parameters = {
        "wavelength": np.linspace(0.5, 2.0, 1000),
        "observing_mode": "IFS",
        "spectral_resolution": [150],
        "lam_low": [0.7],  # No units
        "lam_high": [1.2],  # No units
    }

    filters = parse_filters(parameters)

    assert filters[0].low.unit == u.um
    assert filters[0].high.unit == u.um


def test_parse_filters_legacy_ifs_empty_arrays_raises_error():
    """Test error with empty legacy parameter arrays."""
    parameters = {
        "wavelength": np.linspace(0.5, 2.0, 1000),
        "observing_mode": "IFS",
        "spectral_resolution": [],
        "lam_low": [],
        "lam_high": [],
    }

    with pytest.raises((AssertionError, IndexError, ValueError)):
        parse_filters(parameters)


def test_parse_filters_legacy_ifs_regrid_info_message(caplog):
    """Test that IFS mode logs wavelength grid calculation message."""
    parameters = {
        "wavelength": np.linspace(0.5, 2.0, 1000),
        "observing_mode": "IFS",
        "spectral_resolution": [100],
        "lam_low": [0.6],
        "lam_high": [0.9],
    }

    with caplog.at_level(logging.INFO, logger="pyEDITH"):
        parse_filters(parameters)

    assert any(
        "Calculating a new wavelength grid" in record.message
        for record in caplog.records
    )


def test_parse_filters_legacy_ifs_bounds_already_quantity():
    """Legacy IFS path with lam_low/lam_high already as Quantities skips both
    unit-attach steps"""
    parameters = {
        "wavelength": np.array([0.3, 0.5, 0.9]) * WAVELENGTH,
        "observing_mode": "IFS",
        "spectral_resolution": [100],
        "lam_low": [0.4 * u.um],  # already Quantities
        "lam_high": [0.8 * u.um],  # already Quantities
    }
    result = parse_filters(parameters)
    assert len(result) == 1
    assert result[0].type == "IFS"
    assert np.isclose(result[0].low.to(u.um).value, 0.4)
    assert np.isclose(result[0].high.to(u.um).value, 0.8)


# ============================================================================
# Tests for parse_filters - Filter validation (from old tests)
# ============================================================================


def test_parse_filters_filter_outside_lower_bound(caplog):
    """Test that filter below wavelength range is discarded with warning."""
    from pyEDITH.filters import Filter

    filter1 = Filter("Filter1", low=0.5 * u.um, high=0.7 * u.um, type="IMAGER")
    filter2 = Filter("Filter2", low=0.8 * u.um, high=1.0 * u.um, type="IMAGER")

    parameters = {
        "wavelength": np.linspace(0.75, 1.2, 100),
        "filter_list": [filter1, filter2],
        "observing_mode": "IMAGER",
    }

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        active_filters = parse_filters(parameters)

    assert any(
        "Filter Filter1 discarded" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )
    assert len(active_filters) == 1
    assert active_filters[0].name == "Filter2"


def test_parse_filters_filter_outside_upper_bound(caplog):
    """Test that filter above wavelength range is discarded with warning."""
    from pyEDITH.filters import Filter

    filter1 = Filter("Filter1", low=0.5 * u.um, high=0.7 * u.um, type="IMAGER")
    filter2 = Filter("Filter2", low=0.8 * u.um, high=1.0 * u.um, type="IMAGER")

    parameters = {
        "wavelength": np.linspace(0.4, 0.75, 100),
        "filter_list": [filter1, filter2],
        "observing_mode": "IMAGER",
    }

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        active_filters = parse_filters(parameters)

    assert any(
        "Filter Filter2 discarded" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )
    assert len(active_filters) == 1
    assert active_filters[0].name == "Filter1"


def test_parse_filters_filter_partially_overlapping(caplog):
    """Test that partially overlapping filter is discarded."""
    from pyEDITH.filters import Filter

    filter1 = Filter(
        "Filter1", low=0.5 * u.um, high=0.7 * u.um, type="IFS", resolution=50
    )
    filter2 = Filter(
        "Filter2", low=0.8 * u.um, high=1.0 * u.um, type="IFS", resolution=50
    )

    parameters = {
        "wavelength": np.linspace(0.6, 1.2, 100),
        "filter_list": [filter1, filter2],
        "observing_mode": "IFS",
    }

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        active_filters = parse_filters(parameters)

    print(caplog.records)
    assert any(
        "Filter Filter1 discarded" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )
    assert len(active_filters) == 1
    assert active_filters[0].name == "Filter2"


def test_parse_filters_filter_exact_boundaries():
    """Test filters with wavelength range exactly matching filter boundaries."""
    from pyEDITH.filters import Filter

    filter1 = Filter("Filter1", low=0.5 * u.um, high=0.7 * u.um, type="IMAGER")
    filter2 = Filter("Filter2", low=0.8 * u.um, high=1.0 * u.um, type="IMAGER")

    parameters = {
        "wavelength": np.array([0.5, 1.0]),
        "filter_list": [filter1, filter2],
        "observing_mode": "IMAGER",
    }

    active_filters = parse_filters(parameters)

    assert len(active_filters) == 2


def test_parse_filters_low_resolution_warning_ifs(caplog):
    """Test warning when input spectrum resolution is lower than filter resolution in IFS mode."""
    from pyEDITH.filters import Filter

    high_res_filter = Filter(
        "HighResFilter", low=0.5 * u.um, high=0.7 * u.um, resolution=600, type="IFS"
    )

    low_res_wavelength = np.linspace(0.4, 0.8, 20)
    parameters = {
        "wavelength": low_res_wavelength,
        "filter_list": [high_res_filter],
        "observing_mode": "IFS",
    }

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        parse_filters(parameters)

    assert any(
        "Input spectrum resolution" in record.message
        and "lower than" in record.message
        and "filter resolution" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_parse_filters_adequate_resolution_no_warning_ifs(caplog):
    """Test no warning when input spectrum resolution is adequate for filter in IFS mode."""
    from pyEDITH.filters import Filter

    filter_obj = Filter(
        "MedResFilter", low=0.5 * u.um, high=0.7 * u.um, resolution=300, type="IFS"
    )

    high_res_wavelength = np.linspace(0.4, 0.8, 500)
    parameters = {
        "wavelength": high_res_wavelength,
        "filter_list": [filter_obj],
        "observing_mode": "IFS",
    }

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        parse_filters(parameters)

    resolution_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "Input spectrum resolution" in record.message
        and "lower than" in record.message
    ]
    assert len(resolution_warnings) == 0


def test_parse_filters_low_resolution_no_warning_imager(caplog):
    """Test no warning for low resolution in IMAGER mode."""
    from pyEDITH.filters import Filter

    filter_obj = Filter("ImageFilter", low=0.5 * u.um, high=0.7 * u.um, type="IMAGER")

    low_res_wavelength = np.linspace(0.4, 0.8, 20)
    parameters = {
        "wavelength": low_res_wavelength,
        "filter_list": [filter_obj],
        "observing_mode": "IMAGER",
    }

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        parse_filters(parameters)

    resolution_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "Input spectrum resolution" in record.message
    ]
    assert len(resolution_warnings) == 0


def test_parse_filters_resolution_warning_multiple_filters(caplog):
    """Test resolution warnings for multiple filters with different resolutions."""
    from pyEDITH.filters import Filter

    low_res_filter = Filter(
        "LowResFilter", low=0.5 * u.um, high=0.6 * u.um, resolution=110, type="IFS"
    )

    high_res_filter = Filter(
        "HighResFilter", low=0.7 * u.um, high=0.8 * u.um, resolution=1500, type="IFS"
    )

    input_wavelength = np.linspace(0.4, 0.9, 150)
    parameters = {
        "wavelength": input_wavelength,
        "filter_list": [low_res_filter, high_res_filter],
        "observing_mode": "IFS",
    }

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        parse_filters(parameters)

    resolution_warnings = [
        record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "Input spectrum resolution" in record.message
    ]
    assert len(resolution_warnings) == 1
    assert "HighResFilter" in resolution_warnings[0]
    assert "LowResFilter" not in resolution_warnings[0]


def test_parse_filters_no_valid_filters_raises_error():
    """Test that having no valid filters after filtering raises ValueError."""
    from pyEDITH.filters import Filter

    # Filters completely outside wavelength range
    filter1 = Filter("Filter1", low=0.3 * u.um, high=0.4 * u.um, type="IMAGER")

    parameters = {
        "wavelength": np.linspace(0.5, 1.0, 100),
        "filter_list": [filter1],
        "observing_mode": "IMAGER",
    }

    with pytest.raises(
        ValueError,
        match="No filters can be used. Specify different filters or change spectrum.",
    ):
        parse_filters(parameters)
