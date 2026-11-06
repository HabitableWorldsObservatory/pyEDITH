import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from pyEDITH.cli import (
    main,
    calculate_texp,
    calculate_snr,
    _get_closest_finite_time,
    calculate_snr_from_reference_lambda,
)
from pyEDITH.filters import Filter
from io import StringIO
import astropy.units as u
import logging

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_args():
    """Fixture to create a mock arguments object simulating parsed CLI arguments."""

    class Args:
        edith = "test.edith"
        verbose = False
        time = 60.0

    return Args()


@pytest.fixture
def mock_parameters():
    """Fixture to create a mock parameters dictionary."""
    return {
        "wavelength": [0.5],
        "distance": 10,
        "magV": 5.0,
        "nzodis": 3.0,
        "observing_mode": "IMAGER",
        "observatory_preset": "ToyModel",
        "filter_list": [
            Filter("photometry", center=0.5 * u.um, bandwidth=0.2, type="IMAGER")
        ],
    }


@pytest.fixture
def mock_parameters_ifs():
    """Fixture to create mock parameters for IFS mode."""
    return {
        "wavelength": np.linspace(0.2, 1.8, 100),
        "distance": 10,
        "magV": 5.0,
        "nzodis": 3.0,
        "observing_mode": "IFS",
        "observatory_preset": "ToyModel",
        "filter_list": [Filter("VIS", low=0.4 * u.um, high=1.0 * u.um, resolution=140)],
    }


@pytest.fixture
def mock_parameters_reference():
    """Parameters whose filter spans the 1.0–2.0 µm reference wavelengths
    used by the calculate_snr_from_reference_lambda tests, so Filter.contains()
    is exercised honestly (1.5 inside, 99.0 outside)."""
    return {
        "wavelength": np.linspace(1.0, 2.0, 100),
        "distance": 10,
        "magV": 5.0,
        "nzodis": 3.0,
        "observing_mode": "IFS",
        "observatory_preset": "ToyModel",
        "filter_list": [
            Filter("F150W", low=1.0 * u.um, high=2.0 * u.um, resolution=140)
        ],
    }


@pytest.fixture
def mock_parameters_two_filters():
    return {
        "observing_mode": "IFS",
        "observatory_preset": "ToyModel",
        "wavelength": np.linspace(0.2, 2.0, 100),
        "filter_list": [
            Filter("VIS", low=0.4 * u.um, high=1.0 * u.um, resolution=140),
            Filter("NIR", low=1.0 * u.um, high=1.8 * u.um, resolution=70),
        ],
    }


# ============================================================================
# Tests for main() - No arguments
# ============================================================================


def test_main_no_args_shows_help():
    """Test that main with no arguments displays usage help."""
    with patch("sys.argv", ["edith"]):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            main()
            assert "usage: edith" in fake_out.getvalue()


# ============================================================================
# Tests for main() - Missing required arguments
# ============================================================================


@pytest.mark.parametrize("subcommand", ["etc", "snr", "etc2snr"])
def test_main_missing_edith_argument(subcommand):
    """Missing --edith raises the unified SyntaxError for every subcommand."""
    with patch("sys.argv", ["edith", subcommand]):
        with pytest.raises(SyntaxError, match="--edith argument is required."):
            main()


@patch("pyEDITH.cli.parse_input.parse_input_file")
@patch("pyEDITH.cli.parse_input.parse_parameters")
def test_main_snr_missing_time_argument(mock_parse_params, mock_parse_file):
    """snr with --edith but no --time raises the new SyntaxError."""
    mock_parse_file.return_value = {}
    mock_parse_params.return_value = {}
    with patch("sys.argv", ["edith", "snr", "--edith", "test.edith"]):
        with pytest.raises(
            SyntaxError, match="--time argument is required for snr subfunction."
        ):
            main()


# ============================================================================
# Tests for main() - etc subcommand
# ============================================================================


@patch("pyEDITH.cli.parse_input.parse_input_file")
@patch("pyEDITH.cli.parse_input.parse_parameters")
@patch("pyEDITH.cli.calculate_texp")
def test_main_etc_basic_execution(mock_texp, mock_parse_params, mock_parse_file):
    """etc subcommand calculates and prints exposure time."""
    mock_parse_file.return_value = {}
    mock_parse_params.return_value = {}
    mock_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["edith", "etc", "--edith", "test.edith"]):
        with patch("builtins.print") as mock_print:
            main()
            mock_print.assert_called_with(np.array([1.0]))


# ============================================================================
# Tests for main() - snr subcommand
# ============================================================================


@patch("pyEDITH.cli.parse_input.parse_input_file")
@patch("pyEDITH.cli.parse_input.parse_parameters")
@patch("pyEDITH.cli.calculate_snr")
def test_main_snr_basic_execution(mock_snr, mock_parse_params, mock_parse_file):
    """snr subcommand calculates and prints SNR."""
    mock_parse_file.return_value = {}
    mock_parse_params.return_value = {}
    mock_snr.return_value = (np.array([10.0]), {})

    with patch("sys.argv", ["edith", "snr", "--edith", "test.edith", "--time", "60"]):
        with patch("builtins.print") as mock_print:
            main()
            mock_print.assert_called_with(np.array([10.0]))


# ============================================================================
# Tests for main() - etc2snr subcommand (single-function routing)
# ============================================================================


@patch("pyEDITH.cli.parse_input.parse_input_file")
@patch("pyEDITH.cli.calculate_snr_from_reference_lambda")
def test_main_etc2snr_basic_execution(mock_ref_snr, mock_parse_file):
    """etc2snr with --ref_lambda runs through to the end of main()."""
    mock_parse_file.return_value = {}
    snrs = {"VIS": {"snr": np.array([7.0])}}
    texp = 30.0 * u.s  # scalar Quantity — the engine's real return shape
    mock_ref_snr.return_value = (snrs, texp, {})

    with patch(
        "sys.argv",
        ["edith", "etc2snr", "--edith", "test.edith", "--ref_lambda", "1.5"],
    ):
        with patch("builtins.print"):
            main()
    mock_ref_snr.assert_called_once()


@patch("pyEDITH.cli.parse_input.parse_input_file")
def test_main_etc2snr_no_ref_lambda_raises(mock_parse_file):
    """etc2snr without --ref_lambda raises a descriptive ValueError."""
    mock_parse_file.return_value = {}

    with patch("sys.argv", ["edith", "etc2snr", "--edith", "test.edith"]):
        with pytest.raises(ValueError, match="etc2snr requires --ref_lambda"):
            main()


@patch("pyEDITH.cli.parse_input.parse_input_file")
@patch("pyEDITH.cli.calculate_snr_from_reference_lambda")
def test_main_etc2snr_passes_scalar_float(mock_ref_snr, mock_parse_file):
    """main() passes --ref_lambda to the engine as a single float (not a list)."""
    mock_parse_file.return_value = {}
    mock_ref_snr.return_value = ({"VIS": {"snr": np.array([7.0])}}, 30.0 * u.s, {})

    with patch(
        "sys.argv",
        ["edith", "etc2snr", "--edith", "test.edith", "--ref_lambda", "1.5"],
    ):
        with patch("builtins.print"):
            main()

    mock_ref_snr.assert_called_once()
    passed = mock_ref_snr.call_args[0][1]
    assert passed == 1.5
    assert isinstance(passed, float), "Engine must receive a scalar float, not a list."


# ============================================================================
# Tests for main() - Verbosity levels
# ============================================================================


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.parse_input_file")
@patch("pyEDITH.cli.parse_input.parse_parameters")
@patch("pyEDITH.cli.calculate_texp")
def test_main_default_verbosity_level(
    mock_texp, mock_parse_params, mock_parse_file, mock_set_verbosity
):
    """Test that default verbosity (no flag) sets warning level."""
    mock_parse_file.return_value = {}
    mock_parse_params.return_value = {}
    mock_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("warning")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.parse_input_file")
@patch("pyEDITH.cli.parse_input.parse_parameters")
@patch("pyEDITH.cli.calculate_texp")
def test_main_verbose_flag_sets_info_level(
    mock_texp, mock_parse_params, mock_parse_file, mock_set_verbosity
):
    """Test that -v flag sets info verbosity level."""
    mock_parse_file.return_value = {}
    mock_parse_params.return_value = {}
    mock_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "-v", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("info")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.parse_input_file")
@patch("pyEDITH.cli.parse_input.parse_parameters")
@patch("pyEDITH.cli.calculate_texp")
def test_main_verbose_long_flag_sets_info_level(
    mock_texp, mock_parse_params, mock_parse_file, mock_set_verbosity
):
    """Test that --verbose flag sets info verbosity level."""
    mock_parse_file.return_value = {}
    mock_parse_params.return_value = {}
    mock_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "--verbose", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("info")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.parse_input_file")
@patch("pyEDITH.cli.parse_input.parse_parameters")
@patch("pyEDITH.cli.calculate_texp")
def test_main_double_verbose_flag_sets_debug_level(
    mock_texp, mock_parse_params, mock_parse_file, mock_set_verbosity
):
    """Test that -vv flag sets debug verbosity level."""
    mock_parse_file.return_value = {}
    mock_parse_params.return_value = {}
    mock_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "-vv", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("debug")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.parse_input_file")
@patch("pyEDITH.cli.parse_input.parse_parameters")
@patch("pyEDITH.cli.calculate_texp")
def test_main_quiet_flag_sets_error_level(
    mock_texp, mock_parse_params, mock_parse_file, mock_set_verbosity
):
    """Test that -q flag sets error verbosity level."""
    mock_parse_file.return_value = {}
    mock_parse_params.return_value = {}
    mock_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "-q", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("error")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.parse_input_file")
@patch("pyEDITH.cli.parse_input.parse_parameters")
@patch("pyEDITH.cli.calculate_texp")
def test_main_quiet_long_flag_sets_error_level(
    mock_texp, mock_parse_params, mock_parse_file, mock_set_verbosity
):
    """Test that --quiet flag sets error verbosity level."""
    mock_parse_file.return_value = {}
    mock_parse_params.return_value = {}
    mock_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "--quiet", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("error")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.parse_input_file")
@patch("pyEDITH.cli.parse_input.parse_parameters")
@patch("pyEDITH.cli.calculate_texp")
def test_main_quiet_overrides_verbose(
    mock_texp, mock_parse_params, mock_parse_file, mock_set_verbosity
):
    """Test that --quiet flag takes precedence over -vv flag."""
    mock_parse_file.return_value = {}
    mock_parse_params.return_value = {}
    mock_texp.return_value = (np.array([1.0]), {})

    with patch(
        "sys.argv", ["pyedith", "--quiet", "-vv", "etc", "--edith", "test.edith"]
    ):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("error")


# ============================================================================
# Tests for calculate_texp()
# ============================================================================


def test_calculate_texp_basic_execution(mock_parameters):
    """Test that calculate_texp correctly initializes objects and returns exposure time."""
    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.Observatory") as mock_observatory,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr") as mock_calculate,
    ):
        # Set up mock objects
        mock_observation_instance = MagicMock()
        mock_observation_instance.exptime = np.array([1.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance

        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance

        mock_observatory_instance = MagicMock()
        mock_observatory.return_value = mock_observatory_instance

        # Execute
        texp, validation_variables = calculate_texp(mock_parameters, False)

        # Verify results
        filter_name = mock_parameters["filter_list"][0].name
        assert filter_name in texp
        assert "wavelength" in texp[filter_name]
        assert "exposure_time" in texp[filter_name]
        assert np.array_equal(texp[filter_name]["exposure_time"], np.array([1.0]))
        assert filter_name in validation_variables
        assert validation_variables[filter_name] == {}


def test_calculate_texp_calls_all_components(mock_parameters):
    """Test that calculate_texp calls all required component methods."""
    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.Observatory") as mock_observatory,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr") as mock_calculate,
    ):
        # Set up mock objects
        mock_observation_instance = MagicMock()
        mock_observation_instance.exptime = np.array([1.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance

        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance

        mock_observatory_instance = MagicMock()
        mock_observatory.return_value = mock_observatory_instance

        # Execute
        calculate_texp(mock_parameters, False)

        # Verify all components were called
        mock_observation.assert_called_once()
        mock_scene.assert_called_once()
        mock_calculate.assert_called_once()


def test_calculate_texp_ifs_mode_with_regridding(mock_parameters_ifs):
    """Test that calculate_texp correctly handles IFS mode with wavelength regridding."""
    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.Observatory") as mock_observatory,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr") as mock_calculate,
    ):
        # Set up mock objects
        mock_observation_instance = MagicMock()
        mock_observation_instance.exptime = np.array([1.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance

        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance

        mock_observatory_instance = MagicMock()
        mock_observatory.return_value = mock_observatory_instance

        # Execute
        texp, validation_variables = calculate_texp(mock_parameters_ifs, False)

        # Verify results
        filter_name = mock_parameters_ifs["filter_list"][0].name
        assert filter_name in texp
        assert "wavelength" in texp[filter_name]
        assert "exposure_time" in texp[filter_name]
        assert np.array_equal(texp[filter_name]["exposure_time"], np.array([1.0]))
        assert filter_name in validation_variables
        assert validation_variables[filter_name] == {}

        # Verify scene regridding was called
        mock_scene_instance.regrid_spectra.assert_called_once()


# ============================================================================
# Tests for calculate_snr()
# ============================================================================


def test_calculate_snr_basic_execution(mock_parameters):
    """Test that calculate_snr correctly initializes objects and returns SNR."""
    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.Observatory") as mock_observatory,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr") as mock_calculate,
    ):
        # Set up mock objects
        mock_observation_instance = MagicMock()
        mock_observation_instance.fullsnr = np.array([10.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance

        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance

        mock_observatory_instance = MagicMock()
        mock_observatory.return_value = mock_observatory_instance

        # Execute
        snr, validation_variables = calculate_snr(mock_parameters, 1.0)

        # Verify results
        filter_name = mock_parameters["filter_list"][0].name
        assert filter_name in snr
        assert "wavelength" in snr[filter_name]
        assert "snr" in snr[filter_name]
        assert np.array_equal(snr[filter_name]["snr"], np.array([10.0]))
        assert filter_name in validation_variables
        assert validation_variables[filter_name] == {}


def test_calculate_snr_calls_all_components(mock_parameters):
    """Test that calculate_snr calls all required component methods."""
    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.Observatory") as mock_observatory,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr") as mock_calculate,
    ):
        # Set up mock objects
        mock_observation_instance = MagicMock()
        mock_observation_instance.fullsnr = np.array([10.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance

        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance

        mock_observatory_instance = MagicMock()
        mock_observatory.return_value = mock_observatory_instance

        # Execute
        calculate_snr(mock_parameters, 1.0)

        # Verify all components were called
        mock_observation.assert_called_once()
        mock_scene.assert_called_once()
        mock_observatory.assert_called_once()
        mock_calculate.assert_called_once()


def test_calculate_snr_ifs_mode_with_regridding(mock_parameters_ifs):
    """Test that calculate_snr correctly handles IFS mode with wavelength regridding."""
    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.Observatory") as mock_observatory,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr") as mock_calculate,
    ):
        # Set up mock objects
        mock_observation_instance = MagicMock()
        mock_observation_instance.fullsnr = np.array([10.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance

        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance

        mock_observatory_instance = MagicMock()
        mock_observatory.return_value = mock_observatory_instance

        # Execute
        snr, validation_variables = calculate_snr(mock_parameters_ifs, 1.0)

        # Verify results
        filter_name = mock_parameters_ifs["filter_list"][0].name
        assert filter_name in snr
        assert "wavelength" in snr[filter_name]
        assert "snr" in snr[filter_name]
        assert np.array_equal(snr[filter_name]["snr"], np.array([10.0]))
        assert filter_name in validation_variables
        assert validation_variables[filter_name] == {}

        # Verify scene regridding was called
        mock_scene_instance.regrid_spectra.assert_called_once()


# ============================================================================
# Tests for calculate_snr() - dict (independent) time dispatch
# ============================================================================


def test_calculate_snr_accepts_dict_of_times(mock_parameters):
    """Test that calculate_snr maps a dict of times onto the matching filter."""
    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.Observatory") as mock_observatory,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr"),
    ):
        mock_observation_instance = MagicMock()
        mock_observation_instance.fullsnr = np.array([10.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance
        mock_scene.return_value = MagicMock()
        mock_observatory.return_value = MagicMock()

        filter_name = mock_parameters["filter_list"][0].name
        times_dict = {filter_name: 42.0}

        snr, _ = calculate_snr(mock_parameters, times_dict)

        # The scalar time for this specific filter must have been assigned
        assert mock_observation_instance.obstime == 42.0
        assert filter_name in snr


def test_calculate_snr_dict_missing_filter_raises(mock_parameters):
    """Test that calculate_snr raises KeyError if a filter is absent from the time dict."""
    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.Observatory") as mock_observatory,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr"),
    ):
        mock_observation.return_value = MagicMock()
        mock_scene.return_value = MagicMock()
        mock_observatory.return_value = MagicMock()

        # Provide a dict that does NOT contain the real filter name
        times_dict = {"NonExistentFilter": 42.0}

        with pytest.raises(
            KeyError, match="not found in the provided exposure time dict"
        ):
            calculate_snr(mock_parameters, times_dict)


# ============================================================================
# Tests for _get_closest_finite_time()
# ============================================================================


def test_get_closest_finite_time_exact_match():
    """Test the helper picks the exact bin when ref_wav lands on a grid point."""
    wvl = np.array([1.0, 1.5, 2.0])
    exptime = np.array([10.0, 20.0, 30.0])

    t_val, actual_wvl = _get_closest_finite_time(wvl, exptime, 1.5)

    assert t_val == 20.0
    assert actual_wvl == 1.5


def test_get_closest_finite_time_nearest_match():
    """Test the helper rounds to the nearest bin for an in-between ref_wav."""
    wvl = np.array([1.0, 1.5, 2.0])
    exptime = np.array([10.0, 20.0, 30.0])

    # 1.6 is closest to 1.5
    t_val, actual_wvl = _get_closest_finite_time(wvl, exptime, 1.6)

    assert t_val == 20.0
    assert actual_wvl == 1.5


def test_get_closest_finite_time_scalar_input():
    """Test the helper handles IMAGER-style scalar wavelength/time inputs."""
    t_val, actual_wvl = _get_closest_finite_time(1.0, 42.0, 1.0)

    assert t_val == 42.0
    assert actual_wvl == 1.0


def test_get_closest_finite_time_raises_on_infinite():
    """Test the helper raises ValueError when the closest bin is non-finite."""
    wvl = np.array([1.0, 1.5, 2.0])
    exptime = np.array([np.inf, 20.0, 30.0])

    # 1.0 is closest to the infinite bin
    with pytest.raises(ValueError, match="is infinite"):
        _get_closest_finite_time(wvl, exptime, 1.0)


def test_get_closest_finite_time_allows_inf_away_from_reference():
    """Test that an inf at the band edge is tolerated if ref_wav is elsewhere."""
    wvl = np.array([1.0, 1.5, 2.0])
    exptime = np.array([np.inf, 20.0, 30.0])

    # ref_wav 2.0 is far from the inf edge -> should succeed
    t_val, actual_wvl = _get_closest_finite_time(wvl, exptime, 2.0)

    assert t_val == 30.0
    assert actual_wvl == 2.0


# ============================================================================
# Tests for calculate_snr_from_reference_lambda
# ============================================================================


@patch("pyEDITH.cli.calculate_snr")
@patch("pyEDITH.cli.calculate_texp")
def test_reference_snr_coupled_orchestration(
    mock_texp, mock_snr, mock_parameters_two_filters
):
    """Orchestration (mocked engine): a single reference wavelength yields one
    scalar time from the containing filter, which is then passed to calculate_snr
    and broadcast across all filters.

    NOTE: etc2snr now requires >=2 filters, so this uses the two-filter fixture.
    The driving time is returned as a SCALAR (not a per-filter dict -- that was
    the artefact of the removed independent mode).
    """
    mock_texp.return_value = (
        {
            "VIS": {
                "wavelength": np.array([0.4, 0.5, 0.6]) * u.um,
                "exposure_time": np.array([10.0, 20.0, 30.0]) * u.s,
            },
            "NIR": {
                "wavelength": np.array([1.0, 1.2, 1.5]) * u.um,
                "exposure_time": np.array([40.0, 50.0, 60.0]) * u.s,
            },
        },
        {},
    )
    mock_snr.return_value = (
        {"VIS": {"snr": np.array([7.0])}, "NIR": {"snr": np.array([5.0])}},
        {},
    )

    # Reference 0.5 um lands in VIS; closest-finite time there is 20.0 s.
    snrs, texp, _ = calculate_snr_from_reference_lambda(
        mock_parameters_two_filters, reference_wavelength=0.5
    )

    # Scalar return, taken from the filter containing the reference wavelength.
    assert texp == 20.0 * u.s

    # That SAME scalar is what gets passed to calculate_snr (one shared time).
    assert mock_snr.call_args[0][1] == 20.0 * u.s

    assert "VIS" in snrs and "NIR" in snrs


import numpy as np
import astropy.units as u
import pytest

from pyEDITH.filters import Filter
from pyEDITH.parse_input import parse_parameters

# Workflow functions currently live in cli.py. When they are promoted to
# workflows.py, update this import AND move the "WORKFLOW-CONTRACT" tests below
# into test_workflows.py. The translation tests above stay here.
from pyEDITH.cli import calculate_snr_from_reference_lambda

# ============================================================================
# WORKFLOW-CONTRACT tests (Job B)
#
# These exercise the REAL engine (no mocking of calculate_*). They verify the
# workflow functions' own contracts, independent of argv/argparse. They live
# here per the interim rule "non-equivalence tests go in test_cli"; they will
# migrate to test_workflows.py when those functions leave cli.py.
#
# Distinct in kind from the translation tests above, which DO mock the engine
# and only check main()'s routing. Do not conflate the two.
# ============================================================================


@pytest.fixture
def toymodel_two_filter_params():
    """A two-filter ToyModel scenario (blue 0.5 um, red 1.0 um), built as a
    Python API user would and parsed. Shared source of truth for the coupled
    etc2snr contract tests, so they cannot drift apart."""
    return parse_parameters(
        {
            "observing_mode": "IMAGER",
            "observatory_preset": "ToyModel",
            "wavelength": [0.5, 1.0],
            "snr": 7.0,
            "bandwidth": 0.2,
            "CRb_multiplier": 2.0,
            "photometric_aperture_radius": 0.85,
            "distance": 10.0,
            "Fstar_10pc": [122.9279, 120.0],
            "FstarV_10pc": 115.59984,
            "stellar_radius": 1.0,
            "nzodis": 3.0,
            "ra": 236.0,
            "dec": 2.5,
            "separation": 0.1,
            "Fp/Fs": [1e-10, 1.1e-10],
            "filter_list": [
                Filter("blue", center=0.5 * u.um, bandwidth=0.2, type="IMAGER"),
                Filter("red", center=1.0 * u.um, bandwidth=0.2, type="IMAGER"),
            ],
        }
    )


@pytest.mark.integration
def test_workflow_etc2snr_coupled(toymodel_two_filter_params):
    """calculate_snr_from_reference_lambda: one reference wavelength drives a
    single shared exposure time applied to BOTH filters (the coupled contract).

    Exercises the real engine. The reference band (blue, 0.5 um) must recover
    the target SNR (7.0); the other band (red), evaluated at the SAME time, must
    NOT -- that non-recovery is the whole point of coupling, and its presence is
    what distinguishes this from the removed (vacuous) independent mode.
    """
    snrs, texp, _ = calculate_snr_from_reference_lambda(
        toymodel_two_filter_params, reference_wavelength=0.5
    )

    # The driving time is a single shared scalar Quantity (not a per-filter dict
    # -- that was the artefact of the removed independent mode).
    assert isinstance(texp, u.Quantity)
    assert np.isfinite(texp.to(u.s).value) and texp.to(u.s).value > 0

    # Reference band recovers the target.
    np.testing.assert_allclose(
        snrs["blue"]["snr"].value,
        7.0,
        rtol=1e-3,
        err_msg="Reference band (blue) should recover the target SNR of 7.0.",
    )

    # Non-reference band is evaluated at blue's time and must differ. If it also
    # recovered 7.0, coupling would be computing a tautology -- the very vacuity
    # we removed with independent mode. This assertion is that mode's epitaph.
    assert np.all(np.isfinite(snrs["red"]["snr"].value))
    assert not np.allclose(snrs["red"]["snr"].value, 7.0, rtol=1e-3), (
        "Red band recovered the target SNR -- coupling is behaving like the "
        "removed independent mode and computing nothing of value."
    )


@pytest.mark.integration
def test_workflow_etc2snr_single_filter_raises():
    """The engine's >=2-filter guard fires. A single-filter etc2snr is a
    degenerate round-trip (solve t to hit a target, then recover it), so the
    engine rejects it. Pins the ENGINE contract directly; Step 3's file-level
    validation will layer a SECOND check, not the only one.
    """
    params = parse_parameters(
        {
            "observing_mode": "IMAGER",
            "observatory_preset": "ToyModel",
            "wavelength": 0.5,
            "snr": 7.0,
            "bandwidth": 0.2,
            "CRb_multiplier": 2.0,
            "photometric_aperture_radius": 0.85,
            "distance": 10.0,
            "Fstar_10pc": 122.9279,
            "FstarV_10pc": 115.59984,
            "stellar_radius": 1.0,
            "nzodis": 3.0,
            "ra": 236.0,
            "dec": 2.5,
            "separation": 0.1,
            "Fp/Fs": 1e-10,
            "filter_list": [
                Filter("blue", center=0.5 * u.um, bandwidth=0.2, type="IMAGER"),
            ],
        }
    )

    with pytest.raises(ValueError, match="at least two filters"):
        calculate_snr_from_reference_lambda(params, reference_wavelength=0.5)


@pytest.mark.integration
def test_workflow_etc2snr_wavelength_outside_filters_raises(toymodel_two_filter_params):
    """A reference wavelength outside every filter's band is rejected. Guards
    the engine's `filt is None` branch -- distinct from the <2-filter guard."""
    with pytest.raises(ValueError, match="outside every filter"):
        calculate_snr_from_reference_lambda(
            toymodel_two_filter_params, reference_wavelength=99.0
        )
