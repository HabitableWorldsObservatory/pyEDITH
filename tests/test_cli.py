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


@pytest.mark.parametrize(
    "subcommand, error_message",
    [
        ("etc", "--edith argument is required for etc subfunction."),
        (
            "snr",
            "Both --edith and --time arguments are required for snr subfunction.",
        ),
        ("etc2snr", "--edith argument is required for etc2snr subfunction."),
    ],
)
def test_main_missing_edith_argument(subcommand, error_message):
    """Test that missing --edith argument raises appropriate error for each subcommand."""
    with patch("sys.argv", ["edith", subcommand]):
        with pytest.raises(SyntaxError, match=error_message):
            main()


def test_main_snr_missing_time_argument():
    """Test that snr subcommand requires both --edith and --time arguments."""
    with patch("sys.argv", ["edith", "snr", "--edith", "test.edith"]):
        with pytest.raises(
            SyntaxError,
            match="Both --edith and --time arguments are required for snr subfunction.",
        ):
            main()


# ============================================================================
# Tests for main() - etc subcommand
# ============================================================================


@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_etc_basic_execution(
    mock_calculate_texp, mock_read_configuration, mock_args
):
    """Test that etc subcommand correctly calculates and prints exposure time."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["edith", "etc", "--edith", "test.edith"]):
        with patch("builtins.print") as mock_print:
            main()
            mock_print.assert_called_with(np.array([1.0]))


# ============================================================================
# Tests for main() - snr subcommand
# ============================================================================


@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_snr")
def test_main_snr_basic_execution(
    mock_calculate_snr, mock_read_configuration, mock_args
):
    """Test that snr subcommand correctly calculates and prints SNR."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_snr.return_value = (np.array([10.0]), {})

    with patch("sys.argv", ["edith", "snr", "--edith", "test.edith", "--time", "60"]):
        with patch("builtins.print") as mock_print:
            main()
            mock_print.assert_called_with(np.array([10.0]))


# ============================================================================
# Tests for main() - etc2snr subcommand (single-function routing)
# ============================================================================


@patch("pyEDITH.cli.parse_input.read_configuration")
def test_main_etc2snr_no_ref_lambda_raises(mock_read_configuration):
    """etc2snr without --ref_lambda raises a descriptive ValueError (lines 140-145)."""
    mock_read_configuration.return_value = ({"some": "params"}, {})

    with patch("sys.argv", ["edith", "etc2snr", "--edith", "test.edith"]):
        with pytest.raises(ValueError, match="etc2snr requires --ref_lambda"):
            main()


@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_snr_from_reference_lambda")
def test_main_etc2snr_coupled_routing(mock_ref_snr, mock_read_configuration):
    """--coupled routes with coupled=True and a single parsed float."""
    mock_read_configuration.return_value = ({"some": "params"}, {})
    mock_ref_snr.return_value = ({"VIS": {"snr": np.array([7.0])}}, 30.0, {})

    with patch(
        "sys.argv",
        [
            "edith",
            "etc2snr",
            "--edith",
            "test.edith",
            "--ref_lambda",
            "1.5",
            "--coupled",
        ],
    ):
        with patch("builtins.print"):
            main()

    mock_ref_snr.assert_called_once()
    # Positional: (parameters, reference) ; keyword: coupled=True
    assert mock_ref_snr.call_args[0][1] == [1.5]  # parsed to float, in a list
    assert mock_ref_snr.call_args[1]["coupled"] is True


@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_snr_from_reference_lambda")
def test_main_etc2snr_independent_routing(mock_ref_snr, mock_read_configuration):
    """No --coupled routes with coupled=False and multiple parsed floats."""
    mock_read_configuration.return_value = ({"some": "params"}, {})
    mock_ref_snr.return_value = ({"VIS": {}, "NIR": {}}, {"VIS": 10.0, "NIR": 15.0}, {})

    with patch(
        "sys.argv",
        ["edith", "etc2snr", "--edith", "test.edith", "--ref_lambda", "0.5", "1.2"],
    ):
        with patch("builtins.print"):
            main()

    mock_ref_snr.assert_called_once()
    assert mock_ref_snr.call_args[0][1] == [0.5, 1.2]  # both parsed to float
    assert mock_ref_snr.call_args[1]["coupled"] is False


@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_snr_from_reference_lambda")
def test_main_etc2snr_parses_strings_to_floats(mock_ref_snr, mock_read_configuration):
    """Line 147: args.ref_lambda strings are converted to floats, not left as str."""
    mock_read_configuration.return_value = ({"some": "params"}, {})
    mock_ref_snr.return_value = ({}, {}, {})

    with patch(
        "sys.argv",
        ["edith", "etc2snr", "--edith", "test.edith", "--ref_lambda", "0.75"],
    ):
        with patch("builtins.print"):
            main()

    passed = mock_ref_snr.call_args[0][1]
    assert passed == [0.75]
    assert all(isinstance(x, float) for x in passed)  # genuinely floats, not "0.75"


# ============================================================================
# Tests for main() - Verbosity levels
# ============================================================================


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_default_verbosity_level(
    mock_calculate_texp, mock_read_configuration, mock_set_verbosity
):
    """Test that default verbosity (no flag) sets warning level."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("warning")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_verbose_flag_sets_info_level(
    mock_calculate_texp, mock_read_configuration, mock_set_verbosity
):
    """Test that -v flag sets info verbosity level."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "-v", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("info")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_verbose_long_flag_sets_info_level(
    mock_calculate_texp, mock_read_configuration, mock_set_verbosity
):
    """Test that --verbose flag sets info verbosity level."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "--verbose", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("info")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_double_verbose_flag_sets_debug_level(
    mock_calculate_texp, mock_read_configuration, mock_set_verbosity
):
    """Test that -vv flag sets debug verbosity level."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "-vv", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("debug")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_quiet_flag_sets_error_level(
    mock_calculate_texp, mock_read_configuration, mock_set_verbosity
):
    """Test that -q flag sets error verbosity level."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "-q", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("error")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_quiet_long_flag_sets_error_level(
    mock_calculate_texp, mock_read_configuration, mock_set_verbosity
):
    """Test that --quiet flag sets error verbosity level."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "--quiet", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("error")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_quiet_overrides_verbose(
    mock_calculate_texp, mock_read_configuration, mock_set_verbosity
):
    """Test that --quiet flag takes precedence over -vv flag."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

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
def test_reference_snr_independent_single(
    mock_texp, mock_snr, mock_parameters_reference
):
    """Mode 1: single wavelength drives its own filter (list input)."""
    fname = mock_parameters_reference["filter_list"][0].name
    mock_texp.return_value = (
        {
            fname: {
                "wavelength": np.array([1.0, 1.5, 2.0]) * u.um,
                "exposure_time": np.array([10.0, 20.0, 30.0]) * u.s,
            }
        },
        {},
    )
    mock_snr.return_value = ({fname: {"snr": np.array([7.0])}}, {})

    snrs, times, validation = calculate_snr_from_reference_lambda(
        mock_parameters_reference, [1.5]
    )

    assert times == {fname: 20.0 * u.s}
    assert mock_snr.call_args[0][1] == {fname: 20.0 * u.s}  # dict passed in independent
    assert fname in snrs


@patch("pyEDITH.cli.calculate_snr")
@patch("pyEDITH.cli.calculate_texp")
def test_reference_snr_coupled_scalar(mock_texp, mock_snr, mock_parameters_reference):
    """Mode 3: single wavelength, scalar time applied to ALL filters."""
    fname = mock_parameters_reference["filter_list"][0].name
    mock_texp.return_value = (
        {
            fname: {
                "wavelength": np.array([1.0, 1.5, 2.0]) * u.um,
                "exposure_time": np.array([10.0, 20.0, 30.0]) * u.s,
            }
        },
        {},
    )
    mock_snr.return_value = ({fname: {"snr": np.array([7.0])}}, {})

    snrs, texp, validation = calculate_snr_from_reference_lambda(
        mock_parameters_reference, 2.0, coupled=True
    )

    assert texp == 30.0 * u.s
    assert mock_snr.call_args[0][1] == 30.0 * u.s  # scalar passed in coupled
    assert fname in snrs


@patch("pyEDITH.cli.calculate_snr")
@patch("pyEDITH.cli.calculate_texp")
def test_reference_snr_out_of_range_raises(
    mock_texp, mock_snr, mock_parameters_reference
):
    """A wavelength outside every filter raises ValueError (not KeyError)."""
    fname = mock_parameters_reference["filter_list"][0].name
    mock_texp.return_value = (
        {
            fname: {
                "wavelength": np.array([1.0, 1.5, 2.0]) * u.um,
                "exposure_time": np.array([10.0, 20.0, 30.0]) * u.s,
            }
        },
        {},
    )
    # 99.0 um is outside any sane filter range in the fixture
    with pytest.raises(ValueError, match="outside every filter's range"):
        calculate_snr_from_reference_lambda(mock_parameters_reference, [99.0])

    mock_snr.assert_not_called()


@patch("pyEDITH.cli.calculate_snr")
@patch("pyEDITH.cli.calculate_texp")
def test_reference_snr_warns_too_few_wavelengths(
    mock_texp, mock_snr, mock_parameters_two_filters, caplog
):
    """Independent mode with fewer wavelengths than filters warns about coupled."""
    mock_texp.return_value = (
        {
            "VIS": {
                "wavelength": np.array([0.5, 0.7]) * u.um,
                "exposure_time": np.array([10.0, 11.0]) * u.s,
            },
            "NIR": {
                "wavelength": np.array([1.2, 1.5]) * u.um,
                "exposure_time": np.array([14.0, 15.0]) * u.s,
            },
        },
        {},
    )
    mock_snr.return_value = ({}, {})

    with caplog.at_level(logging.ERROR, logger="pyEDITH"):
        calculate_snr_from_reference_lambda(mock_parameters_two_filters, [0.5])

    assert "coupled=true" in caplog.text.lower()


@patch("pyEDITH.cli.calculate_snr")
@patch("pyEDITH.cli.calculate_texp")
def test_coupled_warns_on_extra_wavelengths(
    mock_texp, mock_snr, mock_parameters_reference, caplog
):
    """Coupled mode with >1 wavelength warns that extras are ignored."""
    fname = mock_parameters_reference["filter_list"][0].name
    mock_texp.return_value = (
        {
            fname: {
                "wavelength": np.array([1.0, 1.5, 2.0]) * u.um,
                "exposure_time": np.array([10.0, 20.0, 30.0]) * u.s,
            }
        },
        {},
    )
    mock_snr.return_value = ({fname: {"snr": np.array([7.0])}}, {})

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        # Two wavelengths in coupled mode → only the first (1.5) is used, extra warned
        snrs, texp, _ = calculate_snr_from_reference_lambda(
            mock_parameters_reference, [1.5, 1.8], coupled=True
        )

    # The warning fired, mentioning the ignored extras
    assert "only the first" in caplog.text.lower()
    assert "ignored" in caplog.text.lower()
    # And the time came from the FIRST wavelength (1.5 → 20.0), not the second
    assert texp == 20.0 * u.s


@patch("pyEDITH.cli.calculate_snr")
@patch("pyEDITH.cli.calculate_texp")
def test_coupled_out_of_range_raises(mock_texp, mock_snr, mock_parameters_reference):
    """Coupled mode with a wavelength outside every filter raises ValueError (line 375)."""
    fname = mock_parameters_reference["filter_list"][0].name
    mock_texp.return_value = (
        {
            fname: {
                "wavelength": np.array([1.0, 1.5, 2.0]) * u.um,
                "exposure_time": np.array([10.0, 20.0, 30.0]) * u.s,
            }
        },
        {},
    )
    # 99.0 um is outside the F150W filter (1.0–2.0 µm)
    with pytest.raises(ValueError, match="outside every filter's range"):
        calculate_snr_from_reference_lambda(
            mock_parameters_reference, 99.0, coupled=True
        )

    mock_snr.assert_not_called()


@patch("pyEDITH.cli.calculate_snr")
@patch("pyEDITH.cli.calculate_texp")
def test_single_filter_coupled_matches_independent(
    mock_texp, mock_snr, mock_parameters_reference
):
    """For one filter, scalar-coupled and single-element-list-independent agree."""
    fname = mock_parameters_reference["filter_list"][0].name
    texp_fixture = (
        {
            fname: {
                "wavelength": np.array([1.0, 1.5, 2.0]) * u.um,
                "exposure_time": np.array([10.0, 20.0, 30.0]) * u.s,
            }
        },
        {},
    )
    mock_texp.return_value = texp_fixture
    mock_snr.return_value = ({fname: {"snr": np.array([7.0])}}, {})

    _, t_coupled, _ = calculate_snr_from_reference_lambda(
        mock_parameters_reference, 1.5, coupled=True
    )
    _, t_independent, _ = calculate_snr_from_reference_lambda(
        mock_parameters_reference, [1.5]
    )

    # Same driving time—coupled returns a scalar, independent a dict of one
    assert t_coupled == 20.0 * u.s
    assert t_independent == {fname: 20.0 * u.s}
    assert t_coupled == t_independent[fname]
