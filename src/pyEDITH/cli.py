from pyEDITH import AstrophysicalScene, Observation, Observatory
from pyEDITH import calculate_exposure_time_or_snr, parse_input
from argparse import ArgumentParser
import numpy as np
from .units import *
import sys
from . import set_verbosity
import os
import logging

logger = logging.getLogger("pyEDITH")

# Declare the environment variables
# os.environ["SCI_ENG_DIR"] = "/path/to/Sci-Eng-Interface/hwo_sci_eng"
# EAC_SUMMARY_DIR points to a directory of <eac>_summary.yaml files (the default
# EAC4-6 source; self-contained, does not require hwome). If unset, EAC4-6 fall
# back to the hwome model.
# os.environ["EAC_SUMMARY_DIR"] = "/path/to/eac_summaries"


def main():
    """
    Main entry point for the py-E.D.I.T.H. command-line interface.

    This function sets up the argument parser and handles the execution
    of different subcommands: etc (Exposure Time Calculator), snr (Signal-to-Noise
    Ratio), and etc2snr (Exposure Time to Signal-to-Noise Ratio).

    The function reads configuration from .edith files and calls the appropriate
    calculation functions based on the subcommand.

    Raises
    ------
    SyntaxError
        If required command-line arguments are missing
    TypeError
        If multiple wavelengths are provided for primary lambda in etc2snr mode
    ValueError
        If secondary parameters are not specified in etc2snr mode or
        if the returned exposure time is infinity
    """
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity: -v for INFO, -vv for DEBUG logs with detailed output",
    )
    parent_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Quiet mode, only show errors"
    )

    parser = ArgumentParser(
        description="Available command line arguments for E.D.I.T.H.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity: -v for INFO, -vv for DEBUG logs with detailed output",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Quiet mode, only show errors"
    )

    subparsers = parser.add_subparsers(
        dest="subfunction", help="Subfunction to execute"
    )

    parser_a = subparsers.add_parser(
        "etc", help="Exposure Time Calculator for a specific lambda (in .edith file)"
    )
    parser_a.add_argument("--edith", type=str, help="an .edith file")

    parser_b = subparsers.add_parser(
        "snr",
        help="SNR calculator for a specific lambda (in .edith file) and for a given \
            exposure time (argument, in seconds)",
    )
    parser_b.add_argument("--edith", type=str, help="an .edith file")
    parser_b.add_argument(
        "--time", type=float, help="the desired observing time in seconds"
    )

    parser_c = subparsers.add_parser(
        "etc2snr",
        help="SNR calculator for a secondary lambda based on an exposure file \
            calculated on a primary lambda (in .edith file)",
    )
    parser_c.add_argument("--edith", type=str, help="an .edith file")

    args = parser.parse_args()

    if args.quiet:
        set_verbosity("error")
    elif args.verbose == 0:
        set_verbosity("warning")  # Default
    elif args.verbose == 1:
        set_verbosity("info")  # -v
    else:  # args.verbose >= 2
        set_verbosity("debug")  # -vv, -vvv, etc.

    if args.subfunction == "etc":
        if not args.edith:
            raise SyntaxError("--edith argument is required for etc subfunction.")

        parameters, _ = parse_input.read_configuration(args.edith)
        texp, _ = calculate_texp(parameters)
        print(texp)

    elif args.subfunction == "snr":
        if not args.edith or args.time is None:
            raise SyntaxError(
                "Both --edith and --time arguments are required for snr subfunction."
            )

        parameters, _ = parse_input.read_configuration(args.edith)
        texp = args.time * TIME
        snr, _ = calculate_snr(parameters, texp)
        print(snr)

    elif args.subfunction == "etc2snr":
        if not args.edith:
            raise SyntaxError("--edith argument is required for etc2snr subfunction.")

        parameters, secondary_parameters = parse_input.read_configuration(
            args.edith, secondary_flag=True
        )

        if not secondary_parameters:
            raise ValueError("The secondary parameters are not specified.")

        if len(parameters["wavelength"]) > 1:
            raise TypeError("Cannot accept multiple lambdas as primary lambda")
        else:
            for key in parameters:
                if key not in secondary_parameters:
                    secondary_parameters[key] = parameters[key]

        logger.info("Calculating texp from primary lambda")
        logger.info(parameters.keys())
        texp, _ = calculate_texp(parameters)
        print("Reference exposure time: ", texp)
        if np.isfinite(texp).all():
            logger.info("Calculating snr on secondary lambda")
            snr, _ = calculate_snr(secondary_parameters, texp)
            print("SNR at the secondary lambda: ", snr)
        else:
            raise ValueError("Returned exposure time is infinity.")
    else:
        parser.print_help()


def calculate_texp(parameters: dict, ETC_validation: bool = False) -> tuple[dict, dict]:
    """
    Calculate exposure time for a planet observation with specified parameters.

    This function initializes Observation, AstrophysicalScene, and Observatory
    objects with the provided parameters, then calculates the required exposure
    time to achieve the specified signal-to-noise ratio at each wavelength.

    Parameters
    ----------
    parameters : dict
        Dictionary containing all input parameters for the calculation
    ETC_validation : bool, optional
        If True, use specific parameter values for validation against the ETC,
        default is False

    Returns
    -------
    tuple[dict, dict]
        A tuple containing:

        results : dict
            Dictionary with filter names as keys, each containing:
            - 'wavelength': wavelength array
            - 'exposure_time': exposure time array for each wavelength

        validation_variables : dict
            Dictionary with filter names as keys, containing validation variables
            with intermediate calculation results

    """

    # Read Filter list
    filters = parse_input.parse_filters(parameters)

    results = {}
    validation_variables = {}
    # Loop by filter
    for f in filters:
        logger.info("Running pyEDITH for Filter: " + f.name)
        # Define Observation and load relevant parameters
        observation = Observation()
        observation.load_configuration(parameters, filter=f)
        observation.set_output_arrays()
        observation.validate_configuration()

        # Define Astrophysical Scene and load relevant parameters,
        # then calculate zodi/exozodi
        scene = AstrophysicalScene()
        scene.load_configuration(parameters)
        scene.calculate_zodi_exozodi(parameters)
        scene.validate_configuration()
        if parameters["observing_mode"] == "IFS":
            scene.regrid_spectra(observation)

        # Create and configure Observatory
        observatory_config = parse_input.get_observatory_config(parameters)
        observatory = Observatory()
        observatory.create_observatory(observatory_config)
        observatory.load_configuration(parameters, observation, scene)
        observatory.validate_configuration()

        # EXPOSURE TIME CALCULATION
        calculate_exposure_time_or_snr(
            observation,
            scene,
            observatory,
            ETC_validation=ETC_validation,
            mode="exposure_time",
        )

        results[f.name] = {
            "wavelength": observation.wavelength,
            "exposure_time": observation.exptime,
        }
        validation_variables[f.name] = observation.validation_variables

    return results, validation_variables


def calculate_snr(parameters: dict, reference_texp: float) -> tuple[dict, dict]:
    """
    Calculate signal-to-noise ratio for a given exposure time.

    This function initializes Observation, AstrophysicalScene, and Observatory
    objects with the provided parameters, then calculates the achievable
    signal-to-noise ratio for the specified exposure time at each wavelength.

    Parameters
    ----------
    parameters : dict
        Dictionary containing all input parameters for the calculation
    reference_texp : float
        Reference exposure time in seconds

    Returns
    -------
    tuple[dict, dict]
        A tuple containing:

        results : dict
            Dictionary with filter names as keys, each containing:
            - 'wavelength': wavelength array
            - 'snr': signal-to-noise ratio array for each wavelength

        validation_variables : dict
            Dictionary with filter names as keys, containing validation variables
            with intermediate calculation results

    """

    # Read Filter list
    filters = parse_input.parse_filters(parameters)

    results = {}
    validation_variables = {}
    # Loop by filter
    for f in filters:
        logger.info("Running pyEDITH for Filter: " + f.name)

        # Define Observation and load relevant parameters
        observation = Observation()
        observation.load_configuration(parameters, filter=f)
        observation.set_output_arrays()

        # Define Astrophysical Scene and load relevant parameters,
        # then calculate zodi/exozodi
        scene = AstrophysicalScene()
        scene.load_configuration(parameters)
        scene.calculate_zodi_exozodi(parameters)

        if parameters["observing_mode"] == "IFS":
            scene.regrid_spectra(observation)

        # Create and configure Observatory
        observatory_config = parse_input.get_observatory_config(parameters)
        observatory = Observatory()
        observatory.create_observatory(observatory_config)
        observatory.load_configuration(parameters, observation, scene)
        observatory.validate_configuration()

        # SNR CALCULATION
        observation.obstime = reference_texp
        calculate_exposure_time_or_snr(
            observation, scene, observatory, mode="signal_to_noise"
        )
        results[f.name] = {
            "wavelength": observation.wavelength,
            "snr": observation.fullsnr,
        }
        validation_variables[f.name] = observation.validation_variables

    return results, validation_variables
