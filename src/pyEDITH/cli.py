from astropy.units import Quantity

from pyEDITH import AstrophysicalScene, Observation, Observatory
from pyEDITH import calculate_exposure_time_or_snr, parse_input
from argparse import ArgumentParser
import numpy as np
from typing import Union
from .units import *
import sys
from . import set_verbosity
import os, copy
import logging

logger = logging.getLogger("pyEDITH")


def main():
    """
    Main entry point for the py-E.D.I.T.H. command-line interface.

    This function sets up the argument parser and handles the execution
    of different subcommands: etc (Exposure Time Calculator), snr (Signal-to-Noise
    Ratio), and etc2snr (Exposure Time to Signal-to-Noise Ratio).

    The etc2snr subcommand supports two modes:
    - Coupled (--ref_lambda): A single wavelength drives one exposure time
      applied to all filters.
    - Independent (--ref_lambdas): Each filter is driven by its own reference
      wavelength and evaluated independently.

    Raises
    ------
    SyntaxError
        If required command-line arguments are missing
    ValueError
        If no reference wavelength option is specified in etc2snr mode
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
        help="SNR calculator derived from an ETC calculation at a reference wavelength",
    )
    parser_c.add_argument("--edith", type=str, help="an .edith file")
    parser_c.add_argument(
        "--ref_lambda",
        nargs="+",  # one OR more values → always a list
        type=float,  # argparse converts each token to float for you
        help="Reference wavelength(s) in microns. Provide one value with "
        "--coupled to drive all filters from a single wavelength, or "
        "several values for independent per-filter targets.",
    )
    parser_c.add_argument(
        "--coupled",
        action="store_true",  # boolean flag; absent → False, present → True
        help="Use a single reference wavelength to drive the exposure time "
        "for ALL filters simultaneously (e.g. dichroic splits).",
    )

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

        parameters, _ = parse_input.read_configuration(args.edith)

        if not args.ref_lambda:
            raise ValueError(
                "etc2snr requires --ref_lambda to specify the reference "
                "wavelength(s). Use one value with --coupled to drive all "
                "filters from a single wavelength, or several values for "
                "independent per-filter targets."
            )

        reference = [float(x) for x in args.ref_lambda]

        mode = "COUPLED" if args.coupled else "INDEPENDENT"
        logger.info(f"Executing {mode} etc2snr workflow")

        snrs, texps, _ = calculate_snr_from_reference_lambda(
            parameters, reference, coupled=args.coupled
        )

        print(f"{mode} exposure time(s): ", texps)
        print(f"{mode} SNRs: ", snrs)
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
        results : dict
            Dictionary with filter names as keys, each containing:
            - 'wavelength': wavelength array
            - 'exposure_time': exposure time array for each wavelength
        validation_variables : dict
            Dictionary with filter names as keys, containing validation variables
    """
    filters = parse_input.parse_filters(parameters)

    results = {}
    validation_variables = {}
    for f in filters:
        logger.info("Running pyEDITH for Filter: " + f.name)
        observation = Observation()
        observation.load_configuration(parameters, filter=f)
        observation.set_output_arrays()
        observation.validate_configuration()

        scene = AstrophysicalScene()
        scene.load_configuration(parameters)
        scene.calculate_zodi_exozodi(parameters)
        scene.validate_configuration()
        scene.regrid_spectra(observation)

        observatory_config = parse_input.get_observatory_config(parameters)
        observatory = Observatory()
        observatory.create_observatory(observatory_config)
        observatory.load_configuration(parameters, observation, scene)
        observatory.validate_configuration()

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


def calculate_snr(
    parameters: dict, reference_texp: Union[float, dict]
) -> tuple[dict, dict]:
    """
    Calculate signal-to-noise ratio for a given exposure time.

    This function initializes Observation, AstrophysicalScene, and Observatory
    objects with the provided parameters, then calculates the achievable
    signal-to-noise ratio for the specified exposure time at each wavelength.

    Parameters
    ----------
    parameters : dict
        Dictionary containing all input parameters for the calculation
    reference_texp : float or dict
        Reference exposure time. If a scalar, the same time is applied to all
        filters (coupled mode). If a dict mapping filter names to times, each
        filter uses its own time (independent mode).

    Returns
    -------
    tuple[dict, dict]
        results : dict
            Dictionary with filter names as keys, each containing:
            - 'wavelength': wavelength array
            - 'snr': signal-to-noise ratio array for each wavelength
        validation_variables : dict
            Dictionary with filter names as keys, containing validation variables
    """
    filters = parse_input.parse_filters(parameters)

    results = {}
    validation_variables = {}

    for f in filters:
        logger.info("Running pyEDITH for Filter: " + f.name)
        observation = Observation()
        observation.load_configuration(parameters, filter=f)
        observation.set_output_arrays()

        scene = AstrophysicalScene()
        scene.load_configuration(parameters)
        scene.calculate_zodi_exozodi(parameters)

        scene.regrid_spectra(observation)

        observatory_config = parse_input.get_observatory_config(parameters)
        observatory = Observatory()
        observatory.create_observatory(observatory_config)
        observatory.load_configuration(parameters, observation, scene)
        observatory.validate_configuration()

        # Handle independent times (dict) vs coupled time (scalar)
        if isinstance(reference_texp, dict):
            if f.name not in reference_texp:
                raise KeyError(
                    f"Filter '{f.name}' not found in the provided exposure time dict."
                )
            observation.obstime = reference_texp[f.name]
        else:
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


def _get_closest_finite_time(wvl_array, exptime_array, ref_wav):
    """
    Internal helper to find the valid exposure time at the wavelength
    closest to ref_wav.

    Searches the provided wavelength grid for the bin nearest to ref_wav and
    returns the corresponding exposure time. Raises an error if the closest
    point is non-finite (e.g. zero transmission at the band edge).

    Parameters
    ----------
    wvl_array : array-like
        Wavelength grid for the filter.
    exptime_array : array-like
        Exposure times corresponding to each wavelength bin.
    ref_wav : float or astropy.units.Quantity
        The reference wavelength to match against.

    Returns
    -------
    t_val : float or astropy.units.Quantity
        The exposure time at the closest wavelength bin.
    actual_wvl : float or astropy.units.Quantity
        The actual wavelength of the matched bin.

    Raises
    ------
    ValueError
        If the exposure time at the closest wavelength is infinite or NaN.
    """
    wvl_array = np.atleast_1d(wvl_array)
    exptime_array = np.atleast_1d(exptime_array)

    idx = np.argmin(np.abs(wvl_array - ref_wav))
    t_val = exptime_array[idx]

    if not np.isfinite(t_val):
        raise ValueError(
            f"Exposure time at the wavelength closest to {ref_wav} is infinite. "
            "This usually indicates zero transmission at this wavelength."
        )
    return t_val, wvl_array[idx]


def calculate_snr_from_reference_lambda(parameters, reference, coupled=False):
    if not isinstance(reference, (list, tuple)):
        reference = [reference]

    # Ensure each element is a wavelength quantity
    converted_reference = []
    for ref in reference:
        converted_reference.append(
            ref.to(WAVELENGTH) if isinstance(ref, u.Quantity) else ref * WAVELENGTH
        )

    reference = converted_reference

    filters = parameters["filter_list"]

    texp_results, _ = calculate_texp(parameters)

    if coupled:
        ref = reference[0]
        if len(reference) > 1:
            logger.warning(
                f"Coupled mode uses only the first wavelength ({ref}); "
                f"the remaining {len(reference) - 1} were ignored."
            )
        filt = next((f for f in filters if f.contains(ref)), None)
        if filt is None:
            raise ValueError(f"Wavelength {ref} is outside every filter's range.")

        texp_scalar, actual = _get_closest_finite_time(
            texp_results[filt.name]["wavelength"],
            texp_results[filt.name]["exposure_time"],
            ref,
        )
        logger.info(f"Coupled time driven by lambda={actual}: {texp_scalar}")

        snr_results, validation = calculate_snr(parameters, texp_scalar)

        return snr_results, texp_scalar, validation

    # --- Independent mode ---
    # Guard against the classic "meant to use coupled" mistake:
    # fewer wavelengths than filters means some filters have no time.
    if len(reference) < len(filters):
        logger.error(
            f"Independent mode: you provided {len(reference)} wavelength(s) "
            f"but the configuration has {len(filters)} filter(s). Each filter "
            f"needs its own target wavelength in this mode. "
            f"If you intended a single wavelength to drive the exposure time "
            f"for ALL filters, pass coupled=True instead."
        )

    times = {}
    for l in reference:
        filt = next((f for f in filters if f.contains(l)), None)
        if filt is None:
            raise ValueError(f"Wavelength {l} is outside every filter's range.")

        t_val, actual = _get_closest_finite_time(
            texp_results[filt.name]["wavelength"],
            texp_results[filt.name]["exposure_time"],
            l,
        )
        times[filt.name] = t_val
        logger.info(f"Filter '{filt.name}' time driven by lambda={actual}: {t_val}")

    snr_results, validation = calculate_snr(parameters, times)
    return snr_results, times, validation
