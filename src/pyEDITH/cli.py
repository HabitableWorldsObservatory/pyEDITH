import astropy.units as u

from pyEDITH import AstrophysicalScene, Observation, Observatory
from pyEDITH import calculate_exposure_time_or_snr, parse_input
from argparse import ArgumentParser
import numpy as np
from typing import Union
from .units import WAVELENGTH, TIME
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

    Raises
    ------
    SyntaxError
        If required command-line arguments are missing
    ValueError
        If no reference wavelength option is specified in etc2snr
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
        type=float,
        help="Reference wavelength in microns. The exposure time is solved on "
        "the filter containing this wavelength and applied to all filters.",
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

    # Guard: no subcommand chosen -> show help and return.
    if args.subfunction is None:
        parser.print_help()
        return

    if not args.edith:
        raise SyntaxError("--edith argument is required.")

    parameters = parse_input.parse_input_file(args.edith)

    if args.subfunction == "etc":

        texp, _ = calculate_texp(parameters)
        print(texp)

    elif args.subfunction == "snr":
        if args.time is None:
            raise SyntaxError("--time argument is required for snr subfunction.")

        texp = args.time * TIME
        snr, _ = calculate_snr(parameters, texp)
        print(snr)

    else:  # args.subfunction == "etc2snr" — the only remaining possibility

        if args.ref_lambda is None:
            raise ValueError(
                "etc2snr requires --ref_lambda to specify the reference "
                "wavelength(s)."
            )

        reference = float(args.ref_lambda)

        snrs, texp, _ = calculate_snr_from_reference_lambda(parameters, reference)

        print(f"Driving exposure time: ", texp)
        print(f"Derived SNRs at prescribed exposure time: ", snrs)


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
    parameters = parse_input.parse_parameters(parameters)
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
        Reference exposure time.

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
    parameters = parse_input.parse_parameters(parameters)
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


def calculate_snr_from_reference_lambda(parameters, reference_wavelength):
    """
    Calculate SNR by first deriving a single exposure time at a reference wavelength.

    Runs the Exposure Time Calculator across all filters, extracts the
    exposure time at the reference wavelength (from whichever filter contains
    it), and then applies that single exposure time to *all* filters to compute
    their achievable signal-to-noise ratios.

    This is the coupled multi-band observation: every filter shares one
    integration time, determined by the reference band (e.g. a dichroic or
    parallel-channel split). It therefore requires **at least two filters** —
    a single-filter case is a degenerate round-trip (solve t to hit a target,
    then recover that same target) and is rejected.

    Parameters
    ----------
    parameters : dict
        Dictionary containing all input parameters for the calculation. Must
        include a ``"filter_list"`` entry of filter objects exposing a
        ``contains(wavelength)`` method and a ``name`` attribute. At least two
        filters are required.
    reference_wavelength : float or astropy.units.Quantity
        The single reference wavelength. A bare number is interpreted in the
        default ``WAVELENGTH`` unit; a Quantity is converted to ``WAVELENGTH``.

    Returns
    -------
    snr_results : dict
        Dictionary keyed by filter name, each value containing ``"wavelength"``
        and ``"snr"`` arrays (see :func:`calculate_snr`).
    texp_scalar : astropy.units.Quantity
        Exposure time at reference lambda
    validation : dict
        Dictionary keyed by filter name, containing validation variables.

    Raises
    ------
    ValueError
        If fewer than two filters are present, or if the reference wavelength
        falls outside every filter's range.

    See Also
    --------
    calculate_texp : The underlying Exposure Time Calculator.
    calculate_snr : Computes SNR given a single exposure time.
    """
    parameters = parse_input.parse_parameters(parameters)
    filters = parse_input.parse_filters(parameters)

    if len(filters) < 2:
        raise ValueError(
            f"etc2snr requires at least two filters (found {len(filters)}). "
            "A single-filter reference calculation is degenerate — it simply "
            "recovers the input SNR. Use the 'etc' or 'snr' workflow for a "
            "single filter instead."
        )

    # Normalise the reference wavelength to a WAVELENGTH quantity.
    ref = (
        reference_wavelength.to(WAVELENGTH)
        if isinstance(reference_wavelength, u.Quantity)
        else reference_wavelength * WAVELENGTH
    )

    filt = next((f for f in filters if f.contains(ref)), None)
    if filt is None:
        raise ValueError(f"Wavelength {ref} is outside every filter's range.")

    parameters_exposuretime = dict(parameters)
    parameters_exposuretime["filter_list"] = [filt]
    texp_results, _ = calculate_texp(parameters_exposuretime)

    texp_scalar, actual = _get_closest_finite_time(
        texp_results[filt.name]["wavelength"],
        texp_results[filt.name]["exposure_time"],
        ref,
    )
    snr_results, validation = calculate_snr(parameters, texp_scalar)
    return snr_results, texp_scalar, validation
