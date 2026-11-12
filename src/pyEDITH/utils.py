from scipy.interpolate import interp1d
import numpy as np
import astropy.units as u
from typing import Dict, Any
import logging
from .units import *

logger = logging.getLogger("pyEDITH")


def average_over_bandpass(params: dict, wavelength_range: list) -> dict:
    """
    Calculate the average of array parameters within a specified wavelength range.

    This function takes a dictionary of parameters and computes the mean value
    of all numpy array parameters (except wavelength) within the specified
    wavelength boundaries. The wavelength array is expected to be stored under
    the key "lam" in the params dictionary.

    Out-of-domain behaviour
    -----------------------
    Some curves (e.g. ``qe_vis`` / ``qe_nir``) are only physically defined over
    part of the wavelength axis. When a requested ``wavelength_range`` does not
    straddle any tabulated point for a given curve, that curve is interpolated
    at the *center* of ``wavelength_range``. If that center lies outside the
    curve's native domain the interpolation deliberately returns ``NaN`` (via
    ``bounds_error=False, fill_value=np.nan``) rather than raising, so the NaN
    can be used downstream to mark the region where a curve does not apply
    (this is how the vis/nir split is reconstructed).

    Parameters
    ----------
    params : dict
        Dictionary containing parameters where numpy arrays represent wavelength-dependent
        quantities. Must include a "lam" key containing the wavelength array.
        These parameters come from EACy and follow that formatting.
    wavelength_range : list
        Two-element list containing the lower and upper wavelength boundaries
        for averaging, expected to have astropy units

    Returns
    -------
    dict
        Modified parameters dictionary with array values replaced by their mean
        values within the specified wavelength range
    """
    # take the average within the specified wavelength range
    numpy_array_variables = {
        key: value for key, value in params.items() if isinstance(value, np.ndarray)
    }

    lam_values = params["lam"].value
    mask = (lam_values >= wavelength_range[0].value) & (
        lam_values <= wavelength_range[1].value
    )
    # Center of the requested bandpass, used only for the empty-slice fallback.
    center_wavelength = 0.5 * (wavelength_range[0].value + wavelength_range[1].value)

    for key, value in numpy_array_variables.items():
        if key != "lam":
            if mask.any():
                params[key] = np.mean(params[key][mask])
            else:
                # No tabulated sample points fall inside the requested
                # bandpass; interpolate the curve at the band center instead.
                # Points outside the curve's native domain return NaN (rather
                # than raising) so the NaN can flag where the curve does not
                # apply -- this is what lets the vis/nir curves be stitched
                # back together element-wise downstream.
                interp_func = interp1d(
                    params["lam"],
                    params[key],
                    bounds_error=False,
                    fill_value=np.nan,
                )
                params[key] = interp_func(center_wavelength)
    return params


def interpolate_over_bandpass(params: dict, wavelengths: list) -> dict:
    """
    Interpolate array parameters onto a new wavelength grid.

    This function takes a dictionary of parameters and interpolates all numpy array
    parameters (except the wavelength array itself) onto a new set of wavelength
    points using 1D linear interpolation. The original wavelength array is expected
    to be stored under the key "lam" in the params dictionary.

    Out-of-domain behaviour
    -----------------------
    Interpolation points that fall outside a curve's native wavelength domain
    return ``NaN`` (via ``bounds_error=False, fill_value=np.nan``) rather than
    raising. This is intentional: curves such as ``qe_vis`` / ``qe_nir`` are
    only defined over part of the spectrum, and the resulting NaNs mark the
    region where each curve does not apply so the vis/nir arrays can be stitched
    together element-wise (e.g. ``qe_vis = [finite, finite, nan]`` and
    ``qe_nir = [nan, nan, finite]``).

    Parameters
    ----------
    params : dict
        Dictionary containing parameters where numpy arrays represent wavelength-dependent
        quantities. Must include a "lam" key containing the original wavelength array.
        These parameters come from EACy and follow that formatting.

    wavelengths : list
        New wavelength points onto which to interpolate the parameter arrays

    Returns
    -------
    dict
        Modified parameters dictionary with array values interpolated onto the new
        wavelength grid specified by the wavelengths parameter
    """

    # take the average within the specified wavelength range
    numpy_array_variables = {
        key: value for key, value in params.items() if isinstance(value, np.ndarray)
    }
    for key, value in numpy_array_variables.items():
        if key != "lam":
            interp_func = interp1d(
                params["lam"],
                params[key],
                bounds_error=False,
                fill_value=np.nan,
            )
            ynew = interp_func(
                wavelengths
            )  # interpolates the CG throughput values onto native wl grid
            params[key] = ynew
    return params


def fill_parameters(
    class_obj: object,
    parameters: dict,
    default_parameters: dict,
    locked_keys: set = None,
    allow_override: set = None,
    rebinned_wavelength: list = None,
) -> None:
    """
    Populate class object attributes with user parameters or default values.

    Parameters
    ----------
    class_obj : object
        Class instance whose attributes will be set.
    parameters : dict
        Dictionary of user-provided parameter values.
    default_parameters : dict
        Dictionary of default (or, for YIP/EAC mode, model-loaded) parameter values.
    locked_keys : set, optional
        Keys that must NOT be overridden by the user by default. For these keys
        the value staged in ``default_parameters`` is always used (e.g. values
        loaded from a YIP or EAC YAML), regardless of what the user supplied,
        UNLESS the key is also present in ``allow_override``.
    allow_override : set, optional
        Keys that are normally locked but that the user has explicitly and
        intentionally requested to override (e.g. via parameters["overrides"]).
        Has no effect on keys that are not in ``locked_keys`` -- those are
        already user-editable. Any name in ``allow_override`` that does not
        correspond to a key in ``default_parameters`` raises a ValueError,
        to catch typos rather than silently ignoring them.
    """

    if locked_keys is None:
        locked_keys = set()
    if allow_override is None:
        allow_override = set()

    unknown_overrides = allow_override - set(default_parameters.keys())
    if unknown_overrides:
        raise ValueError(
            f"'overrides' contains unrecognized parameter name(s): "
            f"{unknown_overrides}"
        )

    def _coerce(user_value, default_value):
        """Match user_value's type/units to default_value's, where applicable."""
        if isinstance(default_value, u.Quantity):
            if isinstance(user_value, u.Quantity):
                return user_value.to(default_value.unit)
            return u.Quantity(user_value, default_value.unit)
        return user_value

    # Step 0: ensure that all the overrides parameters have the correct length
    for key in allow_override:
        if (
            key in parameters
        ):  # the user actually provided the value, otherwise nothing will happen
            param_value = parameters[key]

            # Extract value and unit if it's a Quantity
            if isinstance(param_value, u.Quantity):
                param_unit = param_value.unit
                param_array = param_value.value
            else:
                param_unit = None
                param_array = param_value

            if isinstance(param_array, (np.ndarray, list)) and len(param_array) > 1:
                if len(param_array) == len(
                    parameters["wavelength"]
                ):  # Rebin only parameters that are supposed to be nlambda
                    logger.info(
                        f"Before overriding: rebinning {key} array to regridded lambda..."
                    )
                    rebinned_values = resample_to_wavelength_grid(
                        param_array,
                        from_wavelength=parameters["wavelength"],
                        to_wavelength=rebinned_wavelength,
                        name=key,
                        interpolation="1d",
                    )
                    # Reattach unit if original was a Quantity
                    if param_unit is not None:
                        parameters[key] = rebinned_values * param_unit
                    else:
                        parameters[key] = rebinned_values
                else:
                    raise ValueError(
                        f"Before overriding: {key} does not have length equal to input wavelength. Please check your inputs."
                    )
            else:
                logger.info(
                    f"Before overriding: {key} is a scalar and won't be rebinned."
                )

    # Step 1: replace overrides when possible
    for key, default_value in default_parameters.items():
        is_locked = key in locked_keys
        is_overridden = key in allow_override
        if key in parameters and is_locked and not is_overridden:
            # User tried to set a value that is owned by the model (e.g. YIP/EAC)
            # and did not explicitly request an override.
            logger.warning(
                f"Parameter '{key}' is locked in this mode and "
                f"cannot be user-overridden; using the model-provided value "
                f"instead of the supplied value {parameters[key]!r}."
            )
            setattr(class_obj, key, default_value)

        elif key in parameters and is_locked and is_overridden:
            # User explicitly requested to override a normally-locked value.
            final_value = _coerce(parameters[key], default_value)
            setattr(class_obj, key, final_value)
            logger.warning(
                f"Parameter '{key}' is normally locked in this mode, but was "
                f"explicitly overridden per user request (via 'overrides'). "
                f"Model-provided value was {default_value!r}; using "
                f"user-supplied value: {final_value!r}."
            )

        elif key in parameters and not is_locked:
            # User provided a value and it is allowed to be overridden.
            final_value = _coerce(parameters[key], default_value)
            setattr(class_obj, key, final_value)
            logger.debug(f"Parameter '{key}' set to user-provided value: {final_value}")

        else:
            # Use default / model-provided value (also the path for locked
            # keys the user didn't touch at all).
            setattr(class_obj, key, default_value)
            logger.debug(f"Parameter '{key}' set to default value: {default_value}")


def convert_to_numpy_array(class_obj: object, array_params: list) -> None:
    """
    Convert specified class attributes to numpy arrays with proper dtype.

    This function converts class attributes to numpy arrays with float64 dtype,
    while preserving astropy units for Quantity objects. Non-Quantity attributes
    are converted to plain numpy arrays, while Quantity attributes maintain their
    units but have their values converted to numpy arrays.

    Parameters
    ----------
    class_obj : object
        Class instance whose attributes will be converted
    array_params : list
        List of attribute names to convert to numpy arrays
    """

    for param in array_params:
        attr_value = getattr(class_obj, param)
        if isinstance(attr_value, u.Quantity):
            # If it's already a Quantity, convert to numpy array while preserving units
            setattr(
                class_obj,
                param,
                u.Quantity(
                    np.array(attr_value.value, dtype=np.float64), attr_value.unit
                ),
            )
        else:
            # If it's not a Quantity, convert to numpy array without units
            setattr(class_obj, param, np.array(attr_value, dtype=np.float64))


def validate_attributes(obj: Any, expected_args: Dict[str, Any]) -> None:
    """
    Validate attributes of an object against expected types and units.

    This function checks that an object has all the required attributes and that
    each attribute has the correct type or units. It supports validation of
    integer and float types as well as astropy Quantity objects with specific units.

    Parameters
    ----------
    obj : object
        The object whose attributes are to be validated
    expected_args : dict
        A dictionary where keys are attribute names and values are expected types or units

    Raises
    ------
    AttributeError
        If a required attribute is missing
    TypeError
        If an attribute has an incorrect type
    ValueError
        If a Quantity attribute has incorrect units or if there's an unexpected type specification
    """

    class_name = obj.__class__.__name__

    for arg, expected_type in expected_args.items():
        if not hasattr(obj, arg):
            raise AttributeError(f"{class_name} is missing attribute: {arg}")

        value = getattr(obj, arg)

        if expected_type is int:
            if not isinstance(value, (int, np.integer)):
                raise TypeError(f"{class_name} attribute {arg} should be an integer")
        elif expected_type is float:
            if not isinstance(value, (float, np.floating)):
                raise TypeError(f"{class_name} attribute {arg} should be a float")
        elif isinstance(
            expected_type, (u.UnitBase, u.CompositeUnit, u.IrreducibleUnit)
        ):
            if not isinstance(value, u.Quantity):
                raise TypeError(f"{class_name} attribute {arg} should be a Quantity")
            if value.unit != expected_type:
                raise ValueError(
                    f"{class_name} attribute {arg} has incorrect units. "
                    f"Expected {expected_type}, got {value.unit}"
                )
        else:
            raise ValueError(f"Unexpected type specification for {arg}")


def print_array_info(
    file: object, name: str, arr: np.ndarray, mode: str = "full_info"
) -> None:
    """
    Write detailed information about an array or variable to a file.

    This function writes comprehensive information about a given array or variable
    to a specified file, including its shape, data type, units (if applicable),
    and statistical properties such as minimum and maximum values. The output format
    depends on the specified mode.

    Parameters
    ----------
    file : object
        Open file object to write the information to
    name : str
        Name or identifier of the variable/array being described
    arr : np.ndarray
        The array or variable to analyze and describe. Can be a numpy array,
        an array-like object, or a scalar with or without astropy units
    mode : str, optional
        Output mode that determines the level of detail in the description.
        Default is "full_info" which provides comprehensive information.
        Other modes provide more concise output
    """
    if arr is None:
        return

    if mode == "full_info":
        file.write(f"{name}:\n")

        # Handle units
        if hasattr(arr, "unit"):
            if arr.unit == DIMENSIONLESS:
                file.write(" Unit: dimensionless\n")
            else:
                file.write(f" Unit: {arr.unit}\n")
        else:
            file.write(" Unit: N/A\n")

        # Convert to numpy array if it's not already
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        # Handle shape
        if arr.size == 1:
            file.write(" Shape: scalar\n")
            if np.issubdtype(arr.dtype, np.integer):
                file.write(f" Value: {arr.item():d}\n")
            else:
                if arr.item() is None:
                    file.write("Value: None\n")
                else:
                    file.write(f"Value: {arr.item():.6e}\n")
        else:
            file.write(f" Shape: {arr.shape}\n")
            if arr.size > 0:
                max_val = np.max(arr)
                min_val = np.min(arr)
                max_coords = np.unravel_index(np.argmax(arr), arr.shape)
                min_coords = np.unravel_index(np.argmin(arr), arr.shape)
                file.write(f" Max value: {max_val} at coordinates: {max_coords}\n")
                file.write(f" Min value: {min_val} at coordinates: {min_coords}\n")
            else:
                file.write(" Array is empty\n")
    else:
        # C-like output for non-full_info mode
        file.write(f"{name}: ")

        # Convert to numpy array if it's not already
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        is_int = np.issubdtype(arr.dtype, np.integer)

        # Check if the array has units
        has_units = hasattr(arr, "unit")

        if arr.size == 1:
            if has_units:
                file.write(f"value: {arr.value.item():.6e}\n")
            else:
                if arr.item() is None:
                    file.write("value: None\n")
                else:
                    file.write(f"value: {arr.item():.6e}\n")
        else:
            max_val = np.max(arr)
            min_val = np.min(arr)
            if has_units:
                file.write(
                    f"max value: {max_val.value:.6e}, min value: {min_val.value:.6e}\n"
                )
            else:
                file.write(f"max value: {max_val:.6e}, min value: {min_val:.6e}\n")


def print_all_variables(
    observation: object,
    scene: object,
    observatory: object,
) -> None:
    """
    Write comprehensive debug information to files for observation calculations.

    Reads all intermediate and final arrays from observation.validation_variables
    (populated by the vectorized compute path) and writes them to both
    pyedith_validation.txt and pyedith_full_info.txt.

    Parameters
    ----------
    observation : Observation
        Observation object containing observation-specific parameters and
        a populated `validation_variables` dict.
    scene : AstrophysicalScene
        Scene object containing astrophysical scene parameters.
    observatory : Observatory
        Observatory object containing telescope, coronagraph, and detector parameters.
    """
    logger.debug(
        "Printing all relevant variables in pyedith_validation.txt and pyedith_full_info.txt."
    )

    vv = observation.validation_variables  # shorthand

    for mode in ["validation", "full_info"]:
        with open("pyedith_" + mode + ".txt", "w") as file:
            file.write("Input Objects and Their Relevant Properties:\n")
            file.write("1. Observation:\n")
            for item_name, item in [
                ("observation.wavelength", observation.wavelength),
                ("observation.SNR", observation.SNR),
                ("observation.td_limit", observation.td_limit),
                ("observation.CRb_multiplier", observation.CRb_multiplier),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\n2. Scene:\n")
            for item_name, item in [
                ("scene.mag", scene.mag),
                (
                    "scene.stellar_angular_diameter_arcsec",
                    scene.stellar_angular_diameter_arcsec,
                ),
                ("scene.F0", scene.F0),
                ("scene.Fp_over_Fs", scene.Fp_over_Fs),
                ("scene.Fzodi_list", scene.Fzodi_list),
                ("scene.Fexozodi_list", scene.Fexozodi_list),
                ("scene.Fbinary_list", scene.Fbinary_list),
                ("scene.xp", scene.xp),
                ("scene.yp", scene.yp),
                ("scene.separation", scene.separation),
                ("scene.dist", scene.dist),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\n3. Observatory:\n")
            file.write("Telescope:\n")
            for item_name, item in [
                ("observatory.telescope.diameter", observatory.telescope.diameter),
                (
                    "observatory.telescope.temperature",
                    observatory.telescope.temperature,
                ),
                (
                    "observatory.telescope.toverhead_multi",
                    observatory.telescope.toverhead_multi,
                ),
                (
                    "observatory.telescope.toverhead_fixed",
                    observatory.telescope.toverhead_fixed,
                ),
                ("observatory.total_throughput", observatory.total_throughput),
                ("observatory.epswarmTrcold", observatory.epswarmTrcold),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\nCoronagraph:\n")
            for item_name, item in [
                (
                    "observatory.coronagraph.coronagraph_bandwidth",
                    observatory.coronagraph.coronagraph_bandwidth,
                ),
                ("observatory.coronagraph.Istar", observatory.coronagraph.Istar),
                (
                    "observatory.coronagraph.noisefloor",
                    observatory.coronagraph.noisefloor,
                ),
                ("observatory.coronagraph.npix", observatory.coronagraph.npix),
                ("observatory.coronagraph.pixscale", observatory.coronagraph.pixscale),
                (
                    "observatory.coronagraph.psf_trunc_ratio",
                    getattr(observatory.coronagraph, "psf_trunc_ratio", None),
                ),
                (
                    "observatory.coronagraph.photometric_aperture_throughput",
                    getattr(
                        observatory.coronagraph, "photometric_aperture_throughput", None
                    ),
                ),
                ("observatory.coronagraph.skytrans", observatory.coronagraph.skytrans),
                (
                    "observatory.coronagraph.omega_lod",
                    observatory.coronagraph.omega_lod,
                ),
                ("observatory.coronagraph.xcenter", observatory.coronagraph.xcenter),
                ("observatory.coronagraph.ycenter", observatory.coronagraph.ycenter),
                (
                    "observatory.coronagraph.npsfratios",
                    observatory.coronagraph.npsfratios,
                ),
                ("observatory.coronagraph.nrolls", observatory.coronagraph.nrolls),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\nDetector:\n")
            for item_name, item in [
                (
                    "observatory.detector.pixscale_mas",
                    observatory.detector.pixscale_mas,
                ),
                (
                    "observatory.detector.QE*observatory.detector.dQE",
                    observatory.detector.QE * observatory.detector.dQE,
                ),
                (
                    "observatory.detector.npix_multiplier",
                    observatory.detector.npix_multiplier,
                ),
                ("observatory.detector.DC", observatory.detector.DC),
                ("observatory.detector.RN", observatory.detector.RN),
                ("observatory.detector.tread", observatory.detector.tread),
                ("observatory.detector.CIC", observatory.detector.CIC),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\nCalculated Variables:\n")
            file.write("\n1. Initial Calculations:\n")
            for item_name, item in [
                ("Fs_over_F0", scene.Fs_over_F0),
                ("deltalambda_nm", vv.get("deltalambda_nm")),
                ("lod", vv.get("lod")),
                ("lod_rad", vv.get("lod_rad")),
                ("lod_arcsec", vv.get("lod_arcsec")),
                ("area_cm2", vv.get("area_cm2")),
                ("detpixscale_lod", vv.get("detpixscale_lod")),
                ("stellar_diam_lod", vv.get("stellar_diam_lod")),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\n2. Interpolated Arrays:\n")
            for item_name, item in [
                ("Istar_interp", observatory.coronagraph.Istar),
                ("noisefloor_interp", observatory.coronagraph.noisefloor),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\n3. Coronagraph Performance Measurements:\n")
            for item_name, item in [
                ("pixscale_rad", vv.get("pixscale_rad")),
                ("oneopixscale_arcsec", vv.get("oneopixscale_arcsec")),
                ("det_sep_pix", vv.get("det_sep_pix")),
                ("det_sep", vv.get("det_sep")),
                ("det_Istar", vv.get("det_Istar")),
                ("det_skytrans", vv.get("det_skytrans")),
                (
                    "det_photometric_aperture_throughput",
                    vv.get("det_photometric_aperture_throughput"),
                ),
                ("det_omega_lod", vv.get("det_omega_lod")),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\n4. Detector Noise Calculations:\n")
            for item_name, item in [
                ("det_CRp", vv.get("det_CRp")),
                ("det_CRbs", vv.get("det_CRbs")),
                ("det_CRbz", vv.get("det_CRbz")),
                ("det_CRbez", vv.get("det_CRbez")),
                ("det_CRbbin", vv.get("det_CRbbin")),
                ("det_CRbth", vv.get("det_CRbth")),
                ("det_CR", vv.get("det_CR")),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\n5. Planet Position and Separation:\n")
            for item_name, item in [
                ("ix", vv.get("ix")),
                ("iy", vv.get("iy")),
                ("sp_lod", vv.get("sp_lod")),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\n6. Count Rates and Exposure Time Calculation:\n")
            for item_name, item in [
                ("CRp", vv.get("CRp")),
                ("CRnf", vv.get("CRnf")),
                ("CRbs", vv.get("CRbs")),
                ("CRbz", vv.get("CRbz")),
                ("CRbez", vv.get("CRbez")),
                ("CRbbin", vv.get("CRbbin")),
                ("t_photon_count", vv.get("t_photon_count")),
                ("CRbd", vv.get("CRbd")),
                ("CRbth", vv.get("CRbth")),
                ("CRb", vv.get("CRb")),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\n7. Final Result:\n")
            for item_name, item in [
                ("observation.exptime", observation.exptime),
                ("observation.fullsnr", observation.fullsnr),
            ]:
                print_array_info(file, item_name, item, mode)


def synthesize_observation(
    snr_arr: np.ndarray,
    scene: object,
    random_seed: int = None,
    set_below_zero: float = np.nan,
) -> tuple:
    """
    Synthesize an observation using calculated SNRs for each wavelength bin.

    This function generates a synthetic observation by adding noise to the
    planet-to-star flux ratio based on the provided signal-to-noise ratio
    array. The noise is drawn from a normal distribution and scaled according
    to the SNR values. This function requires that the ETC has been run in
    SNR mode with a given exposure time first.

    Parameters
    ----------
    snr_arr : np.ndarray
        1D array containing SNR for each spectral bin
    scene : AstrophysicalScene
        Scene object containing astrophysical parameters including Fp_over_Fs
    random_seed : int, optional
        Random seed for reproducible noise generation. Default is None
    set_below_zero : float, optional
        Value to assign to measurements below zero. Default is np.nan

    Returns
    -------
    tuple
        A tuple containing:

        obs : np.ndarray
            1D array, spectrum with added noise

        noise : np.ndarray
            1D array, noise for each spectral bin
    """

    # set a random seed if desired
    if random_seed is not None:
        np.random.seed(random_seed)

    noise = scene.Fp_over_Fs / snr_arr
    obs = scene.Fp_over_Fs + noise * np.random.randn(len(noise))

    obs[obs < 0] = (
        set_below_zero  # any observation that is below zero is set to whatever you want
    )

    return obs, noise


def generate_wavelength_grid(
    # input_wls: np.ndarray,
    res: float,
    lam_low: float,
    lam_high: float,
) -> tuple:
    """
    Create a new fixed-resolution wavelength grid within given boundaries.

    This function creates a wavelength grid with constant resolution across
    the specified wavelength range. The grid spacing increases logarithmically
    to maintain constant R = λ/Δλ.
    Parameters
    ----------
    res : float
        Desired spectral resolution, e.g. R = lambda / delta_lambda
    lam_low : float, optional
        Lower boundary of the new grid. Defaults to the input grid minimum.
    lam_high : float, optional
        Upper boundary of the new grid. Defaults to the input grid maximum.

    Returns
    -------
    tuple
        A tuple containing two 1D numpy arrays:

        wavelength_grid : np.ndarray
            New wavelength grid

        delta_wavelength_grid : np.ndarray
            New delta wavelength grid
    """

    x = [lam_low]
    fac = (1 + 2 * res) / (2 * res - 1)
    i = 0
    while x[i] * fac < lam_high:
        x = np.concatenate((x, [x[i] * fac]))
        i = i + 1
    x = np.array(x)
    Dx = x / res
    return np.squeeze(x), np.squeeze(Dx)


def regrid_spec_gaussconv(
    input_wls: np.ndarray,
    input_spec: np.ndarray,
    new_lam: np.ndarray,
    new_dlam: np.ndarray,
) -> np.ndarray:
    """
    Regrid a spectrum onto a new wavelength grid using Gaussian convolution.

    This function regrids a spectrum by convolving with Gaussian kernels to
    account for the spectral resolution at each wavelength point. The convolution
    is performed in log-wavelength space for accurate spectral line handling.

    Parameters
    ----------
    input_wls : np.ndarray
        The wavelength grid supplied by the user
    input_spec : np.ndarray
        The spectrum supplied by the user
    new_lam : np.ndarray
        The new wavelength grid calculated for the ETC
    new_dlam : np.ndarray
        The new delta wavelength grid calculated for the ETC

    Returns
    -------
    np.ndarray
        1D array containing the regridded spectrum with original units preserved
    """

    R_arr = new_lam / new_dlam

    # interpolate original spectrum onto a fine log-lambda grid
    loglam_old = np.log(input_wls)
    interp_flux = interp1d(loglam_old, input_spec, bounds_error=False, fill_value=0.0)

    # make fine log-lambda grid
    dloglam = 1e-5
    loglam_grid = np.arange(loglam_old[0], loglam_old[-1], dloglam)
    lam_grid = np.exp(loglam_grid)
    flux_grid = interp_flux(loglam_grid)

    spec_regrid = np.zeros_like(new_lam)

    for i in range(len(new_lam)):
        lam = new_lam[i]
        R = R_arr[i]

        # get width of gaussian: sigma = FWHM / (2*np.sqrt(2*np.log(2))), where FWHM is dlam, but this is in logspace, so FWHM = 1/R
        sigma_loglam = 1.0 / (R * 2.0 * np.sqrt(2 * np.log(2)))

        # Gaussian kernel in log-space
        kernel_half_width = int(4 * sigma_loglam / dloglam)
        kernel_grid = np.arange(-kernel_half_width, kernel_half_width + 1)
        kernel = np.exp(-0.5 * (kernel_grid * dloglam / sigma_loglam) ** 2)
        kernel /= np.sum(kernel)

        # Find center index
        center_idx = np.searchsorted(lam_grid, lam)

        # Define convolution range safely
        i1 = max(center_idx - kernel_half_width, 0)
        i2 = min(center_idx + kernel_half_width + 1, len(flux_grid))
        k1 = kernel_half_width - (center_idx - i1)
        k2 = kernel_half_width + (i2 - center_idx)

        # Perform local convolution
        flux_segment = flux_grid[i1:i2]
        kernel_segment = kernel[k1:k2]
        spec_regrid[i] = np.sum(flux_segment * kernel_segment)

    return spec_regrid


def resample_to_wavelength_grid(
    values,
    from_wavelength,
    to_wavelength,
    to_delta_wavelength=None,  # (only needed for Gaussian interpolation)
    *,  # Forces all subsequent parameters to be keyword-only (must be called as name="value")
    name: str = "parameter",
    interpolation: str = "1d",  # 1d or Gaussian
):
    """
    Fit an already-shaped array onto a resolved wavelength grid.

    The rule is centralized here so no consumer re-implements it:

      * length 1                        -> broadcast the single value to the grid
      * length == len(to_wavelength)    -> already on the grid, pass through
      * any other length                -> regrid onto the grid

    Note that any length > 1 that is neither 1 nor a mismatch requiring regrid
    for *user-supplied* params has already been vetted by ``parse_parameters``;
    this helper additionally covers *defaults* or values that never passed through it.

    Parameters
    ----------
    values : array-like
        The values to fit onto the grid (a plain array or Quantity value).
    from_wavelength : array-like
        The wavelength grid ``values`` currently live on. For consumers this
        should be ``parsed_params["input_wavelength"]`` (the pre-regrid grid).
    to_wavelength : array-like
        The target (resolved) wavelength grid, i.e. ``observation.wavelength``.
    to_delta_wavelength : array-like, optional
        Bin widths of the target grid, required only when a regrid is needed. (only used for Gaussian interpolation)
    name : str, optional
        Name used in log messages.

    Returns
    -------
    np.ndarray
        A float64 array of length ``len(to_wavelength)``.
    """
    if isinstance(values, u.Quantity):
        unit = values.unit
        values_plain = np.atleast_1d(np.asarray(values.value, dtype=np.float64))
    else:
        unit = None
        values_plain = np.atleast_1d(np.asarray(values, dtype=np.float64))

    to_wavelength = np.asarray(to_wavelength, dtype=np.float64)
    n_target = len(to_wavelength)

    # length 1 -> broadcast
    if len(values_plain) == 1 and n_target > 1:
        result = values_plain[0] * np.ones(n_target, dtype=np.float64)

    # already on the grid -> pass through
    elif len(values_plain) == n_target:
        result = values_plain

    # otherwise -> regrid
    else:
        logger.info(
            f"'{name}' has length {len(values_plain)} but the resolved wavelength grid "
            f"has length {n_target}. Rebinning..."
        )
        if interpolation == "1d":
            interp_func = interp1d(
                np.asarray(from_wavelength, dtype=np.float64), values_plain
            )
            result = interp_func(to_wavelength)

        elif interpolation == "Gaussian":
            if to_delta_wavelength is None:
                raise ValueError(
                    f"'{name}' must be regridded onto the resolved grid, but "
                    f"'to_delta_wavelength' was not provided."
                )
            result = regrid_spec_gaussconv(
                np.asarray(from_wavelength, dtype=np.float64),
                values_plain,
                to_wavelength,
                to_delta_wavelength,
            )

        else:
            raise ValueError(
                "Unknown interpolation type. Possible values are '1d' for 1D "
                "interpolation and 'Gaussian' for Gaussian kernel interpolation "
                "(recommended for spectral quantities)."
            )

    # ------------------------------------------------------------------
    # Unit reattachment: return a Quantity if the input was one.
    # ------------------------------------------------------------------
    if unit is not None:
        return result * unit
    return result


def rebin_channel_curves_to_grid(
    spectral: dict,
    curve_keys: list,
    to_wavelength: np.ndarray,
    to_delta_wavelength: np.ndarray = None,
    interpolation: str = "1d",
    obs_mode: str = "IMAGER",  # temporary
    wavelength_range: list = [0.1, 0.2],  # temporary
) -> dict:
    """
    Rebin channel-native engineering curves onto the resolved observation
    wavelength grid.

    ``spectral`` is expected to be the per-channel spectral sub-dictionary
    of the unified EAC configuration, i.e.
    ``eac_config[obs_mode][active_channel]["spectral"]``, containing a single
    shared ``"wavelength"`` array plus one flat value array per curve
    (e.g. ``"qe"``, ``"dqe"``, ``"optics_throughput"``), all on the same
    native (typically much higher-resolution) grid.

    For each key in ``curve_keys`` found in ``spectral``:
      - If ``obs_mode == "IMAGER"`` (temporary behavior), the curve is
        collapsed to a single bandpass-averaged value via
        ``average_over_bandpass`` and used as-is (no further resampling).
      - Otherwise, the array is sorted, deduplicated, and rebinned onto
        ``to_wavelength`` via ``resample_to_wavelength_grid``.

    Keys not present in ``spectral`` are skipped (e.g. scalar detector
    parameters like ``dc``/``rn``/``cic``, which live outside ``spectral``
    and are used as-is without rebinning).

    Parameters
    ----------
    spectral : dict
        The per-channel spectral sub-dictionary, i.e.
        ``eac_config[obs_mode][active_channel]["spectral"]``. Must contain a
        ``"wavelength"`` key shared by all curves, plus one array per curve
        key requested in ``curve_keys``.
    curve_keys : list of str
        Which keys within ``spectral`` to rebin (e.g. ``["qe", "dqe"]``).
    to_wavelength : np.ndarray
        Target wavelength grid, plain array (e.g. ``observation.wavelength.value``).
    to_delta_wavelength : np.ndarray, optional
        Bin widths of the target grid; required only for Gaussian interpolation.
    interpolation : str
        "1d" or "Gaussian", passed through to ``resample_to_wavelength_grid``.

    Returns
    -------
    dict
        Mapping of curve key -> rebinned array on the ``to_wavelength`` grid
        (or bandpass-averaged single-value array, in the temporary IMAGER path).
    """
    if interpolation == "Gaussian" and to_delta_wavelength is None:
        raise ValueError(
            "to_delta_wavelength must be provided when interpolation='Gaussian'."
        )

    rebinned = {}

    if "wavelength" not in spectral:
        logger.debug(
            "No 'wavelength' array found in spectral data. Skipping all curves."
        )
        return rebinned

    wavelengths_native = np.asarray(spectral["wavelength"], dtype=np.float64)

    for key in curve_keys:
        if key not in spectral:
            logger.debug(f"{key} was not found in YAML files. Skipping...")
            continue  # key was not found

        wavelengths = wavelengths_native
        values = np.asarray(spectral[key], dtype=np.float64)

        if (
            obs_mode == "IMAGER"
        ):  # temporary, to reproduce past behavior (will be replaced once we implement new feature)
            rebinned[key] = [
                average_over_bandpass(
                    {
                        "lam": wavelengths * WAVELENGTH,
                        "value": values,
                    },
                    wavelength_range,
                )["value"]
            ]
            logger.debug(
                f"Bandpass-averaged '{key}' over {wavelength_range} "
                f"(IMAGER temporary path); skipping resampling."
            )
            continue  # skip resampling below; bandpass average is final for IMAGER

        if wavelengths.size > 1:
            sorted_idx = np.argsort(wavelengths)
            wavelengths = wavelengths[sorted_idx]
            values = values[sorted_idx]
            unique_idx = np.concatenate(([True], wavelengths[1:] != wavelengths[:-1]))
            wavelengths = wavelengths[unique_idx]
            values = values[unique_idx]

        rebinned[key] = resample_to_wavelength_grid(
            values,
            from_wavelength=(wavelengths * u.um).to(
                WAVELENGTH
            ),  # in case wavelength ever changes units
            to_wavelength=to_wavelength,
            to_delta_wavelength=to_delta_wavelength,
            name=f"{key}",
            interpolation=interpolation,
        )

        logger.debug(
            f"Rebinned '{key}' from {wavelengths.size} native points to "
            f"{len(to_wavelength)} grid points."
        )

    return rebinned
