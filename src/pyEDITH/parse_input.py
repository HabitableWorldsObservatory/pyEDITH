from typing import Union, Dict, Tuple
from pathlib import Path
import astropy.units as u
import numpy as np

from pyEDITH.filters import Filter, utils
from .units import *
import pandas as pd
import os
import logging

logger = logging.getLogger("pyEDITH")


def parse_input_file(
    file_path: Union[Path, str], secondary_flag: bool
) -> Tuple[Dict, Dict]:
    """
    Parse an input file and extract variables and secondary variables.

    This function reads a configuration file, processes its contents line by line,
    and extracts primary and optional secondary variables. It handles various data
    types including arrays, strings, and numeric values, and performs special
    processing for IFS observing mode parameters. The function handles various data types including arrays, strings, and numbers.
    Comments in the input file should start with ';'.

    Parameters
    ----------
    file_path : Union[Path, str]
        Path to the input file
    secondary_flag : bool
        Flag indicating whether secondary variables are expected

    Returns
    -------
    Tuple[Dict, Dict]
        A tuple containing two dictionaries:

        variables: dict
            Primary variables extracted from the file

        secondary_variables: dict
            Secondary variables extracted from the file
            (any non-specified variable will be the same as in the variables dictionary)

    Raises
    ------
    KeyError
        If some parameters are missing
    FileNotFoundError
        If a specified spectrum file cannot be found
    ValueError
        If there are issues with the spectrum file

    """

    def _parse_array_value(raw_items: str) -> list:
        """
        Parse the contents of a bracketed array (without the brackets) into
        a list of floats or strings, entry by entry.

        Parameters
        ----------
        raw_items : str
            Comma-separated array contents, e.g. "0.55, 0.65" or "photometry, photometry"

        Returns
        -------
        list
            List where each element is a float if convertible, otherwise a
            stripped string (surrounding quotes removed if present).
        """
        parsed = []
        for item in raw_items.split(","):
            item = item.strip()
            # Try numeric conversion first, since this is the common case
            try:
                parsed.append(float(item))
            except ValueError:
                # Strip optional surrounding quotes, e.g. 'photometry' -> photometry
                if (item.startswith("'") and item.endswith("'")) or (
                    item.startswith('"') and item.endswith('"')
                ):
                    item = item[1:-1]
                parsed.append(item)
        return parsed

    with open(file_path, "r") as file:
        content = file.read()

    # Remove comments and empty lines
    lines = [
        line.split(";")[0].strip()
        for line in content.split("\n")
        if line.strip() and not line.strip().startswith(";")
    ]

    variables = {}
    secondary_variables = {}
    has_secondary = False

    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Handle arrays
            if value.startswith("[") and value.endswith("]"):
                value = _parse_array_value(value[1:-1])
            # Handle strings
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            else:
                # Handle numbers
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass  # Keep as string if it's not a number

            if "secondary" in key:
                has_secondary = True
                if secondary_flag:
                    secondary_variables[key[10:]] = (
                        value  # it replaces the default value
                    )
            else:
                variables[key] = value

    if secondary_flag and not has_secondary:
        raise KeyError(
            "Secondary flag is True but no secondary variables found in the input file."
        )

    # Check that observing_mode is present and valid
    if "observing_mode" not in variables:
        raise KeyError(
            "Required parameter 'observing_mode' is not provided in the input file."
        )
    if variables["observing_mode"] not in ["IMAGER", "IFS"]:
        raise ValueError(
            f"Invalid observing mode '{variables['observing_mode']}'. Must be 'IMAGER' or 'IFS'."
        )

    # Handle IFS mode
    if variables.get("observing_mode") == "IFS":
        required_columns = ["wavelength", "Fstar_10pc", "Fp/Fs"]

        # Check if all required columns are provided as lists in the input file
        if all(
            col in variables and isinstance(variables[col], list)
            for col in required_columns
        ):
            # Ensure all lists have the same length
            lengths = [len(variables[col]) for col in required_columns]
            if len(set(lengths)) != 1:
                raise ValueError(
                    f"All of {', '.join(required_columns)} must have the same length"
                )
            variables["nlambda"] = lengths[0]

        # If not all required columns are provided, try to read from spectrum file
        elif "spectrum_file" in variables:
            spectrum_file = variables["spectrum_file"]

            # Check if the file exists and is readable
            if not os.path.isfile(spectrum_file):
                raise FileNotFoundError(f"Spectrum file not found: {spectrum_file}")

            # Read the spectrum file
            spectrum_df = pd.read_csv(variables["spectrum_file"])

            # Ensure the file has exactly 3 columns
            if len(spectrum_df.columns) != 3:
                raise ValueError(
                    f"Spectrum file must contain exactly 3 columns (wavelength, stellar flux, planet contrast), but it has {len(spectrum_df.columns)}"
                )
            # Rename the columns to ensure they have the correct names
            spectrum_df.columns = ["wavelength", "Fstar_10pc", "Fp/Fs"]

            # Verify that all columns can be converted to float
            for column in spectrum_df.columns:
                try:
                    spectrum_df[column] = spectrum_df[column].astype(float)
                except ValueError:
                    raise ValueError(f"Column '{column}' contains non-numeric values")

            # Set the wavelength-dependent parameters from the file
            variables["wavelength"] = spectrum_df["wavelength"].tolist()
            variables["Fstar_10pc"] = spectrum_df["Fstar_10pc"].tolist()
            variables["Fp/Fs"] = spectrum_df["Fp/Fs"].tolist()
            variables["nlambda"] = len(spectrum_df["wavelength"].tolist())

        else:
            raise KeyError(
                "Required parameters 'wavelength', 'Fstar_10pc', and 'Fp/Fs' are not provided. Please write them explicitly or provide a spectrum_file path."
            )

    if variables.get("observing_mode") == "IMAGER" and isinstance(
        variables["wavelength"], list
    ):
        raise KeyError(
            "In IMAGER mode you can only use one wavelength at a time. If you are simulating photometry, please run every single wavelength separately. If you want to model a spectrum, please use IFS mode."
        )

    # ------------------------------------------------------------------
    # Handle Filters feature: reconstruct Filter objects from parallel
    # arrays, mirroring the list of Filter instances passed directly in
    # the Python-IDE path. Filter.type is always taken from observing_mode,
    # so it is never specified per-filter in the input file. Filter
    # resolution is optional for IMAGER but required for IFS, matching
    # the fact that resolution is only physically meaningful for a
    # spectrograph.
    # ------------------------------------------------------------------
    filter_keys_present = [key for key in variables if key.startswith("filter_")]

    if filter_keys_present:
        if "filter_name" not in variables:
            raise KeyError(
                "'filter_name' is required whenever any filter_* parameter is provided."
            )

        has_center_bandwidth = (
            "filter_center" in variables or "filter_bandwidth" in variables
        )
        has_low_high = "filter_low" in variables or "filter_high" in variables

        if has_center_bandwidth and has_low_high:
            raise ValueError(
                "Specify filters using either (filter_center, filter_bandwidth) "
                "or (filter_low, filter_high), not both, in the same input file."
            )
        elif has_center_bandwidth:
            bounds_keys = ["filter_center", "filter_bandwidth"]
        elif has_low_high:
            bounds_keys = ["filter_low", "filter_high"]
        else:
            raise KeyError(
                "Filters require either (filter_center, filter_bandwidth) "
                "or (filter_low, filter_high) to be specified."
            )

        for key in bounds_keys:
            if key not in variables:
                raise KeyError(f"'{key}' is required to fully specify filter bounds.")

        # observing_mode is guaranteed present and valid at this point,
        # since it was checked earlier in this function.
        observing_mode = variables["observing_mode"]

        if observing_mode == "IFS" and "filter_resolution" not in variables:
            raise KeyError(
                "'filter_resolution' is required for filters when observing_mode is 'IFS'."
            )

        all_filter_keys = ["filter_name"] + bounds_keys
        if "filter_resolution" in variables:
            all_filter_keys.append("filter_resolution")

        # Normalize single-filter shorthand (scalar instead of one-item list)
        for key in all_filter_keys:
            if not isinstance(variables[key], list):
                variables[key] = [variables[key]]

        lengths = {len(variables[key]) for key in all_filter_keys}
        if len(lengths) != 1:
            raise ValueError(
                f"All filter_* parameters ({', '.join(all_filter_keys)}) "
                "must have the same length."
            )
        n_filters = lengths.pop()

        filter_name_list = variables.pop("filter_name")
        filter_resolution_list = variables.pop("filter_resolution", [None] * n_filters)

        filter_list = []
        for i in range(n_filters):
            filter_kwargs = {
                "resolution": filter_resolution_list[i],
                "type": observing_mode,
            }
            if bounds_keys == ["filter_center", "filter_bandwidth"]:
                filter_kwargs["center"] = variables["filter_center"][i] * u.um
                filter_kwargs["bandwidth"] = variables["filter_bandwidth"][i]
            else:
                filter_kwargs["low"] = variables["filter_low"][i] * u.um
                filter_kwargs["high"] = variables["filter_high"][i] * u.um

            filter_list.append(Filter(filter_name_list[i], **filter_kwargs))

        for key in bounds_keys:
            variables.pop(key)

        variables["filter_list"] = filter_list

    return variables, secondary_variables


def normalize_list_shapes(parameters, key, default_len=1):
    """
    Normalize the *shape* of a user-supplied parameter without binding it to a
    wavelength grid that may later change.

    Parsing here is deliberately grid-agnostic: we handle scalar broadcasting
    and preserve ``astropy.Quantity`` units, but we do NOT reject an array whose
    length happens to differ from ``default_len``. The final alignment onto the
    (possibly rebinned) resolved grid is deferred to ingestion time.

    Parameters
    ----------
    key : str
        Name of the parameter to look up in ``parameters``.
    default_len : int
        The expected length based on the *current* input grid. Used only for
        scalar broadcasting and (optionally) strict validation.

    """

    value = parameters[key]

    # Function to convert to float array, preserving Quantity if present
    def to_float_array(v):
        if isinstance(v, u.Quantity):
            return u.Quantity(np.array(v.value, dtype=np.float64), v.unit)
        else:
            return np.array(v, dtype=np.float64)

    if default_len > 1:
        # Case 1 & 1a: default_len > 1 but value is a pure scalar or single-element array
        if (
            np.isscalar(value)
            or (isinstance(value, u.Quantity) and value.isscalar)
            or (isinstance(value, (list, np.ndarray, tuple)) and len(value) == 1)
        ):
            logger.warning(
                f"{key} should be a list of length {default_len}. "
                "pyEDITH will create one assuming the input value for all the elements of the list."
            )
            if isinstance(value, u.Quantity):
                # If it's a single-element quantity array/list, treat it as a scalar
                if not value.isscalar and value.size == 1:
                    value = value[0]
                return u.Quantity(np.full(default_len, value.value), value.unit)
            else:
                # If it's a single-element list, array, or scalar
                if (isinstance(value, (list, np.ndarray, tuple))) and len(value) == 1:
                    value = value[0]
                return np.full(default_len, value, dtype=np.float64)

        # Case 2: default_len > 1 and value has length > 1 but != default_len
        # This is an error: the user provided a multi-element array that doesn't match
        elif (
            isinstance(value, (list, np.ndarray, u.Quantity))
            and len(value) > 1
            and len(value) != default_len
        ):
            raise ValueError(
                f"{key} should be a list of length {default_len}, but it has length {len(value)}."
            )

        # If value is already correct length, just convert to float array
        return to_float_array(value)

    else:
        # Case 3: default_len == 1
        # A scalar or single-element value becomes a length-1 array. A multi-element
        # value throws an error.
        if isinstance(value, u.Quantity):
            if value.size > 1:
                raise ValueError(
                    f"{key} has length {value.size} but the expected input size is {default_len}. "
                )
            else:
                # Handle both scalar Quantity and single-element Quantity array
                scalar_value = value.value if value.isscalar else value.value.flat[0]
                return u.Quantity([scalar_value], value.unit)
        elif isinstance(value, (list, np.ndarray, tuple)):
            if len(value) > 1:
                raise ValueError(
                    f"{key} has length {len(value)} but the expected input size is {default_len}. "
                )
            else:
                return to_float_array(value)
        else:
            return to_float_array([value])


def parse_parameters(parameters: dict) -> dict:
    """
    Parse and process input parameters for simulation.

    This function handles various parameter types including wavelength-dependent parameters,
    target-specific parameters, and scalar values. It converts parameters to appropriate
    data types and ensures arrays have the correct dimensions based on the number of
    wavelength points.

    Parameters
    ----------
    parameters : dict
        A dictionary of input parameters
    nlambda : int, optional
        Number of wavelength points, if not specified in parameters

    Returns
    -------
    dict
        A dictionary of parsed and processed parameters, including: arrays of length
        nlambda (wavelength-dependent parameters), Scalar parameters, Coronagraph
        specifications.

    Raises
    ------
    ValueError
        If wavelength information is missing and nlambda is not provided,
        or if array parameters have incorrect dimensions

    Note
    -----
    The function assumes one target (ntargs = 1) for now.
    nmeananom and norbits are defaulted to 1.
    """
    # LEGACY: Protect in case we already passed parsed parameters
    if "_parsed" in parameters.keys():
        return parameters
    parsed_params = {}

    # NLAMBDA
    if "wavelength" in parameters.keys():
        if np.isscalar(parameters["wavelength"]) or (
            isinstance(parameters["wavelength"], u.Quantity)
            and np.isscalar(parameters["wavelength"].value)
        ):
            parsed_params["nlambda"] = 1

        else:
            parsed_params["nlambda"] = len(parameters["wavelength"])
            if parsed_params["nlambda"] == 0:
                raise ValueError("'wavelength' parameter cannot be an empty list.")
        parsed_params["wavelength"] = normalize_list_shapes(
            parameters, "wavelength", parsed_params["nlambda"]
        )

    else:
        raise ValueError(
            "pyEDITH does not have access to wavelength here, please review your input."
        )

    # Use the determined or provided nlambda for array standardization
    nlambda = parsed_params["nlambda"]

    # ------ ARRAYS OF LENGTH NLAMBDA ------

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
        "IFS_eff",
        "mag",  # used to be [ntargs x nlambda], now just [nlambda]
        "Fstar_10pc",
        "Fp/Fs",
        "ez_PPF",
        "delta_mag",  # used to be [nmeananom x norbits x ntargs]
        "F0",  # for validation purposes, the calculation of F0 is different in AYO
        "det_npix_input",  # for validation purposes
    ]

    parsed_params.update(
        {
            key: normalize_list_shapes(parameters, key, nlambda)
            for key in list(set(wavelength_params) & set(parameters.keys()))
        }
    )

    # ------ SCALARS (USED TO BE ARRAYS IN v. 0.2 and earlier) ------
    target_params = [
        "distance",  # used to be [ntargs]
        "magV",  # used to be [ntargs]
        "FstarV_10pc",
        "stellar_radius",  # used to be [ntargs]
        "nzodis",  # used to be [ntargs]
        "ra",  # used to be [ntargs]
        "dec",  # used to be [ntargs]
        "delta_mag_min",  # used to be [ntargs]
        "Fp_min/Fs",
        "semimajor_axis",
        "separation",  # used to be ARRAYS OF LENGTH  nmeananom x norbits x ntargs (but nmeananom and norbits are defaulted to 1
    ]

    for key in list(set(target_params) & set(parameters.keys())):
        parsed_params[key] = float(parameters[key])

    # ---- SCALARS ----
    scalar_params = [
        "photometric_aperture_radius",
        "psf_trunc_ratio",
        "diameter",
        "toverhead_fixed",
        "toverhead_multi",
        "pixscale",  # pixel scale of the coronagraph
        "pixscale_mas",  # pixel scale of the detector
        "contrast",
        "noisefloor_factor",
        "noisefloor_PPF",
        "bandwidth",
        "coronagraph_bandwidth",
        "Tcore",
        "TLyot",
        "unobscured_area",
        "temperature",
        "T_contamination",
        "CRb_multiplier",
        "t_photon_count_input",  # only for ETC validation
        "npix_multiplier",
    ]

    for key in list(set(scalar_params) & set(parameters.keys())):
        value = parameters[key]
        # Handle npix_multiplier deprecation: was array, now scalar
        if (
            key == "npix_multiplier"
            and hasattr(value, "__len__")
            and not isinstance(value, str)
        ):

            logger.warning(
                "DeprecationWarning: Passing 'npix_multiplier' as an array is deprecated and will be removed in a future version. "
                "Please provide it as a scalar value instead. Using the first element for now."
            )

            if isinstance(value, np.ndarray):
                parsed_params[key] = float(value.flat[0])
            else:
                parsed_params[key] = float(value[0])
        else:
            parsed_params[key] = float(parameters[key])

    # ---- INTEGERS ----
    integer_params = ["nrolls"]

    for key in list(set(integer_params) & set(parameters.keys())):
        parsed_params[key] = int(parameters[key])

    if "nchannels" in parameters.keys():
        logger.warning(
            "DeprecationWarning: 'nchannels' is deprecated, disregarding. Results will be comparable to setting nchannels to 1."
        )

    # ----- BOOLEANS ----
    for key in ["az_avg"]:
        if key in parameters.keys():
            value = parameters[key]
            if isinstance(value, str):
                value_lower = value.lower()
                if value_lower in ("true", "1", "yes"):
                    parsed_params[key] = True
                elif value_lower in ("false", "0", "no"):
                    parsed_params[key] = False
                else:
                    raise ValueError(
                        f"Invalid value '{value}' for parameter '{key}'. "
                        f"Expected boolean or one of: 'true', 'false', '1', '0', 'yes', 'no' "
                        f"(case-insensitive)."
                    )
            elif isinstance(value, bool):
                parsed_params[key] = value
            elif isinstance(value, (int, float)):
                if value in (0, 1, 0.0, 1.0):
                    parsed_params[key] = bool(value)
                else:
                    raise ValueError(
                        f"Invalid numeric value '{value}' for parameter '{key}'. "
                        f"Expected 0 or 1 for boolean parameters."
                    )
            else:
                raise TypeError(
                    f"Invalid type {type(value).__name__} for parameter '{key}'. "
                    f"Expected boolean, string, or numeric (0/1)."
                )

    # ---- OBSERVATORY SPECS ----
    for key in [
        "observatory_preset",
        "telescope_type",
        "coronagraph_type",
        "detector_type",
        "observing_mode",
    ]:

        if key in parameters.keys():
            parsed_params[key] = parameters[key]
            if key == "observing_mode" and parameters[key] not in ["IMAGER", "IFS"]:
                raise KeyError("Invalid observing mode. Must be 'IMAGER' or 'IFS'.")

    # ADVANCED FLAG: dictionary of values to be overwritten (despite being locked)
    if "overrides" in parameters.keys():
        overrides_value = parameters["overrides"]
        if isinstance(overrides_value, str):
            overrides_list = [item.strip() for item in overrides_value.split(",")]
            # Filter out empty strings
            overrides_list = [item for item in overrides_list if item]
            if overrides_list:
                parsed_params["overrides"] = overrides_list
        else:
            overrides_list = list(overrides_value)
            if overrides_list:
                parsed_params["overrides"] = overrides_list

    # Update _parsed key
    parsed_params["_parsed"] = True
    return parsed_params


def parse_filters(parameters):

    active_filters = []
    # ALWAYS ensure it has units
    input_wls = parameters["wavelength"]

    if not isinstance(input_wls, u.Quantity):
        input_wls = input_wls * WAVELENGTH

    wl_min = np.min(input_wls).to(WAVELENGTH)
    wl_max = np.max(input_wls).to(WAVELENGTH)

    if "filter_list" in parameters.keys():
        filter_list = parameters["filter_list"]

        # If it's a single filter, convert to list
        if not isinstance(filter_list, list):
            filter_list = [filter_list]

        # Validate each filter in the list
        for i, f in enumerate(filter_list):
            # Check if it's a Filter object (duck typing - check for expected attributes)
            if not hasattr(f, "__dict__"):
                raise TypeError(
                    f"Filter at index {i}: must be a Filter object, "
                    f"but got {type(f).__name__}"
                )

            # TEMPORARY: will go away once we force the user to provide the spectrum
            # check if the value is in-between the filter that I request. Assume it is the average value.
            if (parameters["observing_mode"] == "IMAGER") and (
                wl_min <= f.high and wl_max >= f.low
            ):  # wl_min and wl_max should be the same
                active_filters.append(f)
            # Check spectral resolution compatibility for IFS mode
            elif (parameters["observing_mode"] == "IFS") and (
                wl_min <= f.low and wl_max >= f.high
            ):  # checking that original wavelength range is larger than the required filter
                active_filters.append(f)
                # Calculate input spectrum resolution
                input_dlam = np.gradient(input_wls)
                input_resolution = input_wls / input_dlam

                # Find overlapping wavelength region
                overlap_mask = (input_wls >= f.low) & (input_wls <= f.high)
                if np.any(overlap_mask):
                    # Get median resolution in overlapping region
                    median_input_res = np.median(input_resolution[overlap_mask])

                    # Warn if input resolution is lower than filter resolution
                    if median_input_res < f.resolution:
                        logger.warning(
                            f"Filter {f.name if hasattr(f, 'name') else f}: "
                            f"Input spectrum resolution (R~{median_input_res:.1f}) is lower than "
                            f"filter resolution (R~{f.resolution:.1f}). "
                            f"Interpolation to filter wavelength grid may introduce artifacts."
                        )
            else:
                logger.warning(
                    f"Filter {f.name if hasattr(f, 'name') else f} discarded: "
                    f"wavelength range [{wl_min}, {wl_max}] does not fully cover "
                    f"filter range [{f.low}, {f.high}]"
                )

        if len(active_filters) == 0:
            raise ValueError(
                "No filters can be used. Specify different filters or change spectrum."
            )
        else:
            return active_filters
    else:
        logger.warning("Making a filter from the legacy parameters...")
        # LEGACY FILTER HELPER

        if parameters["observing_mode"] == "IMAGER":
            # Extract scalar value from input_wls, handling both array and scalar cases

            if input_wls.isscalar:
                center = input_wls
            else:
                center = input_wls.flat[0]  # it is an array, pick the first value

            return [
                Filter(
                    name=str(np.round(center.value, 1))
                    + " BW "
                    + str(parameters["bandwidth"]),
                    center=center,
                    bandwidth=parameters["bandwidth"],
                    type="IMAGER",
                )
            ]

        elif parameters["observing_mode"] == "IFS":
            # In IFS mode, create a filter for each value of spectral_resolution/lam_low/lam_high
            logger.info("Calculating a new wavelength grid and re-gridding spectra...")

            deprecated_keys = ["spectral_resolution", "lam_low", "lam_high"]
            if all(key in parameters for key in deprecated_keys):
                logger.warning(
                    "DeprecationWarning: The parameters 'regrid_wavelength', 'spectral_resolution', 'lam_low', and 'lam_high' will be deprecated soon. "
                    "Please update your configuration to use the new Filter object."
                )

                # Check that all required parameters are arrays/lists
                for key in deprecated_keys:
                    val = parameters[key]
                    if not isinstance(val, (list, np.ndarray, u.Quantity)):
                        raise ValueError(
                            f"'{key}' is not an array. "
                            f"All of {', '.join(deprecated_keys)} must be arrays of the same length."
                        )

                assert (
                    len(parameters["spectral_resolution"])
                    == len(parameters["lam_low"])
                    == len(parameters["lam_high"])
                ), f"{', '.join(deprecated_keys)} have different lengths. All must have the same length."
                assert (
                    np.min(input_wls.value) < parameters["lam_low"][0]
                ), "Your minimum input wavelength is greater than first channel lower boundary."
                assert (
                    np.max(input_wls.value) > parameters["lam_high"][-1]
                ), f"Your maximum input wavelength is less than last channel upper boundary."

                for i in range(0, len(parameters["spectral_resolution"])):
                    res = parameters["spectral_resolution"][i]
                    lam_low = parameters["lam_low"][i]
                    lam_high = parameters["lam_high"][i]
                    if not isinstance(lam_low, u.Quantity):
                        lam_low = lam_low * WAVELENGTH
                    if not isinstance(lam_high, u.Quantity):
                        lam_high = lam_high * WAVELENGTH
                    active_filters.append(
                        Filter(
                            str(np.round(lam_low, 1))
                            + "-"
                            + str(np.round(lam_high, 1))
                            + "IFS",
                            low=lam_low,
                            high=lam_high,
                            resolution=res,
                            type="IFS",
                        )
                    )

                return active_filters
            else:
                raise ValueError(
                    f"Could not find filters; attempted legacy behavior but some of the following keys are missing: {deprecated_keys}"
                )


def read_configuration(
    input_file: Union[Path, str], secondary_flag: bool = False
) -> Tuple[Dict, Dict]:
    """
    Read and parse configuration from an input file.

    This function reads the input file, extracts parameters, and then parses both
    the primary and secondary parameters. It serves as a high-level wrapper around
    parse_input_file() and parse_parameters().

    Parameters
    ----------
    input_file : Union[Path, str]
        Path to the input configuration file
    secondary_flag : bool, optional
        Flag indicating whether secondary variables should be processed, default is False

    Returns
    -------
    Tuple[Dict, Dict]
        A tuple containing two dictionaries:

        parsed_parameters: dict
            Parsed primary parameters

        parsed_secondary_parameters: dict
            Parsed secondary parameters (empty if secondary_flag is False)
    """

    parameters, secondary_parameters = parse_input_file(input_file, secondary_flag)
    parsed_parameters = parse_parameters(parameters)

    if secondary_flag:
        # Parse secondary parameters
        parsed_secondary_parameters = parse_parameters(secondary_parameters)
    else:
        parsed_secondary_parameters = {}

    return parsed_parameters, parsed_secondary_parameters


def get_observatory_config(parameters: Dict[str, str]) -> Union[str, Dict[str, str]]:
    """
    Generate observatory configuration from parameters.

    This function extracts observatory configuration information from the parameters
    dictionary. It either returns a preset name if specified or constructs a custom
    configuration dictionary with telescope, coronagraph, and detector components.

    Parameters
    ----------
    parameters : Dict[str, str]
        Dictionary containing configuration parameters

    Returns
    -------
    Union[str, Dict[str, str]]
        Either a string (if using a preset) or a dictionary (for custom configurations)

    Raises
    ------
    ValueError
        If any required component type is not specified
    """

    if "observatory_preset" in parameters:
        config = parameters["observatory_preset"]
    else:
        config = {}
        for component in ["telescope", "coronagraph", "detector"]:
            component_type = parameters.get(f"{component}_type")
            if component_type is None:
                raise ValueError(
                    f"{component.capitalize()} type not specified. Please provide a '{component}_type' parameter or use a preset."
                )
            config[component] = component_type

    print_observatory_config(config)
    return config


def print_observatory_config(config: Union[str, Dict[str, str]]) -> None:
    """
    Print the observatory configuration to the terminal.

    This function formats and displays the observatory configuration in a
    human-readable format, showing either the preset name or the individual
    component selections.

    Parameters
    ----------
    config : Union[str, Dict[str, str]]
        The observatory configuration, either as a string (preset) or
        a dictionary (custom configuration)
    """

    logger.info("Observatory Configuration:")
    if isinstance(config, str):
        logger.info(f"  Using preset: {config}")
    else:
        logger.info(f"  Telescope:   {config['telescope']}")
        logger.info(f"  Coronagraph: {config['coronagraph']}")
        logger.info(f"  Detector:    {config['detector']}")
    logger.info("")  # Add a blank line for better readability
