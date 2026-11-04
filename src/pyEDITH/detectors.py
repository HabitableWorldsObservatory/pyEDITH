from abc import ABC, abstractmethod
import numpy as np
from . import utils
import astropy.units as u
from .units import *
from pyEDITH import parse_input


class Detector(ABC):
    """
    A class representing a detector for astronomical observations.

    This class manages detector-specific parameters and configurations
    used in astronomical simulations and observations.

    Parameters
    ----------
    pixscale_mas : float
        Detector pixel scale in milliarcseconds.
    npix_multiplier : scalar
        Number of detector pixels per image plane "pixel".
    DC : ndarray
        Dark current in counts per pixel per second.
    RN : ndarray
        Read noise in counts per pixel per read.
    tread : ndarray
        Read time in seconds.
    CIC : ndarray
        Clock-induced charge in counts per pixel per photon count.
    QE: ndarray
        Quantum efficiency of detector
    dQE: ndarray
        Effective QE due to degradation, cosmic ray effects, readout inefficiencies
    """

    # Keys that a user is NOT allowed to override for this detector mode.
    # Subclasses override this. An empty set means "everything is user-editable".
    LOCKED_KEYS: set = set()

    @abstractmethod
    def load_configuration(self):
        """
        Load configuration parameters for the detector.

        This abstract method must be implemented by subclasses to define
        how detector configuration parameters are loaded and processed.
        """
        pass  # pragma: no cover

    def validate_configuration(self):
        """
        Check that mandatory variables are present and have the correct format.

        This method validates that all required attributes exist on the detector
        object and that they have the expected types and units. Additional variables
        may be present but are not required for calculations.

        Raises
        ------
        AttributeError
            If a required attribute is missing
        TypeError
            If an attribute has an incorrect type
        ValueError
            If a Quantity attribute has incorrect units
        """

        expected_args = {
            "pixscale_mas": MAS,
            "npix_multiplier": DIMENSIONLESS,
            "DC": DARK_CURRENT,
            "RN": READ_NOISE,
            "tread": READ_TIME,
            "CIC": CLOCK_INDUCED_CHARGE,
            "QE": QUANTUM_EFFICIENCY,
            "dQE": DIMENSIONLESS,
        }
        utils.validate_attributes(self, expected_args)
        # for arg, expected_unit in expected_args.items():
        #     if not hasattr(self, arg):
        #         raise AttributeError(f"Detector is missing attribute: {arg}")
        #     value = getattr(self, arg)
        #     if not isinstance(value, u.Quantity):
        #         raise TypeError(f"Detector attribute {arg} should be a Quantity")
        #     if not value.unit == expected_unit:
        #         raise ValueError(
        #             f"Detector attribute {arg} has incorrect units. Expected {expected_unit}, got {value.unit}"
        #         )


class ToyModelDetector(Detector):
    """
    A toy model detector class that extends the base Detector class.

    This class represents a simplified detector model for use in simulations
    where users can specify all detector parameters manually rather than
    using predefined models from configuration files.

    Parameters
    ----------
    path : str, optional
        Path to configuration files (not used in toy model)
    keyword : str, optional
        Keyword for configuration selection (not used in toy model)
    """

    # In toy-model mode EVERY parameter is user-editable, so nothing is locked.
    LOCKED_KEYS: set = set()
    DEFAULT_CONFIG = {
        "pixscale_mas": None,  # Detector pixel scale in milliarcseconds.
        "npix_multiplier": 1
        * DIMENSIONLESS,  # Number of detector pixels per image plane "pixel".
        "DC": [3e-5] * DARK_CURRENT,  # Dark current (counts pix^-1 s^-1, nlambda array)
        "RN": [0.0] * READ_NOISE,  # Read noise (counts pix^-1 read^-1, nlambda array)
        "tread": [1000] * READ_TIME,  # Read time (s, nlambda array)
        "CIC": [1.3e-3]
        * CLOCK_INDUCED_CHARGE,  # Clock-induced charge (counts pix^-1 photon_count^-1, nlambda array)
        "QE": [0.9] * QUANTUM_EFFICIENCY,  # Quantum efficiency of detector
        "dQE": [0.75]
        * DIMENSIONLESS,  # Effective QE due to degradation, cosmic ray effects, readout inefficiencies
    }

    def __init__(self, path: str = None, keyword: str = None):
        """
        Initialize a ToyModelDetector instance.

        Parameters
        ----------
        path : str, optional
            Path to configuration files (not used in toy model)
        keyword : str, optional
            Keyword for configuration selection (not used in toy model)
        """
        self.path = path
        self.keyword = keyword

    def load_configuration(self, parameters: dict, mediator: object) -> None:
        """
        Load configuration parameters for the toy model detector simulation.

        This method initializes various attributes of the Detector object
        using the provided parameters dictionary. It calculates the default
        detector pixel scale based on telescope parameters and handles
        wavelength-dependent detector properties by expanding single values
        to arrays when multiple wavelengths are specified.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including detector
            specifications and observational parameters
        mediator : ObservatoryMediator
            Mediator object providing access to telescope and observation parameters
        """
        parameters = parse_input.parse_parameters(parameters)

        # Calculate default detector pixel scale based on telescope diameter
        # Uses 0.5 * lambda/D at reference wavelength of 0.5 microns
        self.DEFAULT_CONFIG["pixscale_mas"] = (
            0.5
            * lambda_d_to_arcsec(
                1 * LAMBDA_D,
                0.5 * WAVELENGTH,
                mediator.get_telescope_parameter("diameter").to(LENGTH),
            )
        ).to(MAS)

        # For IFS, the default config won't work. It needs to be propagated at every wavelength.
        # Normalize list shapes just in case.
        array_params = [
            "npix_multiplier",
            "DC",
            "RN",
            "tread",
            "CIC",
            "QE",
            "dQE",
        ]

        self.DEFAULT_CONFIG.update(
            {
                key: parse_input.normalize_list_shapes(
                    self.DEFAULT_CONFIG,
                    key,
                    mediator.get_observation_parameter("nlambda"),
                )
                for key in array_params
            }
        )

        # Load parameters from user input, falling back to defaults if not provided
        utils.fill_parameters(self, parameters, self.DEFAULT_CONFIG, self.LOCKED_KEYS)
        utils.convert_to_numpy_array(self, array_params)


class EACDetector(Detector):
    """
    An EAC detector class that extends the base Detector class.

    This class represents a detector model that loads parameters from the EAC YAML
    configuration files through the module EACy, supporting both imaging and IFS
    (Integral Field Spectroscopy) observing modes.

    Parameters
    ----------
    path : str, optional
        Path to configuration files (not used in EAC detector)
    keyword : str, optional
        Keyword for configuration selection (not used in EAC detector)
    """

    # In EAC mode these quantities are OWNED by the YAML files and must stay
    # consistent with the loaded package. The user is NOT allowed to override
    # them; if they try, fill_parameters will warn and keep the YAML value.
    # Anything not listed here  remains user-editable.
    LOCKED_KEYS: set = {
        "npix_multiplier",
        "DC",
        "RN",
        "tread",
        "CIC",
        "QE",
        "dQE",
    }

    DEFAULT_CONFIG = {
        "pixscale_mas": None,  # Detector pixel scale in milliarcseconds.
        "npix_multiplier": 1
        * DIMENSIONLESS,  # Number of detector pixels per image plane "pixel".
        "DC": None,  # Dark current (counts pix^-1 s^-1, nlambda array)
        "RN": None,  # Read noise (counts pix^-1 read^-1, nlambda array)
        "tread": [1000] * READ_TIME,  # Read time (s, nlambda array) # TO ADD TO YAML
        "CIC": [0]
        * CLOCK_INDUCED_CHARGE,  # Clock-induced charge (counts pix^-1 photon_count^-1, nlambda array) # TO ADD TO YAML
        "QE": None,  # Quantum efficiency of detector
        "dQE": None,  # Effective QE due to degradation, cosmic ray effects, readout inefficiencies
    }

    def __init__(self, path: str = None, keyword: str = None):
        """
        Initialize an EACDetector instance.

        Parameters
        ----------
        path : str, optional
            Path to configuration files (not used in EAC detector)
        keyword : str, optional
            Keyword for configuration selection (not used in EAC detector)
        """
        self.path = path
        self.keyword = keyword

    def load_configuration(self, parameters: dict, mediator: object) -> None:
        """
        Load configuration parameters from the YAML files using EACy.

        This method initializes detector attributes using parameters from EAC YAML
        detector configuration files. It handles both IMAGER and IFS observing modes,
        loading appropriate detector characteristics including dark current, read noise,
        and quantum efficiency. The method automatically selects VIS or NIR detector
        parameters based on the observing wavelength.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including observing mode,
            detector specifications, and observational parameters
        mediator : ObservatoryMediator
            Mediator object providing access to observation, telescope, and coronagraph
            parameters including wavelength arrays and bandwidth specifications

        Raises
        ------
        AssertionError
            If the QE array contains NaN values after processing
        """
        parameters = parse_input.parse_parameters(parameters)

        from eacy import load_detector

        # ****** Update Default Config when necessary ******

        raw_detector_params = load_detector(
            mediator.get_observation_parameter("observing_mode")
        ).__dict__

        if mediator.get_observation_parameter("observing_mode") == "IMAGER":

            detector_params = utils.average_over_bandpass(
                raw_detector_params,
                mediator.get_observation_parameter("wavelength_range"),
            )

        else:  # can only be IFS
            detector_params = utils.interpolate_over_bandpass(
                raw_detector_params, mediator.get_observation_parameter("wavelength")
            )

        # scalar values projected to an array of length nlambda
        dc_arr = np.empty_like(mediator.get_observation_parameter("wavelength").value)
        dc_arr[mediator.get_observation_parameter("wavelength") < 1 * WAVELENGTH] = (
            detector_params["dc_vis"]
        )
        dc_arr[mediator.get_observation_parameter("wavelength") >= 1 * WAVELENGTH] = (
            detector_params["dc_nir"]
        )
        self.DEFAULT_CONFIG["DC"] = (
            dc_arr * DARK_CURRENT
        )  # Dark current (counts pix^-1 s^-1, nlambda array)

        # Dark current (counts pix^-1 s^-1, nlambda array)

        rn_arr = np.empty_like(mediator.get_observation_parameter("wavelength").value)
        rn_arr[mediator.get_observation_parameter("wavelength") < 1 * WAVELENGTH] = (
            detector_params["rn_vis"]
        )
        rn_arr[mediator.get_observation_parameter("wavelength") >= 1 * WAVELENGTH] = (
            detector_params["rn_nir"]
        )
        self.DEFAULT_CONFIG["RN"] = rn_arr * READ_NOISE

        # array values binned at wavelength points must just be stacked
        # combine the vis and nir qe arrays into a single array.
        qe_arr = np.empty_like(mediator.get_observation_parameter("wavelength").value)
        if parameters["observing_mode"] == "IMAGER":
            qe_arr[
                mediator.get_observation_parameter("wavelength") < 1 * WAVELENGTH
            ] = detector_params["qe_vis"]
            qe_arr[
                mediator.get_observation_parameter("wavelength") >= 1 * WAVELENGTH
            ] = detector_params["qe_nir"]
        else:  # IFS case
            qe_arr[
                mediator.get_observation_parameter("wavelength") < 1 * WAVELENGTH
            ] = detector_params["qe_vis"][
                mediator.get_observation_parameter("wavelength") < 1 * WAVELENGTH
            ]
            qe_arr[
                mediator.get_observation_parameter("wavelength") >= 1 * WAVELENGTH
            ] = detector_params["qe_nir"][
                mediator.get_observation_parameter("wavelength") >= 1 * WAVELENGTH
            ]
            # if qe_arr contains NaNs, then likely the wavelength range is outside of the qe range.
            # set the NaN values to zero
            qe_arr = np.nan_to_num(qe_arr)
            # make sure qe_arr does not contain NaNs
            assert ~np.isnan(np.sum(qe_arr)), "QE array contains NaN values"

        self.DEFAULT_CONFIG["QE"] = qe_arr * QUANTUM_EFFICIENCY

        dQE_arr = np.empty_like(mediator.get_observation_parameter("wavelength").value)

        # for now, hardcoded to 0.75 TODO change
        dQE_arr.fill(0.75)
        self.DEFAULT_CONFIG["dQE"] = dQE_arr * DIMENSIONLESS
        # self.DEFAULT_CONFIG["dQE"] = [
        #     0.75
        # ] * DIMENSIONLESS  # Effective QE due to degradation, cosmic ray effects, readout inefficiencies ## TO ADD TO YAML

        # Calculate default detector pixel scale based on telescope

        self.DEFAULT_CONFIG["pixscale_mas"] = (
            0.5
            * lambda_d_to_arcsec(
                1 * LAMBDA_D,
                0.5 * WAVELENGTH,
                mediator.get_telescope_parameter("diameter").to(LENGTH),
            )
        ).to(MAS)

        # fill in tread and CIC to match the length of the wavelength array
        # TODO read from YAML files
        self.DEFAULT_CONFIG["tread"] = (
            np.full_like(
                mediator.get_observation_parameter("wavelength").value,
                self.DEFAULT_CONFIG["tread"][0].value,
                dtype=np.float64,
            )
            * READ_TIME
        )
        self.DEFAULT_CONFIG["CIC"] = (
            np.full_like(
                mediator.get_observation_parameter("wavelength").value,
                self.DEFAULT_CONFIG["CIC"][0].value,
                dtype=np.float64,
            )
            * CLOCK_INDUCED_CHARGE
        )
        # Load parameters, use defaults if not provided
        utils.fill_parameters(
            self,
            parameters,
            self.DEFAULT_CONFIG,
            self.LOCKED_KEYS,
            allow_override=set(
                set(parameters.get("overrides", [])) & self.LOCKED_KEYS
            ),  # finds the keys that should be locked but that the user wants to override
            rebinned_wavelength=mediator.get_observation_parameter("wavelength").value,
        )

        # USED ONLY TO VALIDATE ETCs
        if "t_photon_count_input" in parameters.keys():

            self.t_photon_count_input = (
                parameters["t_photon_count_input"] * SECOND / FRAME
            )

        # USED ONLY TO VALIDATE ETCs
        if "det_npix_input" in parameters.keys():

            self.det_npix_input = parameters["det_npix_input"] * DIMENSIONLESS
