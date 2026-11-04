from abc import ABC, abstractmethod
import numpy as np
from . import utils
import astropy.units as u
from .units import *
from pyEDITH import parse_input


class Telescope(ABC):
    """
    A class representing a telescope for astronomical observations.

    This class provides methods to initialize and configure a telescope
    for simulating observations of exoplanets and their host stars.

    Parameters
    ----------
    diameter : float
        Circumscribed diameter of the telescope aperture in meters.
    Area : float
        Effective collecting area of the telescope in square meters.
    toverhead_fixed : float
        Fixed overhead time in seconds.
    toverhead_multi : float
        Multiplicative overhead time.
    telescope_optical_throughput : numpy.ndarray
        Array of throughput values.
    temperature : float
        Temperature of the warm optics.
    T_contamination : float
        Effective throughput factor to budget for contamination.
    """

    # Keys that a user is NOT allowed to override for this telescope mode.
    # Subclasses override this. An empty set means "everything is user-editable".
    LOCKED_KEYS: set = set()

    @abstractmethod
    def load_configuration(self):
        """
        Load configuration parameters for the telescope.

        This abstract method must be implemented by subclasses to define
        how telescope configuration parameters are loaded and processed.
        """
        pass  # pragma: no cover

    def validate_configuration(self):
        """
        Check that mandatory variables are present and have the correct format.

        This method validates that all required attributes exist on the telescope
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
            "diameter": LENGTH,
            "Area": LENGTH**2,
            "toverhead_fixed": TIME,
            "toverhead_multi": DIMENSIONLESS,
            "telescope_optical_throughput": DIMENSIONLESS,
            "temperature": TEMPERATURE,
            "T_contamination": DIMENSIONLESS,
        }

        utils.validate_attributes(self, expected_args)

        # for arg, expected_unit in expected_args.items():
        #     if not hasattr(self, arg):
        #         raise AttributeError(f"Telescope is missing attribute: {arg}")
        #     value = getattr(self, arg)
        #     if not isinstance(value, u.Quantity):
        #         raise TypeError(f"Telescope attribute {arg} should be a Quantity")
        #     if not value.unit == expected_unit:
        #         raise ValueError(
        #             f"Telescope attribute {arg} has incorrect units. Expected {expected_unit}, got {value.unit}"
        #         )


class ToyModelTelescope(Telescope):
    """
    A toy model telescope class that extends the base Telescope class.

    This class represents a simplified telescope model for use in simulations
    where users can specify telescope parameters manually rather than using
    predefined models from configuration files.

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
        "diameter": 7.87 * LENGTH,  # circumscribed diameter of aperture (m, scalar)
        "unobscured_area": (1.0 - 0.121),  # unobscured area (percentage,scalar)
        "toverhead_fixed": 8.25e3 * TIME,  # fixed overhead time (seconds,scalar)
        "toverhead_multi": 1.1 * DIMENSIONLESS,  # multiplicative overhead time (scalar)
        "telescope_optical_throughput": [0.823]
        * DIMENSIONLESS,  # Optical throughput (nlambda array) [made up from EAC1-ish]
        "temperature": 290 * TEMPERATURE,
        "T_contamination": 0.95 * DIMENSIONLESS,
    }

    def __init__(self, path: str = None, keyword: str = None):
        """
        Initialize a ToyModelTelescope instance.

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
        Load configuration parameters for the toy model telescope simulation.

        This method initializes various attributes of the Telescope object
        using the provided parameters dictionary or default values. It calculates
        the effective collecting area of the telescope based on the diameter
        and unobscured area parameters.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including telescope
            specifications and observational parameters
        mediator : ObservatoryMediator
            Mediator object providing access to other simulation components
        """
        parameters = parse_input.parse_parameters(parameters)

        # Load parameters, use defaults if not provided
        utils.fill_parameters(self, parameters, self.DEFAULT_CONFIG, self.LOCKED_KEYS)

        # Convert to numpy array when appropriate
        array_params = [
            "telescope_optical_throughput",
        ]
        # TODO rebin at new wavelength

        utils.convert_to_numpy_array(self, array_params)

        # Derived parameters
        # effective collecting area of telescope (m^2) # scalar
        self.Area = np.single(np.pi) / 4.0 * self.diameter**2.0 * self.unobscured_area


class EACTelescope(Telescope):
    """
    An EAC telescope class that extends the base Telescope class.

    This class represents a telescope model that loads parameters from EAC YAML
    configuration files through the module EACy, supporting both imaging and IFS
    (Integral Field Spectroscopy) observing modes.

    Parameters
    ----------
    path : str, optional
        Path to configuration files (not used directly)
    keyword : str
        Keyword identifying the specific telescope model to load from EACy files
    """

    # In EAC mode these quantities are OWNED by the YAML files and must stay
    # consistent with the loaded package. The user is NOT allowed to override
    # them; if they try, fill_parameters will warn and keep the YAML value.
    # Anything not listed here  remains user-editable.
    LOCKED_KEYS: set = {
        "diameter",
        "unobscured_area",
        "T_contamination",
        "temperature",
        "telescope_optical_throughput",
    }

    DEFAULT_CONFIG = {
        "diameter": None,  # circumscribed diameter of aperture (m, scalar)
        "unobscured_area": 1.0,  # unobscured area (percentage,scalar) ### NOTE default for now
        "toverhead_fixed": 8.25e3
        * TIME,  # fixed overhead time (seconds,scalar) ### NOTE default for now
        "toverhead_multi": 1.1
        * DIMENSIONLESS,  # multiplicative overhead time (scalar) ### NOTE default for now
        "telescope_optical_throughput": None,  # Optical throughput (nlambda array)
        "T_contamination": 1.0
        * DIMENSIONLESS,  # Effective throughput factor to budget for contamination; NOTE: missing from YAML files
        "temperature": 290
        * TEMPERATURE,  # Temperature of the warm optics; NOTE: missing from YAML files
    }

    def __init__(self, path: str = None, keyword: str = None):
        """
        Initialize an EACTelescope instance.

        Parameters
        ----------
        path : str, optional
            Path to configuration files (not used directly)
        keyword : str
            Keyword identifying the specific telescope model to load from EACy files
        """

        self.path = path
        self.keyword = keyword

    def load_configuration(self, parameters: dict, mediator: object) -> None:
        """
        Load configuration parameters from the YAML files using EACy.

        This method initializes telescope attributes using parameters from EAC YAML
        configuration files. It handles both IMAGER and IFS observing modes,
        loading appropriate telescope characteristics including diameter and optical
        throughput. For IMAGER mode, parameters are averaged over the specified
        wavelength range, while for IFS mode, parameters are interpolated onto
        the observation wavelength grid.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including observing mode
            and telescope specifications
        mediator : ObservatoryMediator
            Mediator object providing access to observation and coronagraph parameters
        """
        parameters = parse_input.parse_parameters(parameters)

        from eacy import load_telescope

        # **** LOAD DEFAULTS FROM EAC YAML FILES AND UPDATE DEFAULT CONFIG ****

        # Load parameters from YAML files
        telescope_params = load_telescope(self.keyword).__dict__
        if mediator.get_observation_parameter("observing_mode") == "IMAGER":

            telescope_params = utils.average_over_bandpass(
                telescope_params, mediator.get_observation_parameter("wavelength_range")
            )

        else:  # IFS case
            # interpolate telescope throughput onto native wavelength grid
            telescope_params = utils.interpolate_over_bandpass(
                telescope_params, mediator.get_observation_parameter("wavelength")
            )

        # Load parameters that you need from the YAML files
        self.DEFAULT_CONFIG["diameter"] = telescope_params["diam_circ"] * LENGTH

        # Ensure telescope_optical_throughput has dimensions nlambda
        if np.isscalar(telescope_params["total_tele_refl"]):
            self.DEFAULT_CONFIG["telescope_optical_throughput"] = (
                np.array([telescope_params["total_tele_refl"]]) * DIMENSIONLESS
            )
        else:
            self.DEFAULT_CONFIG["telescope_optical_throughput"] = (
                np.array(telescope_params["total_tele_refl"]) * DIMENSIONLESS
            )
        # ****** Update Default Config when necessary ******
        # TODO: wavelength_range probably should not depend on the coronagraph bandwidth; let's discuss
        # the coronagraph module needs the telescope module to be initialized first to get the telescope diameter

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

        # Derived parameters
        # effective collecting area of telescope (m^2) # scalar
        self.Area = np.single(np.pi) / 4.0 * self.diameter**2.0 * self.unobscured_area
