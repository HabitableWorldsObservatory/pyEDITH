from abc import ABC, abstractmethod
import numpy as np
from . import utils
import astropy.units as u
from .units import *
from pyEDITH import parse_input
import copy


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
        # * DIMENSIONLESS,  # Optical throughput (nlambda array) [made up from EAC1-ish]
        "temperature": 290 * TEMPERATURE,
        "T_contamination": 0.95 * DIMENSIONLESS,
    }

    def __init__(self, path: str = None, keyword: str = "ToyModel"):
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
        self.DEFAULT_CONFIG = copy.deepcopy(self.DEFAULT_CONFIG)

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
    }

    DEFAULT_CONFIG = {
        "diameter": None,  # circumscribed diameter of aperture (m, scalar)
        "unobscured_area": 1.0,  # unobscured area (percentage,scalar) ### NOTE default for now
        "toverhead_fixed": 8.25e3
        * TIME,  # fixed overhead time (seconds,scalar) ### NOTE default for now
        "toverhead_multi": 1.1
        * DIMENSIONLESS,  # multiplicative overhead time (scalar) ### NOTE default for now
        "T_contamination": 1.0
        * DIMENSIONLESS,  # Effective throughput factor to budget for contamination; NOTE: missing from YAML files
        "temperature": None,  # (now a variable)
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
        self.DEFAULT_CONFIG = copy.deepcopy(self.DEFAULT_CONFIG)

    def load_configuration(self, parameters: dict, mediator: object) -> None:
        """
        Load configuration parameters from unified EAC configuration.

        This method initializes telescope attributes using the unified configuration
        from the Observatory (which handles hwome/eacy loading). It handles both
        IMAGER and IFS observing modes, loading appropriate telescope characteristics
        including diameter and optical throughput. For IMAGER mode, parameters are
        averaged over the specified wavelength range, while for IFS mode, parameters
        are interpolated onto the observation wavelength grid.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including observing mode
            and telescope specifications
        mediator : ObservatoryMediator
            Mediator object providing access to observation and coronagraph parameters

        Raises
        ------
        KeyError
            If the observing mode is not 'IMAGER' or 'IFS'
        """
        parameters = parse_input.parse_parameters(parameters)

        # Get unified EAC configuration from observatory
        eac_config = mediator.get_eac_configuration()

        # For EAC telescopes, configuration must be available
        if eac_config is None:
            raise RuntimeError(
                f"Failed to load EAC configuration for {self.keyword}. "
                f"Cannot proceed with telescope initialization."
            )

        # Extract telescope parameters
        self.DEFAULT_CONFIG["diameter"] = eac_config["diameter"] * LENGTH
        self.DEFAULT_CONFIG["temperature"] = eac_config["temperature"] * TEMPERATURE

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
