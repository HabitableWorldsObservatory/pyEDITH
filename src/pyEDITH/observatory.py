from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path
import numpy as np
import os
from .units import *
from . import utils
from pyEDITH import coronagraphs
from yippy import fetch_yip
from yippy import Coronagraph as yippycoro

import logging
from pyEDITH import detectors, parse_input, telescopes

logger = logging.getLogger("pyEDITH")


class Observatory(ABC):  # abstract class
    """
    Abstract base class for various astronomical observatories.

    This class defines the basic structure for modeling astronomical observatories
    used in performance calculations. It includes abstract methods that must be
    implemented by concrete subclasses and provides common functionality for
    calculating optical throughput, thermal properties, and component validation.

    Parameters
    ----------
    telescope : Telescope
        The telescope component of the observatory
    detector : Detector
        The detector component of the observatory
    coronagraph : Coronagraph
        The coronagraph component of the observatory
    optics_throughput : np.ndarray
        Optical throughput of the telescope and coronagraph system
    epswarmTrcold : np.ndarray
        Warm emissivity times cold transmission factor for thermal noise
    total_throughput : np.ndarray
        Combined optical and detector throughput
    observing_mode : str
        Observing mode ('IFS' or 'IMAGER')
    PRESETS : dict
        Dictionary of predefined observatory configurations
    TOY_MODEL_COMPONENTS : dict
        Default component definitions for toy model simulations

    """

    # Default presents for the currently implemented concepts
    PRESETS = {
        "ToyModel": {
            "telescope": "ToyModel",
            "coronagraph": "ToyModel",
            "detector": "ToyModel",
        },
        "EAC1": {
            "telescope": "EAC1",
            "coronagraph": "eac1_aavc_2d",  # will download from the database
            "detector": "EAC1",
        },
        "EAC5": {
            "telescope": "EAC5",
            "coronagraph": "eac1_aavc_2d",  # temporary, will be replaced by a more suitable one
            "detector": "EAC5",
        },
    }

    # Hardcoded registry for ToyModel since we do not expect it to be used much
    TOY_MODEL_COMPONENTS = {
        "telescopes": {"class": "ToyModelTelescope", "path": None},
        "coronagraphs": {"class": "ToyModelCoronagraph", "path": None},
        "detectors": {"class": "ToyModelDetector", "path": None},
    }

    def __init__(self):
        """
        Initialize an Observatory instance.

        This constructor initializes the basic components of the observatory
        (telescope, detector, and coronagraph) to None. These components
        should be properly initialized in subclasses using the `create_observatory` method.
        """

        self.telescope = None
        self.detector = None
        self.coronagraph = None

    def create_observatory(self, config: Union[str, dict]) -> object:
        """
        Create an observatory based on the given configuration.

        This method creates an Observatory instance with telescope, coronagraph,
        and detector components as specified in the configuration. The configuration
        can be either a preset name or a custom configuration dictionary.

        Preset usage:
            observatory = create_observatory("ToyModel")
            observatory = create_observatory("EAC5")

        Custom config dict:
            'telescope_type': 'EAC5' or 'ToyModel'
            'coronagraph_type': str or yippy.Coronagraph, path/keyword/pre-constructed object
            'detector_type': 'EAC5' or 'ToyModel'

        For remote coronagraphs, use yippy first:
            from yippy import fetch_yip
            coro_path = fetch_yip('remote_name')

        Parameters
        ----------
        config : Union[str,dict]
            Either a preset name or a custom configuration dictionary

        Returns
        -------
        object
            A configured Observatory object

        Raises
        ------
        ValueError
            If the config is invalid or a component is not found
        """
        # Load up the preset (config is a string with the preset name)
        if isinstance(config, str):
            if config not in Observatory.PRESETS:
                raise ValueError(
                    f"Unknown preset: {config}. "
                    f"Available presets: {list(Observatory.PRESETS.keys())}"
                )
            preset = Observatory.PRESETS[config]
            telescope = Observatory._create_telescope(preset["telescope"])
            coronagraph = Observatory._create_coronagraph(preset["coronagraph"])
            detector = Observatory._create_detector(preset["detector"])

        # Else, the user provided details in a config dictionary
        elif isinstance(config, dict):
            # Custom config mode
            required_keys = ["telescope", "coronagraph", "detector"]
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                raise ValueError(f"Config missing required keys: {missing_keys}")

            telescope = Observatory._create_telescope(config["telescope"])
            coronagraph = Observatory._create_coronagraph(config["coronagraph"])
            detector = Observatory._create_detector(config["detector"])

        else:
            raise ValueError("Invalid configuration.")

        self.telescope = telescope
        self.coronagraph = coronagraph
        self.detector = detector

        return

    @staticmethod
    def _create_telescope(keyword: str) -> object:
        """
        Create a telescope component from keyword.

        Parameters
        ----------
        keyword : str
            Telescope identifier ('ToyModel' or 'EAC*')

        Returns
        -------
        Telescope
            Instantiated telescope object

        Raises
        ------
        ValueError
            If the keyword is unknown
        """
        if keyword == "ToyModel":
            return telescopes.ToyModelTelescope()
        elif keyword.startswith("EAC"):
            return telescopes.EACTelescope(keyword=keyword)
        else:
            raise ValueError(
                f"Unknown telescope type: {keyword}. " f"Expected 'ToyModel' or 'EAC*'"
            )

    @staticmethod
    def _create_coronagraph(keyword: Union[str, object, Path]) -> object:
        """
        Create a coronagraph component from various input types.

        Priority order:
        0. If it's already a yippy Coronagraph object, use it directly
        1. If it's a valid path (exists on disk), load from path
        2. If YIP_CORO_DIR is set and YIP_CORO_DIR/keyword exists, use that
        3. Otherwise, attempt to fetch from remote database using yippy

        Parameters
        ----------
        keyword : str, Path or yippy.Coronagraph
            Can be:
            - 'ToyModel' for toy model coronagraph
            - Pre-constructed yippy Coronagraph object
            - Direct path to coronagraph folder
            - Keyword to search in YIP_CORO_DIR or fetch remotely

        Returns
        -------
        Coronagraph
            Instantiated coronagraph object

        Raises
        ------
        FileNotFoundError
            If the coronagraph cannot be found locally or fetched remotely

        """
        # Check if it is ToyModel
        if isinstance(keyword, str) and keyword == "ToyModel":
            return coronagraphs.ToyModelCoronagraph()

        ## YIP FILES NEEDED
        # Priority 0: Check if it's already a yippy Coronagraph object
        if isinstance(keyword, yippycoro):
            logger.info("Using pre-constructed yippy Coronagraph object")
            return coronagraphs.CoronagraphYIP(yippy_coro=keyword)

        # Priority 1: Check if it's a direct path that exists
        if Path(keyword).exists():
            logger.info(f"Using coronagraph from explicit path: {keyword}")
            return coronagraphs.CoronagraphYIP(path=Path(keyword))

        # Priority 2: Check if it exists in YIP_CORO_DIR
        yip_dir = os.environ.get("YIP_CORO_DIR")
        if yip_dir:
            candidate_path = Path(yip_dir, keyword)
            if os.path.exists(candidate_path):
                logger.info(f"Using coronagraph from YIP_CORO_DIR: {candidate_path}")
                return coronagraphs.CoronagraphYIP(path=Path(candidate_path))

        # Priority 3: Not found locally, attempt remote fetch
        logger.warning(
            f"Coronagraph '{keyword}' not found locally. "
            f"Attempting to fetch from remote database..."
        )

        try:
            fetched_path = fetch_yip(keyword)
            logger.info(f"Successfully downloaded coronagraph to: {fetched_path}")
            return coronagraphs.CoronagraphYIP(path=Path(fetched_path))

        except Exception as e:
            raise FileNotFoundError(
                f"Could not find or fetch coronagraph '{keyword}'.\n"
                f"Tried:\n"
                f"  1. Direct path: {keyword}\n"
                f"  2. YIP_CORO_DIR: {os.path.join(yip_dir, keyword) if yip_dir else '2. YIP_CORO_DIR Not set'}\n"
                f"  3. Remote fetch: Failed with error: {str(e)}\n\n"
                f"Solutions:\n"
                f"  - Provide a valid path to a local coronagraph\n"
                f"  - Set YIP_CORO_DIR environment variable and ensure coronagraph exists there\n"
                f"  - Check that the remote identifier is correct\n"
                f"  - Or provide a pre-constructed yippy.Coronagraph object"
            ) from e

    @staticmethod
    def _create_detector(keyword: str) -> object:
        """
        Create a detector component from keyword.

        Parameters
        ----------
        keyword : str
            Detector identifier ('ToyModel' or 'EAC*')

        Returns
        -------
        Detector
            Instantiated Detector object

        Raises
        ------
        ValueError
            If the keyword is unknown
        """
        if keyword == "ToyModel":
            return detectors.ToyModelDetector()
        elif keyword.startswith("EAC"):
            return detectors.EACDetector(keyword=keyword)
        else:
            raise ValueError(
                f"Unknown detector type: {keyword}. " f"Expected 'ToyModel' or 'EAC*'"
            )

    def load_configuration(
        self, parameters: dict, observation: object, scene: object
    ) -> None:
        """
        Load and configure all observatory components.

        This method initializes all observatory components (coronagraph, telescope,
        detector) with the provided parameters and calculates derived quantities
        like throughputs and thermal factors. Creates a mediator for component
        communication and sets the observing mode.

        Parameters
        ----------
        parameters : dict
            Configuration parameters dictionary containing observatory settings
        observation : Observation
            Observation object containing observational parameters
        scene : AstrophysicalScene
            Scene object containing target and environmental parameters
        """

        # Creates a mediator that picks selected variables from other classes
        mediator = ObservatoryMediator(self, observation, scene)

        self.coronagraph.load_configuration(parameters, mediator)
        self.telescope.load_configuration(parameters, mediator)
        self.detector.load_configuration(parameters, mediator)
        self.observing_mode = parameters["observing_mode"]  # IFS or IMAGER

        self.calculate_optics_throughput(parameters, mediator)
        self.calculate_warmemissivity_coldtransmission(parameters, mediator)
        self.calculate_total_throughput()

    def validate_configuration(self) -> None:
        """
        Validate that all observatory components and parameters are correctly configured.

        This method validates all sub-components (telescope, detector, coronagraph)
        and checks that required observatory-level attributes exist with correct
        types and units. Observatory-related parameters include total_throughput,
        optics_throughput, and epswarmTrcold.

        Raises
        ------
        AttributeError
            If required attributes are missing from the observatory or its components
        TypeError
            If an attribute has an incorrect type
        ValueError
            If a Quantity attribute has incorrect units
        """

        self.telescope.validate_configuration()
        self.detector.validate_configuration()
        self.coronagraph.validate_configuration()

        # Observatory-related args
        expected_args = {
            "total_throughput": QUANTUM_EFFICIENCY,
            "optics_throughput": DIMENSIONLESS,
            "epswarmTrcold": DIMENSIONLESS,
        }
        utils.validate_attributes(self, expected_args)
        # for arg, expected_unit in expected_args.items():
        #     if not hasattr(self, arg):
        #         raise AttributeError(f"Observatory is missing attribute: {arg}")
        #     value = getattr(self, arg)
        #     if not isinstance(value, u.Quantity):
        #         raise TypeError(f"Observatory attribute {arg} should be a Quantity")
        #     if not value.unit.is_equivalent(expected_unit):
        #         raise ValueError(
        #             f"Observatory attribute {arg} has incorrect units. Expected {expected_unit}, got {value.unit}"
        #         )

    def calculate_optics_throughput(self, parameters: dict, mediator: object) -> None:
        """
        Calculate the optical throughput of the observatory system.

        This method computes the optical throughput by either using a provided
        total optical throughput value (T_optical) from parameters, or by
        multiplying the telescope and coronagraph throughputs. For IFS mode,
        an additional IFS efficiency factor is applied. If optics_throughput is
        a scalar and wavelength array has multiple elements, the throughput is
        expanded to match the wavelength array length.

        Parameters
        ----------
        parameters : dict
            Configuration parameters dictionary that may contain 'T_optical',
            'observing_mode', and 'IFS_eff' keys
        mediator : ObservatoryMediator
            Mediator object providing access to observation parameters including
            wavelength array
        """
        parameters = parse_input.parse_parameters(parameters)

        if "T_optical" in parameters.keys():
            logger.info("Calculating optics_throughput from input...")
            rebinned = utils.resample_to_wavelength_grid(
                parameters["T_optical"] * DIMENSIONLESS,
                from_wavelength=parameters["wavelength"],  # input wavelength
                to_wavelength=mediator.get_observation_parameter("wavelength").value,
                name="T_optical",
                interpolation="1d",
            )  # signal to noise # nlambda array

            self.optics_throughput = rebinned * DIMENSIONLESS
        else:
            logger.info("Calculating optics throughput from preset...")
            self.optics_throughput = (
                self.telescope.telescope_optical_throughput
                * self.coronagraph.coronagraph_optical_throughput
            )

        if parameters["observing_mode"] == "IFS":
            # multiply by the IFS efficiency if in spectroscopy mode
            # NOTE: this is a placeholder for now. Not yet included in YAML files. Name will probably change.
            # may also move to elsewhere in code.
            ifs_eff = u.Quantity(parameters.get("IFS_eff", 1.0), unit=DIMENSIONLESS)

            # Rebin to proper wavelength grid
            ifs_eff = utils.resample_to_wavelength_grid(
                ifs_eff,
                from_wavelength=mediator.get_scene_parameter("_input_wavelength"),
                to_wavelength=mediator.get_observation_parameter("wavelength"),
                name="ifs_eff",
                interpolation="1d",
            )

            self.optics_throughput *= ifs_eff

        # if optics_throughput is a number and wavelength>1, make it an array of length nlambda
        if len(self.optics_throughput) == 1:
            self.optics_throughput = self.optics_throughput[0] * np.ones_like(
                mediator.get_observation_parameter("wavelength").value
            )

    def calculate_warmemissivity_coldtransmission(
        self, parameters: dict, mediator: object
    ) -> None:
        """
        Calculate the warm emissivity times cold transmission factor.

        This method computes the factor used for thermal noise calculations.
        It either uses a provided 'epswarmTrcold' value from parameters, or
        calculates it as (1 - optics_throughput).

        Parameters
        ----------
        parameters : dict
            Configuration parameters dictionary that may contain 'epswarmTrcold' key
        mediator : ObservatoryMediator
            Mediator object providing access to observation parameters including
            wavelength array
        """

        if "epswarmTrcold" in parameters.keys():
            logger.info("Calculating epswarmTrcold from input...")
            parameters = parse_input.parse_parameters(parameters)

            self.epswarmTrcold = parameters["epswarmTrcold"] * DIMENSIONLESS
        else:
            logger.info("Calculating epswarmTrcold as 1 - optics throughput...")
            self.epswarmTrcold = (
                np.ones_like(mediator.get_observation_parameter("wavelength").value)
                - self.optics_throughput
            )

    def calculate_total_throughput(self) -> None:
        """
        Calculate the total system throughput.

        This method computes the combined optical and detector throughput by
        multiplying the optics throughput with the detector quantum efficiency (QE),
        detector QE, and telescope contamination factor. This total throughput
        is used as a multiplicative factor in noise calculations.
        """

        self.total_throughput = (
            self.optics_throughput
            * self.detector.dQE
            * self.detector.QE
            * self.telescope.T_contamination
        )


class ObservatoryMediator:
    """
    Mediator class facilitating communication between observatory components.

    This class provides a centralized interface for accessing parameters from
    different components (observatory, observation, scene) without creating
    direct dependencies between them. It implements the mediator design pattern
    to decouple component interactions.

    Parameters
    ----------
    observatory : Observatory
        The observatory object
    observation : Observation
        The observation object containing observational parameters
    scene : AstrophysicalScene
        The scene object containing target and environmental parameters
    """

    def __init__(self, observatory: object, observation: object, scene: object):
        """
        Initialize the mediator with references to all major components.

        Parameters
        ----------
        observatory : Observatory
            The observatory object
        observation : Observation
            The observation object
        scene : AstrophysicalScene
            The scene object
        """

        self.observatory = observatory
        self.observation = observation
        self.scene = scene

    def get_telescope_parameter(self, param_name: str):
        """
        Retrieve a parameter from the telescope object.

        Parameters
        ----------
        param_name : str
            Name of the parameter to retrieve

        Returns
        -------
        Any or None
            The parameter value if it exists, None otherwise
        """
        return getattr(self.observatory.telescope, param_name, None)

    def get_coronagraph_parameter(self, param_name: str):
        """
        Retrieve a parameter from the coronagraph object.

        Parameters
        ----------
        param_name : str
            Name of the parameter to retrieve

        Returns
        -------
        Any or None
            The parameter value if it exists, None otherwise
        """
        return getattr(self.observatory.coronagraph, param_name, None)

    def get_detector_parameter(self, param_name: str):
        """
        Retrieve a parameter from the detector object.

        Parameters
        ----------
        param_name : str
            Name of the parameter to retrieve

        Returns
        -------
        Any or None
            The parameter value if it exists, None otherwise
        """

        return getattr(self.observatory.detector, param_name, None)

    def get_observation_parameter(self, param_name: str):
        """
        Retrieve a parameter from the observation object.

        Parameters
        ----------
        param_name : str
            Name of the parameter to retrieve

        Returns
        -------
        Any or None
            The parameter value if it exists, None otherwise
        """
        return getattr(self.observation, param_name, None)

    def get_scene_parameter(self, param_name: str):
        """
        Retrieve a parameter from the scene object.

        Parameters
        ----------
        param_name : str
            Name of the parameter to retrieve

        Returns
        -------
        Any or None
            The parameter value if it exists, None otherwise
        """
        return getattr(self.scene, param_name, None)
