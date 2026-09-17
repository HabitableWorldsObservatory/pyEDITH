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

# Effective QE due to degradation, cosmic ray effects, and readout
# inefficiencies. Not yet tracked by eacy or hwome engineering data;
# this is a placeholder until real per-channel values are available.
# TODO: replace with real values once available from eacy/hwome.
DEFAULT_DQE = 0.75


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
            keyword = preset["telescope"]
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
            keyword = config["telescope"]
        else:
            raise ValueError("Invalid configuration.")

        self.telescope = telescope
        self.coronagraph = coronagraph
        self.detector = detector

        # Load unified EAC configuration (from hwome or eacy)
        if keyword.startswith("EAC"):
            self.configuration = Observatory._load_eac_configuration(keyword)
        else:
            self.configuration = None
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

    @staticmethod
    def _load_eac_configuration(eac_keyword):
        """
        Load EAC configuration from hwome (EAC4-6) or eacy (EAC1-3).

        This is the single source of truth for EAC configuration loading.
        It attempts to use hwome for EAC4-6, falls back to eacy if needed,
        and uses eacy directly for EAC1-3. Returns a unified configuration
        format regardless of source.

        Parameters
        ----------
        eac_keyword : str
            EAC configuration name (e.g., "EAC1", "EAC5", "EAC6")

        Returns
        -------
        dict
            Unified configuration dictionary in hwome-like format, or None if loading fails
        """
        # For EAC4-6, try hwome first
        if eac_keyword in ["EAC4", "EAC5", "EAC6"]:
            try:
                config = Observatory._ingest_from_hwome(eac_name=eac_keyword.lower())
                logger.info(f"Loaded {eac_keyword} configuration from hwome")
                return config
            except ImportError as e:
                logger.warning(
                    f"hwome not available ({e}). Falling back to eacy for {eac_keyword}."
                )
            except Exception as e:
                logger.warning(
                    f"Could not load {eac_keyword} from hwome: {e}. "
                    f"Falling back to eacy."
                )

        # Use eacy (either for EAC1-3 or as fallback for EAC4-6)
        try:
            logger.info(f"Loading {eac_keyword} configuration from eacy")
            config = Observatory._convert_eacy_to_unified_format(eac_keyword)
            return config
        except Exception as e:
            logger.error(f"Failed to load {eac_keyword} configuration from eacy: {e}")
            return None

    @staticmethod
    def _convert_eacy_to_unified_format(eac_keyword):
        """
        Convert eacy data to unified hwome-like configuration format, for
        BOTH observing modes (IMAGER and IFS), matching how hwome-derived
        configs are structured.

        eacy's detector QE/RN/DC genuinely differ between IMAGER and IFS mode
        (different YAML blocks -- see eacy's DETECTOR.load_imager vs
        DETECTOR.load_IFS), so load_detector() is called once per mode here
        to build two distinct mode configs. Telescope and instrument optics
        throughput, however, do NOT depend on mode in eacy (single curve from
        load_telescope/load_instrument) -- that same curve is reused for both
        mode configs below, sliced per-channel to match each mode's QE domain.

        Preserves eacy's native vis/nir split as two distinct channels
        ("vis", "nir") rather than concatenating them into one curve.

        Parameters
        ----------
        eac_keyword : str
            EAC configuration keyword (e.g., "EAC1", "EAC5")

        Returns
        -------
        dict
            Configuration with keys:
            - diameter : float
            - temperature : float (hardcoded default for eacy)
            - IMAGER : dict with "vis"/"nir" channel data (from load_detector("IMAGER"))
            - IFS : dict with "vis"/"nir" channel data (from load_detector("IFS"))
        """
        from eacy import load_telescope, load_detector, load_instrument

        telescope = load_telescope(eac_keyword).__dict__
        instrument = load_instrument("CI").__dict__

        unified_config = {
            "diameter": telescope["diam_circ"],
            "temperature": 290.0,  # Default for eacy (not in eacy data)
        }

        # telescope["lam"] and detector["lam"] are the same array (eacy's shared
        # internal_lam grid) -- one wavelength array serves both optics
        # throughput and QE, for both modes.
        wavelengths_um = (telescope["lam"]).to(WAVELENGTH).value

        telescope_throughput = np.asarray(telescope["total_tele_refl"])
        instrument_throughput = np.asarray(instrument["total_inst_refl"])
        combined_throughput = telescope_throughput * instrument_throughput

        for obs_mode in ("IMAGER", "IFS"):
            detector = load_detector(obs_mode).__dict__

            qe_vis = np.atleast_1d(detector["qe_vis"])
            qe_nir = np.atleast_1d(detector["qe_nir"])

            # eacy's interp_arr fills out-of-domain wavelengths with NaN, and
            # the subsequent np.clip(..., 0, None) preserves NaN (np.clip is
            # built on np.maximum/np.minimum, which propagate NaN). This is
            # how each channel's native wavelength domain is recovered here.
            vis_mask = np.isfinite(qe_vis)
            nir_mask = np.isfinite(qe_nir)

            if not vis_mask.any() or not nir_mask.any():
                raise ValueError(
                    f"Could not determine distinct vis/nir wavelength domains "
                    f"for {eac_keyword} ({obs_mode}). qe_vis or qe_nir has no "
                    f"finite values -- check eacy's NaN-padding convention."
                )

            mode_config = {
                "optics_throughput": {
                    "vis": {
                        "wavelength": list(wavelengths_um[vis_mask]),
                        "throughput": list(combined_throughput[vis_mask]),
                    },
                    "nir": {
                        "wavelength": list(wavelengths_um[nir_mask]),
                        "throughput": list(combined_throughput[nir_mask]),
                    },
                },
                "qe": {
                    "vis": {
                        "wavelength": list(wavelengths_um[vis_mask]),
                        "qe": list(qe_vis[vis_mask]),
                    },
                    "nir": {
                        "wavelength": list(wavelengths_um[nir_mask]),
                        "qe": list(qe_nir[nir_mask]),
                    },
                },
                "dqe": {
                    "vis": {
                        "wavelength": list(wavelengths_um[vis_mask]),
                        "qe": np.ones_like(qe_vis[vis_mask]) * DEFAULT_DQE,
                    },
                    "nir": {
                        "wavelength": list(wavelengths_um[nir_mask]),
                        "qe": np.ones_like(qe_nir[nir_mask]) * DEFAULT_DQE,
                    },
                },
                "dc": {
                    "vis": float(detector["dc_vis"]),
                    "nir": float(detector["dc_nir"]),
                },
                "rn": {
                    "vis": float(detector["rn_vis"]),
                    "nir": float(detector["rn_nir"]),
                },
                "cic": {
                    "vis": 0.0,  # eacy's cic_vis/cic_nir are always None ("NOT IMPLEMENTED YET")
                    "nir": 0.0,
                },
                "pixscale_mas": {"vis": None, "nir": None},  # filled later
            }

            unified_config[obs_mode] = mode_config

        return unified_config

    @staticmethod
    def _ingest_from_hwome(eac_name="eac5"):
        """
        Load configuration data from hwome for a given EAC configuration.

        This method replaces the deprecated eacy package with hwome for loading
        the most up-to-date telescope, detector, and instrument configurations.

        Prerequisites
        -------------
        Requires installation of: hwome_data, hwome-roam, hwome_core
        Environment variable HWOME_DATA_PATH must be set to the hwome_data directory.

        Parameters
        ----------
        eac_name : str, optional
            Name of the EAC configuration to load (e.g., "eac5", "eac1").
            Default is "eac5".

        Returns
        -------
        dict
            Configuration dictionary containing:
            - diameter : float
                Telescope circumscribed diameter in meters
            - temperature : float
                Median temperature of the optical path in Kelvin
            - IMAGER : dict
                Configuration for imaging mode with wavelength-dependent parameters
            - IFS : dict
                Configuration for spectroscopy mode with wavelength-dependent parameters

        Notes
        -----
        The returned configuration is structured to be compatible with the existing
        pyEDITH telescope, detector, and coronagraph loading mechanisms.
        """

        from hwome.roam.analyzer import Analyzer
        from hwome.core.navigator import search_configuration

        system = Analyzer()
        system.load_configuration(f"{eac_name}.yaml")

        nav = (
            system.system.Mask
        )  # generic -we could iterate all the masks, but we only need one since they don't have real values (throughput 0.99)
        mask_name = next(iter(nav.name.values())).value

        hwo_configuration = {}

        # Physical parameters (telescope-level)
        hwo_configuration["diameter"] = float(
            system.system.Telescope.circumscribing_diameter.q.to("m").value
        )
        optical_path = system.system.OpticalPath.select(
            Instrument="CI",
            Channel="CI_VIS_IFS",
            Filter="CI_4F874",
            Mask=mask_name,
        )
        hwo_configuration["temperature"] = float(np.median(optical_path.temperature.v))

        # Load configuration for each observation type
        for observation_type in ["IMAGER", "IFS"]:
            configuration_by_obs = {}
            obs_type = "di" if observation_type == "IMAGER" else "ifs"

            # Search for available channels matching this observation type
            options = search_configuration(
                channel_type="cg_" + obs_type,
                wavelength_range_nm=[400, 1800],
                center_nm=None,
            )

            # Initialize parameter dictionaries for each channel
            configuration_by_obs["optics_throughput"] = {}
            configuration_by_obs["pixscale_mas"] = {}
            configuration_by_obs["dc"] = {}
            configuration_by_obs["rn"] = {}
            configuration_by_obs["cic"] = {}
            configuration_by_obs["qe"] = {}
            configuration_by_obs["dqe"] = {}

            # Process each channel
            for chan_name, cdict in options.items():

                # Wavelength-dependent arrays
                channel_throughput = []
                channel_wavelength = []
                channel_qe = []

                # Iterate through filters in this channel
                for filt_name, fdict in cdict.items():
                    parts = fdict["path"].split(".")

                    optical_path = system.system.OpticalPath.select(
                        Instrument=parts[0],
                        Channel=parts[1],
                        Filter=parts[-1],
                        Mask=mask_name,
                    )
                    tp = optical_path.throughput(include_detector=True)

                    # Separate optical throughput from detector QE
                    optical_throughput = np.prod(
                        tp.v[:-1], axis=0
                    )  # Use [:-1] to not include the detector QE which is always last
                    qe = tp.v[-1]

                    # Apply wavelength mask for this filter
                    mask = (tp.w > (fdict["center"] - fdict["width"] / 2)) & (
                        tp.w < (fdict["center"] + fdict["width"] / 2)
                    )

                    channel_throughput.extend(list(optical_throughput[mask]))
                    channel_wavelength.extend(list(tp.w.value[mask]))
                    channel_qe.extend(list(qe[mask]))

                # Sort by wavelength and apply same sorting to all arrays
                sorted_indices = np.argsort(channel_wavelength)
                channel_wavelength = [channel_wavelength[i] for i in sorted_indices]
                channel_throughput = [channel_throughput[i] for i in sorted_indices]
                channel_qe = [channel_qe[i] for i in sorted_indices]

                # Store wavelength-dependent parameters
                configuration_by_obs["optics_throughput"][chan_name] = {
                    "wavelength": ((channel_wavelength * u.nm).to(WAVELENGTH)).value,
                    "throughput": channel_throughput,
                }
                configuration_by_obs["qe"][chan_name] = {
                    "wavelength": ((channel_wavelength * u.nm).to(WAVELENGTH)).value,
                    "qe": channel_qe,
                }

                configuration_by_obs["dqe"][chan_name] = {
                    "wavelength": ((channel_wavelength * u.nm).to(WAVELENGTH)).value,
                    "dqe": np.ones_like(channel_qe) * DEFAULT_DQE,
                }  # hardcoded for now

                # Retrieve scalar parameters for this channel
                nav_chan = system.resolve("CI." + chan_name)

                configuration_by_obs["pixscale_mas"][chan_name] = float(
                    (
                        nav_chan.Detector.pixel_pitch.v
                        / nav_chan.focal_length.v
                        * u.radian
                    )
                    .to(u.mas)
                    .value
                )
                configuration_by_obs["dc"][chan_name] = float(
                    nav_chan.Detector.dark_current.v
                )  # ct/px/s
                configuration_by_obs["rn"][chan_name] = float(
                    nav_chan.Detector.read_noise.v
                )
                configuration_by_obs["cic"][chan_name] = float(
                    nav_chan.Detector.cic.v
                )  # ct/px

            hwo_configuration[observation_type] = configuration_by_obs

        return hwo_configuration

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

        # Select active channel
        if self.configuration is not None:
            obs_mode = parameters["observing_mode"]
            self.active_channel = Observatory._select_active_channel(
                self.configuration[obs_mode], observation.wavelength_range
            )
            if self.active_channel is not None:
                utils.rebin_channel_curves_to_grid(
                    self.configuration[obs_mode],
                    self.active_channel,
                    curve_keys=[
                        "qe",
                        "dqe",
                        "optics_throughput",
                    ],  # TODO extend to ["dc", "rn", "cic"] once those
                    # become per-wavelength arrays in the YAML
                    to_wavelength=observation.wavelength.value,
                    to_delta_wavelength=(
                        observation.delta_wavelength.value
                        if observation.delta_wavelength is not None
                        else None
                    ),
                    interpolation=(
                        "Gaussian" if observation.delta_wavelength is not None else "1d"
                    ),
                )
        else:
            self.active_channel = None

        self.telescope.load_configuration(parameters, mediator)
        self.coronagraph.load_configuration(parameters, mediator)
        self.detector.load_configuration(parameters, mediator)
        self.observing_mode = parameters["observing_mode"]  # IFS or IMAGER

        self.calculate_optics_throughput(parameters, mediator)
        self.calculate_warmemissivity_coldtransmission(parameters, mediator)
        self.calculate_total_throughput()

    @staticmethod
    def _select_active_channel(
        mode_config, wavelength_range, quantity_key="optics_throughput"
    ):
        """
        Determine which single channel's engineering data applies to filter_obj.

        Parameters
        ----------
        mode_config : dict
            The per-mode unified configuration dict (e.g. eac_config["IFS"]),
            containing at least mode_config[quantity_key] = {chan_name: {...}}.
        wavelength_range : np.array
            The active observation wavelength range.
        quantity_key : str, optional
            Which quantity's channel dict to use for the domain check
            (default "optics_throughput"). Assumes all quantities
            (optics_throughput, qe, dc, rn, cic) share the same channel names
            for a given mode_config -- not yet independently validated.

        Returns
        -------
        str
            The selected channel name.

        Raises
        ------
        ValueError
            If filter_obj.channel is set but not present in mode_config, or if
            no channel (or more than one channel) fully contains the filter's
            wavelength range.
        """
        available_channels = list(mode_config[quantity_key].keys())

        domain_matches = []
        for chan_name, chan_data in mode_config[quantity_key].items():
            wl = np.asarray(
                (chan_data["wavelength"] * u.um).to(WAVELENGTH)
            )  # in case WAVELENGTH ever changes
            if wl.size == 0:
                continue
            if (
                wavelength_range[0].value >= wl.min()
                and wavelength_range[1].value <= wl.max()
            ):
                domain_matches.append(chan_name)

        if not domain_matches:
            raise ValueError(
                f"Wavelength range does not fall "
                f"entirely within any channel's native wavelength domain: "
                f"{available_channels}."
            )
        if len(domain_matches) > 1:
            raise ValueError(
                f"Wavelength range spans multiple "
                f"channels {domain_matches}. Please define separate filters, "
                f"each contained within a single instrument channel."
            )
        return domain_matches[0]

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
            "total_throughput": QUANTUM_EFFICIENCY,  # technically, already multiplied by QE terms TODO extrapolate terms
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
            self.optics_throughput = parameters["T_optical"] * DIMENSIONLESS
        else:
            logger.info("Calculating optics throughput from EAC YAML files...")
            eac_config = mediator.get_eac_configuration()
            if eac_config is not None:
                obs_mode = mediator.get_observation_parameter("observing_mode")
                mode_config = eac_config[obs_mode]
                active_channel = mediator.get_active_channel()

                self.optics_throughput = (
                    np.asarray(mode_config["optics_throughput"][active_channel])
                    * DIMENSIONLESS
                )
            else:
                raise ValueError(
                    "Could not calculate optics throughput from the YAML files."
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

    def get_eac_configuration(self):
        """
        Retrieve the EAC configuration from the observatory.

        This returns the unified configuration dictionary that may have been
        loaded from hwome (EAC4-6) or eacy (EAC1-3), in a consistent format.

        Returns
        -------
        dict or None
            The EAC configuration dictionary if available, None otherwise
        """
        return getattr(self.observatory, "configuration", None)

    def get_active_channel(self):
        """
        Retrieve the currently selected instrument channel name for the
        active filter, as resolved once by Observatory.load_configuration.

        Returns
        -------
        str or None
            The active channel name, or None if not using an EAC configuration.
        """
        return getattr(self.observatory, "active_channel", None)
