from abc import ABC, abstractmethod
import numpy as np
from . import utils
from astropy import units as u
from .units import *
from scipy.interpolate import interp1d
from yippy import Coronagraph as yippycoro
from lod_unit import lod
import logging
from pyEDITH import parse_input
import copy

logger = logging.getLogger("pyEDITH")


def generate_radii(numx: int, numy: int = 0) -> np.ndarray:
    """
    Generate a 2D distribution of radii from the center of a matrix.

    This function creates a 2D numpy array representing the radial distance
    from the center for each element. The origin is assumed to be in the
    center of the matrix. Radii are calculated assuming 1 pixel = 1 unit.
    The number of pixels can be odd or even.

    Parameters
    -----------
    numx : int
        Number of pixels in the x direction.
    numy : int, optional
        Number of pixels in the y direction. If 0 (default), it is set equal to numx.

    Returns
    --------
    np.ndarray
        A 2D numpy array of shape (numy, numx) containing the radial distances
        from the center for each pixel.

    Note
    ----
    - The function handles both odd and even dimensions.
    - Formerly  in rgen.pro
    """

    xo2 = int(numx / 2)  # if it is odd, gets the int (lower value)

    if numy == 0:
        numy = numx
    yo2 = int(numy / 2)

    xn1 = int(xo2)
    # xn2 = int(xo2-1)
    yn1 = int(yo2)
    # yn2 = int(yo2-1)

    oddx = 0
    oddy = 0
    if (xo2 + xo2) < numx:
        oddx = 1
        # xn2 = xo2
        xo2 += 1

    if (yo2 + yo2) < numy:
        oddy = 1
        # yn2 = yo2
        yo2 += 1

    xa = np.arange(0, xo2) + 0.5
    ya = np.arange(0, yo2) + 0.5
    if oddx == 1:
        xa -= 0.5
    if oddy == 1:
        ya -= 0.5

    xb = np.array([xa for i in np.arange(numy)])

    yb = np.array([ya for i in np.arange(numx)]).T  # rotate by 90 deg CCW

    # FLIPPED BECAUSE IDL WORKS WITH COLUMNS X ROWS AND PYTHON THE OPPOSITE.
    # TODO consider changing this

    x = np.zeros((numy, numx))

    x[:, int(xn1) : int(numx)] = xb

    if oddx == 1:
        x[:, 0 : int(xn1) + 1] = -np.flip(xb, axis=1)
    else:
        x[:, 0 : int(xn1)] = -np.flip(xb, axis=1)

    y = np.zeros((numy, numx))
    y[int(yn1) : int(numy), :] = yb

    if oddy == 1:
        y[0 : int(yn1) + 1, :] = -np.flip(yb, axis=0)
    else:
        y[0 : int(yn1), :] = -np.flip(yb, axis=0)

    r = np.sqrt(x * x + y * y)
    return r


class Coronagraph(ABC):
    """
    A base class representing a generic coronagraph.

    This class defines the basic structure and methods common to all coronagraphs.
    Specific coronagraph models should inherit from this class and implement
    their own `generate_secondary_parameters` method.
    Right now, there are two coronagraph sub-classes:

        -> **ToyModelCoronagraph**
        This is a simplistic coronagraph setup where the user can specify all
        coronagraph parameters. Use this for testing coronagraph parameters
        not defined by a Yield Input Package (files containing models of
        realistic coronagraph responses).

        -> **CoronagraphYIP**
        This is a coronagraph setup that is defined by a Yield Input Package (YIP),
        which contains models of realistic coronagraph responses. User this for
        testing specific coronagraph cases. Requires a path to a YIP.

    Parameters
    -----------
    Istar : np.ndarray
        Star intensity distribution.
    noisefloor : np.ndarray
        Noise floor of the coronagraph.
    photometric_aperture_throughput : np.ndarray
        Photometric aperture throughput.
    omega_lod : np.ndarray
        Solid angle of the photometric aperture in λ/D units.
    skytrans : np.ndarray
        Sky transmission.
    pixscale : float
        Pixel scale in λ/D units.
    npix : int
        Number of pixels in the image.
    xcenter : float
        X-coordinate of the image center.
    ycenter : float
        Y-coordinate of the image center.
    coronagraph_bandwidth : float
        Bandwidth of the coronagraph.
    stellar_angular_diameter : np.ndarray
        Angular diameters of the target objects.
    photometric_aperture_radius : float
        Photometric aperture radius (in units of lambda/D).
    psf_trunc_ratio : np.ndarray
        PSF truncation ratio.
    npsfratios : int
        Number of PSF ratios.
    nrolls : int
        Number of roll angles.
    """

    # Keys that a user is NOT allowed to override for this coronagraph mode.
    # Subclasses override this. An empty set means "everything is user-editable".
    LOCKED_KEYS: set = set()

    @abstractmethod
    def load_configuration(self) -> None:  # pragma: no cover
        """
        Load configuration parameters for the coronagraph.

        This abstract method must be implemented by subclasses to define
        how coronagraph configuration parameters are loaded and processed.
        """
        pass

    def validate_configuration(self) -> None:
        """
        Check that mandatory variables are present and have the correct format.

        This method validates that all required attributes exist on the coronagraph
        object and that they have the expected types and units.  It specifically
        verifies that either photometric_aperture_radius or psf_trunc_ratio is
        defined but not necessarily both.
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
            "Istar": DIMENSIONLESS,  # contrast
            "noisefloor": DIMENSIONLESS,  # contrast
            "photometric_aperture_throughput": DIMENSIONLESS,  # throughput
            "omega_lod": LAMBDA_D**2,
            "skytrans": DIMENSIONLESS,  # throughput
            "pixscale": LAMBDA_D,
            "npix": int,
            "xcenter": PIXEL,
            "ycenter": PIXEL,
            "coronagraph_bandwidth": float,
            "npsfratios": int,
            "nrolls": int,
            "coronagraph_spectral_resolution": DIMENSIONLESS,
        }

        # Either photometric_aperture_radius or psf_trunc_ratio must be present
        if (
            hasattr(self, "photometric_aperture_radius")
            and getattr(self, "photometric_aperture_radius") is not None
        ):
            expected_args["photometric_aperture_radius"] = LAMBDA_D
        elif (
            hasattr(self, "psf_trunc_ratio")
            and getattr(self, "psf_trunc_ratio") is not None
        ):
            expected_args["psf_trunc_ratio"] = DIMENSIONLESS
        else:
            raise AttributeError(
                "Coronagraph must have either 'photometric_aperture_radius' or 'psf_trunc_ratio' attribute."
            )
        utils.validate_attributes(self, expected_args)


class ToyModelCoronagraph(Coronagraph):
    """
    A toy model coronagraph class that extends the base Coronagraph class.

    This class implements a simplified coronagraph model with basic functionality
    for generating secondary parameters and setting up coronagraph characteristics.
    It allows users to specify all coronagraph parameters manually rather than
    using predefined models from Yield Input Packages.

    Parameters
    ----------
    path : str, optional
        Path to configuration files (not used in toy model)
    """

    # In toy-model mode EVERY parameter is user-editable, so nothing is locked.
    LOCKED_KEYS: set = set()

    DEFAULT_CONFIG = {
        "pixscale": 0.25 * LAMBDA_D,
        "contrast": 1.05e-13
        * DIMENSIONLESS,  # noise floor contrast of coronagraph (uniform over dark hole and unitless)
        "noisefloor_factor": 0.03
        * DIMENSIONLESS,  #  1 sigma systematic noise floor expressed as a multiplicative factor to the contrast (unitless)
        "coronagraph_bandwidth": 0.2,  # fractional bandwidth of coronagraph (unitless)
        "photometric_aperture_radius": 0.85 * LAMBDA_D,
        "Tcore": 0.2968371
        * DIMENSIONLESS,  # core throughput of coronagraph (uniform over dark hole, unitless, scalar)
        "TLyot": 0.65 * DIMENSIONLESS,  # Lyot transmission of the coronagraph
        "nrolls": 1,  # number of rolls
        # * DIMENSIONLESS,  # Coronagraph throughput [made up from EAC1-ish]
        "coronagraph_spectral_resolution": 1
        * DIMENSIONLESS,  # Set to default. It is used to limit the bandwidth if the coronagraph has a specific spectral window.
    }

    def __init__(self, path: str = None):
        """
        Initialize a ToyModelCoronagraph instance.

        Parameters
        ----------
        path : str, optional
            Path to configuration files (not used in toy model)
        """
        self.path = path
        self.DEFAULT_CONFIG = copy.deepcopy(self.DEFAULT_CONFIG)

    def load_configuration(self, parameters: dict, mediator: object) -> None:
        """
        Load configuration parameters for the toy model coronagraph simulation.

        This method sets up all coronagraph parameters using either user-provided
        values or default configurations. It calculates derived parameters such as
        pixel coordinates, radial separations, and throughput maps based on the
        specified inner and outer working angles.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including coronagraph
            specifications, observational parameters, and noise characteristics
        mediator : ObservatoryMediator
            Mediator object providing access to observation and scene parameters
        """
        parameters = parse_input.parse_parameters(parameters)
        # Load parameters, use defaults if not provided
        utils.fill_parameters(
            self, parameters, self.DEFAULT_CONFIG, locked_keys=self.LOCKED_KEYS
        )

        # # Convert to numpy array when appropriate
        # utils.convert_to_numpy_array(self, array_params)

        # Derived parameters
        self.npsfratios = 1
        self.npix = int(2 * 60 / self.pixscale)
        self.xcenter = self.npix / 2.0 * PIXEL
        self.ycenter = self.npix / 2.0 * PIXEL

        self.r = (
            generate_radii(self.npix, self.npix) * self.pixscale
        )  # create an array of circumstellar separations in units of lambd/D centered on star

        self.omega_lod = (
            np.full(
                (self.npix, self.npix, self.npsfratios),
                float(np.pi) * self.photometric_aperture_radius**2,
            )
            * (self.photometric_aperture_radius.unit) ** 2
        )  # size of photometric aperture at all separations (npix,npix,len(psftruncratio))

        self.skytrans = (
            np.full((self.npix, self.npix), self.TLyot) * self.TLyot.unit
        )  # skytrans at all separations

        self.photometric_aperture_throughput = (
            np.full((self.npix, self.npix, self.npsfratios), self.Tcore)
            * self.Tcore.unit
        )  # core throughput at all separations (npix,npix,len(psftruncratio))
        # j = np.where(r lt self.IWA or r gt self.OWA)
        # find separations interior to IWA or exterior to OWA
        # if j[0] ne -1 then photometric_aperture_throughput1[j] = 0.0

        # put in the right dimensions (3d arrays), but third dimension
        # is 1 (number of psf_trunc_ratio)
        # self.omega_lod = np.array([self.omega_lod])
        # self.photometric_aperture_throughput = np.array([self.photometric_aperture_throughput])

        self.PSFpeak = (
            0.025 * self.TLyot
        )  # this is an approximation based on PAPLC results

        """
        OLD VERSION
        Istar = np.full((self.npix, self.npix), self.contrast * self.PSFpeak)
        self.Istar = (
            np.zeros((self.npix, self.npix, len(self.angdiams))) * DIMENSIONLESS
        )
        for z in range(len(self.angdiams)):
            self.Istar[:, :, z] = Istar

        self.noisefloor = (
            self.noisefloor_factor * self.contrast
        )  # 1 sigma systematic noise floor expressed as a contrast (uniform over dark hole and unitless) # scalar
        n_floor = np.full((self.npix, self.npix), self.noisefloor * self.PSFpeak)
        self.noisefloor = (
            np.zeros((self.npix, self.npix, len(self.angdiams))) * DIMENSIONLESS
        )
        for z in range(len(self.angdiams)):
            self.noisefloor[:, :, z] = n_floor
        """
        self.Istar = (
            np.full((self.npix, self.npix), self.contrast * self.PSFpeak)
            * DIMENSIONLESS
        )

        if "noisefloor_factor" not in parameters.keys():
            if "noisefloor_PPF" in parameters.keys():
                raise KeyError(
                    "Noisefloor_PPF mode not implemented in ToyModel coronagraph. Please use the noisefloor_factor method."
                )
            logger.warning(
                "noisefloor_factor value not provided. Using the default value: "
                + str(self.DEFAULT_CONFIG["noisefloor_factor"])
            )

        logger.info(
            "Calculating noisefloor by multiplying noisefloor_factor="
            + str(self.noisefloor_factor)
            + ", contrast="
            + str(self.contrast)
            + ", PSFpeak="
            + str(self.PSFpeak)
        )
        # 1 sigma systematic noise floor expressed as a contrast (uniform over dark hole and unitless) * PSF peak # scalar
        self.noisefloor = (
            np.full(
                (self.npix, self.npix),
                self.noisefloor_factor * self.contrast * self.PSFpeak,
            )
            * DIMENSIONLESS
        )


class CoronagraphYIP(Coronagraph):
    """
    A coronagraph class that uses a Yield Input Package (YIP) for realistic modeling.

    This class implements a coronagraph simulation based on Yield Input Package files
    that contain pre-computed coronagraph responses, stellar intensity distributions,
    and off-axis PSFs. It combines coronagraph optical throughput from HWO yaml files
    with detailed coronagraph response data from YIP files.

    Parameters
    ----------
    path : str, optional
        Path to the YIP files containing coronagraph response data
    """

    # In YIP mode these quantities are OWNED by the YIP / yippy and must stay
    # consistent with the loaded package. The user is NOT allowed to override
    # them; if they try, fill_parameters will warn and keep the YIP value.
    # Anything not listed here (e.g. noisefloor_PPF, Tcore, az_avg,
    # coronagraph_spectral_resolution) remains user-editable.
    LOCKED_KEYS: set = {
        "pixscale",
        "npix",
        "xcenter",
        "ycenter",
        "skytrans",
        "r",
        "npsfratios",
        "nrolls",
        "omega_lod",
        "photometric_aperture_throughput",
        "Istar",
        "noisefloor",
        "coronagraph_bandwidth",
    }

    DEFAULT_CONFIG = {
        "noisefloor_PPF": 30.0,  # 30.0 #  divide Istar by this to get the noise floor (unitless)
        "nrolls": 1,  # number of rolls
        "Tcore": 0.2968371
        * DIMENSIONLESS,  # core throughput within off-axis PSF (only used with photometric_aperture_radius method of calculating Omega)
        "coronagraph_spectral_resolution": 1
        * DIMENSIONLESS,  # Set to default. It is used to limit the bandwidth if the coronagraph has a specific spectral window.
        "az_avg": True,  # azimuthally average the contrast maps and noise floor if True
    }

    def __init__(self, path: str = None, yippy_coro: yippycoro = None):
        """
        Initialize a CoronagraphYIP instance.

        Parameters
        ----------
        path : str, optional
            Path to the YIP files containing coronagraph response data
        """

        # Ensure that either a path or a yippy_coro is provided
        if path is None and yippy_coro is None:
            raise ValueError("Either a path or a yippy_coro must be provided")
        if yippy_coro is not None and path is not None:
            raise ValueError("Only one of path or yippy_coro can be provided")

        self.path = path
        self.yippy_coro = yippy_coro
        self.DEFAULT_CONFIG = copy.deepcopy(self.DEFAULT_CONFIG)

    def load_configuration(self, parameters: dict, mediator: object) -> None:
        """
        Load configuration parameters from YIP files and user specifications.

        This method loads coronagraph parameters from multiple sources: EACy YAML files
        for optical throughput, YIP files for detailed coronagraph response including
        stellar intensity and PSF data, and user-provided parameters. It handles both
        imaging and IFS observing modes with appropriate wavelength averaging or
        interpolation.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including observing mode,
            bandwidth specifications, and coronagraph overrides
        mediator : ObservatoryMediator
            Mediator object providing access to observation and scene parameters
            including wavelength, PSF truncation ratios, and stellar properties

        Raises
        ------
        KeyError
            If neither psf_trunc_ratio nor photometric_aperture_radius are specified
        AssertionError
            If stellar angular diameter is outside valid bounds (0 <= diameter < 1 λ/D)
        """
        parameters = parse_input.parse_parameters(parameters)

        # Load photometric aperture radius or psf truncation ratio from user. Fail if not provided
        psf_trunc = parameters.get("psf_trunc_ratio")
        phot_aperture = parameters.get("photometric_aperture_radius")

        # Check that at least one is provided
        if psf_trunc is None and phot_aperture is None:
            raise KeyError(
                "Either 'photometric_aperture_radius' or 'psf_trunc_ratio' must be provided in the parameters."
            )

        # Prefer psf_trunc_ratio if both are provided
        if psf_trunc is not None:
            self.psf_trunc_ratio = psf_trunc * DIMENSIONLESS
            if phot_aperture is not None:
                logger.warning(
                    "Both 'photometric_aperture_radius' and 'psf_trunc_ratio' provided. "
                    "Using 'psf_trunc_ratio' and ignoring 'photometric_aperture_radius'."
                )
            self.photometric_aperture_radius = None
        else:
            self.psf_trunc_ratio = None
            self.photometric_aperture_radius = phot_aperture * LAMBDA_D

        # ***** Load the YIP using yippy *****
        obs_trunc_ratio = self.psf_trunc_ratio
        if obs_trunc_ratio is not None:
            # Strip units so yippy receives a plain float
            obs_trunc_float = float(obs_trunc_ratio)
        else:
            obs_trunc_float = None

        if self.yippy_coro is not None:
            yippy_obj = self.yippy_coro
            if (
                obs_trunc_float is not None
                and yippy_obj.psf_trunc_ratio != obs_trunc_float
            ):
                logger.warning(
                    f"Pre-constructed yippy_coro has psf_trunc_ratio="
                    f"{yippy_obj.psf_trunc_ratio}, but observation specifies "
                    f"{obs_trunc_float}. The pre-constructed value will be used."
                )
        else:
            yippy_obj = yippycoro(self.path, psf_trunc_ratio=obs_trunc_float)

        # ***** Set the bandwith *****
        # get bandwidth info from yippy, if it exists in the YIP files
        if (
            hasattr(yippy_obj, "header")
            and hasattr(yippy_obj.header, "lambda0")
            and hasattr(yippy_obj.header, "minlam")
            and hasattr(yippy_obj.header, "maxlam")
        ):
            coronagraph_bandwidth = (
                yippy_obj.header.maxlam - yippy_obj.header.minlam
            ) / yippy_obj.header.lambda0
            self.DEFAULT_CONFIG["coronagraph_bandwidth"] = coronagraph_bandwidth.value
        else:
            logger.warning(
                "No bandwidth info in YIPs, setting coronagraph_bandwidth = 0.2"
            )

            self.DEFAULT_CONFIG["coronagraph_bandwidth"] = 0.2

        # get nrolls from yippy, if it exists in the YIP files
        if hasattr(yippy_obj, "nrolls"):
            self.DEFAULT_CONFIG["nrolls"] = yippy_obj.nrolls
        else:
            # if yippy did not find nrolls, assume YIP has 360deg coverage and default to 1
            logger.warning("No nrolls in YIPs, setting nrolls = 1.")
            self.DEFAULT_CONFIG["nrolls"] = 1

        # ***** Set all parameters that are defined in the YIP (unpack YIP metadata) *****
        self.DEFAULT_CONFIG["pixscale"] = (
            yippy_obj.header.pixscale.value * LAMBDA_D
        )  # has units of lam/D / pix
        self.DEFAULT_CONFIG["npix"] = yippy_obj.header.naxis1
        self.DEFAULT_CONFIG["xcenter"] = yippy_obj.header.xcenter * PIXEL
        self.DEFAULT_CONFIG["ycenter"] = yippy_obj.header.ycenter * PIXEL

        # Sky transmission map for extended sources
        self.DEFAULT_CONFIG["skytrans"] = yippy_obj.sky_trans() * DIMENSIONLESS

        # Separation grid from yippy (replaces generate_radii)
        self.DEFAULT_CONFIG["r"] = yippy_obj.separation_map() * LAMBDA_D

        self.DEFAULT_CONFIG["npsfratios"] = len([self.psf_trunc_ratio])

        if self.psf_trunc_ratio is not None:

            logger.info("Using psf_trunc_ratio to calculate Omega...")

            # Throughput and core area from yippy (replaces zoom+threshold+interp)
            omega_lod = yippy_obj.core_area_map()[..., np.newaxis]
            photometric_aperture_throughput = yippy_obj.throughput_map()[
                ..., np.newaxis
            ]

        elif (
            self.psf_trunc_ratio is None
            and self.photometric_aperture_radius is not None
        ):
            logger.info("Using photometric_aperture_radius to calculate Omega...")

            # Use the photometric_aperture_radius method of calculating Omega.
            # this method also requires you to set Tcore or uses the default one (does not use the YIP to calculate this)
            # ***** Tcore *****
            if "Tcore" in parameters.keys():
                logger.info("Using user-defined Tcore...")
                setattr(
                    self,
                    "Tcore",
                    parameters.get("Tcore") * DIMENSIONLESS,
                )
            else:
                logger.info("Using default Tcore...")
                setattr(
                    self,
                    "Tcore",
                    self.DEFAULT_CONFIG["Tcore"],
                )

            # simple omega calculation, omega = pi * (photometric_aperture_radius)**2, where photometric_aperture_radius is in lambda/D

            omega_lod = (
                np.full(
                    (self.DEFAULT_CONFIG["npix"], self.DEFAULT_CONFIG["npix"], 1),
                    float(np.pi) * self.photometric_aperture_radius**2,
                )
                * (self.photometric_aperture_radius.unit) ** 2
            )  # size of photometric aperture at all separations (npix,npix,len(psftruncratio))

            photometric_aperture_throughput = (
                np.full(
                    (self.DEFAULT_CONFIG["npix"], self.DEFAULT_CONFIG["npix"], 1),
                    self.Tcore,
                )
                * self.Tcore.unit
            )  # core throughput at all separations (npix,npix,len(psftruncratio))

        omega_lod = np.maximum(omega_lod, 0)
        photometric_aperture_throughput = np.maximum(photometric_aperture_throughput, 0)

        self.DEFAULT_CONFIG["omega_lod"] = omega_lod * DIMENSIONLESS
        self.DEFAULT_CONFIG["photometric_aperture_throughput"] = (
            photometric_aperture_throughput * DIMENSIONLESS
        )

        stellar_angular_diameter_arcsec = mediator.get_scene_parameter(
            "stellar_angular_diameter_arcsec"
        )
        lam = mediator.get_observation_parameter("wavelength")

        tele_diam = mediator.get_telescope_parameter("diameter")

        # TODO use lam (observing wavelength range) to calculate the lod value.
        # This implies increasing dimensionality of related parameters.
        stellar_angular_diameter_lod = arcsec_to_lambda_d(
            stellar_angular_diameter_arcsec, 0.55 * WAVELENGTH, tele_diam
        )
        # check that the user-provided stellar radius is within the valid bounds (0,1)
        assert (
            stellar_angular_diameter_lod >= 0.0 and stellar_angular_diameter_lod < 1.0
        ), "Stellar diameter is outside bounds 0 <= diam (lam/D) < 1. Cannot interpolate Istar from YIP."

        stellar_diam = stellar_angular_diameter_lod.value * lod

        # Load az_avg if the user provided it, otherwise use default
        setattr(self, "az_avg", parameters.get("az_avg", self.DEFAULT_CONFIG["az_avg"]))
        if self.az_avg:
            # Radial profile projection (replaces 100x rotate-and-average)
            self.DEFAULT_CONFIG["Istar"] = (
                yippy_obj.core_mean_intensity_map(stellar_diam) * DIMENSIONLESS
            )
        else:
            self.DEFAULT_CONFIG["Istar"] = (
                yippy_obj.stellar_intens(stellar_diam)[:, :] * DIMENSIONLESS
            )

        if "noisefloor_PPF" in parameters.keys():
            logger.info("Setting the noise floor via user-supplied noisefloor_PPF...")
            self.DEFAULT_CONFIG["noisefloor_PPF"] = parameters["noisefloor_PPF"]
        else:
            if "noisefloor_factor" in parameters.keys():
                raise ValueError(
                    "Noisefloor_factor mode not implemented in CoronagraphYIP "
                    "coronagraph. Please use the noisefloor_PPF method."
                )
            logger.warning(
                "noisefloor_PPF value not provided. Using the default value: "
                + str(self.DEFAULT_CONFIG["noisefloor_PPF"])
            )

        self.DEFAULT_CONFIG["noisefloor"] = (
            self.DEFAULT_CONFIG["Istar"] / self.DEFAULT_CONFIG["noisefloor_PPF"]
        )

        # ***** REPLACE PARAMETERS WITH USER-SPECIFIED ONES (no override allowed) ****
        utils.fill_parameters(
            self,
            parameters,
            self.DEFAULT_CONFIG,
            locked_keys=self.LOCKED_KEYS,
        )
