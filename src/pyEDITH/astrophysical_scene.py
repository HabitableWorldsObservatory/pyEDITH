import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as const
from .units import *
from . import utils
from astropy.coordinates import SkyCoord
import logging
from pyEDITH import parse_input

logger = logging.getLogger("pyEDITH")
# def calc_flux_zero_point_synphot(lam: u.Quantity):

#     # calculates a zeropoint Vega spectrum using boxcar filters
#     from synphot import SourceSpectrum, SpectralElement, Observation, Box1D
#     from synphot.exceptions import DisjointError
#     import numpy as np
#     import astropy.units as u

#     # load the vega spectrum
#     vega = SourceSpectrum.from_vega()

#     # convert wavelength grid to angstrom
#     lam = lam.to(u.AA)
#     dlam = np.gradient(lam)
#     dlam = dlam.to(u.AA)

#     vega_fluxes = np.empty(len(lam))

#     # treat the wavelength bins as box filters
#     for i in range(len(lam)):
#         filt_ctr = lam[i]
#         filt_width = dlam[i]

#         # boxcar filter
#         boxcar = Box1D(amplitude=1, x_0=filt_ctr.value, width=filt_width.value)
#         filt = SpectralElement(boxcar, wavelengths=lam)

#         # calculate an observation of Vega with filter
#         try:
#             obs = Observation(vega, filt, binset=lam)
#         except :
#             #print("Box filter extends beyond wavelength range. Setting flux as nan for this spectral bin.")
#             vega_fluxes[i] = np.nan
#             continue

#         # integrate observation
#         flux_photlam = obs.effstim(flux_unit="photlam")

#         # convert to pyEDITH units
#         flux = flux_photlam.to(u.photon/u.s/u.cm**2/u.nm)

#         vega_fluxes[i] = flux.value

#     return vega_fluxes * u.photon/u.s/u.cm**2/u.nm


def calc_flux_zero_point(
    lambd: u.Quantity,
    output_unit: str = "pcgs",
    perlambd: bool = False,
    AB: bool = False,
) -> u.Quantity:
    """
    Calculate the flux zero point for given wavelengths.

    This function calculates the flux zero point for a given set of wavelengths.
    By default, it returns Johnson zero points. If the 'AB' flag is set, it will
    return AB flux zero points.

    Parameters
    ----------
    lambd : astropy.units.Quantity
        Wavelengths. Must be in units of length.
    output_unit : str, optional
        Output unit type. Possible values are:
        - "jy" for output in Jy
        - "cgs" for output in erg s^-1 cm^-2 Hz^-1
        - "pcgs" for output in photons s^-1 cm^-2 Hz^-1
        Default is "pcgs".
    perlambd : bool, optional
        If True, convert all output values to per wavelength instead of per frequency.
        Default is False.
    AB : bool, optional
        If True, use AB magnitude system instead of Johnson.
        Default is False.

    Returns
    -------
    astropy.units.Quantity
        Flux zero points for the given wavelengths in the specified units.

    Raises
    ------
    ValueError
        If output_unit is not specified or is invalid, or if incompatible options are selected.
    """

    if output_unit not in ["jy", "cgs", "pcgs"]:
        raise ValueError(
            'Invalid output_unit value. Possible values are: "jy" for output in Jy, '
            '"cgs" for output in erg s^-1 cm^-2 Hz^-1, '
            '"pcgs" for output in photons s^-1 cm^-2 Hz^-1.'
        )

    if output_unit == "jy" and perlambd:
        raise ValueError("Cannot set Jy and perlambd")

    # Ensure lambd is in the correct units (cm for CGS)
    lambd = lambd.to(CM)

    # Convert constants to CGS
    h_cgs = const.h.cgs.value
    c_cgs = const.c.cgs.value

    # Calculate the zero point
    if AB:
        # AB magnitude system
        f0 = 3631 * JANSKY  # AB zero point
    else:
        # Johnson magnitude system
        known_lambd = (
            np.array(
                [0.36, 0.44, 0.55, 0.71, 0.97, 1.25, 1.60, 2.22, 3.54, 4.80, 10.6, 21.0]
            )
            * WAVELENGTH
        )
        known_zeropoint_jy = (
            np.array([1823, 4130, 3781, 2941, 2635, 1603, 1075, 667, 288, 170, 36, 9.4])
            * JANSKY
        )

        # Interpolation in log-log space
        interp = interp1d(
            np.log10(known_lambd.value),
            np.log10(known_zeropoint_jy.value),
            kind="cubic",
            fill_value="extrapolate",
        )
        logf0 = interp(np.log10(lambd.to_value(WAVELENGTH)))
        f0 = 10.0**logf0 * JANSKY

    # Convert to desired output units
    if output_unit == "cgs":
        # Convert to erg / (s * cm^2 * Hz)
        f0 = f0.to(SPECTRAL_FLUX_DENSITY_CGS)
    elif output_unit == "pcgs":
        # Convert to photons / (s * cm^2 * Hz)
        f0 = f0.to(
            PHOTON_FLUX_DENSITY_HZ,
            equivalencies=u.spectral_density(lambd),
        )

    # Convert to per wavelength if requested
    if perlambd:
        if output_unit == "cgs":
            f0 = f0.to(
                SPECTRAL_FLUX_DENSITY_CGS_WAVELENGTH,
                equivalencies=u.spectral_density(lambd),
            )
        else:  # can either be cgs or pcgs in the perlambd case (Jy and perlambd are incompatible)
            f0 = f0.to(
                PHOTON_FLUX_DENSITY_CGS_WAVELENGTH,
                equivalencies=u.spectral_density(lambd),
            )

    logger.info(
        f"Flux zero point calculated at lambda of interest in units of {f0.unit}"
    )

    return f0


def calc_exozodi_flux(
    M_V: u.Quantity,
    vmag: u.Quantity,
    nexozodis: u.Quantity,
    lambd: u.Quantity,
    lambdmag: u.Quantity,
) -> u.Quantity:
    """
    Calculate the exozodiacal light flux for given stellar and observational parameters.

    This function computes the exozodiacal light flux for a set of stars, taking into
    account their absolute and apparent magnitudes, the number of exozodis, and the
    observational wavelengths.

    Parameters
    ----------
    M_V : Quantity
        V band absolute magnitude of stars.
    vmag : Quantity
        V band apparent magnitude of stars.
    nexozodis : Quantity
        Number of exozodis for each star.
    lambd : Quantity
        Wavelength in microns (vector of length nlambda).
    lambdmag : Quantity
        Apparent magnitude of stars at each lambd (array nlambda).
    F0lambd : Quantity
        Flux zero point at wavelength lambd (vector of length nlambda).

    Returns
    -------
    Quantity
        Exozodi surface brightness in units of photons s^-1 cm^-2 arcsec^-2 nm^-1 / F0.
        This is equivalent to 10^(-0.4*magOmega_EZ).

    """
    if len(lambd) > 1:
        if len(lambd) != lambdmag.shape[0]:
            raise ValueError(
                "ERROR. lambdmag must have dimensions nstars x nlambda, where nlambda is the length of lambd."
            )

    vmag_1zodi = (
        22.0 * SURFACE_BRIGHTNESS
    )  # V band surface brightness of 1 zodi in mag arcsec^-2
    vflux_1zodi = 10.0 ** (
        -0.4 * vmag_1zodi.value
    )  # V band flux (modulo the F0 factor out front)

    M_V_sun = 4.83 * MAGNITUDE  # V band absolute magnitude of Sun

    # Now, multiply by the ratio of the star's counts received at lambd to those
    # received at V band
    nlambda = len(lambd)

    flux_exozodi_lambd = (
        nexozodis
        * vflux_1zodi
        * 10.0 ** (-0.4 * (M_V - M_V_sun).value)
        * 10.0 ** (-0.4 * (lambdmag - vmag).value)
    ) * DIMENSIONLESS

    return (
        flux_exozodi_lambd * INV_SQUARE_ARCSEC
    )  # CONVERTING FROM FLUX TO SURFACE BRIGHTNESS (modulo F0 that gets multiplied later)


def calc_zodi_flux(
    dec: u.Quantity,
    ra: u.Quantity,
    lambd: u.Quantity,
    F0: u.Quantity,
    # starshade: bool = False,
    # ss_elongation: u.Quantity = None,
) -> u.Quantity:
    """

    Calculate the zodiacal light flux for given celestial coordinates and wavelengths.

    This function computes the zodiacal light flux based on the target's position in the sky,
    observation wavelengths, and whether a starshade is used. It uses the model from
    Leinert et al. (1998) to calculate the zodiacal light intensity.

    Parameters
    ----------
    dec : Quantity
        Declination of targets in degrees (J2000 equatorial coordinate).
    ra : Quantity
        Right ascension of targets in degrees (J2000 equatorial coordinate).
    lambd : Quantity
        Wavelengths in microns (vector of length nlambda).
    F0 : Quantity
        Flux zero points at wavelengths lambd (vector of length nlambda).

    Returns
    -------
    np.ndarray
        Zodi surface brightness in units of photons s^-1 cm^-2 arcsec^-2 nm^-1 / F0.
        This is equivalent to 10^(-0.4*magOmega_ZL).
        - Multiply by F0 to get photons s^-1 cm^-2 arcsec^-2 nm^-1
        - Multiply by energy of photons to get erg s^-1 cm^-2 arcsec^-2 nm^-1
        The output array has dimensions (nlambda, nstars).

    Raises
    ------
    ValueError
        If F0 and lambd have different lengths, or if starshade mode is inconsistent with ss_elongation.

    Note
    ----
    - The function uses the zodiacal light model from Leinert et al. (1998).
    - For coronagraph mode, it assumes observations near solar longitude of 135 degrees.
    - Starshade functionality is currently not fully implemented.

    References:

    Leinert, C., et al. (1998). The 1997 reference of diffuse night sky brightness.
    Astronomy and Astrophysics Supplement Series, 127(1), 1-99.
    """

    # if starshade and ss_elongation is None:
    #     raise ValueError(
    #         "ERROR. You have set the STARSHADE flag. Must specify SS_ELONGATION in degrees."
    #     )
    # if not starshade and ss_elongation is not None:
    #     raise ValueError(
    #         "ERROR. You must enable STARSHADE mode if you are setting SS_ELONGATION in degrees."
    #     )

    # Convert equatorial coordinates to ecliptic coordinates
    coords = SkyCoord(ra=ra, dec=dec, frame="icrs")
    ecl_coords = coords.barycentrictrueecliptic
    beta = ecl_coords.lat.rad

    # all we need is the sine of the latitude
    # Use absolute value of the sin of beta (symmetry about ecliptic plane)
    sinbeta = np.abs(np.sin(beta))

    # SOURCE: Leinert et al. (1998)
    # Define solar longitude and beta values for interpolation
    beta_vector = np.array([0.0, 5, 10, 15, 20, 25, 30, 45, 60, 75]) * DEG
    sollong_vector = (
        np.array(
            [
                0,
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                60,
                75,
                90,
                105,
                120,
                135,
                150,
                165,
                180.0,
            ]
        )
        * DEG
    )

    # Table 17 values (assumed to be in some brightness units)
    table17 = (
        np.array(
            [
                [-1, -1, -1, 3140, 1610, 985, 640, 275, 150, 100],
                [-1, -1, -1, 2940, 1540, 945, 625, 271, 150, 100],
                [-1, -1, 4740, 2470, 1370, 865, 590, 264, 148, 100],
                [11500, 6780, 3440, 1860, 1110, 755, 525, 251, 146, 100],
                [6400, 4480, 2410, 1410, 910, 635, 454, 237, 141, 99],
                [3840, 2830, 1730, 1100, 749, 545, 410, 223, 136, 97],
                [2480, 1870, 1220, 845, 615, 467, 365, 207, 131, 95],
                [1650, 1270, 910, 680, 510, 397, 320, 193, 125, 93],
                [1180, 940, 700, 530, 416, 338, 282, 179, 120, 92],
                [910, 730, 555, 442, 356, 292, 250, 166, 116, 90],
                [505, 442, 352, 292, 243, 209, 183, 134, 104, 86],
                [338, 317, 269, 227, 196, 172, 151, 116, 93, 82],
                [259, 251, 225, 193, 166, 147, 132, 104, 86, 79],
                [212, 210, 197, 170, 150, 133, 119, 96, 82, 77],
                [188, 186, 177, 154, 138, 125, 113, 90, 77, 74],
                [179, 178, 166, 147, 134, 122, 110, 90, 77, 73],
                [179, 178, 165, 148, 137, 127, 116, 96, 79, 72],
                [196, 192, 179, 165, 151, 141, 131, 104, 82, 72],
                [230, 212, 195, 178, 163, 148, 134, 105, 83, 72],
            ]
        )
        * SPECTRAL_RADIANCE
    )
    # For coronagraph, assume observations near solar longitude of 135 degrees
    j = np.argmin(np.abs(sollong_vector - 135 * DEG))
    k = np.argmin(np.abs(sollong_vector - 90 * DEG))

    # Interpolate to get zodi brightness factor
    from scipy.interpolate import interp1d

    interp = interp1d(
        np.sin(beta_vector),
        table17[j] / table17[k, 0],
        kind="cubic",
        fill_value="extrapolate",
    )
    f = interp(sinbeta) * DIMENSIONLESS

    # Wavelength dependence (fits to Table 19 in Leinert et al 1998)
    zodi_lambd = (
        np.array(
            [
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                0.9,
                1.0,
                1.2,
                2.2,
                3.5,
                4.8,
                12,
                25,
                60,
                100,
                140,
            ]
        )
        * WAVELENGTH
    )
    zodi_blambd = (
        np.array(
            [
                2.5e-8,
                5.3e-7,
                2.2e-6,
                2.6e-6,
                2.0e-6,
                1.3e-6,
                1.2e-6,
                8.1e-7,
                1.7e-7,
                5.2e-8,
                1.2e-7,
                7.5e-7,
                3.2e-7,
                1.8e-8,
                3.2e-9,
                6.9e-10,
            ]
        )
        * SPECTRAL_RADIANCE
    )

    # Convert to erg s^-1 cm^-2 arcsec^-2 nm^-1
    zodi_blambd = zodi_blambd.to(SPECTRAL_RADIANCE_CGS)

    # Interpolate to get zodi brightness at desired wavelengths
    interp = interp1d(
        np.log10(zodi_lambd.value), np.log10(zodi_blambd.value), kind="cubic"
    )
    blambd = 10 ** interp(np.log10(lambd.to_value(WAVELENGTH))) * zodi_blambd.unit

    # Convert to photon flux
    # I90fabsfco = blambd / (u.h * u.c / lambd)

    I90fabsfco = blambd.to(
        PHOTON_SPECTRAL_RADIANCE,
        equivalencies=u.spectral_density(lambd),
    )
    # Divide by F0
    I90fabsfco = I90fabsfco / F0

    # Calculate final zodi flux
    nlambda = len(lambd)
    flux_zodi = f * I90fabsfco

    return flux_zodi  # 1/arcsec^2 (UNITS OF SPECTRAL RADIANCE)


class AstrophysicalScene:
    """
    A class representing an astrophysical scene for exoplanet observation simulations.

    This class encapsulates various astrophysical parameters and methods to calculate
    zodi and exozodi fluxes for a set of target stars and their potential exoplanets.

    Parameters
    ----------
    dist : float
        Distance to stars in parsecs
    vmag : float
        Stellar magnitudes at V band
    mag : ndarray
        Stellar magnitudes at desired wavelengths
    stellar_angular_diameter_arcsec : float
        Angular diameter of stars in arcseconds
    nzodis : float
        Amount of exozodi around target stars in "zodis"
    ra : float
        Right ascension of target stars in degrees
    dec : float
        Declination of target stars in degrees
    separation : float
        Separation of planets in arcseconds
    xp : float
        X-coordinate of planets in arcseconds
    yp : float
        Y-coordinate of planets in arcseconds
    deltamag : ndarray
        Magnitude difference between planets and host stars
    min_deltamag : float
        Brightest planet to resolve at the IWA
    F0V : float
        Flux zero point for V band
    wavelength:
        Wavelength array
    F0 : ndarray
        Flux zero points for prescribed wavelengths
    M_V : float
        Absolute V band magnitudes of target stars
    Fzodi_list : ndarray
        Zodiacal light fluxes
    Fexozodi_list : ndarray
        Exozodiacal light fluxes
    Fbinary_list : ndarray
        Binary star fluxes (currently ignored)
    Fp_over_Fs : ndarray
        Flux of planets

    """

    def __init__(self):
        """
        Initialize the AstrophysicalScene object with default values for output arrays.
        """
        # Constants
        # Zero Point Flux in the V Band
        self.F0V = (
            calc_flux_zero_point(
                lambd=0.55 * WAVELENGTH, output_unit="pcgs", perlambd=True
            )
        ).to(
            PHOTON_FLUX_DENSITY, equivalencies=u.spectral_density(0.55 * WAVELENGTH)
        )  # convert to photons cm^-2 nm^-1 s^-1
        return

    def load_configuration(self, parameters: dict) -> None:
        """
        Load configuration parameters for the simulation from a dictionary.

        This method initializes various attributes of the AstrophysicalScene object
        using the provided parameters dictionary.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including target star
            parameters, planet parameters, and observational parameters.
        Returns
        -------
        None

        Raises
        ------
        KeyError
            If a required parameter is missing from the input dictionary.
        """

        parameters = parse_input.parse_parameters(parameters)

        # -------- INPUTS ---------

        # Input wavelength
        self._input_wavelength = np.asarray(parameters["wavelength"], dtype=np.float64)

        # distance to star (pc) # used to be (ntargs array) now scalar
        self.dist = parameters["distance"] * DISTANCE

        # Calculate zero point flux at lambda
        if "F0" in parameters.keys():
            self.F0 = parameters["F0"] * PHOTON_FLUX_DENSITY
        else:
            self.F0 = calc_flux_zero_point(
                lambd=parameters["wavelength"] * WAVELENGTH,
                output_unit="pcgs",
                perlambd=True,
            ).to(
                PHOTON_FLUX_DENSITY,
                equivalencies=u.spectral_density(parameters["wavelength"] * WAVELENGTH),
            )  # [nlambda]

        # look for delta_mag_min or Fp_min/Fs. If not there, default to vals. This hides this feature from the user.
        if "delta_mag_min" in parameters:
            self.min_deltamag = parameters["delta_mag_min"] * MAGNITUDE
        else:
            self.min_deltamag = 26 * MAGNITUDE
        if "Fp_min/Fs" in parameters:
            self.Fp_min_over_Fs = parameters["Fp_min/Fs"] * DIMENSIONLESS
        else:
            self.Fp_min_over_Fs = 1e-10 * DIMENSIONLESS

        # Determine if user provided magnitudes or fluxes

        if all(param in parameters for param in ["magV", "mag", "delta_mag"]):
            # User provided magnitudes
            # stellar mag (not absolute mag!!) at V band # used to be (ntargs array) now scalar
            self.vmag = parameters["magV"] * MAGNITUDE

            self.mag = parameters["mag"] * MAGNITUDE

            # difference in mag between planet and host star
            # (nmeananom x norbits x ntargs array)
            self.deltamag = parameters["delta_mag"] * MAGNITUDE

            # brightest planet to resolve w/ photon counting detector evaluated at
            # the IWA, sets the time between counts (ntargs array)

            # Convert magnitudes to RELATIVE fluxes (modulo F0)
            self.Fs_over_F0 = 10 ** (-0.4 * self.mag.value) * DIMENSIONLESS
            self.Fp_over_Fs = 10 ** (-0.4 * self.deltamag.value) * DIMENSIONLESS
            self.Fp_min_over_Fs = 10 ** (-0.4 * self.min_deltamag.value) * DIMENSIONLESS

        elif all(param in parameters for param in ["Fstar_10pc", "Fp/Fs"]):

            # Load fluxes
            Fstar_10pc = parameters["Fstar_10pc"] * PHOTON_FLUX_DENSITY

            if (
                "FstarV_10pc" not in parameters
                and parameters["observing_mode"] == "IFS"
            ):
                logger.warning(
                    "`FstarV_10pc` not specified in parameters. Calculating internally..."
                )
                # from synphot import SourceSpectrum, SpectralElement, Observation
                # from synphot.models import Empirical1D
                # # if given a spectrum, calculate the v-band flux using synphot
                # tempSpec = SourceSpectrum(Empirical1D, points=parameters["wavelength"] * WAVELENGTH, lookup_table=Fstar_10pc)
                # johnson_v = SpectralElement.from_filter('johnson_v')
                # obsTemp = Observation(tempSpec, johnson_v)
                #
                # # unit conversion is necessary because V-band flux is in units of [photons/s/cm^2],
                # # and not PHOTON_FLUX_DENSITY since an integration over wavelength was performed.
                # # Doing this to make the units line up.
                # Fstar_V_10pc = obsTemp.integrate(flux_unit=PHOTON_FLUX_DENSITY).value * PHOTON_FLUX_DENSITY

                # the above is for calculating the zero-point over a bandpass. What is actually going on here is
                # that the zeropoint calculation is for a single wavelength, lam=0.55 um.
                # let's change this up and interpolate
                func = interp1d(parameters["wavelength"], Fstar_10pc)
                Fstar_V_10pc = (
                    func(0.55) * PHOTON_FLUX_DENSITY
                )  # interpolate to single wavelength, not a bandpass

            elif (
                "FstarV_10pc" not in parameters
                and parameters["observing_mode"] == "IMAGER"
            ):
                raise KeyError("FstarV_10pc missing in parameters.")
            else:
                Fstar_V_10pc = parameters["FstarV_10pc"] * PHOTON_FLUX_DENSITY

            Fstar_absolute = Fstar_10pc * (10 * DISTANCE / self.dist) ** 2

            Fstar_V_absolute = Fstar_V_10pc * (10 * DISTANCE / self.dist) ** 2

            self.Fp_over_Fs = parameters["Fp/Fs"] * DIMENSIONLESS
            # Convert to relative fluxes (used internally)
            self.Fs_over_F0 = (Fstar_absolute / self.F0).to(DIMENSIONLESS)

            # Calculate vmag (needed for zodi)
            self.vmag = -2.5 * np.log10(Fstar_V_absolute / self.F0V) * MAGNITUDE
            self.mag = -2.5 * np.log10(Fstar_absolute / self.F0) * MAGNITUDE

            # Calculate deltamag and min_deltamag (used only to validate)
            self.deltamag = -2.5 * np.log10(self.Fp_over_Fs) * MAGNITUDE
            self.min_deltamag = -2.5 * np.log10(self.Fp_min_over_Fs) * MAGNITUDE
        else:
            missing_mag_params = [
                param
                for param in ["magV", "mag", "delta_mag"]
                if param not in parameters
            ]
            missing_flux_params = [
                param
                for param in ["Fstar_10pc", "FstarV_10pc", "Fp/Fs"]
                if param not in parameters
            ]

            raise KeyError(
                f"Insufficient parameters provided. You must provide either:\n"
                f"1. All magnitude parameters: {', '.join(['magV', 'mag', 'delta_mag'])}\n"
                f"   Missing: {', '.join(missing_mag_params)}\n"
                f"OR\n"
                f"2. All flux parameters: {', '.join(['Fstar_10pc', 'FstarV_10pc', 'Fp/Fs',])}\n"
                f"   Missing: {', '.join(missing_flux_params)}"
            )

        # convert stellar radius into stellar angular diameter (in arcsec)
        stellar_radius_arcsec = (
            to_arcsec(
                (parameters["stellar_radius"] * const.R_sun).to(LENGTH),
                (self.dist.to(LENGTH)),
            )
            * ARCSEC
        )

        # angular diameter of star (arcsec) # used to be (ntargs array) now scalar
        self.stellar_angular_diameter_arcsec = 2 * stellar_radius_arcsec

        # amount of exozodi around target star ("zodis") # used to be (ntargs array) now scalar
        self.nzodis = parameters["nzodis"] * ZODI

        # right ascension of target star used to estimate zodi (deg) # used to be (ntargs array) now scalar
        self.ra = parameters["ra"] * DEG

        # declination of target star used to estimate zodi (deg) # used to be (ntargs array) now scalar
        self.dec = parameters["dec"] * DEG

        # Planet parameters
        # separation of planet (arcseconds) (nmeananom x norbits x ntargs array)
        # NOTE FOR NOW IT IS ASSUMED TO BE ON THE X AXIS
        # SO THAT XP = SP (input) and YP = 0

        if "separation" in parameters.keys():
            self.separation = parameters["separation"] * ARCSEC
        elif "semimajor_axis" in parameters.keys():
            self.separation = (
                to_arcsec(
                    (parameters["semimajor_axis"] * const.au).to(LENGTH),
                    (self.dist.to(LENGTH)),
                )
                * ARCSEC
            )
        else:
            raise KeyError(
                "Either separation [arcsec] or semimajor_axis [AU] must be provided."
            )

        self.xp = self.separation.copy()
        self.yp = self.separation.copy() * 0.0

        # set the exozodi PPF
        if "ez_PPF" in parameters.keys():
            # It has already been parsed to be of length nlambda
            self.ez_PPF = parameters["ez_PPF"] * DIMENSIONLESS

        else:
            logger.warning(
                "ez_PPF not set. Assuming EZ subtraction to Poisson limit (ez_PPF = inf)"
            )
            self.ez_PPF = np.inf * np.ones_like(self.Fp_over_Fs) * DIMENSIONLESS

    def calculate_zodi_exozodi(self, parameters: dict) -> None:
        """
        Calculate zodiacal and exozodiacal light fluxes for the given observation.

        This method computes the flux zero points, zodiacal light fluxes, and
        exozodiacal light fluxes for the target stars based on the provided
        observation parameters.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including target star
            parameters, planet parameters, and observational parameters.
            Must include a "wavelength" key containing the wavelength array

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If the scene has not been properly configured (missing required attributes)
        KeyError
            If parameters dictionary is missing required keys

        """
        parameters = parse_input.parse_parameters(parameters)

        # Validate that scene has been properly configured
        required_attributes = ["vmag", "dist", "dec", "ra", "F0", "nzodis", "mag"]
        missing_attrs = [
            attr for attr in required_attributes if not hasattr(self, attr)
        ]
        if missing_attrs:
            raise AttributeError(
                f"AstrophysicalScene must be configured before calculating zodi/exozodi. "
                f"Missing attributes: {', '.join(missing_attrs)}. "
                f"Please call load_configuration() first."
            )

        # Validate parameters dictionary
        # if "wavelength" not in parameters:
        #     raise KeyError("parameters dictionary must contain 'wavelength' key")
        # NOTE As of July 06, 2026, this is taken into account into parse_params

        # calculate flux at zero point for the V band and the prescribed lambda

        self.M_V = (
            self.vmag - 5.0 * np.log10(self.dist.value) * MAGNITUDE + 5.0 * MAGNITUDE
        )  # calculate absolute V band mag of target

        self.Fzodi_list = calc_zodi_flux(
            self.dec, self.ra, parameters["wavelength"] * WAVELENGTH, self.F0
        )  # [nlambda]

        self.Fexozodi_list = calc_exozodi_flux(
            self.M_V,
            self.vmag,
            self.nzodis,
            parameters["wavelength"] * WAVELENGTH,
            self.mag,
        )  # [nlambda]

        self.Fbinary_list = (
            np.full((len(parameters["wavelength"])), 0.0) * DIMENSIONLESS
        )  # this code ignores stray light from binaries # [nlambda]

    def validate_configuration(self):
        """
        Check that mandatory variables are there and have the right format.
        There can be other variables, but they are not needed for the calculation.
        """
        expected_args = {
            "dist": DISTANCE,
            # "vmag": MAGNITUDE,
            # "mag": MAGNITUDE,
            "stellar_angular_diameter_arcsec": ARCSEC,
            "nzodis": ZODI,
            "ra": DEG,
            "dec": DEG,
            "separation": ARCSEC,
            "xp": ARCSEC,
            "yp": ARCSEC,
            # "deltamag": MAGNITUDE,
            # "min_deltamag": MAGNITUDE,
            "F0V": PHOTON_FLUX_DENSITY,
            "F0": PHOTON_FLUX_DENSITY,
            # "M_V": MAGNITUDE,
            "Fzodi_list": INV_SQUARE_ARCSEC,
            "Fexozodi_list": INV_SQUARE_ARCSEC,
            "Fbinary_list": DIMENSIONLESS,
            "Fp_over_Fs": DIMENSIONLESS,
            "Fs_over_F0": DIMENSIONLESS,
            "ez_PPF": DIMENSIONLESS,
        }
        utils.validate_attributes(self, expected_args)

    def regrid_spectra(self, observation):
        """function to re-grid onto a new wavelength grid if the user specified this option"""
        # spectra to regrid: F0, Fzodi_list, Fexozodi_list, Fbinary_list, Fp_over_Fs, Fs_over_F0
        logger.info("Re-gridding spectra onto ETC wavelength grid...")
        for attr_name in [
            "F0",
            "Fzodi_list",
            "Fexozodi_list",
            "Fbinary_list",
            "Fp_over_Fs",
            "Fs_over_F0",
            "ez_PPF",
            "mag",
            "deltamag",
        ]:
            setattr(
                self,
                attr_name,
                utils.resample_to_wavelength_grid(
                    getattr(self, attr_name),
                    self._input_wavelength,
                    observation.wavelength.value,
                    observation.delta_wavelength.value,
                    name=attr_name,
                    interpolation="Gaussian",
                ),
            )
