from typing import Tuple
import numpy as np
from pyEDITH import AstrophysicalScene, Observation, Observatory
from pyEDITH.coronagraphs import CoronagraphYIP, ToyModelCoronagraph
import astropy.constants as c
import astropy.units as u
from astropy.modeling import models
from .units import *
from . import utils, set_verbosity
import pickle
import logging

logger = logging.getLogger("pyEDITH")


def calculate_CRp(
    F0: u.Quantity,
    Fs_over_F0: u.Quantity,
    Fp_over_Fs: u.Quantity,
    area: u.Quantity,
    Upsilon: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
) -> u.Quantity:
    """
    Calculate the planet count rate.

    This function computes the detected count rate from a planet based on
    the stellar and planetary flux, telescope characteristics, and coronagraph
    performance parameters.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point [photons / (s * cm^2 * nm)]
    Fs_over_F0 : u.Quantity
        Stellar flux [dimensionless]
    Fp_over_Fs : u.Quantity
        Planet flux relative to star [dimensionless]
    area : u.Quantity
        Collecting area of the telescope [cm^2]
    Upsilon : u.Quantity
        Core throughput of the coronagraph [dimensionless]
    throughput : u.Quantity
        Throughput of the system (includes QE) [electrons/photons]
    dlambda : u.Quantity
        Bandwidth [um]
    Returns
    -------
    u.Quantity
        Planet count rate [electrons / s]
    """

    return (F0 * Fs_over_F0 * Fp_over_Fs * area * Upsilon * throughput * dlambda).to(
        COUNT_RATE,
        equivalencies=EQUIV_ANGLE,
    )


def calculate_CRbs(
    F0: u.Quantity,
    Fs_over_F0: u.Quantity,
    Istar: u.Quantity,
    area: u.Quantity,
    pixscale: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
) -> u.Quantity:
    """
    Calculate the stellar leakage count rate.

    This function computes the detected count rate from stellar leakage based
    on the stellar flux, coronagraph performance, and telescope parameters.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point [photons / (s * cm^2 * nm)]
    Fs_over_F0 : u.Quantity
        Stellar flux [dimensionless]
    Istar : u.Quantity
        Stellar intensity at the given pixel [dimensionless]
    area : u.Quantity
        Collecting area of the telescope [cm^2]
    pixscale : u.Quantity
        Pixel scale of the detector [lambda/D]
    throughput : u.Quantity
        Throughput of the system (includes QE) [electrons/photons]
    dlambda : u.Quantity
        Bandwidth [um]

    Returns
    -------
    u.Quantity
        Stellar leakage count rate [electrons / s]
    """

    return (F0 * Fs_over_F0 * Istar * area * throughput * dlambda / (pixscale**2)).to(
        COUNT_RATE,
        equivalencies=EQUIV_ANGLE,
    )


def calculate_CRbz(
    F0: u.Quantity,
    Fzodi: u.Quantity,
    lod_arcsec: u.Quantity,
    skytrans: u.Quantity,
    area: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
) -> u.Quantity:
    """
    Calculate the local zodiacal light count rate.

    This function computes the detected count rate from local zodiacal light
    based on the zodiacal intensity, sky transmission, and telescope parameters.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point [photons / (s * cm^2 * nm)]
    Fzodi : u.Quantity
        Zodiacal light flux [dimensionless]
    lod_arcsec : u.Quantity
        Lambda/D in arcseconds [arcsec]
    skytrans : u.Quantity
        Sky transmission [dimensionless]
    area : u.Quantity
        Collecting area of the telescope [cm^2]
    throughput : u.Quantity
        Throughput of the system (includes QE) [electrons/photons]
    dlambda : u.Quantity
        Bandwidth [um]

    Returns
    -------
    u.Quantity
        Local zodiacal light count rate [electrons / s]
    """

    return (F0 * Fzodi * skytrans * area * throughput * dlambda * lod_arcsec**2).to(
        COUNT_RATE,
        equivalencies=EQUIV_ANGLE,
    )


def calculate_CRbez(
    F0: u.Quantity,
    Fexozodi: u.Quantity,
    lod_arcsec: u.Quantity,
    skytrans: u.Quantity,
    area: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
    dist: u.Quantity,
    sp: u.Quantity,
) -> u.Quantity:
    """
    Calculate the exozodiacal light count rate.

    This function computes the detected count rate from exozodiacal light
    based on the exozodiacal intensity, system geometry, and telescope parameters.
    It scales the exozodiacal intensity based on the distance to the star
    and the angular separation.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point [photons / (s * cm^2 * nm)]
    Fexozodi : u.Quantity
        Exozodiacal light flux at reference position [dimensionless]
    lod_arcsec : u.Quantity
        Lambda/D in arcseconds [arcsec]
    skytrans : u.Quantity
        Sky transmission [dimensionless]
    area : u.Quantity
        Collecting area of the telescope [cm^2]
    throughput : u.Quantity
        Throughput of the system (includes QE) [electrons/photons]
    dlambda : u.Quantity
        Bandwidth [um]
    dist : u.Quantity
        Distance to the star [pc]
    sp : u.Quantity
        Separation of the planet [arcsec]

    Returns
    -------
    u.Quantity
        Exozodiacal light count rate [electrons / s]
    """
    # Calculate Fexozodi at the separation (scale the value of Fexozodi at 1 AU
    # to the separation in AU)
    scaling_factor = u.AU / arcsec_to_au(sp, dist)
    return (
        F0
        * (Fexozodi * scaling_factor**2)
        * skytrans
        * area
        * throughput
        * dlambda
        * lod_arcsec**2
    ).to(
        COUNT_RATE,
        equivalencies=EQUIV_ANGLE,
    )  # this is to simplify the arcsec^2/arcsec^2 that somehow does not simplify by itself


def calculate_CRbbin(
    F0: u.Quantity,
    Fbinary: u.Quantity,
    skytrans: u.Quantity,
    area: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
) -> u.Quantity:
    """
    Calculate the count rate from neighboring stars.

    This function computes the detected count rate from binary or neighboring stars
    based on their flux, sky transmission, and telescope parameters.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point [photons / (s * cm^2 * nm)]
    Fbinary : u.Quantity
        Flux from neighboring stars [dimensionless]
    skytrans : u.Quantity
        Sky transmission [dimensionless]
    area : u.Quantity
        Collecting area of the telescope [cm^2]
    throughput : u.Quantity
        Throughput of the system (includes QE) [electrons/photons]
    dlambda : u.Quantity
        Bandwidth [um]

    Note
    ----
    IMPORTANT: Currently Fbinary is 0 by default, so this term is zero.
    It will need to be checked again in the future.

    Returns
    -------
    u.Quantity
        Count rate from neighboring stars [electrons / s]
    """

    return (F0 * Fbinary * skytrans * area * throughput * dlambda).to(
        COUNT_RATE,
        equivalencies=EQUIV_ANGLE,
    )


def calculate_CRbth(
    lam: u.Quantity,
    area: u.Quantity,
    dlambda: u.Quantity,
    temp: u.Quantity,
    lod_rad: u.Quantity,
    emis: u.Quantity,
    QE: u.Quantity,
    dQE: u.Quantity,
) -> u.Quantity:
    """
    Calculate background thermal count rate.

    This function computes the detected count rate from thermal emission
    of the telescope and instrument components based on their temperature,
    emissivity, and other system parameters. It uses a blackbody radiation
    model to calculate the thermal photon flux.

    Parameters
    ----------
    lam : u.Quantity
        Wavelength of observation [um]
    area : u.Quantity
        Collecting area of the telescope [cm^2]
    dlambda : u.Quantity
        Bandwidth [um]
    temp : u.Quantity
        Telescope mirror temperature [K]
    lod_rad : u.Quantity
        Lambda/D in radians [rad]
    emis : u.Quantity
        Effective emissivity for the observing system [dimensionless]
    QE : u.Quantity
        Quantum efficiency [electron/photon]
    dQE : u.Quantity
        Effective QE due to degradation [dimensionless]

    Returns
    -------
    u.Quantity
        Count rate from thermal background [electrons / s]
    """

    # Calculate blackbody radiation
    bb = models.BlackBody(temperature=temp, scale=1 * SPECTRAL_RADIANCE_CGS_AA)
    Blambda_energy = bb(lam)

    # Convert to photon spectral radiance
    Blambda_photon = (Blambda_energy).to(
        PHOTON_SPECTRAL_RADIANCE_SR, equivalencies=u.spectral_density(lam)
    )

    # Calculate thermal background count rate
    return (Blambda_photon * dlambda * area * (lod_rad * lod_rad) * emis * QE * dQE).to(
        COUNT_RATE
    )


def calculate_t_photon_count(
    det_npix: u.Quantity,
    det_CR: u.Quantity,
) -> u.Quantity:
    """
    Calculate the photon counting time.

    This function computes the average time needed to detect one photon per pixel
    based on the detector count rate and number of pixels.

    Parameters
    ----------
    det_npix : u.Quantity
        Number of detector pixels [pix]
    det_CR : u.Quantity
        Detector count rate [photons / s]

    Returns
    -------
    u.Quantity
        Photon counting time (average time to detect one photon per pixel) [s * pixel / ph]
    """

    counts_per_second_per_pixel = det_CR / det_npix  #  electron / s / pix
    # NOTE: I am just extrapolating that 6.73 has units [pix*frame/electron]
    t_photon_count = 1.0 / (
        6.73 * PIXEL * FRAME / ELECTRON * counts_per_second_per_pixel
    )
    return t_photon_count


def calculate_CRbd(
    det_npix: u.Quantity,
    det_DC: u.Quantity,
    det_RN: u.Quantity,
    det_tread: u.Quantity,
    det_CIC: u.Quantity,
    t_photon_count: u.Quantity,
) -> u.Quantity:
    """
    Calculate the detector noise count rate.

    This function computes the total detector noise count rate by combining
    contributions from dark current, read noise, and clock-induced charge.

    Parameters
    ----------
    det_npix : u.Quantity
        Number of detector pixels [pix]
    det_DC : u.Quantity
        Dark current [electron / pix / s]
    det_RN : u.Quantity
        Read noise [electron / pix / read]
    det_tread : u.Quantity
        Read time [s]
    det_CIC : u.Quantity
        Clock-induced charge [electron / pix / photon]
    t_photon_count : u.Quantity
        Photon counting time [pix * s / electron]

    Returns
    -------
    u.Quantity
        Detector noise count rate [electrons / s]
    """

    # Using the variance of the read noise but keeping the same units as det_RN alone.
    read_noise_variance = det_RN * det_RN.value
    return (
        (det_DC + read_noise_variance / det_tread + det_CIC / t_photon_count) * det_npix
    ).to(
        COUNT_RATE,
        equivalencies=EQUIV_ANGLE,
    )


def calculate_CRnf(
    F0: u.Quantity,
    Fs_over_F0: u.Quantity,
    area: u.Quantity,
    pixscale: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
    noisefloor: u.Quantity,
) -> u.Quantity:
    """
    Calculate the noise floor count rate.

    This function computes the count rate corresponding to the noise floor
    based on the stellar flux, telescope parameters, and the specified noise floor level.
    The noise floor represents the limiting systematic noise that cannot be reduced
    through longer integration times.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point [photons / (s * cm^2 * nm)]
    Fs_over_F0 : u.Quantity
        Stellar flux [dimensionless]
    area : u.Quantity
        Collecting area of the telescope [cm^2]
    pixscale : u.Quantity
        Pixel scale of the detector [lambda/D]
    throughput : u.Quantity
        Throughput of the system [dimensionless]
    dlambda : u.Quantity
        Bandwidth [um]
    noisefloor : u.Quantity
        Noise floor level [dimensionless]

    Returns
    -------
    u.Quantity
        Noise floor count rate [photons / s]
    """

    return (
        F0 * Fs_over_F0 * area * throughput * dlambda / (pixscale**2)
    ) * noisefloor  # Update 1.7.1: SNR now moved outside of this function


def calculate_CRnf_ez(
    CRbez: u.Quantity,
    ez_PPF: u.Quantity,
) -> u.Quantity:
    """
    Calculate the exozodi noise floor count rate.

    This function computes the noise floor contribution from exozodiacal light
    when it cannot be subtracted to the Poisson noise limit. It accounts for
    post-processing capabilities through the ez_PPF factor.

    Parameters
    ----------
    CRbez : u.Quantity
        Count rate of the exozodi [photons / s]
    ez_PPF : u.Quantity
        Post-processing factor for exozodi [dimensionless]

    Returns
    -------
    u.Quantity
        Exozodi noise floor count rate [photons / s]
    """

    return CRbez / ez_PPF  # Update 1.7.1: SNR now moved outside of this function


def measure_coronagraph_performance_at_IWA(
    photometric_aperture_throughput: u.Quantity,
    Istar_interp: u.Quantity,
    skytrans: u.Quantity,
    omega_lod: u.Quantity,
    npix: int,
    xcenter: u.Quantity,
    ycenter: u.Quantity,
    oneopixscale_arcsec: u.Quantity,
) -> Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    """
    Measure the performance of the coronagraph at the Inner Working Angle (IWA).

    This function determines the IWA and calculates various coronagraph performance
    parameters at that point. It identifies the IWA by finding where the photometric
    aperture throughput falls to half its maximum value and then measures stellar
    intensity, sky transmission, and other parameters in a 2-pixel annulus at the IWA.

    Parameters
    ----------
    photometric_aperture_throughput : u.Quantity
        Photometric aperture fractions [dimensionless]
    Istar_interp : u.Quantity
        Interpolated stellar intensity [dimensionless]
    skytrans : u.Quantity
        Sky transmission [dimensionless]
    omega_lod : u.Quantity
        Solid angle of photometric aperture [(lambda/D)^2]
    npix : int
        Number of pixels in each dimension
    xcenter : u.Quantity
        X-coordinate of the center [pixel]
    ycenter : u.Quantity
        Y-coordinate of the center [pixel]
    oneopixscale_arcsec : u.Quantity
        Inverse of pixel scale [1/arcsec]

    Returns
    -------
    Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity]
        det_sep_pix: Separation at the IWA [pixel]
        det_sep: Separation at the IWA [arcsec]
        det_Istar: Maximum stellar intensity at the IWA [dimensionless]
        det_skytrans: Maximum sky transmission at the IWA [dimensionless]
        det_photometric_aperture_throughput: Maximum photometric aperture fraction at the IWA [dimensionless]
        det_omega_lod: Solid angle corresponding to max photometric_aperture_throughput at the IWA [(lambda/D)^2]
    """

    # Find psf_trunc_ratio closest to 0.3
    # Commenting this out for now since this is not used.
    # bestiratio = np.argmin(
    #     np.abs(psf_trunc_ratio - 0.3)
    # )  # NOT USED, in EDITH only one psf_trunc_ratio
    bestiratio = 0  # len = 1 array, so only one index to choose

    # Find maximum photometric_aperture_throughput in first half of image
    maxphotometric_aperture_throughput = np.max(
        photometric_aperture_throughput[: npix // 2, int(ycenter.value), bestiratio]
    )

    # Find IWA = where photometric_aperture_throughput is half the value of the maximum
    row = photometric_aperture_throughput[:, int(ycenter.value), bestiratio]
    iwa_index = np.where(
        row[: int(xcenter.value)] > 0.5 * maxphotometric_aperture_throughput
    )[0][-1]
    det_sep_pix = abs((iwa_index + 0.5) - xcenter.value) * PIXEL
    det_sep = det_sep_pix / oneopixscale_arcsec  # translates to arcsec

    # Calculate max values in 2-pixel annulus at det_sep
    y, x = np.ogrid[:npix, :npix]
    dist_from_center = (
        np.sqrt((x - xcenter.value + 0.5) ** 2 + (y - ycenter.value + 0.5) ** 2)
    ) * PIXEL

    mask = np.abs(dist_from_center.value - det_sep_pix.value) < 2
    # NOTE: We use .value because we want indices

    det_Istar = np.max(Istar_interp[mask]) * DIMENSIONLESS
    det_skytrans = np.max(skytrans[mask]) * DIMENSIONLESS

    photometric_aperture_throughput_masked = photometric_aperture_throughput[
        :, :, bestiratio
    ][mask]
    det_photometric_aperture_throughput = (
        np.max(photometric_aperture_throughput_masked) * DIMENSIONLESS
    )
    det_omega_lod = omega_lod[:, :, bestiratio][mask][
        np.argmax(photometric_aperture_throughput_masked)
    ]

    return (
        det_sep_pix,
        det_sep,
        det_Istar,
        det_skytrans,
        det_photometric_aperture_throughput,
        det_omega_lod,
    )


def calculate_exposure_time_or_snr(
    observation: Observation,
    scene: AstrophysicalScene,
    observatory: Observatory,
    ETC_validation: bool = False,
    mode: str = "exposure_time",
) -> None:
    """
    Calculate exposure time or signal-to-noise ratio for an observation.

    This function performs detailed calculations of exposure time or signal-to-noise
    ratio for each wavelength in an observation, accounting for multiple noise sources,
    coronagraph performance, and detector characteristics. The function handles both
    'exposure_time' mode (calculating required exposure time for a given SNR) and
    'signal_to_noise' mode (calculating achievable SNR for a given exposure time).
    The function stores calculated photon counts, exposure times or SNR values
    directly in the observation object. For planets outside the working angle
    range or below the noise floor, infinity values are assigned.

    Parameters
    ----------
    observation : Observation
        Observation object containing observation parameters including wavelength,
        target SNR or exposure time, and bandwidth information
    scene : AstrophysicalScene
        AstrophysicalScene object containing scene parameters including planet
        contrast, stellar properties, and zodiacal light levels
    observatory : Observatory
        Observatory object containing telescope, detector, and coronagraph parameters
    ETC_validation : bool, optional
        If True, use specific parameter values for validation against the ETC,
        default is False
    mode : str, optional
        Calculation mode, either 'exposure_time' (to calculate required exposure
        time for a given SNR) or 'signal_to_noise' (to calculate achievable SNR
        for a given exposure time), default is 'exposure_time'

    Raises
    ------
    ValueError
        If an invalid mode is specified or if the observing_mode is not
        'IMAGER' or 'IFS'

    """
    if ETC_validation:
        set_verbosity(level="debug")
        logger.debug("Verbosity level locked to 'DEBUG' to execute validation.")
    # Check modes
    if mode not in ["exposure_time", "signal_to_noise"]:
        raise ValueError("Invalid mode. Use 'exposure_time' or 'signal_to_noise'.")

    observation.validation_variables = {}
    observation.photon_counts = {}

    # -----------------------------------------------------------------------
    # Compute all wavelength-dependent scalars as 1D arrays
    # -----------------------------------------------------------------------

    # --- Telescope collecting area (wavelength-independent) ---
    area_cm2 = observatory.telescope.Area.to(AREA)

    # --- Bandwidth per wavelength channel ---
    if observatory.observing_mode == "IMAGER":
        # select minimum bandpass possible
        bw = np.minimum(
            observatory.coronagraph.coronagraph_bandwidth, observation.filter_bandwidth
        )

        # Warn if bandwidth is wider than what the coronagraph allows
        if observatory.coronagraph.coronagraph_bandwidth < observation.filter_bandwidth:
            logger.warning(
                "Bandwidth larger than what the coronagraph allows for one or more "
                "wavelength channels. Selecting widest possible bandwidth..."
            )
        deltalambda_nm = bw * observation.wavelength.to(NM)

    elif observatory.observing_mode == "IFS":
        deltalambda_nm = observation.delta_wavelength.to(NM)
    else:
        raise ValueError("Invalid observation mode. Choose 'IMAGER' or 'IFS'.")

    # --- λ/D in radians and arcseconds (one value per wavelength) ---
    lod = 1 * LAMBDA_D
    lod_rad_arr = lambda_d_to_radians(
        lod,
        observation.wavelength.to(LENGTH),
        observatory.telescope.diameter.to(LENGTH),
    )  # shape (nlambda,), units rad
    lod_rad_arr = u.Quantity(np.atleast_1d(lod_rad_arr.value), lod_rad_arr.unit)
    lod_arcsec_arr = lod_rad_arr.to(ARCSEC)  # shape (nlambda,)

    # --- Detector pixel scale in λ/D (one value per wavelength) ---
    detpixscale_lod_arr = arcsec_to_lambda_d(
        observatory.detector.pixscale_mas.to(ARCSEC),
        observation.wavelength.to(LENGTH),
        observatory.telescope.diameter.to(LENGTH),
    )  # shape (nlambda,), units LAMBDA_D
    detpixscale_lod_arr = u.Quantity(
        np.atleast_1d(detpixscale_lod_arr.value), detpixscale_lod_arr.unit
    )

    # --- Planet pixel position in the coronagraph image (one per wavelength) ---
    # pixscale_rad: [LAMBDA_D] * [rad/LAMBDA_D] = [rad/pix]
    pixscale_rad_arr = (
        observatory.coronagraph.pixscale * lod_rad_arr
    )  # shape (nlambda,)
    oneopixscale_arcsec_arr = (1 * PIXEL) / pixscale_rad_arr.to(
        ARCSEC
    )  # shape (nlambda,) [pix/arcsec]

    ix_arr = (
        scene.xp * oneopixscale_arcsec_arr + observatory.coronagraph.xcenter
    ).value  # shape (nlambda,), float pixel indices
    iy_arr = (
        scene.yp * oneopixscale_arcsec_arr + observatory.coronagraph.ycenter
    ).value  # shape (nlambda,), float pixel indices

    ix_arr = np.atleast_1d(ix_arr)
    iy_arr = np.atleast_1d(iy_arr)

    # Integer (floored) pixel indices for coronagraph map lookups
    ix_int = np.floor(ix_arr).astype(int)  # shape (nlambda,)
    iy_int = np.floor(iy_arr).astype(int)  # shape (nlambda,)

    # --- Pixel validity mask (checks if the planet is within coronagraph pixels) ---
    npix = observatory.coronagraph.npix
    pixel_valid_mask = np.atleast_1d(
        (ix_int >= 0) & (ix_int < npix) & (iy_int >= 0) & (iy_int < npix)
    )  # shape (nlambda,), bool

    # Clamp indices so fancy indexing never goes out of bounds on invalid channels
    # (values for those channels will be masked out anyway)
    ix_safe = np.clip(ix_int, 0, npix - 1)
    iy_safe = np.clip(iy_int, 0, npix - 1)

    # --- Vectorized coronagraph map lookups  [shape (nlambda,)] ---

    # NOTE: noisefloor_interp: technically the Y axis
    # is rows and the X axis is columns,
    # that is why they are inverted
    # NOTE: Evaluate if int(round(iy)) is better than
    # np.floor. Kept np.floor for consistency
    Istar_at_planet = np.atleast_1d(
        observatory.coronagraph.Istar[iy_safe, ix_safe]
    )  # dimensionless
    skytrans_at_planet = np.atleast_1d(
        observatory.coronagraph.skytrans[iy_safe, ix_safe]
    )  # dimensionless
    noisefloor_at_planet = np.atleast_1d(
        observatory.coronagraph.noisefloor[iy_safe, ix_safe]
    )  # dimensionless

    # NOTE: npsfratios == 1 in practice; iratio = 0 throughout.
    iratio = 0  # only one psf truncation ratio supported

    # omega_lod at the planet position, shape (nlambda,)
    omega_lod_at_planet = u.Quantity(
        np.atleast_1d(
            observatory.coronagraph.omega_lod[iy_safe, ix_safe, iratio].value
        ),
        observatory.coronagraph.omega_lod.unit,
    )
    # photometric aperture throughput at the planet position, shape (nlambda,)
    Upsilon_at_planet = np.atleast_1d(
        observatory.coronagraph.photometric_aperture_throughput[
            iy_safe, ix_safe, iratio
        ]
    )

    # -----------------------------------------------------------------------
    # Measure coronagraph performance close to IWA
    # -----------------------------------------------------------------------
    # Compute per wavelength via the existing helper (now vectorized over nlambda
    # because measure_coronagraph_performance_at_IWA accepts scalar oneopixscale;
    # we call it in a list-comprehension to preserve the existing function signature) #TODO update after validation

    det_results = [
        measure_coronagraph_performance_at_IWA(
            observatory.coronagraph.photometric_aperture_throughput,
            observatory.coronagraph.Istar,
            observatory.coronagraph.skytrans,
            observatory.coronagraph.omega_lod,
            npix,
            observatory.coronagraph.xcenter,
            observatory.coronagraph.ycenter,
            oneopixscale_arcsec_arr[ilambd],
        )
        for ilambd in range(len(observation.wavelength))
    ]

    # Convert to Quantity arrays
    det_sep_pix_arr = u.Quantity([r[0] for r in det_results])
    det_sep_arr = u.Quantity([r[1] for r in det_results])
    det_Istar_arr = u.Quantity([r[2] for r in det_results])
    det_skytrans_arr = u.Quantity([r[3] for r in det_results])
    det_photometric_aperture_throughput_arr = u.Quantity([r[4] for r in det_results])
    det_omega_lod_arr = u.Quantity([r[5] for r in det_results])

    # --- Number of detector pixels
    if ETC_validation:
        logger.debug("Fixing det_npix for validation...")

        det_npix = observatory.detector.det_npix_input * PIXEL
    else:

        # Number of detector pixels (wavelength-independent scalar)
        det_npix = (
            observatory.detector.npix_multiplier
            * det_omega_lod_arr
            / (detpixscale_lod_arr**2)
        ) * PIXEL

    # --- det_* quantities used for t_photon_count estimate ---
    # (measured at the IWA, wavelength-dependent because pixscale changes)
    # Here we calculate detector noise, as it may depend on count rates
    # We don't know the count rates yet, so we make estimates based on
    # values near the IWA

    # Detector noise from signal itself (we budget for 10x
    # the planet count rate for the minimum detectable planet)
    det_CRp_arr = calculate_CRp(
        scene.F0,
        scene.Fs_over_F0,
        10 * scene.Fp_min_over_Fs,
        area_cm2,
        det_photometric_aperture_throughput_arr,
        observatory.total_throughput,
        deltalambda_nm,
    )

    det_CRbs_arr = calculate_CRbs(
        scene.F0,
        scene.Fs_over_F0,
        det_Istar_arr,
        area_cm2,
        observatory.coronagraph.pixscale,
        observatory.total_throughput,
        deltalambda_nm,
    )

    det_CRbz_arr = calculate_CRbz(
        scene.F0,
        scene.Fzodi_list,
        lod_arcsec_arr,
        det_skytrans_arr,
        area_cm2,
        observatory.total_throughput,
        deltalambda_nm,
    )

    det_CRbez_arr = calculate_CRbez(
        scene.F0,
        scene.Fexozodi_list,
        lod_arcsec_arr,
        det_skytrans_arr,
        area_cm2,
        observatory.total_throughput,
        deltalambda_nm,
        scene.dist,
        det_sep_arr,
    )

    det_CRbbin_arr = calculate_CRbbin(
        scene.F0,
        scene.Fbinary_list,
        det_skytrans_arr,
        area_cm2,
        observatory.total_throughput,
        deltalambda_nm,
    )

    det_CRbth_arr = (
        calculate_CRbth(
            observation.wavelength,
            area_cm2,
            deltalambda_nm,
            observatory.telescope.temperature,
            lod_rad_arr,
            observatory.epswarmTrcold,
            observatory.detector.QE,
            observatory.detector.dQE,
        )
        * det_omega_lod_arr
    )

    det_CR_arr = (
        det_CRp_arr
        + det_CRbs_arr
        + det_CRbz_arr
        + det_CRbez_arr
        + det_CRbbin_arr
        + det_CRbth_arr
    )

    # --- t_photon_count estimate ---
    t_photon_count_arr = calculate_t_photon_count(det_npix, det_CR_arr)
    if ETC_validation:
        logger.debug("Fixing t_photon_count for validation...")
        t_photon_count_arr = observatory.detector.t_photon_count_input

    # -----------------------------------------------------------------------
    # Photometric aperture size check mask
    # -----------------------------------------------------------------------
    # omega_lod must be larger than one detector pixel solid angle
    phot_aperture_valid_mask = np.atleast_1d(
        omega_lod_at_planet > detpixscale_lod_arr**2
    )

    # -----------------------------------------------------------------------
    # Vectorized count rate calculations at the planet position
    # -----------------------------------------------------------------------

    # --- PLANET COUNT RATE ---
    CRp_arr = calculate_CRp(
        scene.F0,
        scene.Fs_over_F0,
        scene.Fp_over_Fs,
        area_cm2,
        Upsilon_at_planet,
        observatory.total_throughput,
        deltalambda_nm,
    )
    observation.photon_counts["CRp"] = CRp_arr.value

    # --- STELLAR LEAKAGE ---
    CRbs_arr = calculate_CRbs(
        scene.F0,
        scene.Fs_over_F0,
        Istar_at_planet,
        area_cm2,
        observatory.coronagraph.pixscale,
        observatory.total_throughput,
        deltalambda_nm,
    )
    observation.photon_counts["CRbs"] = CRbs_arr.value * omega_lod_at_planet.value
    # --- ZODIACAL LIGHT ---
    CRbz_arr = calculate_CRbz(
        scene.F0,
        scene.Fzodi_list,
        lod_arcsec_arr,
        skytrans_at_planet,
        area_cm2,
        observatory.total_throughput,
        deltalambda_nm,
    )
    observation.photon_counts["CRbz"] = CRbz_arr.value * omega_lod_at_planet.value

    # --- EXOZODIACAL LIGHT ---
    CRbez_arr = calculate_CRbez(
        scene.F0,
        scene.Fexozodi_list,
        lod_arcsec_arr,
        skytrans_at_planet,
        area_cm2,
        observatory.total_throughput,
        deltalambda_nm,
        scene.dist,
        scene.separation,
    )

    observation.photon_counts["CRbez"] = CRbez_arr.value * omega_lod_at_planet.value
    observation.photon_counts["omega_lod"] = omega_lod_at_planet.value

    # --- BINARY / NEIGHBORING STARS ---
    CRbbin_arr = calculate_CRbbin(
        scene.F0,
        scene.Fbinary_list,
        skytrans_at_planet,
        area_cm2,
        observatory.total_throughput,
        deltalambda_nm,
    )

    observation.photon_counts["CRbbin"] = CRbbin_arr.value * omega_lod_at_planet.value

    # --- THERMAL BACKGROUND ---
    CRbth_arr = calculate_CRbth(
        observation.wavelength,
        area_cm2,
        deltalambda_nm,
        observatory.telescope.temperature,
        lod_rad_arr,
        observatory.epswarmTrcold,
        observatory.detector.QE,
        observatory.detector.dQE,
    )
    observation.photon_counts["CRbth"] = CRbth_arr.value * omega_lod_at_planet.value

    CRbd_arr = calculate_CRbd(
        det_npix,
        observatory.detector.DC,
        observatory.detector.RN,
        observatory.detector.tread,
        observatory.detector.CIC,
        t_photon_count_arr,
    )

    observation.photon_counts["CRbd"] = CRbd_arr.value

    # --- NOISE FLOOR ---
    # Calculate CRnf without SNR. Update 1.7.1: SNR now moved outside of this function
    CRnf_s_arr = (
        calculate_CRnf(
            scene.F0,
            scene.Fs_over_F0,
            area_cm2,
            observatory.coronagraph.pixscale,
            observatory.total_throughput,
            deltalambda_nm,
            noisefloor_at_planet,
        )
        * omega_lod_at_planet
    )
    observation.photon_counts["CRnf_s"] = CRnf_s_arr.value

    CRnf_ez_arr = calculate_CRnf_ez(
        CRbez_arr * omega_lod_at_planet.value,
        scene.ez_PPF,
    )
    observation.photon_counts["CRnf_ez"] = CRnf_ez_arr.value

    CRnf_arr = np.sqrt(CRnf_s_arr**2 + CRnf_ez_arr**2)
    observation.photon_counts["CRnf"] = CRnf_arr.value

    # -----------------------------------------------------------------------
    # TOTAL BACKGROUND
    # -----------------------------------------------------------------------

    # Parameters that need to be multiplied by omega_lod
    CRb_arr = (
        CRbs_arr + CRbz_arr + CRbez_arr + CRbbin_arr + CRbth_arr
    ) * omega_lod_at_planet
    observation.photon_counts["CRb"] = CRb_arr.value

    # Add detector noise
    CRb_arr = CRb_arr + CRbd_arr
    observation.photon_counts["CRb+det"] = CRb_arr.value

    # -----------------------------------------------------------------------
    # Compute exposure time or SNR
    # -----------------------------------------------------------------------
    if mode == "exposure_time":
        cp_arr = (
            (CRp_arr + observation.CRb_multiplier * CRb_arr)
            / (
                CRp_arr * CRp_arr - observation.SNR**2 * CRnf_arr * CRnf_arr
            )  # Update 1.7.1: SNR now outside CRnf
            * u.electron
        )

        exptime_arr = (
            observation.SNR**2 * cp_arr * observatory.telescope.toverhead_multi
            + observatory.telescope.toverhead_fixed
        ).to(u.s)

        # Enforce limits
        exptime_arr = u.Quantity(
            np.atleast_1d(np.where(exptime_arr < 0, np.inf, exptime_arr.value)), u.s
        )  # set all negative values (if any) to infinity
        exptime_arr = u.Quantity(
            np.atleast_1d(
                np.where(exptime_arr > observation.td_limit, np.inf, exptime_arr.value)
            ),
            u.s,
        )  # set all values higher than the limit to infinity

        # multiply by number of required rolls to achieve 360 deg coverage
        # if nrolls=1 nothing happens
        exptime_arr = exptime_arr * observatory.coronagraph.nrolls

        observation.exptime = exptime_arr.decompose()

    else:  # mode is "signal_to_noise": we can just say else because we checked that there can only be two keywords

        # cp_arr not used in this mode. Note: This will make the science time in
        # validation variables be 0!
        cp_arr = 0
        time_factors = (
            observation.obstime / observatory.coronagraph.nrolls
            - observatory.telescope.toverhead_fixed
        ) / (
            observatory.telescope.toverhead_multi
            * (CRp_arr + observation.CRb_multiplier * CRb_arr)
        )
        time_factors = time_factors.decompose()

        # UNITS:
        # ([s]/[]-[s])/([electron/s]+[]*[electron/s])
        # [s]/[electron/s]=[s^2/electron]

        # Signal-to-noise
        # observation.fullsnr[ilambd] = (
        #     np.sqrt(
        #         (time_factors * CRp**2)
        #         / (1 * ELECTRON + time_factors * CRnf**2)
        #     )
        #     * DIMENSIONLESS
        # )
        # rewrote the above equation to properly evaluate the SNR when time = inf

        fullsnr_arr = u.Quantity(
            np.atleast_1d(
                np.sqrt(CRp_arr.value**2 / (1 / time_factors.value + CRnf_arr.value**2))
            ),
            DIMENSIONLESS,
        )
        observation.fullsnr = fullsnr_arr

        # UNITS:
        # ([s^2/electron]*[electron/s]^2)/([electron]+[s^2/electron]*[electron/s]^2)=
        # [electron]/[electron] = []

    # -----------------------------------------------------------------------
    # Apply validity masks: set invalid channels to infinity
    # -----------------------------------------------------------------------

    # Channels where the planet falls outside the coronagraph pixel grid
    out_of_bounds = np.atleast_1d(~pixel_valid_mask)
    if np.any(out_of_bounds):
        logger.error(
            "Planet outside coronagraph YIP image for one or more wavelength "
            "channels. Hardcoded infinity results."
        )

    # Channels where the photometric aperture is too small
    small_aperture = np.atleast_1d(pixel_valid_mask & ~phot_aperture_valid_mask)
    if np.any(small_aperture):
        logger.error(
            "Photometric aperture is not large enough for one or more wavelength "
            "channels. Hardcoded infinity results."
        )

    # Channels below the noise floor (exposure_time mode only)
    below_noise_floor = np.atleast_1d(
        (pixel_valid_mask & phot_aperture_valid_mask & (CRp_arr <= CRnf_arr))
        if mode == "exposure_time"
        else np.zeros(len(observation.wavelength), dtype=bool)
    )
    if np.any(below_noise_floor):
        logger.error(
            "Count rate of the planet smaller than the noise floor for one or more "
            "wavelength channels. Hardcoded infinity results."
        )

    invalid_mask = np.atleast_1d(out_of_bounds | small_aperture | below_noise_floor)

    if mode == "exposure_time":
        exptime_vals = np.atleast_1d(observation.exptime.to_value(u.s)).astype(float)
        invalid_mask = np.atleast_1d(np.asarray(invalid_mask, dtype=bool))
        exptime_vals = np.where(invalid_mask, np.inf, exptime_vals)
        observation.exptime = u.Quantity(exptime_vals, u.s)
    else:  # mode is "signal_to_noise": we can just say else because we checked that there can only be two keywords
        fullsnr_vals = np.atleast_1d(
            u.Quantity(observation.fullsnr, DIMENSIONLESS).value
        ).astype(float)
        invalid_mask = np.atleast_1d(np.asarray(invalid_mask, dtype=bool))
        fullsnr_vals = np.where(invalid_mask, np.inf, fullsnr_vals)
        observation.fullsnr = u.Quantity(fullsnr_vals, DIMENSIONLESS)
        observation.SNR = observation.fullsnr.copy()

    # -----------------------------------------------------------------------
    # Verbose / validation output
    # -----------------------------------------------------------------------
    sciencetime_arr = (
        observation.SNR**2 * cp_arr
        if mode == "exposure_time"
        else u.Quantity(np.zeros(len(observation.wavelength)), u.s)
    )

    observation.validation_variables = {
        # --- scene / star ---
        "F0": scene.F0,
        "magstar": scene.mag,
        "dist": scene.dist,
        "nzodis": scene.nzodis,
        "Fs_over_F0": scene.Fs_over_F0 * scene.F0,
        "Fp": scene.Fs_over_F0 * scene.F0 * scene.Fp_over_Fs,
        "Fzodi": scene.Fzodi_list * scene.F0,
        "Fexozodi": scene.Fexozodi_list
        * scene.F0
        / (scene.separation**2 * scene.dist**2),
        # --- telescope / optics ---
        "D": observatory.telescope.diameter,
        "A_cm": area_cm2,
        "toverhead_fixed": observatory.telescope.toverhead_fixed,
        "toverhead_multi": observatory.telescope.toverhead_multi,
        "T_optical": observatory.optics_throughput,
        # --- wavelength grid ---
        "wavelength": observation.wavelength.to(NM),
        "deltalambda_nm": deltalambda_nm,
        "snr": observation.SNR,
        # --- geometry ---
        "lod_rad": lod_rad_arr,
        "lod_arcsec": lod_arcsec_arr,
        "detpixscale_lod": detpixscale_lod_arr,
        "oneopixscale_arcsec": oneopixscale_arcsec_arr,
        "ix": ix_arr,
        "iy": iy_arr,
        # --- detector ---
        "det_DC": observatory.detector.DC,
        "det_RN": observatory.detector.RN,
        "det_CIC": observatory.detector.CIC,
        "det_tread": observatory.detector.tread,
        "det_pixscale_mas": observatory.detector.pixscale_mas,
        "det_omega_lod": det_omega_lod_arr,
        "det_npix": det_npix,
        "t_photon_count": t_photon_count_arr,
        "dQE": observatory.detector.dQE,
        "QE": observatory.detector.QE,
        # --- coronagraph maps at planet position ---
        "omega_lod": omega_lod_at_planet,
        "T_core or photometric_aperture_throughput": Upsilon_at_planet,
        "Istar": Istar_at_planet,
        "Istar*oneopixscale2 in (l/D)^-2": Istar_at_planet
        * (1 / observatory.coronagraph.pixscale) ** 2,
        "skytrans": skytrans_at_planet,
        "skytrans*oneopixscale2  in (l/D)^-2": skytrans_at_planet
        * (1 / observatory.coronagraph.pixscale) ** 2,
        # --- count rates (already include omega where appropriate) ---
        "CRp": CRp_arr,
        "CRbs": CRbs_arr * omega_lod_at_planet,
        "CRbz": CRbz_arr.value * omega_lod_at_planet,
        "CRbez": CRbez_arr.value * omega_lod_at_planet,
        "CRbbin": CRbbin_arr * omega_lod_at_planet,
        "CRbth": CRbth_arr * omega_lod_at_planet,
        "CRb": CRb_arr,
        "CRbd": CRbd_arr,
        "CRnf": CRnf_arr,  # Now stored WITHOUT SNR factor
        # --- results ---
        "sciencetime": sciencetime_arr,
        "exptime": observation.exptime if mode == "exposure_time" else None,
        "fullsnr": observation.fullsnr if mode == "signal_to_noise" else None,
    }
    # Store the full per-wavelength vectors of every intermediate variable.
    if logger.isEnabledFor(logging.DEBUG):
        # IF DEBUG, print them into a file
        utils.print_all_variables(observation, scene, observatory)

        # Save the photon counts for later analysis
        pickle.dump(observation.photon_counts, open("photon_counts.pk", "wb"))

    return
