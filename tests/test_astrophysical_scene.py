import numpy as np
import pytest
import astropy.units as u
from pyEDITH.units import (
    LUMINOSITY,
    DISTANCE,
    MAGNITUDE,
    ARCSEC,
    ZODI,
    DEG,
    WAVELENGTH,
    PHOTON_FLUX_DENSITY,
    DIMENSIONLESS,
    INV_SQUARE_ARCSEC,
    SPECTRAL_FLUX_DENSITY_CGS,
)
from pyEDITH import Filter

from pyEDITH.astrophysical_scene import (
    AstrophysicalScene,
    calc_flux_zero_point,
    calc_exozodi_flux,
    calc_zodi_flux,
)
import logging

# ============================================================================
# Tests for calc_flux_zero_point
# ============================================================================


def test_calc_flux_zero_point_default_units():
    """Test that default output has correct photon flux density units."""
    wavelength = 500 * u.nm
    f0 = calc_flux_zero_point(wavelength)

    assert isinstance(f0, u.Quantity)
    assert f0.unit.is_equivalent(u.photon * u.s**-1 * u.cm**-2 * u.Hz**-1)


@pytest.mark.parametrize(
    "output_unit,expected_unit",
    [
        ("pcgs", u.photon * u.s**-1 * u.cm**-2 * u.Hz**-1),
        ("cgs", SPECTRAL_FLUX_DENSITY_CGS),
        ("jy", u.Jy),
    ],
)
def test_calc_flux_zero_point_output_units(output_unit, expected_unit):
    """Test various output unit conversions."""
    wavelength = 500 * u.nm
    f0 = calc_flux_zero_point(wavelength, output_unit=output_unit)
    assert f0.unit.is_equivalent(expected_unit)


def test_calc_flux_zero_point_perlambd_pcgs():
    """Test per-wavelength vs per-frequency output for photon CGS units."""
    wavelength = 500 * u.nm

    f0_freq = calc_flux_zero_point(wavelength, output_unit="pcgs", perlambd=False)
    f0_wave = calc_flux_zero_point(wavelength, output_unit="pcgs", perlambd=True)

    assert f0_freq.unit != f0_wave.unit
    assert f0_wave.unit.is_equivalent(PHOTON_FLUX_DENSITY)


def test_calc_flux_zero_point_perlambd_cgs():
    """Test per-wavelength vs per-frequency output for energy CGS units."""
    wavelength = 500 * u.nm

    f0_freq = calc_flux_zero_point(wavelength, output_unit="cgs", perlambd=False)
    f0_wave = calc_flux_zero_point(wavelength, output_unit="cgs", perlambd=True)

    assert f0_freq.unit != f0_wave.unit
    assert f0_wave.unit.is_equivalent(u.erg / (u.s * u.cm**3))


def test_calc_flux_zero_point_ab_vs_johnson():
    """Test AB magnitude system produces correct zero point vs Johnson."""
    wavelength = 500 * u.nm

    f0_johnson = calc_flux_zero_point(wavelength, output_unit="jy", AB=False)
    f0_ab = calc_flux_zero_point(wavelength, output_unit="jy", AB=True)

    assert np.isclose(f0_ab.value, 3631, rtol=1e-6)
    assert f0_johnson.value != f0_ab.value


def test_calc_flux_zero_point_array_wavelengths():
    """Test calculation with array of wavelengths returns correct shape."""
    wavelengths = np.array([0.3, 0.5, 1, 5]) * WAVELENGTH

    f0_array = calc_flux_zero_point(wavelengths, output_unit="pcgs", perlambd=True)

    assert isinstance(f0_array, u.Quantity)
    assert f0_array.shape == wavelengths.shape
    assert f0_array.unit.is_equivalent(PHOTON_FLUX_DENSITY)


def test_calc_flux_zero_point_invalid_output_unit():
    """Test that invalid output unit string raises ValueError."""
    wavelength = 500 * u.nm

    with pytest.raises(ValueError):
        calc_flux_zero_point(wavelength, output_unit="invalid")


def test_calc_flux_zero_point_jy_with_perlambd_raises():
    """Test that Jansky output with per-wavelength flag raises ValueError."""
    wavelength = 500 * u.nm

    with pytest.raises(ValueError):
        calc_flux_zero_point(wavelength, output_unit="jy", perlambd=True)


def test_calc_flux_zero_point_logs_calculation(caplog):
    """Test that flux zero point calculation is logged at INFO level."""
    wavelength = 500 * u.nm

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pyEDITH"):
        calc_flux_zero_point(wavelength)
        assert any(
            "Flux zero point calculated" in record.message for record in caplog.records
        )


# ============================================================================
# Tests for calc_exozodi_flux
# ============================================================================


def test_calc_exozodi_flux_basic_calculation():
    """Test basic exozodi flux calculation returns expected values and units."""
    M_V = 5 * MAGNITUDE
    vmag = 6 * MAGNITUDE
    nexozodis = 1 * ZODI
    lambd = np.array([0.5, 0.6, 0.7]) * WAVELENGTH
    lambdmag = np.array([6, 5.9, 5.8]) * MAGNITUDE

    exozodi_flux = calc_exozodi_flux(M_V, vmag, nexozodis, lambd, lambdmag)

    assert isinstance(exozodi_flux, u.Quantity)
    assert exozodi_flux.unit.is_equivalent(INV_SQUARE_ARCSEC)
    assert len(exozodi_flux) == len(lambd)


def test_calc_exozodi_flux_single_wavelength():
    """Test exozodi flux calculation with single wavelength."""
    M_V = 5 * MAGNITUDE
    vmag = 6 * MAGNITUDE
    nexozodis = 1 * ZODI
    lambd = np.array([0.5]) * WAVELENGTH
    lambdmag = np.array([6]) * MAGNITUDE

    single_flux = calc_exozodi_flux(M_V, vmag, nexozodis, lambd, lambdmag)

    assert isinstance(single_flux, u.Quantity)
    assert single_flux.unit.is_equivalent(INV_SQUARE_ARCSEC)
    assert len(single_flux) == 1


def test_calc_exozodi_flux_scales_linearly():
    """Test that exozodi flux scales linearly with number of zodis."""
    M_V = 5 * MAGNITUDE
    vmag = 6 * MAGNITUDE
    lambd = np.array([0.5, 0.6, 0.7]) * WAVELENGTH
    lambdmag = np.array([6, 5.9, 5.8]) * MAGNITUDE

    flux_1zodi = calc_exozodi_flux(M_V, vmag, 1 * ZODI, lambd, lambdmag)
    flux_3zodi = calc_exozodi_flux(M_V, vmag, 3 * ZODI, lambd, lambdmag)

    assert np.allclose(flux_3zodi.value, 3 * flux_1zodi.value)


def test_calc_exozodi_flux_zero_zodi():
    """Test that zero zodis produces zero flux."""
    M_V = 5 * MAGNITUDE
    vmag = 6 * MAGNITUDE
    lambd = np.array([0.5, 0.6, 0.7]) * WAVELENGTH
    lambdmag = np.array([6, 5.9, 5.8]) * MAGNITUDE

    zero_flux = calc_exozodi_flux(M_V, vmag, 0 * ZODI, lambd, lambdmag)

    assert np.all(zero_flux.value == 0)


def test_calc_exozodi_flux_mismatched_array_lengths():
    """Test that mismatched wavelength and magnitude arrays raise ValueError."""
    M_V = 5 * MAGNITUDE
    vmag = 6 * MAGNITUDE
    nexozodis = 1 * ZODI
    lambd = np.array([0.5, 0.6, 0.7]) * WAVELENGTH
    lambdmag = np.array([6, 5.9]) * MAGNITUDE  # Intentionally shorter

    with pytest.raises(ValueError):
        calc_exozodi_flux(M_V, vmag, nexozodis, lambd, lambdmag)


def test_calc_exozodi_flux_unit_conversion():
    """Test that flux can be converted to dimensionless units."""
    M_V = 5 * MAGNITUDE
    vmag = 6 * MAGNITUDE
    nexozodis = 1 * ZODI
    lambd = np.array([0.5, 0.6, 0.7]) * WAVELENGTH
    lambdmag = np.array([6, 5.9, 5.8]) * MAGNITUDE

    exozodi_flux = calc_exozodi_flux(M_V, vmag, nexozodis, lambd, lambdmag)
    flux_dimensionless = exozodi_flux.to(
        DIMENSIONLESS, equivalencies=u.dimensionless_angles()
    )

    assert isinstance(flux_dimensionless, u.Quantity)
    assert flux_dimensionless.unit == DIMENSIONLESS


def test_calc_exozodi_flux_realistic_values():
    """Test exozodi flux calculation with realistic astrophysical values."""
    M_V = 4.144 * MAGNITUDE
    vmag = 5.444 * MAGNITUDE
    nexozodis = 3 * ZODI
    lambd = np.array([0.5]) * WAVELENGTH
    lambdmag = np.array([5.687]) * MAGNITUDE

    exozodi_flux = calc_exozodi_flux(M_V, vmag, nexozodis, lambd, lambdmag)

    assert np.all(exozodi_flux.value > 0)
    assert np.isclose(exozodi_flux.value, 7.149e-09)


# ============================================================================
# Tests for calc_zodi_flux
# ============================================================================


def test_calc_zodi_flux_basic_calculation():
    """Test basic zodiacal flux calculation returns expected values and units."""
    dec = 45 * DEG
    ra = 180 * DEG
    lambd = np.array([0.5, 1]) * WAVELENGTH
    F0 = np.array([13476, 3.82e03]) * PHOTON_FLUX_DENSITY

    zodi_flux = calc_zodi_flux(dec, ra, lambd, F0)

    assert isinstance(zodi_flux, u.Quantity)
    assert zodi_flux.unit.is_equivalent(INV_SQUARE_ARCSEC)
    assert len(zodi_flux) == len(lambd)


def test_calc_zodi_flux_single_wavelength():
    """Test zodiacal flux calculation with single wavelength."""
    dec = 45 * DEG
    ra = 180 * DEG
    lambd = np.array([0.5]) * WAVELENGTH
    F0 = np.array([13476]) * PHOTON_FLUX_DENSITY

    single_flux = calc_zodi_flux(dec, ra, lambd, F0)

    assert isinstance(single_flux, u.Quantity)
    assert single_flux.unit.is_equivalent(INV_SQUARE_ARCSEC)
    assert len(single_flux) == 1


def test_calc_zodi_flux_unit_conversion():
    """Test that zodiacal flux can be converted to dimensionless units."""
    dec = 45 * DEG
    ra = 180 * DEG
    lambd = np.array([0.5, 1]) * WAVELENGTH
    F0 = np.array([13476, 3.82e03]) * PHOTON_FLUX_DENSITY

    zodi_flux = calc_zodi_flux(dec, ra, lambd, F0)
    flux_dimensionless = zodi_flux.to(
        DIMENSIONLESS, equivalencies=u.dimensionless_angles()
    )

    assert isinstance(flux_dimensionless, u.Quantity)
    assert flux_dimensionless.unit == DIMENSIONLESS


def test_calc_zodi_flux_realistic_values():
    """Test zodiacal flux calculation with realistic sky coordinates."""
    dec = 79.5648101633 * DEG
    ra = 101.5589542028 * DEG
    lambd = np.array([0.5]) * WAVELENGTH
    F0 = np.array([13400]) * PHOTON_FLUX_DENSITY

    zodi_flux = calc_zodi_flux(dec, ra, lambd, F0)

    assert np.all(zodi_flux.value > 0)
    assert np.isclose(zodi_flux.value, 3.52136205e-10, rtol=1e-4)


# ============================================================================
# Tests for AstrophysicalScene initialization
# ============================================================================


def test_astrophysical_scene_initialization():
    """Test AstrophysicalScene initializes with correct F0V value."""
    scene = AstrophysicalScene()

    assert hasattr(scene, "F0V")
    assert np.isclose(scene.F0V.value, 10374.996)
    assert scene.F0V.unit == PHOTON_FLUX_DENSITY


# ============================================================================
# Tests for AstrophysicalScene.load_configuration
# ============================================================================


def test_load_configuration_with_magnitudes():
    """Test loading configuration with magnitude-based inputs."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": 0.5,
        "distance": 10,
        "magV": 5.0,
        "mag": 5.1,
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 180.0,
        "dec": 0.0,
        "separation": 0.1,
        "delta_mag": 20.0,
        "delta_mag_min": 25,
    }

    scene.load_configuration(parameters)
    assert scene.dist == 10 * DISTANCE
    assert scene.vmag == 5.0 * MAGNITUDE
    assert np.allclose(scene.mag.value, 5.1)
    assert scene.mag.unit == MAGNITUDE
    assert np.isclose(scene.stellar_angular_diameter_arcsec.value, 0.00093009345219)
    assert scene.nzodis == 3.0 * ZODI
    assert scene.ra == 180.0 * DEG
    assert scene.dec == 0.0 * DEG
    assert scene.separation == 0.1 * ARCSEC
    assert np.allclose(scene.deltamag.value, 20.0)
    assert scene.deltamag.unit == MAGNITUDE
    assert scene.min_deltamag == 25.0 * MAGNITUDE


def test_load_configuration_magnitude_array_handling():
    """Test that magnitude arrays are properly loaded and converted."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": [0.5, 0.55],
        "distance": 10,
        "magV": 5.0,
        "mag": [5.1, 5.2],
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 180.0,
        "dec": 0.0,
        "separation": 0.1,
        "delta_mag": 20.0,
        "delta_mag_min": 25,
    }

    scene.load_configuration(parameters)

    assert isinstance(scene.mag, u.Quantity)
    assert len(scene.mag) == 2
    assert scene.mag.unit == MAGNITUDE
    assert np.allclose(scene.mag.value, [5.1, 5.2])
    assert np.allclose(scene.Fs_over_F0.value, 10 ** (-0.4 * np.array([5.1, 5.2])))


def test_load_configuration_with_custom_F0():
    """Test loading configuration with custom F0 value."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": 0.5,
        "distance": 10,
        "magV": 5.0,
        "mag": 5.1,
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 180.0,
        "dec": 0.0,
        "separation": 0.1,
        "delta_mag": 20.0,
        "delta_mag_min": 25,
        "F0": 13400,
    }

    scene.load_configuration(parameters)
    assert np.allclose(scene.F0.value, 13400)  # parsed to len(wavelength)
    assert scene.F0.unit == PHOTON_FLUX_DENSITY


def test_load_configuration_with_semimajor_axis():
    """Test loading configuration using semimajor_axis instead of separation."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": 0.5,
        "distance": 10,
        "magV": 5.0,
        "mag": 5.1,
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 180.0,
        "dec": 0.0,
        "semimajor_axis": 1.0,  # 1 AU
        "delta_mag": 20.0,
        "delta_mag_min": 25,
    }

    scene.load_configuration(parameters)

    assert np.isclose(scene.separation.value, 0.1)  # definition of parsec
    assert scene.separation.unit == ARCSEC


def test_load_configuration_missing_separation_and_semimajor_axis():
    """Test that missing both separation and semimajor_axis raises KeyError."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": 0.5,
        "distance": 10,
        "magV": 5.0,
        "mag": 5.1,
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 180.0,
        "dec": 0.0,
        "delta_mag": 20.0,
        "delta_mag_min": 25,
    }

    with pytest.raises(
        KeyError,
        match="Either separation \\[arcsec\\] or semimajor_axis \\[AU\\] must be provided.",
    ):
        scene.load_configuration(parameters)


def test_load_configuration_with_flux_inputs():
    """Test loading configuration with flux-based inputs."""
    scene = AstrophysicalScene()
    flux_parameters = {
        "wavelength": 0.5,
        "distance": 14.8,
        "Fstar_10pc": 1.128e02,
        "FstarV_10pc": 1.244e02,
        "Fp/Fs": 6.3e-8,
        "Fp_min/Fs": 1e-10,
        "stellar_radius": 0.95,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
    }

    scene.load_configuration(flux_parameters)

    assert scene.dist == flux_parameters["distance"] * DISTANCE
    assert np.all(
        scene.Fp_over_Fs == np.array(flux_parameters["Fp/Fs"]) * DIMENSIONLESS
    )
    assert scene.Fp_min_over_Fs == flux_parameters["Fp_min/Fs"] * DIMENSIONLESS
    assert isinstance(scene.Fs_over_F0, u.Quantity)
    assert len(scene.Fs_over_F0) == 1
    assert scene.Fs_over_F0.unit == DIMENSIONLESS


def test_load_configuration_flux_stellar_diameter():
    """Test that stellar angular diameter is calculated correctly from flux inputs."""
    scene = AstrophysicalScene()
    flux_parameters = {
        "wavelength": 0.5,
        "distance": 14.8,
        "Fstar_10pc": 1.128e02,
        "FstarV_10pc": 1.244e02,
        "Fp/Fs": 6.3e-8,
        "Fp_min/Fs": 1e-10,
        "stellar_radius": 0.95,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
    }

    scene.load_configuration(flux_parameters)

    assert np.isclose(scene.stellar_angular_diameter_arcsec.value, 0.00059701944566)


def test_load_configuration_flux_calculations():
    """Test that flux-to-magnitude conversions are calculated correctly."""
    scene = AstrophysicalScene()
    flux_parameters = {
        "wavelength": 0.5,
        "distance": 14.8,
        "Fstar_10pc": 1.128e02,
        "FstarV_10pc": 1.244e02,
        "Fp/Fs": 6.3e-8,
        "Fp_min/Fs": 1e-10,
        "stellar_radius": 0.95,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
    }

    scene.load_configuration(flux_parameters)

    # Verify Fs_over_F0 calculation
    expected_fs_over_f0 = (
        np.array(flux_parameters["Fstar_10pc"])
        * (10 * DISTANCE / scene.dist) ** 2
        / scene.F0.value
    )
    assert np.allclose(scene.Fs_over_F0.value, expected_fs_over_f0)

    # Verify F0 properties
    assert np.all(scene.F0.unit == PHOTON_FLUX_DENSITY)


def test_load_configuration_magnitude_calculations_from_flux():
    """Test that magnitudes are correctly derived from flux inputs."""
    scene = AstrophysicalScene()
    flux_parameters = {
        "wavelength": 0.5,
        "distance": 14.8,
        "Fstar_10pc": 1.128e02,
        "FstarV_10pc": 1.244e02,
        "Fp/Fs": 6.3e-8,
        "Fp_min/Fs": 1e-10,
        "stellar_radius": 0.95,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
    }

    scene.load_configuration(flux_parameters)

    # Check calculated V magnitude
    calculated_vmag = -2.5 * np.log10(
        flux_parameters["FstarV_10pc"]
        * (10 * DISTANCE / scene.dist) ** 2
        / scene.F0V.value
    )
    assert np.isclose(scene.vmag.value, calculated_vmag, rtol=1e-6)

    # Check calculated magnitudes
    calculated_mag = -2.5 * np.log10(
        np.array(flux_parameters["Fstar_10pc"])
        * (10 * DISTANCE / scene.dist) ** 2
        / scene.F0.value
    )
    assert np.allclose(scene.mag.value, calculated_mag, rtol=1e-6)


def test_load_configuration_delta_mag_calculations_from_flux():
    """Test that delta magnitudes are correctly derived from flux ratios."""
    scene = AstrophysicalScene()
    flux_parameters = {
        "wavelength": 0.5,
        "distance": 14.8,
        "Fstar_10pc": 1.128e02,
        "FstarV_10pc": 1.244e02,
        "Fp/Fs": 6.3e-8,
        "Fp_min/Fs": 1e-10,
        "stellar_radius": 0.95,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
    }

    scene.load_configuration(flux_parameters)

    # Check calculated delta_mag
    calculated_delta_mag = -2.5 * np.log10(np.array(flux_parameters["Fp/Fs"]))
    assert np.allclose(scene.deltamag.value, calculated_delta_mag, rtol=1e-6)

    # Check calculated min_delta_mag
    calculated_min_delta_mag = -2.5 * np.log10(flux_parameters["Fp_min/Fs"])
    assert np.isclose(scene.min_deltamag.value, calculated_min_delta_mag, rtol=1e-6)


def test_load_configuration_insufficient_parameters():
    """Test that loading with insufficient parameters raises KeyError."""
    scene = AstrophysicalScene()

    with pytest.raises(KeyError):
        scene.load_configuration(
            {
                "distance": 1.0,
                "wavelength": 0.5,
                "observing_mode": "IMAGER",
                "bandwidth": 0.2,
            }
        )


def test_load_configuration_mixed_magnitude_flux_inputs():
    """Test that mixing magnitude and flux inputs raises KeyError."""
    scene = AstrophysicalScene()
    mixed_parameters = {
        "wavelength": 0.5,
        "distance": 10,
        "magV": 5.0,
        "Fstar_10pc": 1.128e02,
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 180.0,
        "dec": 0.0,
        "separation": 0.1,
        "delta_mag": 20.0,
        "delta_mag_min": 25,
    }

    with pytest.raises(KeyError):
        scene.load_configuration(mixed_parameters)


def test_load_configuration_single_wavelength():
    """Test loading configuration with single wavelength."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": [0.5],
        "distance": 10,
        "magV": 5.0,
        "mag": [5.1],
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 180.0,
        "dec": 0.0,
        "separation": 0.1,
        "delta_mag": 20.0,
        "delta_mag_min": 25,
    }

    scene.load_configuration(parameters)

    assert len(scene.mag) == 1
    assert len(scene.Fs_over_F0) == 1


def test_load_configuration_ifs_mode_missing_fstarv(caplog):
    """Test that missing FstarV_10pc in IFS mode triggers interpolation warning."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "distance": 10.0,
        "Fstar_10pc": [1.128e02, 1.244e02, 1.13e02],
        "Fp/Fs": [6.3e-8, 6.4e-8, 6.5e-8],
        "Fp_min/Fs": 1e-10,
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "observing_mode": "IFS",
    }

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        scene.load_configuration(parameters)
        assert any(
            "not specified in parameters. Calculating internally..." in record.message
            for record in caplog.records
        )


def test_load_configuration_ifs_mode_fstarv_interpolation():
    """Test that FstarV_10pc is correctly interpolated in IFS mode."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "distance": 10.0,
        "Fstar_10pc": [1.128e02, 1.244e02, 1.13e02],
        "Fp/Fs": [6.3e-8, 6.4e-8, 6.5e-8],
        "Fp_min/Fs": 1e-10,
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "observing_mode": "IFS",
    }

    scene.load_configuration(parameters)

    # The interpolated value at 0.55 um should be close to 1.244e02
    expected_fstarv = 1.244e02 * PHOTON_FLUX_DENSITY
    calculated_fstarv = scene.Fs_over_F0[1] * scene.F0[1]  # At 0.55 um
    assert np.isclose(calculated_fstarv, expected_fstarv, rtol=1e-6)


def test_load_configuration_imager_mode_missing_fstarv():
    """Test that missing FstarV_10pc in IMAGER mode raises KeyError."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": 0.5,
        "distance": 10.0,
        "Fstar_10pc": 1.128e02,
        "Fp/Fs": 6.3e-8,
        "Fp_min/Fs": 1e-10,
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "observing_mode": "IMAGER",
    }

    with pytest.raises(KeyError, match="FstarV_10pc missing in parameters."):
        scene.load_configuration(parameters)


def test_load_configuration_ez_ppf_scalar():
    """Test that scalar ez_PPF is propagated to all wavelengths."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "distance": 10.0,
        "Fstar_10pc": [1.128e02, 1.244e02, 1.13e02],
        "FstarV_10pc": 1.244e02,
        "Fp/Fs": [6.3e-8, 6.4e-8, 6.5e-8],
        "Fp_min/Fs": 1e-10,
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "ez_PPF": 100.0,
    }

    scene.load_configuration(parameters)

    assert np.array_equal(
        scene.ez_PPF, parameters["ez_PPF"] * np.ones_like(parameters["Fp/Fs"])
    )


def test_load_configuration_ez_ppf_array():
    """Test that array ez_PPF is correctly loaded."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "distance": 10.0,
        "Fstar_10pc": [1.128e02, 1.244e02, 1.13e02],
        "FstarV_10pc": 1.244e02,
        "Fp/Fs": [6.3e-8, 6.4e-8, 6.5e-8],
        "Fp_min/Fs": 1e-10,
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "ez_PPF": [100.0, 100.0, 100.0],
    }

    scene.load_configuration(parameters)

    assert np.array_equal(
        scene.ez_PPF, parameters["ez_PPF"] * np.ones_like(parameters["Fp/Fs"])
    )


def test_load_configuration_ez_ppf_mismatched_length():
    """Test that mismatched ez_PPF array length raises AssertionError."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "distance": 10.0,
        "Fstar_10pc": [1.128e02, 1.244e02, 1.13e02],
        "FstarV_10pc": 1.244e02,
        "Fp/Fs": [6.3e-8, 6.4e-8, 6.5e-8],
        "Fp_min/Fs": 1e-10,
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "ez_PPF": [100.0, 100.0],  # Wrong length
    }

    with pytest.raises(
        ValueError, match="ez_PPF should be a list of length 3, but it has length 2."
    ):
        scene.load_configuration(parameters)


# ============================================================================
# Tests for AstrophysicalScene.calculate_zodi_exozodi
# ============================================================================


def test_calculate_zodi_exozodi_creates_flux_attributes():
    """Test that calculate_zodi_exozodi creates required flux attributes."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": 0.5,
        "distance": 14.8,
        "magV": 5.84,
        "mag": 5.687,
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "delta_mag": 25.5,
        "delta_mag_min": 25.0,
    }

    scene.load_configuration(parameters)
    scene.calculate_zodi_exozodi(parameters)

    assert hasattr(scene, "Fzodi_list")
    assert hasattr(scene, "Fexozodi_list")
    assert hasattr(scene, "Fbinary_list")


def test_calculate_zodi_exozodi_correct_array_lengths():
    """Test that flux arrays have correct length matching wavelength array."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "distance": 14.8,
        "magV": 5.84,
        "mag": [5.687, 5.632, 5.577],
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "delta_mag": 25.5,
        "delta_mag_min": 25.0,
    }

    scene.load_configuration(parameters)
    scene.calculate_zodi_exozodi(parameters)

    nlambda = len(parameters["wavelength"])
    assert len(scene.Fzodi_list) == nlambda
    assert len(scene.Fexozodi_list) == nlambda
    assert len(scene.Fbinary_list) == nlambda


def test_calculate_zodi_exozodi_flux_values():
    """Test that calculated fluxes have expected properties."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "distance": 14.8,
        "magV": 5.84,
        "mag": [5.687, 5.632, 5.577],
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "delta_mag": 25.5,
        "delta_mag_min": 25.0,
    }

    scene.load_configuration(parameters)
    scene.calculate_zodi_exozodi(parameters)

    assert np.all(scene.Fzodi_list.value >= 0)
    assert np.all(scene.Fexozodi_list.value >= 0)
    assert np.all(scene.Fbinary_list.value == 0)  # Binary flux should be zero


def test_calculate_zodi_exozodi_correct_units():
    """Test that flux arrays have correct units."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": 0.5,
        "distance": 14.8,
        "magV": 5.84,
        "mag": 5.687,
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "delta_mag": 25.5,
        "delta_mag_min": 25.0,
    }

    scene.load_configuration(parameters)
    scene.calculate_zodi_exozodi(parameters)

    assert scene.Fzodi_list.unit.is_equivalent(INV_SQUARE_ARCSEC)
    assert scene.Fexozodi_list.unit.is_equivalent(INV_SQUARE_ARCSEC)
    assert scene.Fbinary_list.unit == DIMENSIONLESS


def test_calculate_zodi_exozodi_absolute_magnitude():
    """Test that absolute magnitude M_V is calculated correctly."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "distance": 14.8,
        "magV": 5.84,
        "mag": [5.687, 5.632, 5.577],
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "delta_mag": 25.5,
        "delta_mag_min": 25.0,
    }

    scene.load_configuration(parameters)
    scene.calculate_zodi_exozodi(parameters)

    expected_M_V = (
        scene.vmag - 5 * np.log10(scene.dist.value) * MAGNITUDE + 5 * MAGNITUDE
    )
    assert np.isclose(scene.M_V.value, expected_M_V.value)


def test_calculate_zodi_exozodi_single_wavelength():
    """Test calculate_zodi_exozodi with single wavelength."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": [0.5],
        "distance": 14.8,
        "magV": 5.84,
        "mag": [5.687],
        "stellar_radius": 1,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "delta_mag": 25.5,
        "delta_mag_min": 25.0,
    }

    scene.load_configuration(parameters)
    scene.calculate_zodi_exozodi(parameters)

    assert len(scene.Fzodi_list) == 1
    assert len(scene.Fexozodi_list) == 1
    assert len(scene.Fbinary_list) == 1


def test_calculate_zodi_exozodi_missing_parameters():
    """Test that calculate_zodi_exozodi raises error when scene is not configured."""
    scene = AstrophysicalScene()

    # Try to calculate without configuring the scene first
    with pytest.raises(AttributeError, match="must be configured before"):
        scene.calculate_zodi_exozodi(
            {
                "wavelength": [0.5, 0.55, 0.6],
            }
        )


# ============================================================================
# Tests for AstrophysicalScene.validate_configuration
# ============================================================================


def test_validate_configuration_with_valid_scene():
    """Test that validate_configuration passes with a properly configured scene."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": 0.5,
        "distance": 14.8,
        "magV": 5.84,
        "mag": 5.687,
        "stellar_radius": 0.95,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "delta_mag": 25.5,
        "delta_mag_min": 25.0,
    }

    scene.load_configuration(parameters)
    scene.calculate_zodi_exozodi(parameters)

    # Should not raise any exception
    scene.validate_configuration()


@pytest.mark.parametrize(
    "attr",
    [
        "dist",
        "stellar_angular_diameter_arcsec",
        "nzodis",
        "ra",
        "dec",
        "separation",
        "F0V",
        "F0",
        "Fzodi_list",
        "Fexozodi_list",
        "Fbinary_list",
        "Fp_over_Fs",
        "Fs_over_F0",
    ],
)
def test_validate_configuration_missing_attributes(attr):
    """Test that validate_configuration raises AttributeError for missing attributes."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": 0.5,
        "distance": 14.8,
        "magV": 5.84,
        "mag": 5.687,
        "stellar_radius": 0.95,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "delta_mag": 25.5,
        "delta_mag_min": 25.0,
    }

    scene.load_configuration(parameters)
    scene.calculate_zodi_exozodi(parameters)

    # Remove the attribute
    temp = getattr(scene, attr)
    delattr(scene, attr)

    with pytest.raises(
        AttributeError, match=f"AstrophysicalScene is missing attribute: {attr}"
    ):
        scene.validate_configuration()

    # Restore for cleanup
    setattr(scene, attr, temp)


@pytest.mark.parametrize(
    "attr,incorrect_value",
    [
        ("dist", 10),
        ("stellar_angular_diameter_arcsec", 0.01),
        ("nzodis", 3),
        ("ra", 236),
        ("dec", 2),
        ("separation", 0.1),
        ("F0V", 1e8),
        ("F0", 1e8),
        ("Fzodi_list", 1e-7),
        ("Fexozodi_list", 1e-8),
        ("Fbinary_list", 0),
        ("Fp_over_Fs", 1e-5),
        ("Fs_over_F0", 1),
    ],
)
def test_validate_configuration_incorrect_types(attr, incorrect_value):
    """Test that validate_configuration raises TypeError for non-Quantity attributes."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": 0.5,
        "distance": 14.8,
        "magV": 5.84,
        "mag": 5.687,
        "stellar_radius": 0.95,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "delta_mag": 25.5,
        "delta_mag_min": 25.0,
    }

    scene.load_configuration(parameters)
    scene.calculate_zodi_exozodi(parameters)

    temp = getattr(scene, attr)
    setattr(scene, attr, incorrect_value)

    with pytest.raises(
        TypeError, match=f"AstrophysicalScene attribute {attr} should be a Quantity"
    ):
        scene.validate_configuration()

    setattr(scene, attr, temp)


@pytest.mark.parametrize(
    "attr,incorrect_value",
    [
        ("dist", 10 * u.km),
        ("stellar_angular_diameter_arcsec", 0.01 * u.rad),
        ("nzodis", 3 * u.m),
        ("ra", 236 * u.rad),
        ("dec", 2 * u.rad),
        ("separation", 0.1 * u.m),
        ("F0V", 1e8 * u.W / (u.m**2)),
        ("F0", 1e8 * u.W / (u.m**2)),
        ("Fzodi_list", 1e-7 * u.W / (u.m**2)),
        ("Fexozodi_list", 1e-8 * u.W / (u.m**2)),
        ("Fbinary_list", 0 * u.m),
        ("Fp_over_Fs", 1e-5 * u.m),
        ("Fs_over_F0", 1 * u.m),
    ],
)
def test_validate_configuration_incorrect_units(attr, incorrect_value):
    """Test that validate_configuration raises ValueError for incorrect units."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": 0.5,
        "distance": 14.8,
        "magV": 5.84,
        "mag": 5.687,
        "stellar_radius": 0.95,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "delta_mag": 25.5,
        "delta_mag_min": 25.0,
    }

    scene.load_configuration(parameters)
    scene.calculate_zodi_exozodi(parameters)

    temp = getattr(scene, attr)
    setattr(scene, attr, incorrect_value)

    with pytest.raises(ValueError, match="has incorrect units"):
        scene.validate_configuration()

    setattr(scene, attr, temp)


@pytest.mark.parametrize(
    "attr",
    [
        "dist",
        "stellar_angular_diameter_arcsec",
        "nzodis",
        "ra",
        "dec",
        "separation",
        "F0V",
        "F0",
        "Fzodi_list",
        "Fexozodi_list",
        "Fbinary_list",
        "Fp_over_Fs",
        "Fs_over_F0",
    ],
)
def test_validate_configuration_invalid_string_values(attr):
    """Test that validate_configuration raises error for string values."""
    scene = AstrophysicalScene()
    parameters = {
        "wavelength": 0.5,
        "distance": 14.8,
        "magV": 5.84,
        "mag": 5.687,
        "stellar_radius": 0.95,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "delta_mag": 25.5,
        "delta_mag_min": 25.0,
    }

    scene.load_configuration(parameters)
    scene.calculate_zodi_exozodi(parameters)

    original_value = getattr(scene, attr)
    setattr(scene, attr, "invalid string")

    with pytest.raises((TypeError, ValueError)):
        scene.validate_configuration()

    setattr(scene, attr, original_value)


# ============================================================================
# Tests for AstrophysicalScene.regrid_spectra
# ============================================================================


def test_regrid_spectra_basic_functionality():
    """Test that regrid_spectra correctly bins down to observation wavelength grid."""
    scene = AstrophysicalScene()

    parameters = {
        "wavelength": np.linspace(0.5, 1.7, 1000),
    }

    class MockObservation:
        wavelength = np.linspace(0.5, 1.7, 100) * WAVELENGTH
        nlambda = len(wavelength)
        delta_wavelength = np.gradient(wavelength)

    observation = MockObservation()

    # Set up scene with random spectra
    scene._input_wavelength = (
        np.asarray(parameters["wavelength"], dtype=np.float64) * WAVELENGTH
    )

    scene.F0 = np.random.randn(len(parameters["wavelength"])) * PHOTON_FLUX_DENSITY
    scene.Fzodi_list = (
        np.random.randn(len(parameters["wavelength"])) * PHOTON_FLUX_DENSITY
    )
    scene.Fexozodi_list = (
        np.random.randn(len(parameters["wavelength"])) * PHOTON_FLUX_DENSITY
    )
    scene.Fbinary_list = (
        np.random.randn(len(parameters["wavelength"])) * PHOTON_FLUX_DENSITY
    )
    scene.Fp_over_Fs = (
        np.random.randn(len(parameters["wavelength"])) * PHOTON_FLUX_DENSITY
    )
    scene.Fs_over_F0 = (
        np.random.randn(len(parameters["wavelength"])) * PHOTON_FLUX_DENSITY
    )
    scene.ez_PPF = 100 * np.ones(len(scene.Fexozodi_list)) * DIMENSIONLESS

    scene.mag = np.random.randn(len(parameters["wavelength"])) * MAGNITUDE

    scene.deltamag = np.random.randn(len(parameters["wavelength"])) * MAGNITUDE

    scene.regrid_spectra(observation)

    # Check that all arrays match observation wavelength grid
    assert len(scene.F0) == observation.nlambda
    assert len(scene.Fzodi_list) == observation.nlambda
    assert len(scene.Fexozodi_list) == observation.nlambda
    assert len(scene.Fbinary_list) == observation.nlambda
    assert len(scene.Fp_over_Fs) == observation.nlambda
    assert len(scene.Fs_over_F0) == observation.nlambda
    assert len(scene.mag) == observation.nlambda
    assert len(scene.deltamag) == observation.nlambda


def test_regrid_spectra_preserves_units():
    """Test that regrid_spectra preserves units of all quantities."""
    scene = AstrophysicalScene()

    parameters = {
        "wavelength": np.linspace(0.5, 1.7, 1000),
    }

    class MockObservation:
        wavelength = np.linspace(0.5, 1.7, 100) * WAVELENGTH
        nlambda = len(wavelength)
        delta_wavelength = np.gradient(wavelength)

    observation = MockObservation()

    # Set up scene with specific units
    scene._input_wavelength = (
        np.asarray(parameters["wavelength"], dtype=np.float64) * WAVELENGTH
    )

    scene.F0 = np.random.randn(len(parameters["wavelength"])) * PHOTON_FLUX_DENSITY
    scene.Fzodi_list = (
        np.random.randn(len(parameters["wavelength"])) * INV_SQUARE_ARCSEC
    )
    scene.Fexozodi_list = (
        np.random.randn(len(parameters["wavelength"])) * INV_SQUARE_ARCSEC
    )
    scene.Fbinary_list = np.random.randn(len(parameters["wavelength"])) * DIMENSIONLESS
    scene.Fp_over_Fs = np.random.randn(len(parameters["wavelength"])) * DIMENSIONLESS
    scene.Fs_over_F0 = np.random.randn(len(parameters["wavelength"])) * DIMENSIONLESS
    scene.ez_PPF = 100 * np.ones(len(scene.Fexozodi_list)) * DIMENSIONLESS
    scene.mag = np.random.randn(len(parameters["wavelength"])) * MAGNITUDE
    scene.deltamag = np.random.randn(len(parameters["wavelength"])) * MAGNITUDE

    original_units = {
        "F0": scene.F0.unit,
        "Fzodi_list": scene.Fzodi_list.unit,
        "Fexozodi_list": scene.Fexozodi_list.unit,
        "Fbinary_list": scene.Fbinary_list.unit,
        "Fp_over_Fs": scene.Fp_over_Fs.unit,
        "Fs_over_F0": scene.Fs_over_F0.unit,
        "mag": scene.mag.unit,
        "deltamag": scene.deltamag.unit,
    }

    scene.regrid_spectra(observation)

    # Check that units are preserved
    assert scene.F0.unit == original_units["F0"]
    assert scene.Fzodi_list.unit == original_units["Fzodi_list"]
    assert scene.Fexozodi_list.unit == original_units["Fexozodi_list"]
    assert scene.Fbinary_list.unit == original_units["Fbinary_list"]
    assert scene.Fp_over_Fs.unit == original_units["Fp_over_Fs"]
    assert scene.Fs_over_F0.unit == original_units["Fs_over_F0"]
    assert scene.mag.unit == original_units["mag"]
    assert scene.deltamag.unit == original_units["deltamag"]


def test_regrid_spectra_1d_case_imager():
    """Test regrid_spectra with 1d interpolation for an imager.

    An imaging filter has no spectral dispersion, so it is characterised by a
    single point at the filter's central wavelength. With only one wavelength
    and no bandwidth to bin over, observation.delta_wavelength is None, which
    drives regrid_spectra down the '1d' interpolation branch.
    """
    scene = AstrophysicalScene()

    parameters = {
        "wavelength": np.linspace(0.5, 1.7, 1000),
    }

    class MockObservation:
        # Imager: a single sample at the filter central wavelength.
        wavelength = np.array([1.1]) * WAVELENGTH
        nlambda = len(wavelength)
        delta_wavelength = None  # no dispersion -> '1d' branch

    observation = MockObservation()

    # Set up scene with random spectra on the fine input grid
    scene._input_wavelength = (
        np.asarray(parameters["wavelength"], dtype=np.float64) * WAVELENGTH
    )

    scene.F0 = np.random.randn(len(parameters["wavelength"])) * PHOTON_FLUX_DENSITY
    scene.Fzodi_list = (
        np.random.randn(len(parameters["wavelength"])) * PHOTON_FLUX_DENSITY
    )
    scene.Fexozodi_list = (
        np.random.randn(len(parameters["wavelength"])) * PHOTON_FLUX_DENSITY
    )
    scene.Fbinary_list = (
        np.random.randn(len(parameters["wavelength"])) * PHOTON_FLUX_DENSITY
    )
    scene.Fp_over_Fs = (
        np.random.randn(len(parameters["wavelength"])) * PHOTON_FLUX_DENSITY
    )
    scene.Fs_over_F0 = (
        np.random.randn(len(parameters["wavelength"])) * PHOTON_FLUX_DENSITY
    )
    scene.ez_PPF = 100 * np.ones(len(scene.Fexozodi_list)) * DIMENSIONLESS

    scene.mag = np.random.randn(len(parameters["wavelength"])) * MAGNITUDE

    scene.deltamag = np.random.randn(len(parameters["wavelength"])) * MAGNITUDE

    scene.regrid_spectra(observation)

    # Each spectrum should collapse to the single imager wavelength point
    assert len(scene.F0) == observation.nlambda
    assert len(scene.Fzodi_list) == observation.nlambda
    assert len(scene.Fexozodi_list) == observation.nlambda
    assert len(scene.Fbinary_list) == observation.nlambda
    assert len(scene.Fp_over_Fs) == observation.nlambda
    assert len(scene.Fs_over_F0) == observation.nlambda
    assert len(scene.mag) == observation.nlambda
    assert len(scene.deltamag) == observation.nlambda

    # nlambda should be exactly 1 for an imager
    assert observation.nlambda == 1
