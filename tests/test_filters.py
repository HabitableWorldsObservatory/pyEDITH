import pytest
import numpy as np
import astropy.units as u
from pyEDITH.filters import Filter  # , FULL_CHANNEL_FILTERS


def test_filter_from_low_high():
    """Test creating filter with explicit bounds."""
    f = Filter("test", low=0.4 * u.um, high=0.8 * u.um, resolution=100)

    assert f.name == "test"
    assert np.isclose(f.low.value, 0.4)
    assert f.low.unit == u.um
    assert np.isclose(f.high.value, 0.8)
    assert f.high.unit == u.um
    assert np.isclose(f.center.value, 0.6)
    assert f.center.unit == u.um
    assert np.isclose(f.bandwidth.value, 0.6667, atol=0.001)
    assert np.isclose(f.width.value, 0.4)
    assert f.width.unit == u.um


def test_filter_from_center_bandwidth():
    """Test creating filter from center wavelength and bandwidth."""
    f = Filter("test", center=0.6 * u.um, bandwidth=0.5, resolution=100)

    assert np.isclose(f.center.value, 0.6)
    assert f.center.unit == u.um
    assert np.isclose(f.low.value, 0.45)
    assert f.low.unit == u.um
    assert np.isclose(f.high.value, 0.75)
    assert f.high.unit == u.um
    assert np.isclose(f.bandwidth.value, 0.5)
    assert np.isclose(f.width.value, 0.3)
    assert f.width.unit == u.um


def test_filter_missing_parameters():
    """Test that filter raises error with insufficient parameters."""
    with pytest.raises(ValueError, match="Must provide either"):
        Filter("test", low=0.4 * u.um, resolution=100)

    with pytest.raises(ValueError, match="Must provide either"):
        Filter("test", center=0.6 * u.um, resolution=100)


def test_filter_imager_type():
    """Test broadband imaging filter (no wavelength array)."""
    f = Filter(
        "photometric", center=0.55 * u.um, bandwidth=0.2, resolution=None, type="IMAGER"
    )

    assert f.type == "IMAGER"
    assert f.wavelength == f.center
    assert f.delta_wavelength is None


def test_filter_ifs_type():
    """Test IFS filter creates wavelength array."""
    f = Filter(
        "spectroscopic", low=0.5 * u.um, high=0.6 * u.um, resolution=100, type="IFS"
    )

    assert f.type == "IFS"
    assert hasattr(f.wavelength, "__len__")  # Is array
    assert f.delta_wavelength is not None
    assert len(f.wavelength) > 1


def test_filter_invalid_type():
    """Test invalid filter type raises error."""
    with pytest.raises(ValueError, match="Type must be either"):
        Filter("bad", center=0.5 * u.um, bandwidth=0.1, resolution=100, type="INVALID")


def test_ifs_without_resolution_demotes_to_imager(caplog):
    """IFS with no resolution cannot build a grid; it is demoted to broadband."""
    import logging

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        f = Filter("f", low=0.5 * u.um, high=0.6 * u.um, resolution=None, type="IFS")

    assert f.type == "IMAGER"  # mutated
    assert len(f.wavelength) == 1  # broadband
    assert f.delta_wavelength is None
    assert any("requires a resolution" in r.message for r in caplog.records)


def test_imager_with_resolution_warns_but_ignores(caplog):
    """IMAGER with a resolution set: the value is meaningless and ignored, with a warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        f = Filter(
            "f", center=0.55 * u.um, bandwidth=0.2, resolution=140, type="IMAGER"
        )

    assert f.type == "IMAGER"
    assert len(f.wavelength) == 1  # still broadband
    assert f.delta_wavelength is None  # resolution had no effect
    assert any("ignores the supplied" in r.message for r in caplog.records)


def test_wavelength_coverage():
    """Test that wavelength array covers the filter bounds."""
    f = Filter("test", low=1.0 * u.um, high=1.5 * u.um, resolution=50)

    assert f.wavelength.min() >= f.low
    assert f.wavelength.max() <= f.high


def test_spectral_resolution():
    """Test that generated wavelengths match requested resolution."""
    R = 100
    f = Filter("test", low=0.5 * u.um, high=0.6 * u.um, resolution=R)

    # Check resolution at centerpoint: R = λ/Δλ
    center_idx = len(f.wavelength) // 2
    measured_R = f.wavelength[center_idx] / f.delta_wavelength[center_idx]

    assert np.isclose(measured_R.value, R, rtol=0.1)


def test_wavelength_units():
    """Test that wavelength arrays have correct units."""
    f = Filter("test", low=500 * u.nm, high=600 * u.nm, resolution=50)

    assert f.wavelength.unit == u.um
    assert f.delta_wavelength.unit == u.um


def test_contains_inside():
    """Test wavelength inside filter bounds."""
    f = Filter("test", low=0.4 * u.um, high=0.8 * u.um, resolution=50)

    assert f.contains(0.5 * u.um)
    assert f.contains(0.6 * u.um)
    assert f.contains(0.7 * u.um)


def test_contains_outside():
    """Test wavelength outside filter bounds."""
    f = Filter("test", low=0.4 * u.um, high=0.8 * u.um, resolution=50)

    assert not f.contains(0.3 * u.um)
    assert not f.contains(0.9 * u.um)


def test_contains_boundary():
    """Test wavelength at filter boundaries."""
    f = Filter("test", low=0.4 * u.um, high=0.8 * u.um, resolution=50)

    assert f.contains(0.4 * u.um)
    assert f.contains(0.8 * u.um)


def test_contains_unit_conversion():
    """Test contains() works with different units."""
    f = Filter("test", low=400 * u.nm, high=800 * u.nm, resolution=50)

    assert f.contains(0.5 * u.um)
    assert f.contains(600 * u.nm)


def test_repr():
    """Test string representation."""
    f = Filter("VIS", low=0.4 * u.um, high=0.8 * u.um, resolution=100)

    repr_str = repr(f)

    assert (
        "Filter(name='VIS', type='IFS', range=0.400 um-0.800 um, center=0.600 um, R=100)"
        in repr_str
    )

    f = Filter("VIS", low=0.4 * u.um, high=0.8 * u.um, type="IMAGER")

    repr_str = repr(f)
    assert (
        "Filter(name='VIS', type='IMAGER', center=0.600 um, bandwidth=66.67%, width=0.400 um)"
        in repr_str
    )


# def test_full_channel_filters_exist():
#     """Test that predefined filters are available."""
#     assert "UV" in FULL_CHANNEL_FILTERS
#     assert "VIS" in FULL_CHANNEL_FILTERS
#     assert "NIR" in FULL_CHANNEL_FILTERS


# def test_full_channel_filters_coverage():
#     """Test that predefined filters cover expected wavelength ranges."""
#     uv = FULL_CHANNEL_FILTERS["UV"]
#     vis = FULL_CHANNEL_FILTERS["VIS"]
#     nir = FULL_CHANNEL_FILTERS["NIR"]

#     # UV: 0.2-0.4 μm
#     assert uv.low == 0.2 * u.um
#     assert uv.high == 0.4 * u.um

#     # VIS: 0.4-1.0 μm
#     assert vis.low == 0.4 * u.um
#     assert vis.high == 1.0 * u.um

#     # NIR: 1.0-1.8 μm
#     assert nir.low == 1.0 * u.um
#     assert nir.high == 1.8 * u.um


# def test_full_channel_filters_resolutions():
#     """Test that predefined filters have expected resolutions."""
#     assert FULL_CHANNEL_FILTERS["UV"].resolution == 7
#     assert FULL_CHANNEL_FILTERS["VIS"].resolution == 140
#     assert FULL_CHANNEL_FILTERS["NIR"].resolution == 70


def test_narrow_line_filter():
    """Test narrow-band filter for emission line imaging."""
    # H-alpha filter at 656.3 nm with ~1% bandwidth
    f = Filter(
        "H-alpha",
        center=656.3 * u.nm,
        bandwidth=0.01,
        resolution=None,
        type="IMAGER",
    )

    assert f.contains(656.3 * u.nm)
    assert not f.contains(650 * u.nm)
    assert not f.contains(660 * u.nm)


def test_high_resolution_spectroscopy():
    """Test high-resolution spectrograph."""
    # R=50000 spectrograph in visible
    f = Filter("echelle", low=0.5 * u.um, high=0.6 * u.um, resolution=50000, type="IFS")

    assert len(f.wavelength) > 1000  # Should have many spectral elements
    assert f.delta_wavelength is not None


def test_wide_photometric_band():
    """Test wide photometric band like SDSS g-band."""
    # SDSS g-band: ~400-550 nm
    f = Filter(
        "g-band", low=400 * u.nm, high=550 * u.nm, resolution=None, type="IMAGER"
    )

    assert f.center.value == pytest.approx(0.475, rel=0.01)
    assert f.center.unit == u.um
    assert f.contains(470 * u.nm)


def test_unnamed_filter_gets_numeric_name():
    """Test that filters without names get auto-generated numeric names."""
    f = Filter(low=0.5 * u.um, high=0.6 * u.um, resolution=100)

    assert f.name is not None
    assert f.name.isdigit()  # Name should be a string representation of a number


def test_unnamed_filters_get_unique_names():
    """Test that multiple unnamed filters get unique sequential names."""
    # Reset the counter if it exists
    if hasattr(Filter, "_filter_count"):
        delattr(Filter, "_filter_count")

    f1 = Filter(low=0.4 * u.um, high=0.5 * u.um, resolution=100)
    f2 = Filter(low=0.5 * u.um, high=0.6 * u.um, resolution=100)
    f3 = Filter(low=0.6 * u.um, high=0.7 * u.um, resolution=100)

    # Names should be different
    assert f1.name != f2.name
    assert f2.name != f3.name
    assert f1.name != f3.name

    # Names should be sequential numbers
    assert f1.name == "1"
    assert f2.name == "2"
    assert f3.name == "3"


def test_named_and_unnamed_filters_mix():
    """Test that named and unnamed filters can coexist."""
    # Reset counter
    if hasattr(Filter, "_filter_count"):
        delattr(Filter, "_filter_count")

    f1 = Filter("custom_name", low=0.4 * u.um, high=0.5 * u.um, resolution=100)
    f2 = Filter(low=0.5 * u.um, high=0.6 * u.um, resolution=100)
    f3 = Filter("another_name", low=0.6 * u.um, high=0.7 * u.um, resolution=100)

    assert f1.name == "custom_name"
    assert f2.name == "1"  # First unnamed filter
    assert f3.name == "another_name"


def test_unnamed_filter_counter_persists():
    """Test that the unnamed filter counter persists across instances."""
    # Get current counter value
    initial_count = getattr(Filter, "_filter_count", 0)

    f1 = Filter(low=0.4 * u.um, high=0.5 * u.um, resolution=100)
    count_after_f1 = int(f1.name)

    f2 = Filter(low=0.5 * u.um, high=0.6 * u.um, resolution=100)
    count_after_f2 = int(f2.name)

    # Counter should increment
    assert count_after_f2 == count_after_f1 + 1
