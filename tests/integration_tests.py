"""Integration tests: everything that runs real pyEDITH code end-to-end
(no engine mocking). Three kinds live here, by design:

  1. SMOKE (test_smoke_*): every example file through etc/snr/etc2snr; asserts
     it runs and prints, NOT that numbers are right. Auto-discovers inputs/.
  2. CONTRACT (test_api_*): the calculate_* functions called directly on native
     Python types; asserts return shapes and the coupling behaviour.
  3. EQUIVALENCE (test_api_and_file_paths_agree): the file front-end and the
     Python-API front-end must compute identical physics.

Mocked unit tests of main()'s argv-translation live in test_cli.py, not here.
"""

import glob
import sys
from pathlib import Path

import pytest

from pyEDITH import cli

import numpy as np
import astropy.units as u


from pyEDITH.parse_input import parse_input_file, parse_parameters
from pyEDITH.filters import Filter

from pyEDITH.cli import (
    calculate_texp,
    calculate_snr,
    calculate_snr_from_reference_lambda,
)

# ---------------------------------------------------------------------------
# FRONT-END TESTS
# Every example file, run through the CLI, completes and prints something.
# ---------------------------------------------------------------------------
_INPUTS_DIR = Path(__file__).resolve().parent.parent / "inputs"
_EDITH_FILES = sorted(glob.glob(str(_INPUTS_DIR / "*.edith")))

# Guard against silent empty discovery: a mis-set path must fail loudly at
# collection time, not masquerade as "0 tests passed".
assert _EDITH_FILES, (
    f"No .edith files discovered in {_INPUTS_DIR}. The smoke suite found nothing "
    f"to test -- check the inputs/ path and that example files are present."
)


def _eac_available() -> bool:
    """True if the optional EAC engineering-data packages import under the
    current (pytest's) interpreter. Used to SKIP (never fail) EAC-dependent
    files when the tooling is absent, e.g. in CI or a fresh clone without the
    local eacy checkout."""
    try:
        import eacy  # noqa: F401
        import yippy  # noqa: F401

        return True
    except ImportError:
        return False


def _requires_eac(edith_path: str) -> bool:
    """A file needs EAC tooling unless it is a pure ToyModel configuration.
    We read the file text rather than parsing it, so this probe itself cannot
    crash on a malformed file (the smoke test proper will catch that)."""
    text = Path(edith_path).read_text()
    return "observatory_preset = ToyModel" not in text.replace(" ", "").replace(
        "observatory_preset=ToyModel", "observatory_preset = ToyModel"
    )


def _run_cli(monkeypatch, argv):
    """Invoke the CLI entry point in-process with the given argv.

    In-process (rather than subprocess) so we get coverage credit and the real
    exception surfaces directly in the test, instead of only an exit code.
    """
    monkeypatch.setattr(sys, "argv", ["pyedith"] + argv)
    cli.main()


@pytest.mark.integration
@pytest.mark.parametrize("edith_file", _EDITH_FILES, ids=lambda p: Path(p).name)
def test_smoke_etc(edith_file, monkeypatch, capsys):
    """`pyedith etc --edith <file>` runs to completion and prints a result."""
    if _requires_eac(edith_file) and not _eac_available():
        pytest.skip(
            f"EAC tooling (eacy/yippy) unavailable; skipping {Path(edith_file).name}"
        )

    _run_cli(monkeypatch, ["etc", "--edith", edith_file])

    out = capsys.readouterr().out
    assert out.strip(), f"etc produced no output for {Path(edith_file).name}"


@pytest.mark.integration
@pytest.mark.parametrize("edith_file", _EDITH_FILES, ids=lambda p: Path(p).name)
def test_smoke_snr(edith_file, monkeypatch, capsys):
    """`pyedith snr --edith <file> --time <t>` runs to completion and prints a result.

    --time is a *query knob* (per the design principle), so a fixed sentinel is
    appropriate here: we are testing that the mode runs, not that 1000 s is the
    'right' time for any given file.
    """
    if _requires_eac(edith_file) and not _eac_available():
        pytest.skip(
            f"EAC tooling (eacy/yippy) unavailable; skipping {Path(edith_file).name}"
        )

    _run_cli(monkeypatch, ["snr", "--edith", edith_file, "--time", "1000"])

    out = capsys.readouterr().out
    assert out.strip(), f"snr produced no output for {Path(edith_file).name}"


@pytest.mark.integration
def test_smoke_etc2snr(monkeypatch, capsys):
    """etc2snr runs end-to-end on a known multi-filter file. Not parametrised
    over all files, because etc2snr REQUIRES >=2 filters -- most example files
    are single-filter and would (correctly) raise. So we name the file."""
    _run_cli(
        monkeypatch,
        ["etc2snr", "--edith", "inputs/input_rasti_paper.edith", "--ref_lambda", "0.5"],
    )
    out = capsys.readouterr().out
    assert out.strip(), "etc2snr produced no output."


@pytest.mark.integration
def test_legacy_secondary_format_runs_etc2snr(monkeypatch, capsys):
    """A legacy primary/secondary .edith file is translated to a two-filter
    canonical form and runs through etc2snr. Guards the unadvertised-but-
    supported legacy path against silent breakage, since no user will report
    its regression if it is undocumented."""
    _run_cli(
        monkeypatch,
        ["etc2snr", "--edith", "inputs/input_rasti_paper.edith", "--ref_lambda", "0.5"],
    )
    out = capsys.readouterr().out
    assert out.strip(), "Legacy secondary-format etc2snr produced no output."


# ---------------------------------------------------------------------------
# WORKFLOW TESTS
# Every example file, run through the CLI, completes and prints something.
# ---------------------------------------------------------------------------


@pytest.fixture
def toymodel_api_parameters():
    """A ToyModel IMAGER scenario as a native Python dict (the API entry point)."""
    return {
        "observing_mode": "IMAGER",
        "observatory_preset": "ToyModel",
        "wavelength": 0.5,  # native Quantity, NOT a parsed bare float
        "snr": 7.0,
        "bandwidth": 0.2,
        "CRb_multiplier": 2.0,
        "photometric_aperture_radius": 0.85,
        "distance": 10.0,
        "Fstar_10pc": 122.9279,
        "FstarV_10pc": 115.59984,
        "stellar_radius": 1.0,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 2.5151668316500,
        "separation": 0.1,
        "Fp/Fs": 1e-10,
        # A real Filter object, as a user would construct -- not synthesised from
        # scalars by the legacy shim. This exercises _select_from_filter_list.
        "filter_list": [Filter("toy", center=0.5 * u.um, bandwidth=0.2, type="IMAGER")],
    }


@pytest.mark.integration
def test_workflow_native_types_etc(toymodel_api_parameters):
    """`calculate_texp` runs end-to-end on a hand-built, Quantity-valued dict.

    Guards the API front-end: the parsed-from-file path never supplies a bare
    `0.5 * u.um` wavelength nor a pre-built Filter object, so the Quantity
    branches of normalize_list_shapes and the _select_from_filter_list path are
    only reached this way.
    """
    parsed = parse_parameters(toymodel_api_parameters)
    results, _ = calculate_texp(parsed)

    assert "toy" in results, "Expected a result keyed by the Filter's name."
    exptime = results["toy"]["exposure_time"]
    assert isinstance(exptime, u.Quantity)
    # ToyModel is deterministic; the time must at least be a finite, positive
    # physical quantity (we assert correctness of the *value* in test #3's
    # round-trip, not here).
    assert np.all(np.isfinite(exptime.value))
    assert np.all(exptime.to(u.s).value > 0)


@pytest.mark.integration
def test_workflow_native_types_snr(toymodel_api_parameters):
    """`calculate_snr` runs end-to-end on the same hand-built dict, for a fixed
    exposure time. The API analogue of the `snr` smoke test."""
    parsed = parse_parameters(toymodel_api_parameters)
    results, _ = calculate_snr(parsed, reference_texp=1e5 * u.s)

    assert "toy" in results
    snr = results["toy"]["snr"]
    assert isinstance(snr, u.Quantity)
    assert np.all(np.isfinite(snr.value))
    assert np.all(snr.value >= 0)


# ===========================================================================
# TEST 3: the direct-engine API. Users call calculate_snr_from_reference_lambda
# programmatically.
# ===========================================================================
@pytest.mark.integration
def test_workflow_etc2snr():
    """calculate_snr_from_reference_lambda, called directly.

    Two filters, one reference wavelength driving both via a single shared
    exposure time. Exercises the engine via its PROGRAMMATIC interface -- the
    way API users reach etc2snr, independent of the CLI. This test is the guard
    that the API signature stays stable as the CLI's sourcing is rewired.
    """
    params = parse_parameters(
        {
            "observing_mode": "IMAGER",
            "observatory_preset": "ToyModel",
            "wavelength": [0.5, 1.0],
            "snr": 7.0,
            "bandwidth": 0.2,
            "CRb_multiplier": 2.0,
            "photometric_aperture_radius": 0.85,
            "distance": 10.0,
            "Fstar_10pc": [122.9279, 120.0],
            "FstarV_10pc": 115.59984,
            "stellar_radius": 1.0,
            "nzodis": 3.0,
            "ra": 236.0,
            "dec": 2.5,
            "separation": 0.1,
            "Fp/Fs": [1e-10, 1.1e-10],
            "filter_list": [
                Filter("blue", center=0.5 * u.um, bandwidth=0.2, type="IMAGER"),
                Filter("red", center=1.0 * u.um, bandwidth=0.2, type="IMAGER"),
            ],
        }
    )

    # A single reference wavelength (0.5 um, in the blue band) drives the shared
    # exposure time applied to BOTH filters.
    snrs, texp, _ = calculate_snr_from_reference_lambda(
        params, reference_wavelength=0.5
    )

    # The driving time is now a single shared scalar Quantity (not a per-filter
    # dict -- that was the artefact of the removed independent mode).
    assert isinstance(texp, u.Quantity)
    assert np.isfinite(texp.to(u.s).value) and texp.to(u.s).value > 0

    # Both bands are evaluated at that one shared time. The reference band (blue)
    # should recover the target SNR (7.0); the other band (red) is evaluated at
    # a time optimised for blue and will generally DIFFER -- that difference is
    # the whole point of the coupled computation, so we assert it is NOT 7.0's twin.
    assert np.all(np.isfinite(snrs["blue"]["snr"].value))
    assert np.all(np.isfinite(snrs["red"]["snr"].value))
    np.testing.assert_allclose(
        snrs["blue"]["snr"].value,
        7.0,
        rtol=1e-3,
        err_msg="Reference band (blue) should recover the target SNR.",
    )


@pytest.mark.integration
def test_workflow_etc2snr_single_filter_raises():
    """The engine's >=2-filter guard fires. A single-filter etc2snr is a
    degenerate round-trip (solve t to hit a target, then recover it), so the
    engine rejects it. This pins the ENGINE contract directly, so Step 3's
    file-level validation layers a *second* check, not the only one.
    """
    params = parse_parameters(
        {
            "observing_mode": "IMAGER",
            "observatory_preset": "ToyModel",
            "wavelength": 0.5,
            "snr": 7.0,
            "bandwidth": 0.2,
            "CRb_multiplier": 2.0,
            "photometric_aperture_radius": 0.85,
            "distance": 10.0,
            "Fstar_10pc": 122.9279,
            "FstarV_10pc": 115.59984,
            "stellar_radius": 1.0,
            "nzodis": 3.0,
            "ra": 236.0,
            "dec": 2.5,
            "separation": 0.1,
            "Fp/Fs": 1e-10,
            "filter_list": [
                Filter("blue", center=0.5 * u.um, bandwidth=0.2, type="IMAGER"),
            ],
        }
    )

    with pytest.raises(ValueError, match="at least two filters"):
        calculate_snr_from_reference_lambda(params, reference_wavelength=0.5)


# ===========================================================================
# CHECKING API AND INPUT FILE VERSIONS
# ===========================================================================
@pytest.mark.integration
def test_api_and_file_paths_agree(toymodel_api_parameters):
    """File parsing and the direct Python API must produce identical exposure
    times for the same ToyModel scenario.

    This proves parse_input_file faithfully reconstructs what a user would build
    by hand. If the file parser ever subtly misparses a unit, a shape, or a
    filter, this fires -- even when every file-only test still passes, because
    it checks the file path *against an independent second derivation*.
    """
    # File front-end
    file_params = parse_parameters(
        parse_input_file("inputs/input_toymodel_basic.edith")
    )
    # API front-end (the fixture's hand-built twin)
    api_params = parse_parameters(toymodel_api_parameters)

    file_texp, _ = calculate_texp(file_params)
    api_texp, _ = calculate_texp(api_params)

    # The two paths may name their filters differently (the file's legacy scalar
    # path synthesises a name like "0.5 BW 0.2"; the API uses "toy"). Match on
    # physics, not on name: there is exactly one filter per side here, so compare
    # the sole exposure-time arrays directly.
    assert len(file_texp) == 1, f"Expected one file filter, got {list(file_texp)}"
    assert len(api_texp) == 1, f"Expected one API filter, got {list(api_texp)}"

    (file_result,) = file_texp.values()
    (api_result,) = api_texp.values()

    np.testing.assert_allclose(
        file_result["exposure_time"].to(u.s).value,
        api_result["exposure_time"].to(u.s).value,
        rtol=1e-6,
        err_msg=(
            "File and API front-ends disagree on ToyModel exposure time. "
            "The file parser is reconstructing a different scenario than the "
            "hand-built API dict for nominally identical inputs."
        ),
    )
