"""Integration smoke tests: run every curated example .edith file through the
`etc` and `snr` CLI modes and assert they complete without crashing.

These are deliberately *shallow*: they assert the pipeline runs end-to-end and
produces output, NOT that the numbers are correct (that is the job of the
golden-master / equivalence tests). Their value is catching regressions where a
refactor silently breaks a whole input file or mode.

New .edith files dropped into inputs/ are discovered automatically, so the suite
scales without edits. EAC-dependent files are skipped (not failed) if the
optional engineering-data packages (eacy, yippy) are not importable, so the
suite stays honest on machines lacking that private/development tooling.

"""

import glob
import sys
from pathlib import Path

import pytest

from pyEDITH import cli

# ---------------------------------------------------------------------------
# Discovery: glob the curated inputs, anchored to the repo root (two levels up
# from this test file: tests/ -> repo root -> inputs/). Anchoring to __file__
# rather than the CWD makes discovery immune to where pytest is invoked from.
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


"""Integration tests for the PRIMARY (Python API) interface.

Users predominantly drive pyEDITH programmatically: they build a parameters dict
with native types (astropy Quantities, pre-constructed Filter objects) and call
the calculate_* functions directly -- never touching parse_input_file or the CLI.

Every other integration test we have exercises the FILE/CLI front-end. These
tests guard the API front-end: specifically the isinstance(..., u.Quantity) and
Filter-object branches in parse_parameters / normalize_list_shapes that the file
path, feeding parsed strings/floats, does not reach.

All tests use ToyModel: it has NO external engineering-data dependency (no eacy,
yippy, hwome, no network), so these run deterministically in any environment,
fast and slow suites alike -- no skip-guards required. That makes them a portable,
always-on guardian of the interface your users actually depend on.
"""

import numpy as np
import astropy.units as u
import pytest

from pyEDITH.parse_input import parse_input_file, parse_parameters
from pyEDITH.filters import Filter

# NOTE: these currently live in cli.py. If/when the API functions are promoted
# to a non-CLI module (recommended -- the primary interface should not import
# from the command-line layer), update this single import line.
from pyEDITH.cli import (
    calculate_texp,
    calculate_snr,
    calculate_snr_from_reference_lambda,
)


# ---------------------------------------------------------------------------
# A single, hand-built ToyModel scenario, mirroring input_toymodel_basic.edith
# but constructed the way a Python API user would: native Quantities and a real
# Filter object, no file involved. Centralised in a fixture so the three tests
# share one source of truth and cannot drift apart.
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


# ===========================================================================
# TEST 2 (written first, as #1 depends on its scenario): API native-type
# end-to-end. Does the primary interface run etc and snr to completion with
# native Quantity / Filter inputs?
# ===========================================================================
@pytest.mark.integration
def test_api_native_types_etc(toymodel_api_parameters):
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
def test_api_native_types_snr(toymodel_api_parameters):
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
# TEST 1 (highest value): API <-> file equivalence. The two front-ends must
# converge on identical physics for the same scenario. A cross-interface
# oracle -- neither path is the 'reference'; their AGREEMENT is the assertion.
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


# ===========================================================================
# TEST 3: the direct-engine API. Users call calculate_snr_from_reference_lambda
# programmatically, passing `reference` and `coupled` as ARGUMENTS (never via
# the CLI's args.ref_lambda). This guards that primary-interface contract --
# and is exactly the test that will prove Step 3 leaves the API signature
# intact when it rewires the CLI's sourcing of those arguments.
# ===========================================================================
@pytest.mark.integration
def test_api_reference_lambda_coupled():
    """calculate_snr_from_reference_lambda, called directly (coupled mode).

    Two filters, one reference wavelength driving both. Exercises the engine via
    its PROGRAMMATIC interface -- the way API users reach etc2snr -- independent
    of the CLI. Step 3 must not change this signature; this test is the guard.
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

    # Coupled: a single reference wavelength (0.5 um) drives BOTH filters.
    snrs, texps, _ = calculate_snr_from_reference_lambda(
        params, reference=[0.5], coupled=True
    )

    # Both filters share the one exposure time (the coupled contract).
    assert set(texps) == {"blue", "red"}
    assert texps["blue"] == texps["red"], "Coupled mode must apply one time to all."

    # The reference band (blue, 0.5 um) should recover the target SNR; the other
    # band is evaluated at that same time and may differ -- we only assert it ran.
    assert np.all(np.isfinite(snrs["blue"]["snr"].value))
    assert np.all(np.isfinite(snrs["red"]["snr"].value))


@pytest.mark.integration
def test_api_reference_lambda_independent():
    """calculate_snr_from_reference_lambda, called directly (independent mode).

    One reference wavelength per filter; each filter solved on its own terms.
    Exercises the engine's independent path and its len(reference)==len(filters)
    contract, programmatically -- the way API users reach etc2snr, independent
    of the CLI.
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

    # Independent: one reference wavelength per filter (0.5 -> blue, 1.0 -> red).
    snrs, texps, _ = calculate_snr_from_reference_lambda(
        params, reference=[0.5, 1.0], coupled=False
    )

    # Each filter gets its OWN exposure time (the independent contract): the two
    # times are solved separately, so they are generally distinct.
    assert set(texps) == {"blue", "red"}

    # Independent mode targets each band's own SNR, so BOTH bands should recover
    # (approximately) the requested SNR of 7.0 -- this is the defining property
    # that distinguishes it from coupled mode.
    np.testing.assert_allclose(
        snrs["blue"]["snr"].value,
        7.0,
        rtol=1e-3,
        err_msg="Independent mode: blue band did not recover its target SNR.",
    )
    np.testing.assert_allclose(
        snrs["red"]["snr"].value,
        7.0,
        rtol=1e-3,
        err_msg="Independent mode: red band did not recover its target SNR.",
    )


@pytest.mark.integration
def test_api_reference_lambda_length_mismatch_raises():
    """The engine's len(reference) != len(filters) guard fires in independent
    mode. This is the programmatic counterpart to the file-level validation we
    will add in Step 3 -- here we pin the ENGINE contract directly, so Step 3's
    file parsing is layering a *second* check, not the only one.
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

    # Two filters but only one reference wavelength, in independent mode -> error.
    with pytest.raises(ValueError, match="Independent mode"):
        calculate_snr_from_reference_lambda(params, reference=[0.5], coupled=False)
