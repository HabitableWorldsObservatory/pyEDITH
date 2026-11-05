# Frequently Asked Questions (FAQ)

Welcome to the **pyEDITH** documentation FAQ. This page addresses common setup issues, coronagraph configurations, wavelength/interpolation errors, and model usage recommendations.

---

## Installation & Environment Configuration

:::{dropdown} What are the `SCI_ENG_DIR` and `YIP_CORO_DIR` environment variables?
- **`SCI_ENG_DIR`**: Points to your local clone of the [Sci-Eng-Interface repository](https://github.com/HWO-GOMAP-Working-Groups/Sci-Eng-Interface/tree/main). This directory is required for running the science engine interface.
- **`YIP_CORO_DIR`**: Optional directory used only if you are loading custom coronagraph yield input packages (YIPs). By default, `yippy` handles caching automatically in `YIPPY_CACHE_DIR`.

For full setup instructions, see the [Installation Guide](installation.md).

:::

:::{dropdown} How do I set up the required environment variables?
Add the following exports to your shell configuration file (`.bashrc` or `.zshrc`) and replace the paths with your local directory structure:

```bash
export SCI_ENG_DIR="/path/to/Sci-Eng-Interface/hwo_sci_eng"
export YIP_CORO_DIR="/path/to/yip/folder" # Optional if using custom YIP files
```

After updating the file, run `source ~/.bashrc` (or `source ~/.zshrc`) to update your current terminal session.

:::

---

## Coronagraph Configurations

:::{dropdown} How is the Inner Working Angle (IWA) handled, and why do I get an IWA flag when placing a planet?
The physical IWA limit is determined by the specific coronagraph selected. To observe targets at smaller angular separations, switch to a coronagraph preset that allows for a smaller physical IWA.

In earlier versions of `pyEDITH`, we used internal limits (`minimum_IWA` $\approx 1\lambda/D$ and `maximum_OWA` $\approx 30\lambda/D$) strictly to skip unnecessary exposure-time calculations for targets outside viable instrument bounds. This has been removed in newer versions.

:::

:::{dropdown} How can I inspect coronagraph parameters or switch coronagraph presets?
Coronagraph performance characteristics are stored within Yield Input Package (YIP) files managed by `yippy`. Refer to the [Yippy Guide](yippy_guide.md) for instructions on inspecting parameters or passing custom YIP files into `pyEDITH`.

:::

:::{dropdown} What happens if coronagraph (YIP) files are not found locally?
`pyEDITH` uses the `yippy` package to handle coronagraph input packages. If a specified coronagraph preset (such as `eac1_aavc_2d`) is not found in your local directory or cache, `pyEDITH` will automatically attempt to fetch and cache it from the remote database.

:::

---

## Wavelength Channels

:::{dropdown} Why do I get `ValueError: A value in x_new is below the interpolation range's minimum value` when setting low wavelength bounds?
This error originates from `scipy.interpolate.interp1d`. It occurs when the channel wavelength bounds (`lam_low`, `lam_high`) extend beyond the wavelength range of the input spectrum file (e.g., `earth_refl_demo`).

**Solution:** Ensure your input stellar/planetary spectrum covers a slightly wider spectral range than the requested channel bounds. For example, to evaluate down to $\lambda = 0.4\,\mu\text{m}$, the input spectrum must contain data points below $0.4\,\mu\text{m}$ (e.g., starting at $\le 0.35\,\mu\text{m}$).

:::

:::{dropdown} How should UV observations be configured in pyEDITH?
UV coronagraphy trade studies for HWO are ongoing, and instrument parameters remain preliminary:

- **Resolution:** Low spectral resolution ($R = 7$) is generally assumed for UV spectroscopy mode.
- **Photometry Option:** Alternatively, UV observations can be simulated in photometry/imaging mode using discrete broad photometric bands rather than spectroscopy mode.
- **Throughput and Quantum Efficiency:** The current HWO specifications do not include prescriptions for optical paths and detector quantum efficiency for the UV channel. You must make your own assumptions and feed `T_optical` (total optical throughput) and `QE` (quantum efficiency) for a hypothetical UV channel. For guidelines on how to vary observatory parameters, please refer to the [online documentation](imaging_tutorial.ipynb#3.2.-Vary-observatory-parameters).
- **UV Coronagraph YIP files:** You should make some assumptions about the coronagraph as well. We are working on a ToyModel UV coronagraph YIP file; if you need early access, please [get in touch](mailto:eleonora.alei@nasa.gov).

:::

:::{dropdown} What resolution should I assume for the various channels?
Coronagraphy trade studies for HWO are ongoing, and instrument parameters remain preliminary. In literature, the following resolutions have been used to describe HWO:

- $R = 7$ (UV)
- $R = 140$ (Visible)
- $R = 70$ (NIR)

More up-to-date information can be found on [HWO’s website](https://science.nasa.gov/astrophysics/programs/habitable-worlds-observatory/).

```

```{dropdown} What wavelength boundaries should I assume for the various channels?
Coronagraphy trade studies for HWO are ongoing, and instrument parameters remain preliminary. In literature, common wavelength boundaries are:

- **UV:** between ~200–300 nm and ~500–530 nm
- **Visible:** between ~500–530 nm and ~1000–1030 nm
- **NIR:** between ~1000–1030 nm and ~1700–1800 nm

More up-to-date information can be found on [HWO’s website](https://science.nasa.gov/astrophysics/programs/habitable-worlds-observatory/).

```

---

## Input Parameters & Data Handling

```{dropdown} What input spectra can I use in pyEDITH, and how do I load spectral templates?
You can use any custom stellar or planetary spectrum (such as planet-to-star flux ratio $F_p/F_s$ or stellar flux $F_{\text{star}}$) as long as it spans your requested wavelength grid.

For convenience, standard planetary and stellar spectral templates can be retrieved directly from the connected [`hwotools`](https://github.com/HabitableWorldsObservatory/hwotools) GitHub repository or local files. As demonstrated in the [ExoEarth Spectroscopy Tutorial](exoearth_spectroscopy_tutorial.ipynb), templates (e.g., Modern Earth, Archean Earth, or stellar models) can be loaded and interpolated onto your target wavelength grid:

```python
import numpy as np

# Load template wavelength and flux ratio data from hwotools or local data files
# Interpolate onto your active pyEDITH wavelength grid
lam_grid = parameters["wavelength"]
parameters["Fp/Fs"] = np.interp(lam_grid, template_wavelength, template_fp_fs)
```

```

:::{dropdown} Can input parameters be single floats when evaluating multiple wavelengths?
Yes. Parameters that vary with wavelength (such as `wavelength`, `snr`, `Fstar_10pc`, `Fp/Fs`) can accept both floats and lists. If wavelength is provided as a list and another wavelength-dependent parameter is supplied as a single float, `pyEDITH` automatically expands that float into a list matching the length of wavelength.

:::

:::{dropdown} What information is stored in `validation_output`?
The `validation_output` dictionary contains detailed intermediate calculated physical and instrument values, including:

* Zero-point flux (`F0`), stellar flux (`Fs_over_F0`), and planet flux (`Fp`)
* Optical throughput (`T_optical`), core area throughput (`T_core`), and sky transmission (`skytrans`)
* Individual count rate noise components (`CRp`, `CRbs`, `CRbz`, `CRbez`, `CRbd`, `CRnf`)
* Instrument properties such as detector pixel scale (`det_pixscale_mas`) and dark current (`det_DC`)

:::

---

## Troubleshooting & Logging

:::{dropdown} How do I suppress detailed log messages during parameter grid searches?
To clean up console output when running parameter sweeps or loops, adjust the verbosity level using `set_verbosity()`:

```python
from pyEDITH import set_verbosity

# Options: 'debug' (all logs), 'info' (default), 'warning' (warnings & errors), 'quiet' (errors only)
set_verbosity(level='quiet')
```

:::

:::{dropdown} Why do I get warnings about `ez_PPF` or `noisefloor_PPF`?

* **ez_PPF warning**: Indicates exozodi subtraction precision limit wasn’t specified. `pyEDITH` defaults to assuming exozodi subtraction down to the Poisson noise limit ($ez\_PPF = \infty$). Different values of $ez\_PPF$ were used in [Currie et al. 2026](https://iopscience.iop.org/article/10.3847/1538-3881/ae8c31) and we refer to that paper for details.
* **noisefloor_PPF warning**: Indicates post-processing factor for noise floor was omitted in the input parameters, so `pyEDITH` applies the default value of 30.0.

:::

:::{dropdown} Why do I get a warning that a parameter is "locked" and cannot be overridden?
When using certain modes (such as EAC configurations or Yield Input Packages), specific parameters are locked to model-provided values to prevent accidental mismatches. If you attempt to pass in a different value for a locked parameter, `pyEDITH` will ignore your input, use the default model value, and issue a warning.

**How to intentionally override a locked parameter:**
If you intentionally want to bypass a locked parameter (e.g., for a specific trade study), you must explicitly add that parameter’s name to an `overrides` list in your parameters dictionary. `pyEDITH` will then accept your custom value and log a warning stating the parameter was explicitly overridden per user request.

*Notes on Overrides:*
- To prevent typos, any unrecognized names in your overrides list will raise a `ValueError`.
- If you override a parameter with an array of values, the array length must match your input wavelength array so it can be correctly rebinned.

More info on the [online documentation](imaging_tutorial.ipynb#3.2.-Vary-observatory-parameters).

:::

:::{dropdown} What should I keep in mind when looping over observatory architectures or separations?
Whenever you change parameters that alter the underlying optical response or scene geometry (such as telescope diameter $D$, planet separation, or coronagraph architecture), you must re-initialize the `Observatory` object within your loop. Coronagraph response curves, core areas, and background terms vary dynamically with these geometry changes.

:::

---

## Updates & Releases

:::{dropdown} How can I stay informed when a new release of `pyEDITH` is published?
To receive notifications regarding code updates, releases, and discussions, click the **Watch** button (selecting *All Activity* or *Releases only*) at the top of the [`pyEDITH` GitHub repository](https://github.com/HabitableWorldsObservatory/pyEDITH).

:::

---

*Cannot find a question? [Check the open issues!](https://github.com/HabitableWorldsObservatory/pyEDITH/issues?q=is%3Aissue+state%3Aopen)*
