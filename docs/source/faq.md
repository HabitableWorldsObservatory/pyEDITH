# Frequently Asked Questions (FAQ)

Welcome to the **pyEDITH** documentation FAQ. This page addresses common setup issues, coronagraph configurations, wavelength/interpolation errors, and model usage recommendations.

---

## 1. Installation & Environment Configuration

<details>
<summary><strong>What are the <code>SCI_ENG_DIR</code> and <code>YIP_CORO_DIR</code> environment variables?</strong></summary>

* **`SCI_ENG_DIR`**: Points to your local clone of the [Sci-Eng-Interface repository](https://github.com/HWO-GOMAP-Working-Groups/Sci-Eng-Interface/tree/main). This directory is required for running the science engine interface.
* **`YIP_CORO_DIR`**: Optional directory used only if you are loading custom coronagraph yield input packages (YIPs). By default, `yippy` handles caching automatically in `YIPPY_CACHE_DIR`.

For full setup instructions, see the [Installation Guide](https://pyedith.readthedocs.io/en/latest/installation.html).
</details>

<details>
<summary><strong>How do I set up the required environment variables?</strong></summary>

Add the following exports to your shell configuration file (`.bashrc` or `.zshrc`) and replace the paths with your local directory structure:

```bash
export SCI_ENG_DIR="/path/to/Sci-Eng-Interface/hwo_sci_eng"
export YIP_CORO_DIR="/path/to/yip/folder" # Optional if using custom YIP files
```

After updating the file, run `source ~/.bashrc` (or `source ~/.zshrc`) to update your current terminal session.
</details>

## 2. Coronagraph Configurations

<details>
<summary><strong>How is the Inner Working Angle (IWA) handled, and why do I get an IWA flag when placing a planet?</strong></summary>
The physical IWA limit is determined by the specific coronagraph selected.  To observe targets at smaller angular separations, switch to a coronagraph preset that allows for a smaller physical IWA.

In earlier versions of `pyEDITH`, we used internal limits (`minimum_IWA` $\approx 1\lambda/D$ and `maximum_OWA` $\approx 30\lambda/D$) strictly to skip unnecessary exposure-time calculations for targets outside viable instrument bounds. This has been removed in newer versions. 
</details>

<details>
<summary><strong>How can I inspect coronagraph parameters or switch coronagraph presets?</strong></summary>

Coronagraph performance characteristics are stored within Yield Input Package (YIP) files managed by `yippy`. Refer to the [Yippy Guide](https://pyedith.readthedocs.io/en/latest/yippy_guide.html) for instructions on inspecting parameters or passing custom YIP files into `pyEDITH`.
</details>

<details>
<summary><strong>What happens if coronagraph (YIP) files are not found locally?</strong></summary>

`pyEDITH` uses the yippy package to handle coronagraph input packages. If a specified coronagraph preset (such as `eac1_aavc_2d`) is not found in your local directory or cache, `pyEDITH` will automatically attempt to fetch and cache it from the remote database.
</details>

---

## 3. Wavelength Channels

<details>
<summary><strong>Why do I get <code>ValueError: A value in x_new is below the interpolation range's minimum value</code> when setting low wavelength bounds?</strong></summary>

This error originates from `scipy.interpolate.interp1d`. It occurs when the channel wavelength bounds (`lam_low`, `lam_high`) extend beyond the wavelength range of the input spectrum file (e.g., `earth_refl_demo`).

* **Solution:** Ensure your input stellar/planetary spectrum covers a slightly wider spectral range than the requested channel bounds. For example, to evaluate down to $\lambda = 0.4\,\mu\text{m}$, the input spectrum must contain data points below $0.4\,\mu\text{m}$ (e.g., starting at $\le 0.35\,\mu\text{m}$).
</details>

<details>
<summary><strong>How should UV observations be configured in pyEDITH?</strong></summary>

UV coronagraphy trade studies for HWO are ongoing, and instrument parameters remain preliminary:
* **Resolution:** Low spectral resolution ($R = 7$) is currently assumed for UV spectroscopy mode.
* **Photometry Option:** Alternatively, UV observations can be simulated in photometry/imaging mode using discrete broad photometric bands rather than spectroscopy mode.
</details>

<details>
<summary><strong>What resolution should I assume for the various channels?</strong></summary>

Coronagraphy trade studies for HWO are ongoing, and instrument parameters remain preliminary. In literature, the following resolutions have been used to describe HWO:
* R=7 (UV)
* R=140 (visible)
* R=70 (NIR).

More up-to-date information can be found on [HWO's website](https://science.nasa.gov/astrophysics/programs/habitable-worlds-observatory/).
</details>

<details>
<summary><strong>What wavelength boundaries should I assume for the various channels?</strong></summary>

Coronagraphy trade studies for HWO are ongoing, and instrument parameters remain preliminary. In literature, common wavelength boundaries are:
* UV: between ~200 - 300 nm and ~500 - 530 nm
* Visible: between ~500 - 530 nm and ~1000 - 1030 nm
* NIR: between ~1000 - 1030 nm and ~1700 - 1800 nm.

More up-to-date information can be found on [HWO's website](https://science.nasa.gov/astrophysics/programs/habitable-worlds-observatory/).
</details>

---
## 4. Input Parameters & Data Handling

<details>
<summary><strong>Can input parameters be single floats when evaluating multiple wavelengths?</strong></summary>

Yes. Parameters that vary with wavelength (such as `wavelength`, `snr`, `Fstar_10pc`, `Fp/Fs`) can accept both floats and lists. If wavelength is provided as a list and another wavelength-dependent parameter is supplied as a single float, `pyEDITH` automatically expands that float into a list matching the length of wavelength.
</details>

<details>
<summary><strong>What information is stored in <code>validation_output</code>?</strong></summary>

The `validation_output` dictionary contains detailed intermediate calculated physical and instrument values, including:
* Zero-point flux (`F0`), stellar flux (`Fs_over_F0`), and planet flux (`Fp`)
* Optical throughput (`T_optical`), core area throughput (`T_core`), and sky transmission (`skytrans`)
* Individual count rate noise components (`CRp`, `CRbs`, `CRbz`, `CRbez`, `CRbd`, `CRnf`)
* Instrument properties such as detector pixel scale (`det_pixscale_mas`) and dark current (`det_DC`)
</details>

---

## 5. Calculations & Multi-Band Strategy

<details>
<summary><strong>How do I compute SNR for secondary bandpasses during simultaneous observations?</strong></summary>

To calculate SNR for secondary channels using an exposure time anchored to a primary reference wavelength, use `calculate_snr_from_reference_lambda`:

```python
from pyEDITH import calculate_snr_from_reference_lambda, Filter

parameters = {
    'wavelength': [0.5, 1.0],
    'snr': 7,
    'Fstar_10pc': [115.6, 100.0],
    'Fp/Fs': [1e-10, 1e-10],
    'filter_list': [
        Filter("VIS1", center=0.5, bandwidth=0.1, type="IMAGER"),
        Filter("NIR", center=1.0, bandwidth=0.1, type="IMAGER")
    ],
    # ... include remaining required scene/observatory parameters
}

# 0.5 microns serves as the reference bandpass
snr_dict, ref_texp, validation_output = calculate_snr_from_reference_lambda(
    parameters, 
    ref_lam=0.5, 
    coupled=True
)
```
</details>
---

## 6. Troubleshooting & Logging

<details>
<summary><strong>How do I suppress detailed log messages during parameter grid searches?</strong></summary>

To clean up console output when running parameter sweeps or loops, adjust the verbosity level using `set_verbosity()`:

```python
from pyEDITH import set_verbosity

# Options: 'debug' (all logs), 'info' (default), 'warning' (warnings & errors), 'quiet' (errors only)
set_verbosity(level='quiet')
```
</details>

<details>
<summary><strong>Why do I get warnings about <code>ez_PPF</code> or <code>noisefloor_PPF</code>?</strong></summary>

* **ez_PPF warning**: Indicates exozodi subtraction precision limit wasn't specified. `pyEDITH` defaults to assuming exozodi subtraction down to the Poisson noise limit (`ez_PPF=∞`). Different values of `ez_PPF` were used in (Currie et al. 2026)[https://iopscience.iop.org/article/10.3847/1538-3881/ae8c31] and we refer to that for details.

* **noisefloor_PPF warning**: Indicates post-processing factor for noise floor was omitted in the input parameters, so `pyEDITH` applies the default value of 30.0.
</details>

What should I keep in mind when looping over observatory architectures or separations?
Whenever you change parameters that alter the underlying optical response or scene geometry (such as telescope diameter D, planet separation, or coronagraph architecture), you must re-initialize the Observatory object within your loop. Coronagraph response curves, core areas, and background terms vary dynamically with these geometry changes.
Would you like to paste the specific email exchanges you replied to so we can extract additional Q&As to add to this page?

## 5. Updates & Releases

### Q: How can I stay informed when a new release of `pyEDITH` is published?

**A:** To receive notifications regarding code updates, releases, and discussions, click the **Watch** button (selecting *All Activity* or *Releases only*) at the top of the `pyEDITH` GitHub repository.


---
*Cannot find a question? (Check the open issues!)[https://github.com/HabitableWorldsObservatory/pyEDITH/issues?q=is%3Aissue+state%3Aopen]*
