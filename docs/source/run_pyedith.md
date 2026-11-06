# Run pyEDITH

## Running pyEDITH via Terminal

pyEDITH provides a command-line interface with three functionalities: `etc`
(Exposure Time Calculator), `snr` (Signal-to-Noise Ratio), and `etc2snr`
(Exposure Time to Signal-to-Noise Ratio). You supply a configuration file;
examples live in `inputs/`.

**Exposure Time Calculation:**
```bash
pyedith etc --edith path/to/your/config.edith
```

Computes the exposure time needed to reach the target SNR in the config file.

**SNR Calculation:**
```bash
pyedith snr --edith path/to/your/config.edith --time 100000
```

Computes the SNR for a given exposure time (`100000` seconds here). `--time` is
a *query knob*: vary it on the command line without editing the config file.

**ETC to SNR Calculation:**
```bash
pyedith etc2snr --edith path/to/your/config.edith --ref_lambda 0.5
```

Solves the exposure time at the reference wavelength (`0.5` microns, on the
filter containing it) and applies that single shared time to *all* filters,
reporting each band's SNR. Requires **at least two filters** (a single-filter
config is rejected -- use `etc` or `snr`). `--ref_lambda` is a *query knob*.
See `inputs/input_multiple_filters.edith`.

**Verbose output:** add `-v` to any command.

### Specifying Multiple Filters

A multi-filter config (required by `etc2snr`) uses parallel `filter_*` arrays;
the i-th entry of each array describes the i-th filter.

- `filter_name` — required whenever any `filter_*` key is present.
- Band edges, in **one** of two mutually exclusive styles:
  - `filter_center` + `filter_bandwidth` (microns), **or**
  - `filter_low` + `filter_high` (microns).
- `filter_resolution` — required for `IFS`, optional for `IMAGER`.

Rules: all `filter_*` arrays must be the same length; a single filter may be a
scalar; wavelengths are in microns; filter `type` comes from `observing_mode`.

Note the distinction: `wavelength` is the **spectral grid** the filters sample
from (with `Fstar_10pc` and `Fp/Fs` giving the spectrum on that grid, same
length as `wavelength`) -- it is *not* a list of bandpasses. The bandpasses are
defined by the `filter_*` keys. See `inputs/input_multiple_filters.edith` for a working 
example.


> **Note — Legacy format.** Files using `secondary_wavelength` / `secondary_*`
> keys (the originally published two-band format) are still read and translated
> internally to the modern two-filter form, but are **deprecated**. Use
> `filter_*` for new work.

## Running pyEDITH from Python (preferred)

This mode offers much more flexibility to run the ETC. We refer to our tutorials (up next in this documentation, or in the `tutorials/` folder).
