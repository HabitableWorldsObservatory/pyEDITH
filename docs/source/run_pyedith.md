# Run pyEDITH

## Running pyEDITH via Terminal

pyEDITH provides a command-line interface with three main functionalities: `etc` (Exposure Time Calculator), `snr` (Signal-to-Noise Ratio), and `etc2snr` (Exposure Time to Signal-to-Noise Ratio). You will need to compile a configuration file that the code will read. You can find some examples in `inputs/`.

- Exposure Time Calculation:
```
pyedith etc --edith path/to/your/config.edith
```

- SNR Calculation:

```
pyedith snr --edith path/to/your/config.edith --time 100000
```
Here, 100000 is the exposure time in seconds.

- ETC to SNR Calculation:
```
pyedith etc2snr --edith path/to/your/config.edith
```
In this case, the config file will need to also have a secondary set of parameters see Figure A1 in [Alei, Currie et al. 2026](https://ui.adsabs.harvard.edu/abs/2026arXiv260828477A/abstract).


Add the -v flag to any command for verbose output:

```
pyedith etc --edith path/to/your/config.edith -v
```


## Running pyEDITH from Python (preferred)

This mode offers much more flexibility to run the ETC. We refer to our tutorials (up next in this documentation, or in the `tutorials/` folder).


### Running on Google Colab

If you prefer to run `pyEDITH` in a cloud environment without setting up local environment variables, open a new [Google Colab notebook](https://colab.research.google.com/) and execute the following initialization block in your first cell:

```python
# 1. Install pyEDITH and dependencies
!pip install pyEDITH

# 2. Clone required engineering interfaces
import os, sys
if not os.path.exists('/content/Sci-Eng-Interface'):
    !git clone -q [https://github.com/HWO-GOMAP-Working-Groups/Sci-Eng-Interface.git](https://github.com/HWO-GOMAP-Working-Groups/Sci-Eng-Interface.git) /content/Sci-Eng-Interface

# 3. Set environment variables for the session
os.environ["SCI_ENG_DIR"] = "/content/Sci-Eng-Interface/hwo_sci_eng"
os.environ["YIP_CORO_DIR"]="/content/yips"
```

Once executed, you can import and use `pyEDITH` normally for the rest of your notebook session.