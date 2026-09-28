from abc import ABC, abstractmethod
import numpy as np
from .units import *
import astropy.units as u
from . import utils
import logging
from pyEDITH import parse_input

logger = logging.getLogger("pyEDITH")


class Filter:
    """
    Represents a spectral channel with wavelength bounds and resolution.
    """

    def __init__(
        self,
        name: str = None,
        low: u.Quantity = None,
        high: u.Quantity = None,
        center: u.Quantity = None,
        bandwidth: float = None,
        resolution: float = None,
        type: str = "IFS",
    ):
        """
        If type is 'IMAGER', save a filter object with wavelength and bandwidth.
        If type is 'IFS', create a filter from either:
        - low, high, resolution (explicit bounds)
        - center, bandwidth, resolution (center wavelength + fractional bandwidth)

        And generate wavelength array for the filters given those parameters.

        Parameters
        ----------
        name : str
            Filter identifier (e.g., "UV", "500nm")
        type : str
            Type of filter ("IFS" or "IMAGER")
        low : u.Quantity, optional
            Lower wavelength bound
        high : u.Quantity, optional
            Upper wavelength bound
        center : u.Quantity, optional
            Center wavelength
        bandwidth : float, optional
            Fractional bandwidth (e.g., 0.2 for 20%)
        resolution : float, optional
            Spectral resolution R = λ/Δλ

        """

        # ---- Set up from user specs ----
        if name is not None:
            self.name = name
        else:
            # Generate a unique number for unnamed filters
            if not hasattr(Filter, "_filter_count"):
                Filter._filter_count = 0
            Filter._filter_count += 1
            self.name = str(Filter._filter_count)
        self.type = type
        self.resolution = resolution

        if low is not None and high is not None:
            self.low = (
                low.to(WAVELENGTH) if isinstance(low, u.Quantity) else low * WAVELENGTH
            )
            self.high = (
                high.to(WAVELENGTH)
                if isinstance(high, u.Quantity)
                else high * WAVELENGTH
            )
            self.center = (self.low + self.high) / 2
            self.bandwidth = ((self.high - self.low) / self.center).to(DIMENSIONLESS)
            self.width = self.high - self.low

        elif center is not None and bandwidth is not None:
            self.center = (
                center.to(WAVELENGTH)
                if isinstance(center, u.Quantity)
                else center * WAVELENGTH
            )
            self.bandwidth = bandwidth * DIMENSIONLESS
            self.low = self.center * (1 - self.bandwidth.value / 2)
            self.high = self.center * (1 + self.bandwidth.value / 2)
            self.width = self.high - self.low

        else:
            raise ValueError(
                f"Filter '{name}': Must provide either (low, high) or (center, bandwidth)"
            )

        # ---- Check and fix type ----
        if self.type not in ["IMAGER", "IFS"]:
            raise ValueError(
                f"Filter '{self.name}': Type must be either IMAGER "
                f"(broadband photometry) or IFS (spectroscopy)."
            )

        # An IFS filter with no resolution cannot build a wavelength grid,
        # so we demote it to broadband IMAGER behaviour.
        if self.type == "IFS" and self.resolution is None:
            logger.warning(
                f"Filter {self.name}: IFS type requires a resolution, "
                f"but none was set. Defaulting to IMAGER (broadband photometry)."
            )
            self.type = "IMAGER"

        # An IMAGER that was handed a resolution is over-specified; the value
        # is meaningless for broadband photometry, so we ignore it (with a nudge).
        elif self.type == "IMAGER" and self.resolution is not None:
            logger.warning(
                f"Filter {self.name}: IMAGER type ignores the supplied "
                f"resolution (R={self.resolution}); using broadband photometry."
            )

        # ---- Calculate wavelength ----

        if self.type == "IMAGER":
            ## BROADBAND PHOTOMETRY
            self.wavelength = np.array([self.center.value]) * WAVELENGTH
            self.delta_wavelength = None

        else:
            ## SPECTROSCOPY
            lam, dlam = utils.generate_wavelength_grid(
                res=self.resolution,
                lam_low=self.low.value,
                lam_high=self.high.value,
            )
            self.wavelength = lam * WAVELENGTH
            self.delta_wavelength = dlam * WAVELENGTH

    def contains(self, wavelength: u.Quantity) -> bool:
        """Check if a wavelength falls within this filter."""
        wavelength = wavelength.to(WAVELENGTH)
        return (wavelength >= self.low) and (wavelength <= self.high)

    def __repr__(self):
        if self.type == "IMAGER" or self.resolution is None:
            return (
                f"Filter(name='{self.name}', type='{self.type}', "
                f"center={self.center:.3f}, bandwidth={self.bandwidth.value:.2%}, "
                f"width={self.width:.3f})"
            )
        else:

            return (
                f"Filter(name='{self.name}', type='{self.type}', "
                f"range={self.low:.3f}-{self.high:.3f}, center={self.center:.3f}, "
                f"R={self.resolution})"
            )


# # # Predefined FULL_CHANNEL_FILTERS (to reproduce legacy behavior)
# FULL_CHANNEL_FILTERS = {
#     "UV": Filter("UV", low=0.2 * u.um, high=0.4 * u.um, resolution=7),
#     "VIS": Filter("VIS", low=0.4 * u.um, high=1.0 * u.um, resolution=140),
#     "NIR": Filter("NIR", low=1.0 * u.um, high=1.8 * u.um, resolution=70),
# }


# def get_hwome_channels(channel_names: list = None) -> list:
#     """will be implemented when hwome is operational."""
#     return NotImplementedError
