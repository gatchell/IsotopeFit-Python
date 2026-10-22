"""Core data model.

These classes replace the MATLAB ``handles`` struct and its sub-structs.
They hold data only; processing lives in the other modules.

Index convention
----------------
MATLAB indices are 1-based. Everything in this package is 0-based. The
conversion happens only at the file boundary (``isotopefit.io``), so
``Molecule.minind`` here is one less than ``minind`` in an ``.ifd`` file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import numpy as np
from scipy.interpolate import PPoly

from . import calibration as _cal

# Order of the flags in MATLAB's ``guistatusvector``.
GUI_STATUS_FLAGS = (
    "file_loaded",
    "molecules_loaded",
    "calibrated",
    "bg_corrected",
    "drift_corrected",
    "changed",
    "fitted",
    "cs_selected",
)
DEFAULT_GUI_STATUS = (1, 1, 1, 1, 0, 0, 1, 0)


@dataclass
class PeakShape:
    """Peak shape stored as a MATLAB piecewise polynomial (``pp`` form).

    ``coefs[j]`` holds the coefficients of piece ``j`` in descending powers
    of ``(x - breaks[j])``, exactly as MATLAB's ``mkpp``/``ppval`` use them.
    The mass axis of a shape is normalised: the maximum sits at 0 and the
    FWHM spans -0.5 .. 0.5.
    """

    breaks: np.ndarray
    coefs: np.ndarray

    def __post_init__(self) -> None:
        self.breaks = np.asarray(self.breaks, dtype=float).ravel()
        self.coefs = np.atleast_2d(np.asarray(self.coefs, dtype=float))
        if self.coefs.shape[0] != self.breaks.size - 1:
            raise ValueError("PeakShape needs len(breaks) == len(coefs) + 1")

    @property
    def order(self) -> int:
        return self.coefs.shape[1]

    def to_ppoly(self) -> PPoly:
        # MATLAB's ppval extrapolates with the outermost pieces; so does PPoly.
        return PPoly(self.coefs.T, self.breaks, extrapolate=True)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate like MATLAB's ``ppval``."""
        return self.to_ppoly()(np.asarray(x, dtype=float))

    def copy(self) -> PeakShape:
        return PeakShape(self.breaks.copy(), self.coefs.copy())

    @classmethod
    def gaussian(cls) -> PeakShape:
        """The default peak shape shipped with IsotopeFit (``shapes.mat``)."""
        from scipy.io import loadmat

        with resources.as_file(resources.files(__package__) / "data" / "shapes.mat") as p:
            g = loadmat(p, squeeze_me=True, struct_as_record=False)["gaussian"]
        return cls(g.breaks, g.coefs)


@dataclass
class Molecule:
    """One molecule (or cluster) with its isotope pattern.

    ``peakdata`` is an (n, 2) array of [mass, abundance] per isotopologue.
    """

    name: str
    peakdata: np.ndarray
    minmass: float = np.nan
    maxmass: float = np.nan
    com: float = np.nan  # centre of mass of the isotope pattern
    minind: int = -1  # 0-based index into the spectrum
    maxind: int = -1
    area: float = 0.0
    areaerror: float = np.nan
    rootindex: int = -1  # 0-based position in the original molecule list
    extra: dict[str, Any] = field(default_factory=dict)  # unknown .ifd fields, kept for round trips

    def __post_init__(self) -> None:
        self.peakdata = np.asarray(self.peakdata, dtype=float).reshape(-1, 2)


@dataclass
class Calibration:
    """Mass offset and resolution as functions of mass, plus the peak shape.

    ``*_method`` is one of ``"Flat"``, ``"Polynomial"``, ``"Spline"``,
    ``"PCHIP"`` (case-insensitive, as in MATLAB). ``*_param`` is the flat
    value or the polynomial degree.
    """

    comlist: np.ndarray = field(default_factory=lambda: np.empty(0))
    massoffsetlist: np.ndarray = field(default_factory=lambda: np.empty(0))
    resolutionlist: np.ndarray = field(default_factory=lambda: np.empty(0))
    massoffset_method: str = "Flat"
    resolution_method: str = "Flat"
    massoffset_param: float = 0.0
    resolution_param: float = 3000.0
    namelist: list[str] = field(default_factory=list)
    shape: PeakShape = field(default_factory=PeakShape.gaussian)

    def __post_init__(self) -> None:
        for name in ("comlist", "massoffsetlist", "resolutionlist"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float).ravel())

    def massoffset(self, mass: np.ndarray) -> np.ndarray:
        """``massoffsetbycalibration``"""
        return _cal.calibration_curve(
            self.comlist, self.massoffsetlist, self.massoffset_param, self.massoffset_method, mass
        )

    def resolution(self, mass: np.ndarray) -> np.ndarray:
        """``resolutionbycalibration``"""
        return _cal.calibration_curve(
            self.comlist, self.resolutionlist, self.resolution_param, self.resolution_method, mass
        )

    def sigma(self, mass: np.ndarray) -> np.ndarray:
        """``sigmabycalibration``: Gaussian sigma corresponding to the resolution."""
        mass = np.asarray(mass, dtype=float)
        return mass / self.resolution(mass) / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass
class BackgroundCorrection:
    """Background estimate: the points (``bgm``, ``bgy``) are PCHIP-interpolated."""

    startmass: float = -np.inf
    endmass: float = np.inf
    ndiv: int = 50
    percent: float = 70.0
    bgm: np.ndarray = field(default_factory=lambda: np.empty(0))
    bgy: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __post_init__(self) -> None:
        self.bgm = np.asarray(self.bgm, dtype=float).ravel()
        self.bgy = np.asarray(self.bgy, dtype=float).ravel()


@dataclass
class Project:
    """Everything an ``.ifd`` file stores, plus the processed spectrum.

    ``raw_peakdata`` is the (n, 2) [mass, signal] array as loaded.
    ``start``/``stop`` select the cropped region as a Python slice
    (MATLAB's ``startind:endind`` becomes ``start=startind-1, stop=endind``).
    ``peakdata`` is the cropped, background-subtracted, mass-corrected
    spectrum that fitting works on; call :meth:`update_peakdata` after
    changing calibration or background.
    """

    raw_peakdata: np.ndarray
    start: int = 0
    stop: int | None = None
    molecules: list[Molecule] = field(default_factory=list)
    calibration: Calibration = field(default_factory=Calibration)
    bgcorrection: BackgroundCorrection = field(default_factory=BackgroundCorrection)
    gui_status: dict[str, bool] = field(
        default_factory=lambda: dict(zip(GUI_STATUS_FLAGS, map(bool, DEFAULT_GUI_STATUS)))
    )
    fileinfo: dict[str, str] = field(default_factory=dict)
    peakdata: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.raw_peakdata = np.asarray(self.raw_peakdata, dtype=float).reshape(-1, 2)
        if self.stop is None:
            self.stop = self.raw_peakdata.shape[0]

    def update_peakdata(self) -> np.ndarray:
        """Crop, subtract background and correct the mass axis (as MATLAB's ``load_ifd``)."""
        from .spectrum import crop, subtract_background, subtract_mass_offset

        pd = crop(self.raw_peakdata, self.start, self.stop)
        pd = subtract_background(pd, self.bgcorrection)
        pd = subtract_mass_offset(pd, self.calibration)
        self.peakdata = pd
        return pd
