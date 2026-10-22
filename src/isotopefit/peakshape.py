"""Measuring the peak shape from the spectrum (the computations behind ``peak_shape_generator.m``).

The idea: in a mass range, the measured signal is the peak shape convolved
with a comb of delta peaks, one per isotopologue at its mass, weighted by
the fitted area times abundance. Dividing the Fourier transforms recovers
the peak shape (the "convolution core"). The user then places spline knots
on the core and fits them; the resulting spline becomes the new shape after
normalising it to unit area, centring its maximum at 0 and scaling its
FWHM to 1.

Workflow::

    core = convolution_core_for(peakdata_section, molecules, calibration)
    x, y = core.knots_x, core.knots_y                    # start from the current shape
    x, y = fit_knots(core, x, y, fixed)                  # "Fit" button
    shape = shape_from_knots(x, y)                       # what the editor shows
    new_cal = finalize_shape(shape, calibration, core)   # "OK" button
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from .model import Calibration, Molecule, PeakShape
from .pattern import adapt_peak_shape
from .spectrum import mass2ind


def molecule_stems(molecules: Sequence[Molecule], massaxis: np.ndarray) -> np.ndarray:
    """``create_molecule_stems``: area x abundance of every isotopologue at its nearest bin.

    Peaks falling on the first or last bin are left out, as in MATLAB.
    """
    massaxis = np.asarray(massaxis, dtype=float)
    out = np.zeros(massaxis.size)
    for mol in molecules:
        for mass, abundance in mol.peakdata:
            i = mass2ind(massaxis, mass)
            if 0 < i < massaxis.size - 1:
                out[i] += mol.area * abundance
    return out


def convolution_core(peakdata: np.ndarray, stems: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``get_convolution_core``: deconvolve the signal by the stems with FFTs.

    Returns ``(core, x)`` with ``x`` centred on 0. As in MATLAB, ``x`` spans
    twice the mass range of the section; :func:`convolution_core_for`
    compensates for that.
    """
    n = peakdata.shape[0]
    core = np.real(np.fft.ifftshift(np.fft.ifft(np.fft.fft(peakdata[:, 1]) / np.fft.fft(stems))))
    x = (np.arange(n) / (n - 1) - 0.5) * 2 * n
    return core, x * np.mean(np.diff(peakdata[:, 0]))


@dataclass
class ConvolutionCore:
    """The measured peak shape in the normalised units of :class:`PeakShape`."""

    masses: np.ndarray  # normalised mass axis, maximum of K at 0
    K: np.ndarray  # measured core, scaled to the height of the current shape
    fwhm: float  # FWHM in mass units at the centre of the section
    knots_x: np.ndarray  # inner breaks of the current shape: starting knots
    knots_y: np.ndarray


def convolution_core_for(peakdata: np.ndarray, molecules: Sequence[Molecule],
                         calibration: Calibration) -> ConvolutionCore:
    """Prepare the core as ``peak_shape_generator.m`` does when it opens.

    ``peakdata`` is the (mass-corrected) section of the spectrum to use and
    ``molecules`` the fitted molecules in it.
    """
    peakdata = np.asarray(peakdata, dtype=float)
    centre = float(np.mean(peakdata[:, 0]))
    fwhm = centre / float(calibration.resolution(centre))
    K, masses = convolution_core(peakdata, molecule_stems(molecules, peakdata[:, 0]))
    # convolution_core's axis spans twice the section, hence 2 * fwhm: the
    # result has the unit FWHM of a PeakShape.
    masses = masses / (2 * fwhm)
    shape = calibration.shape
    K = K * (np.max(shape(masses)) / np.max(K))
    masses = masses - masses[np.argmax(K)]
    knots_x = shape.breaks[1:-1].copy()
    return ConvolutionCore(masses, K, fwhm, knots_x, shape(knots_x))


def spline_shape(x: np.ndarray, y: np.ndarray) -> PeakShape:
    """``interpfcn`` of the peak shape editor.

    Not-a-knot cubic spline through the points (repeated x values dropped),
    with the outermost pieces set to zero so the shape ends cleanly.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.concatenate([[True], np.diff(x) != 0])
    cs = CubicSpline(x[keep], y[keep], bc_type="not-a-knot")
    coefs = cs.c.T.copy()  # MATLAB pp order: one row per piece, highest power first
    coefs[0] = 0
    coefs[-1] = 0
    return PeakShape(cs.x, coefs)


def shape_from_knots(x: np.ndarray, y: np.ndarray) -> PeakShape:
    """``update_shape``: sort the knots, pin both ends to zero, add a zero point outside each end."""
    order = np.argsort(x, kind="stable")
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order].copy()
    y[0] = y[-1] = 0.0
    return spline_shape(np.concatenate([[x[0] - 1], x, [x[-1] + 1]]),
                        np.concatenate([[0.0], y, [0.0]]))


def _apply_offsets(offsets, x, y, fixed):
    free = ~fixed
    n = int(free.sum())
    x, y = x.copy(), y.copy()
    x[free] -= offsets[:n]
    y[free] -= offsets[n:]
    return x, y


def _msd(offsets, masses, K, x, y, fixed) -> float:
    xx, yy = _apply_offsets(offsets, x, y, fixed)
    order = np.argsort(np.concatenate([[xx[0] - 1], xx, [xx[-1] + 1]]), kind="stable")
    px = np.concatenate([[xx[0] - 1], xx, [xx[-1] + 1]])[order]
    py = np.concatenate([[0.0], yy, [0.0]])[order]
    try:
        model = spline_shape(px, py)(masses)
    except ValueError:  # knots collapsed onto each other
        return np.inf
    return float(np.sum((K - model) ** 2))


def fit_knots(core: ConvolutionCore, x: np.ndarray, y: np.ndarray,
              fixed: Sequence[bool] | None = None, xsearch: float = 0.5,
              ysearch: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``fitpoints``: move the free knots so the spline matches the measured core.

    Each free knot may move by up to ``xsearch`` in x and ``ysearch`` in y.
    The first and last knots are always fixed. MATLAB uses ``fmincon``
    (interior point); this uses SciPy's SLSQP, so results agree in quality
    rather than digit for digit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fixed = np.zeros(x.size, dtype=bool) if fixed is None else np.asarray(fixed, dtype=bool).copy()
    fixed[0] = fixed[-1] = True
    n = int((~fixed).sum())
    if n == 0:
        return x.copy(), y.copy()
    i0 = mass2ind(core.masses, x[0])
    i1 = mass2ind(core.masses, x[-1])
    masses, K = core.masses[i0:i1 + 1], core.K[i0:i1 + 1]
    bounds = [(-xsearch, xsearch)] * n + [(-ysearch, ysearch)] * n
    res = minimize(_msd, np.zeros(2 * n), args=(masses, K, x, y, fixed), method="SLSQP",
                   bounds=bounds, options={"ftol": 1e-10, "maxiter": 10000})
    return _apply_offsets(res.x, x, y, fixed)


def _zeros(pp, lo, hi):
    r = pp.roots(extrapolate=False)
    r = np.unique(r[np.isfinite(r)])
    return r[(r >= lo) & (r <= hi)]


def shape_properties(shape: PeakShape) -> dict[str, float]:
    """Area, position of the maximum and FWHM of a shape (for checks)."""
    pp = shape.to_ppoly()
    area = float(pp.integrate(shape.breaks[0], shape.breaks[-1]))
    x = np.linspace(shape.breaks[0], shape.breaks[-1], 200001)
    yv = pp(x)
    xmax = float(x[np.argmax(yv)])
    above = x[yv >= yv.max() / 2]
    return {"area": area, "xmax": xmax, "fwhm": float(above[-1] - above[0])}


def finalize_shape(shape: PeakShape, calibration: Calibration,
                   core: ConvolutionCore | None = None) -> Calibration:
    """``okclick``: normalise a fitted shape and return the updated calibration.

    The shape is scaled to unit area, shifted so its maximum (searched in
    -0.5..0.5) is at 0, and stretched to unit FWHM (half-maximum crossings
    searched in -2..2). The resolutions are rescaled by the same FWHM so
    calculated peaks keep their width.

    Unlike MATLAB, a Flat resolution parameter is rescaled too; MATLAB only
    rescaled the list, which changed the peak width for Flat calibrations.
    """
    pp = copy.deepcopy(shape).to_ppoly()
    upper = core.masses[-1] if core is not None else shape.breaks[-1]
    scale = float(pp.integrate(shape.breaks[0], upper))
    coefs = shape.coefs / scale
    breaks = shape.breaks.copy()
    pp = PeakShape(breaks, coefs).to_ppoly()

    zmax = _zeros(pp.derivative(), -0.5, 0.5)
    if zmax.size == 0:
        raise ValueError("No maximum of the peak shape between -0.5 and 0.5")
    breaks = breaks - zmax[0]
    pp = PeakShape(breaks, coefs).to_ppoly()
    half = pp(0.0) / 2
    shifted = PeakShape(breaks, coefs.copy())
    shifted.coefs[:, -1] -= half
    cross = _zeros(shifted.to_ppoly(), -2.0, 2.0)
    if cross.size < 2:
        raise ValueError("Could not find the half-maximum points of the peak shape")
    width = float(cross[-1] - cross[0])

    out = copy.deepcopy(calibration)
    out.shape = adapt_peak_shape(PeakShape(breaks, coefs), 1.0 / width, 1.0)
    out.resolutionlist = calibration.resolutionlist / width
    if calibration.resolution_method.lower() == "flat":
        out.resolution_param = calibration.resolution_param / width
    return out


class PeakShapeSession:
    """State of the peak shape editor (``peak_shape_generator.m``), without any GUI.

    ``peakdata`` is the section of the (mass-corrected) spectrum to use and
    ``molecules`` the fitted molecules in it. Knots are kept sorted; the
    first and last are pinned to zero and never move in a fit.
    """

    def __init__(self, peakdata: np.ndarray, molecules: Sequence[Molecule],
                 calibration: Calibration):
        if not molecules:
            raise ValueError("No molecules in this mass range.")
        self.calibration = calibration
        self.core = convolution_core_for(peakdata, molecules, calibration)
        self.x = self.core.knots_x.copy()
        self.y = self.core.knots_y.copy()
        self.fixed = np.zeros(self.x.size, dtype=bool)
        self.shape = calibration.shape.copy()  # what is shown until the first edit

    def _update(self) -> None:
        order = np.argsort(self.x, kind="stable")
        self.x, self.y, self.fixed = self.x[order], self.y[order].copy(), self.fixed[order]
        self.y[0] = self.y[-1] = 0.0
        self.shape = shape_from_knots(self.x, self.y)

    def add(self, x: float, y: float) -> int:
        self.x = np.append(self.x, float(x))
        self.y = np.append(self.y, float(y))
        self.fixed = np.append(self.fixed, False)
        self._update()
        return int(np.flatnonzero(self.x == float(x))[0])

    def remove(self, indices: Iterable[int]) -> bool:
        """Remove knots; refused if fewer than three would remain."""
        keep = np.setdiff1d(np.arange(self.x.size), np.asarray(list(indices), dtype=int))
        if keep.size < 3:
            return False
        self.x, self.y, self.fixed = self.x[keep], self.y[keep], self.fixed[keep]
        self._update()
        return True

    def move(self, index: int, x: float, y: float) -> None:
        self.x[index], self.y[index] = float(x), float(y)
        self._update()

    def set_fixed(self, index: int, fixed: bool) -> None:
        self.fixed[index] = bool(fixed)

    def nearest(self, x: float, y: float, xscale: float = 1.0, yscale: float = 1.0) -> tuple[int, float]:
        """Index of and (scaled) distance to the knot nearest to (x, y)."""
        d = np.hypot((self.x - x) / xscale, (self.y - y) / yscale)
        i = int(np.argmin(d))
        return i, float(d[i])

    def fit(self) -> None:
        self.x, self.y = fit_knots(self.core, self.x, self.y, self.fixed)
        self._update()

    def result(self) -> Calibration:
        """The calibration with the normalised new shape (the OK button)."""
        return finalize_shape(self.shape, self.calibration, self.core)
