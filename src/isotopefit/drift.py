"""Drift correction of the time-of-flight axis over a measurement (port of ``driftcorrection.m``).

A TOFWERK file stores the individual spectra (``/FullSpectra/TofData``,
writes × buffers × segments × time bins). If the flight times drift during
a measurement, the peaks of the summed spectrum smear out. The correction:

1. splits the time-bin range between a start and an end mass into
   ``ndiv`` divisions;
2. in each division, finds the position of every spectrum (write, or buffer
   of the first write) from the maximum of a cross-correlation, searched
   within ±``shiftsearch`` bins;
3. for every spectrum, fits a weighted 2nd-order polynomial through the
   division positions (weights: the cross-correlation maxima), as a function
   of the time bin;
4. moves every spectrum back by the rounded polynomial and sums the
   spectra again.

Differences from the MATLAB version, which in tests on spectra with a known
drift made the peaks broader instead of narrower:

* MATLAB correlates *consecutive* spectra and adds up whole-bin shifts, so
  a drift slower than half a bin per spectrum is never seen and rounding
  errors accumulate. Here every spectrum is correlated with the sum of all
  spectra (median removed), the maximum is located to a fraction of a bin
  with a parabola, and the spectra are aligned to their average position;
* MATLAB's shift has the opposite sign of the peak movement while the
  correction reads ``data(t + shift)``, so peaks move further away;
* MATLAB applies the polynomial of spectrum i+1 to spectrum i and leaves the
  last spectrum out of the sum;
* MATLAB's time-domain polynomial uses the first bin of each division instead
  of its centre.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from ._matlab import round_half_away
from .model import Project
from .spectrum import mass2ind

ProgressCallback = Callable[[int, int], None]


@dataclass
class DriftSettings:
    startmass: float = 5.0  # below this, bins are ignored (avoid the switching peak)
    endmass: float = np.inf
    ndiv: int = 4  # number of divisions of the mass range
    use_buffers: bool = False  # spectroscopy files: the buffers of the first write
    shiftsearch: int = 10  # largest shift searched, in bins


# ------------------------------------------------------------------ sources of spectra


class SpectrumSource(Protocol):
    n_spectra: int
    n_bins: int

    def spectrum(self, i: int) -> np.ndarray: ...


class ArraySource:
    """Spectra from an array (n_spectra, n_bins); for tests and data already in memory."""

    def __init__(self, data: np.ndarray):
        self.data = np.asarray(data, dtype=float)
        self.n_spectra, self.n_bins = self.data.shape

    def spectrum(self, i: int) -> np.ndarray:
        return self.data[i]

    def close(self) -> None:
        pass


class H5Source:
    """The writes (summed over buffers) or the buffers of the first write of a TOFWERK file.

    h5py sees ``TofData`` as (writes, buffers, segments, bins), the reverse of
    MATLAB. MATLAB reads segment 1 only; segments are summed here, which is
    the same for the usual files with one segment.
    """

    def __init__(self, path: str | os.PathLike, use_buffers: bool = False):
        import h5py

        self.path = os.fspath(path)
        self.use_buffers = use_buffers
        self._h5 = h5py.File(self.path, "r")
        self._data = self._h5["/FullSpectra/TofData"]
        writes, bufs, _segments, bins = self._data.shape
        self.n_spectra = bufs if use_buffers else writes
        self.n_bins = bins

    def spectrum(self, i: int) -> np.ndarray:
        if self.use_buffers:
            block = self._data[0, i, :, :]
        else:
            block = self._data[i, :, :, :].reshape(-1, self.n_bins)
        return np.asarray(block, dtype=float).reshape(-1, self.n_bins).sum(axis=0)

    def sum_spectrum(self) -> np.ndarray:
        return np.asarray(self._h5["/FullSpectra/SumSpectrum"], dtype=float).ravel()

    def close(self) -> None:
        self._h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------------ the steps


def divisions(massaxis: np.ndarray, startmass: float, endmass: float, ndiv: int):
    """First, last (inclusive) and centre bin of every division, 0-based, as in MATLAB."""
    massaxis = np.asarray(massaxis, dtype=float)
    endmass = min(endmass, massaxis[-1])
    s = mass2ind(massaxis, startmass) + 1  # MATLAB 1-based
    e = mass2ind(massaxis, endmass) + 1
    i = np.arange(1, ndiv + 1)
    lo = round_half_away(s + (e - s) / ndiv * (i - 1)).astype(int)
    hi = round_half_away(s + (e - s) / ndiv * i).astype(int)
    centre = round_half_away((lo + hi) / 2).astype(int)
    return lo - 1, hi - 1, centre - 1


def offset_against(ref: np.ndarray, s: np.ndarray, shiftsearch: int = 10) -> tuple[float, float]:
    """How many bins later the peaks of ``s`` sit than those of ``ref`` (to a fraction of a bin).

    Cross-correlation sum_t ref(t)·s(t+lag) for lag = -shiftsearch..shiftsearch,
    after removing the median of both, with a parabola through the maximum.
    Returns the offset and the correlation maximum (used as weight).
    """
    r = ref - np.median(ref)
    x = s - np.median(s)
    L = r.size
    lags = np.arange(-shiftsearch, shiftsearch + 1)
    c = np.array([np.dot(r[max(0, -g):L - max(0, g)], x[max(0, g):L - max(0, -g)]) for g in lags])
    k = int(np.argmax(c))
    off = float(lags[k])
    if 0 < k < c.size - 1:
        den = c[k - 1] - 2 * c[k] + c[k + 1]
        if den < 0:
            off += 0.5 * (c[k - 1] - c[k + 1]) / den
    return off, float(c[k])


def _wpolyfit(x, y, w, degree=2) -> np.ndarray:
    """Weighted least-squares polynomial (MATLAB ``fit(..., 'poly2', Weights=w)``)."""
    x, y, w = (np.asarray(v, dtype=float) for v in (x, y, w))
    degree = min(degree, x.size - 1)
    if np.all(w <= 0) or not np.all(np.isfinite(w)):
        w = np.ones_like(w)
    p = np.polyfit(x, y, degree, w=np.sqrt(np.clip(w, 0, None)))
    return np.concatenate([np.zeros(2 - degree), p])  # always 3 coefficients


@dataclass
class DriftAnalysis:
    """Everything the drift correction window shows.

    ``positions[d, w]`` is how many bins later spectrum w sits than the sum
    of all spectra, in division d; ``weights[d, w]`` is the correlation
    maximum used to weight that position in the polynomial fit.
    """

    settings: DriftSettings
    massaxis: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    centre: np.ndarray
    windows: list[np.ndarray]  # per division: (n_spectra, width) signal
    positions: np.ndarray  # (ndiv, n_spectra)
    weights: np.ndarray  # (ndiv, n_spectra)
    coms: np.ndarray  # mass at the centre of every division
    mass_polys: list[np.ndarray] = field(default_factory=list)  # per spectrum, x = mass
    time_polys: list[np.ndarray] = field(default_factory=list)  # per spectrum, x = 1-based bin

    @property
    def n_spectra(self) -> int:
        return self.positions.shape[1]

    def correction(self, w: int, n_bins: int) -> np.ndarray:
        """Integer shift (bins) applied to spectrum w, over all bins."""
        t = np.arange(1, n_bins + 1, dtype=float)
        return round_half_away(np.polyval(self.time_polys[w], t)).astype(int)


def analyse(source: SpectrumSource, massaxis: np.ndarray, settings: DriftSettings,
            progress: ProgressCallback | None = None) -> DriftAnalysis:
    """Steps 1-3: divisions, position of every spectrum, polynomials."""
    massaxis = np.asarray(massaxis, dtype=float)
    if massaxis.size != source.n_bins:
        raise ValueError(f"The mass axis has {massaxis.size} points but the spectra have "
                         f"{source.n_bins} bins; load the spectrum from the same .h5 file.")
    if source.n_spectra < 2:
        raise ValueError("The file holds only one spectrum; there is no drift to correct.")
    if settings.ndiv < 1:
        raise ValueError("Use at least one division.")
    lo, hi, centre = divisions(massaxis, settings.startmass, settings.endmass, settings.ndiv)
    n = source.n_spectra
    windows = [np.empty((n, h - l + 1)) for l, h in zip(lo, hi)]
    for w in range(n):  # one pass over the file, keep only the division windows
        spec = source.spectrum(w)
        for d, (l, h) in enumerate(zip(lo, hi)):
            windows[d][w] = spec[l:h + 1]
        if progress:
            progress(w + 1, n)

    nd = len(lo)
    positions = np.zeros((nd, n))
    weights = np.zeros((nd, n))
    for d in range(nd):
        ref = windows[d].sum(axis=0)
        for w in range(n):
            positions[d, w], weights[d, w] = offset_against(ref, windows[d][w],
                                                            settings.shiftsearch)

    coms = massaxis[centre]
    tbins = centre + 1.0  # 1-based bin numbers
    result = DriftAnalysis(settings, massaxis, lo, hi, centre, windows, positions, weights, coms)
    result.mass_polys = [_wpolyfit(coms, positions[:, w], weights[:, w]) for w in range(n)]
    result.time_polys = [_wpolyfit(tbins, positions[:, w], weights[:, w]) for w in range(n)]
    return result


def corrected_sum(source: SpectrumSource, analysis: DriftAnalysis,
                  progress: ProgressCallback | None = None) -> np.ndarray:
    """Step 4: every spectrum moved back by its correction, summed.

    Bins whose shifted position falls outside the spectrum keep their own value.
    """
    n_bins, n = source.n_bins, source.n_spectra
    t = np.arange(n_bins)
    total = np.zeros(n_bins)
    for w in range(n):
        idx = t + analysis.correction(w, n_bins)
        idx = np.where((idx >= 0) & (idx < n_bins), idx, t)
        total += source.spectrum(w)[idx]
        if progress:
            progress(w + 1, n)
    return total


def apply_correction(project: Project, corrected: np.ndarray) -> None:
    """Use the corrected sum spectrum, then redo background and mass correction (``menudc``)."""
    corrected = np.asarray(corrected, dtype=float).ravel()
    if corrected.size != project.raw_peakdata.shape[0]:
        raise ValueError("The corrected spectrum does not match the project's spectrum.")
    project.raw_peakdata = project.raw_peakdata.copy()
    project.raw_peakdata[:, 1] = corrected
    project.update_peakdata()
    project.gui_status.update(drift_corrected=True, calibrated=True, bg_corrected=True,
                              fitted=False, changed=True)


def drift_correct(project: Project, h5path: str | os.PathLike | None = None,
                  settings: DriftSettings | None = None,
                  progress: ProgressCallback | None = None) -> DriftAnalysis:
    """Analyse the drift in the project's ``.h5`` file and apply the correction to the project."""
    settings = settings or DriftSettings()
    path = h5path or project.fileinfo.get("h5completepath")
    if not path:
        raise ValueError("The original .h5 file of this project is not known.")
    with H5Source(path, settings.use_buffers) as src:
        analysis = analyse(src, project.raw_peakdata[:, 0], settings, progress)
        corrected = corrected_sum(src, analysis)
    apply_correction(project, corrected)
    project.fileinfo["h5completepath"] = os.path.abspath(os.fspath(path))
    return analysis
