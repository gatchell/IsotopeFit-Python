"""Basic spectrum operations.

Ports of ``croppeakdata.m``, ``subtractbg.m``, ``subtractmassoffset.m``,
``guessarea.m`` and ``mass2ind.m``. A spectrum ("peakdata") is an (n, 2)
array of [mass, signal].
"""

from __future__ import annotations

import weakref

import numpy as np

from ._matlab import colon, interp1_pchip

# Fitting asks about the same mass axis hundreds of times, and checking
# 400 000 points each time adds up, so the answer for the last axis is kept.
_ORDER_CACHE: list = [None, 0, 0, 0]  # weak reference, data address, size, result


def increasing_from(axis: np.ndarray) -> int:
    """Smallest k such that ``axis[k:]`` is strictly increasing (remembered for the last array).

    Mass axes rebuilt from a TOFWERK calibration start with a few bins before
    time zero whose masses run backwards; after those, the axis increases.
    """
    a = np.asarray(axis)
    if a.size < 2:
        return 0
    ref, addr, size, k = _ORDER_CACHE
    key = a.__array_interface__["data"][0]
    if ref is not None and ref() is a and addr == key and size == a.size:
        return k
    bad = np.flatnonzero(~(a[1:] > a[:-1]))
    k = int(bad[-1]) + 1 if bad.size else 0
    try:
        _ORDER_CACHE[:] = [weakref.ref(a), key, a.size, k]
    except TypeError:  # arrays that cannot be weakly referenced
        pass
    return k


def is_increasing(axis: np.ndarray) -> bool:
    """True if ``axis`` is strictly increasing."""
    return increasing_from(axis) == 0


def crop(peakdata: np.ndarray, start: int, stop: int) -> np.ndarray:
    """``croppeakdata``: rows ``start:stop`` (Python slice, 0-based)."""
    return np.array(peakdata[start:stop], dtype=float)


def subtract_background(peakdata: np.ndarray, bg) -> np.ndarray:
    """``subtractbg``: subtract the PCHIP-interpolated background.

    ``bg`` is a :class:`~isotopefit.model.BackgroundCorrection`. Nothing is
    subtracted unless it holds at least two points.
    """
    out = np.array(peakdata, dtype=float)
    if bg.bgm.size > 1:
        out[:, 1] -= interp1_pchip(bg.bgm, bg.bgy, out[:, 0])
    return out


def subtract_mass_offset(peakdata: np.ndarray, calibration, step: float = 0.01) -> np.ndarray:
    """``subtractmassoffset``: move the mass axis by the calibrated offset.

    The offset is defined at the *corrected* mass: m_old = m_new + mo(m_new).
    As in MATLAB, this is inverted by tabulating m_old on a grid of m_new
    (spacing ``step``) and interpolating back with PCHIP.
    """
    out = np.array(peakdata, dtype=float)
    mass = out[:, 0]
    xaxis = colon(mass.min(), step, mass.max())
    yaxis = xaxis + calibration.massoffset(xaxis)
    out[:, 0] = interp1_pchip(yaxis, xaxis, mass)
    return out


def guess_area(peakdata: np.ndarray) -> float:
    """``guessarea``: left-Riemann integral of the signal, clipped at 0."""
    if len(peakdata) <= 1:
        return 0.0
    return max(0.0, float(np.sum(peakdata[:-1, 1] * np.diff(peakdata[:, 0]))))


def mass2ind(massaxis: np.ndarray, mass: float) -> int:
    """``mass2ind``: 0-based index of the nearest mass (first one on ties).

    Binary search where the axis increases, a scan over the rest.
    """
    a = np.asarray(massaxis)
    k = increasing_from(a) if np.isfinite(mass) else a.size
    if k > a.size // 2:
        return int(np.argmin(np.abs(a - mass)))
    t = a[k:]
    i = int(np.searchsorted(t, mass, side="left"))
    if i <= 0:
        best = 0
    elif i >= t.size:
        best = t.size - 1
    else:
        best = i - 1 if abs(t[i - 1] - mass) <= abs(t[i] - mass) else i
    best += k
    if k:
        h = int(np.argmin(np.abs(a[:k] - mass)))
        if abs(a[h] - mass) <= abs(a[best] - mass):  # earlier index wins ties
            return h
    return best
