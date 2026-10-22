"""Calculated spectra: isotope patterns convolved with the peak shape.

Ports of ``peak_width_adaption.m``, ``pattern_func.m``,
``multispecparameters.m`` and ``multispec.m``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .model import Molecule, PeakShape
from .spectrum import increasing_from, mass2ind


def adapt_peak_shape(shape: PeakShape, fwhm: float, area: float) -> PeakShape:
    """``peak_width_adaption``: stretch the unit shape to ``fwhm`` and scale to ``area``.

    The stored shape has unit FWHM and unit area. Stretching the x axis by
    ``fwhm`` rescales the polynomial coefficients by ``fwhm**-power``; the
    final division by ``fwhm`` restores unit area before scaling by ``area``.
    """
    breaks = shape.breaks * fwhm
    powers = np.arange(shape.order - 1, -1, -1)  # MATLAB hard-codes [3 2 1 0]
    coefs = shape.coefs / fwhm ** powers[np.newaxis, :]
    if fwhm < 0:  # kept from MATLAB: "robust" against negative resolution
        breaks = breaks[::-1]
        coefs = coefs[::-1]
    return PeakShape(breaks, coefs * area / fwhm)


def pattern(
    molecule: Molecule,
    area: float,
    resolution: float | np.ndarray,
    massoffset: float,
    massaxis: np.ndarray,
    shape: PeakShape,
) -> np.ndarray:
    """``pattern_func``: calculated signal of one molecule on ``massaxis``.

    Every isotopologue contributes the peak shape, with FWHM =
    ``com / resolution``, centred at its mass, scaled by its abundance and
    by ``area``. The whole pattern is shifted by ``massoffset``.

    ``resolution`` may also be an array along ``massaxis``; as in MATLAB,
    the value at the molecule's centre of mass is then used for all its
    peaks.
    """
    massaxis = np.asarray(massaxis, dtype=float)
    out = np.zeros(massaxis.shape)
    if np.ndim(resolution) == 0:
        res = float(resolution)
    else:
        res = float(np.asarray(resolution)[mass2ind(massaxis - massoffset, molecule.com)])
    peak = adapt_peak_shape(shape, molecule.com / res, area)
    pp = peak.to_ppoly()

    # Where the axis increases (usually all of it, after a few bins before time
    # zero in old files), each peak's points are found by binary search.
    k = increasing_from(massaxis)
    if k > massaxis.size // 2:
        k = massaxis.size
    x = massaxis - massoffset
    tail = x[k:]
    lo_b, hi_b = peak.breaks[0], peak.breaks[-1]
    for m, abundance in molecule.peakdata:
        lo, hi = lo_b + m, hi_b + m
        if k:
            idx = np.flatnonzero((x[:k] > lo) & (x[:k] < hi))
            out[idx] += abundance * pp(x[idx] - m)
        if tail.size:
            i0 = k + int(np.searchsorted(tail, lo, side="right"))
            i1 = k + int(np.searchsorted(tail, hi, side="left"))
            out[i0:i1] += abundance * pp(x[i0:i1] - m)
    return out


def multispec_parameters(
    massaxis: np.ndarray,
    molecules: Sequence[Molecule],
    parameters: np.ndarray,
    shape: PeakShape,
) -> np.ndarray:
    """``multispecparameters``: sum of patterns with areas taken from ``parameters``.

    ``parameters`` = [area_1, ..., area_n, resolution, massoffset].
    """
    out = np.zeros(np.shape(massaxis))
    for mol, area in zip(molecules, parameters[:-2]):
        out += pattern(mol, area, parameters[-2], parameters[-1], massaxis, shape)
    return out


def multispec(
    molecules: Sequence[Molecule],
    resolution: float | np.ndarray,
    massoffset: float,
    massaxis: np.ndarray,
    shape: PeakShape,
) -> np.ndarray:
    """``multispec``: sum of patterns using each molecule's stored area."""
    out = np.zeros(np.shape(massaxis))
    for mol in molecules:
        out += pattern(mol, mol.area, resolution, massoffset, massaxis, shape)
    return out
