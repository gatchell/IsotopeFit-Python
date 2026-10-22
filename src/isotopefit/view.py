"""Helpers for displaying and fitting parts of a project.

Kept free of Qt so they can be tested and used from scripts; the GUI calls
them. Ports of the plotting helpers in ``IsotopeFit.m`` and of
``findinvolvedmolecules.m``.
"""

from __future__ import annotations

import copy
import os
import re
from typing import Sequence

import numpy as np

from .fitting import ProgressCallback, fit_with_calibration
from .model import Calibration, Molecule, Project
from .molecules import molecules_in_mass_range
from .pattern import pattern
from .spectrum import increasing_from
from .settings import Settings


def fit_calibration(project: Project) -> Calibration:
    """The calibration used on the already mass-corrected spectrum (mass offset 0)."""
    cal = copy.copy(project.calibration)
    cal.massoffset_method, cal.massoffset_param = "Flat", 0.0
    return cal


def molecules_in_view(project: Project, lo: float, hi: float, searchrange: float = 1.0) -> np.ndarray:
    """Indices of molecules whose pattern (± searchrange sigma) overlaps [lo, hi]."""
    if not project.molecules:
        return np.empty(0, dtype=int)
    return molecules_in_mass_range(project.molecules, lo, hi, fit_calibration(project), searchrange)


def calculated_spectrum(project: Project, lo: float, hi: float,
                        indices: Sequence[int] | None = None,
                        searchrange: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Sum of the patterns of the molecules (default: all overlapping the range) on [lo, hi].

    Each molecule uses the calibrated resolution at its centre of mass,
    like MATLAB's ``multispec`` with a resolution axis. Each pattern is
    evaluated only on the points its peak shape reaches.
    """
    pd = project.peakdata
    sel = (pd[:, 0] >= lo) & (pd[:, 0] <= hi)
    x = pd[sel, 0]
    y = np.zeros_like(x)
    if indices is None:
        indices = molecules_in_view(project, lo, hi, searchrange)
    cal = project.calibration
    k = increasing_from(x)
    if k > x.size // 2:  # no long increasing part: evaluate everywhere
        k = x.size
    head = np.arange(k)
    tail = x[k:]
    b0, b1 = cal.shape.breaks[0], cal.shape.breaks[-1]
    for i in indices:
        m = project.molecules[i]
        res = float(cal.resolution(m.com))
        if not m.peakdata.size:
            continue
        fwhm = abs(m.com / res)
        reach_lo = min(b0, -b1) * fwhm  # peak shape support, either orientation
        reach_hi = max(b1, -b0) * fwhm
        i0 = k + int(np.searchsorted(tail, m.peakdata[:, 0].min() + reach_lo, side="left"))
        i1 = k + int(np.searchsorted(tail, m.peakdata[:, 0].max() + reach_hi, side="right"))
        idx = np.concatenate([head, np.arange(i0, i1)]) if k else slice(i0, i1)
        y[idx] += pattern(m, m.area, res, 0.0, x[idx], cal.shape)
    return x, y


_NUMBER = r"(\d+(?:\.\d*)?|\.\d+)"
_RANGE = re.compile(rf"^\s*{_NUMBER}\s*(?:(?:-|–|—|\.\.|to|\s)\s*{_NUMBER})?\s*u?\s*$", re.I)


def parse_goto(text: str, molecules: Sequence[Molecule]):
    """What a "Go to" entry means: ``("range", lo, hi)``, ``("molecule", index)`` or None.

    A single mass shows a few u around it (more at high mass); a range such
    as ``300-320`` shows exactly that; anything else is looked up as a
    molecule name, first exactly, then as part of a name (ignoring case).
    """
    text = text.strip()
    if not text:
        return None
    m = _RANGE.match(text)
    if m:
        a = float(m.group(1))
        if m.group(2) is None:
            half = max(1.5, 0.005 * a)
            return ("range", a - half, a + half)
        b = float(m.group(2))
        lo, hi = min(a, b), max(a, b)
        if hi == lo:
            hi = lo + 1.0
        return ("range", lo, hi)
    names = [mol.name.lower() for mol in molecules]
    key = text.lower()
    for candidate in (key, f"[{key}]"):
        if candidate in names:
            return ("molecule", names.index(candidate))
    for i, n in enumerate(names):
        if key in n:
            return ("molecule", i)
    return None


def involved_molecules(molecules: Sequence[Molecule], indices: Sequence[int],
                       calibration: Calibration, searchrange: float) -> np.ndarray:
    """``findinvolvedmolecules``: the selected molecules plus all that overlap them.

    Consecutive selected indices are treated as one block; every molecule
    overlapping a block's mass range (± searchrange sigma) is added.
    """
    indices = np.unique(np.asarray(indices, dtype=int))
    if indices.size == 0:
        return indices
    mins = np.array([m.minmass for m in molecules])
    maxs = np.array([m.maxmass for m in molecules])
    mins = mins - searchrange * calibration.sigma(mins)
    maxs = maxs + searchrange * calibration.sigma(maxs)
    breaks = np.flatnonzero(np.diff(indices) != 1)
    starts = np.concatenate([[indices[0]], indices[breaks + 1]])
    ends = np.concatenate([indices[breaks], [indices[-1]]])
    involved = set(indices.tolist())
    for s, e in zip(starts, ends):
        lo, hi = mins[s:e + 1].min(), maxs[s:e + 1].max()
        involved.update(molecules_in_mass_range(molecules, lo, hi, calibration, searchrange).tolist())
    return np.array(sorted(involved), dtype=int)


def fit_molecule_subset(project: Project, indices: Sequence[int], settings: Settings,
                        mode: str = "ranges",
                        progress: ProgressCallback | None = None) -> np.ndarray:
    """Fit the given molecules and everything overlapping them; returns the fitted indices.

    Like the GUI's "Fit" with "Selected" or "View": results are written into
    ``project.molecules``.
    """
    cal = fit_calibration(project)
    idx = involved_molecules(project.molecules, indices, cal, settings.searchrange)
    if idx.size == 0:
        return idx
    fitted = fit_with_calibration(
        [project.molecules[i] for i in idx], project.peakdata, cal, mode=mode,
        searchrange=settings.searchrange, deltam=settings.deltamass,
        deltar=settings.deltaresolution / 100.0, method=settings.fittingmethod_main,
        progress=progress,
    )
    for i, m in zip(idx, fitted):
        project.molecules[i] = m
    project.gui_status["changed"] = True
    return idx


def export_spectrum(project: Project, path: str | os.PathLike, with_fit: bool = False) -> None:
    """Write the calibrated spectrum (and optionally the fitted one) as tab-separated columns."""
    pd = project.peakdata
    cols = [pd[:, 0], pd[:, 1]]
    header = "Mass (Dalton)\tSignal (a.u.)"
    if with_fit:
        _, fit = calculated_spectrum(project, pd[:, 0].min(), pd[:, 0].max(),
                                     indices=range(len(project.molecules)))
        sel = (pd[:, 0] >= pd[:, 0].min()) & (pd[:, 0] <= pd[:, 0].max())
        full = np.zeros(pd.shape[0])
        full[sel] = fit
        cols.append(full)
        header += "\tFit (a.u.)"
    np.savetxt(path, np.column_stack(cols), delimiter="\t", fmt="%e", header=header, comments="")
