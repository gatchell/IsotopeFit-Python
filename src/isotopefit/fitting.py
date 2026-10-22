"""Fitting molecule areas (and optionally resolution and mass offset) to a spectrum.

Ports of ``fitwithcalibration.m``, ``fitranges.m``, ``fitmolecules.m`` and
the ``get_fit_params_using_*.m`` methods.

Every method works on a parameter vector
``[area_1, ..., area_n, resolution, massoffset]``, as in MATLAB.

Methods
-------
``linear_system``
    Resolution and mass offset fixed; areas by non-negative least squares.
    The default, and much the fastest.
``simplex_lin_combi``
    Nelder-Mead over resolution and mass offset (within the bounds); areas
    by non-negative least squares at every step. No error estimates.
``simplex``
    Nelder-Mead over all parameters. No error estimates (the MATLAB error
    routine for this method fails with an error, see README).
``linear_system_baseline``
    Like ``linear_system`` plus a constant baseline, unconstrained (areas
    may go negative). Meant for energy scans. MATLAB appended the baseline
    to ``baseline.txt``; here it is returned in ``FitRange.extra``.
"""

from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import nnls

from ._matlab import fminsearchbnd, round_half_away
from .model import Calibration, Molecule, PeakShape, Project
from .molecules import (FitRange, find_mass_range, find_mass_range2, find_ranges,
                        molecules_in_mass_range)
from .pattern import multispec_parameters, pattern
from .settings import Settings

log = logging.getLogger(__name__)

FITTING_METHODS = ("linear_system", "simplex_lin_combi", "simplex", "linear_system_baseline")
ProgressCallback = Callable[[int, int], None]  # (done, total)

#: fitranges.m and fitmolecules.m use a fixed +-3 sigma window to select data points.
DATA_WINDOW_SIGMA = 3.0


@dataclass
class _Fit:
    params: np.ndarray
    errors: np.ndarray
    baseline: float | None = None


def design_matrix(massaxis: np.ndarray, molecules: Sequence[Molecule], resolution: float,
                  massoffset: float, shape: PeakShape) -> np.ndarray:
    """One column per molecule: its pattern with unit area."""
    M = np.empty((np.size(massaxis), len(molecules)))
    for j, mol in enumerate(molecules):
        M[:, j] = pattern(mol, 1.0, resolution, massoffset, massaxis, shape)
    return M


def _area_errors(M: np.ndarray, areas: np.ndarray, spec: np.ndarray) -> np.ndarray:
    """Standard error (one standard deviation) of the areas.

    err_j = sqrt( (M'M)^-1_jj * RSS / (n - m) ), computed through the QR
    factor of M, for n data points and m molecules. (MATLAB's
    ``get_fit_params_using_linear_system.m`` reports 1.96 times this, a 95 %
    interval, and divides by n instead of n - m.)
    """
    n, m = M.shape
    if n <= m:  # no degrees of freedom left
        return np.full(m, np.nan)
    R = np.linalg.qr(M, mode="r")
    rss = float(np.sum((M @ areas - spec) ** 2))
    with np.errstate(all="ignore"):
        try:
            S = np.linalg.inv(R)
        except np.linalg.LinAlgError:  # singular: an area is not determined
            return np.full(m, np.inf)
        return np.sqrt(np.sum(S * S, axis=1) * rss / (n - m))


def _linear_system(spec, massaxis, shape, molecules, parameters, with_errors=True) -> _Fit:
    M = design_matrix(massaxis, molecules, parameters[-2], parameters[-1], shape)
    areas, _ = nnls(M, spec, maxiter=50 * max(M.shape[1], 1))
    params = np.array(parameters, dtype=float)
    params[:-2] = areas
    errors = np.zeros(len(parameters))
    errors[:-2] = _area_errors(M, areas, spec) if with_errors else np.nan
    return _Fit(params, errors)


def _linear_system_baseline(spec, massaxis, shape, molecules, parameters) -> _Fit:
    M = design_matrix(massaxis, molecules, parameters[-2], parameters[-1], shape)
    M = np.column_stack([M, np.full(M.shape[0], 1.0 / (massaxis[-1] - massaxis[0]))])
    A = np.linalg.lstsq(M, spec, rcond=None)[0]
    params = np.array(parameters, dtype=float)
    params[:-2] = A[:-1]
    rss = float(np.sum((M @ A - spec) ** 2))
    dof = M.shape[0] - M.shape[1]  # data points minus molecules and baseline
    with np.errstate(all="ignore"):
        err = (np.sqrt(np.diag(np.linalg.pinv(M.T @ M)) * rss / dof) if dof > 0
               else np.full(M.shape[1], np.nan))
    errors = np.zeros(len(parameters))
    errors[:-2] = err[:-1]
    return _Fit(params, errors, baseline=float(A[-1]))


def _msd_without_areas(spec, massaxis, shape, molecules, res_mo) -> float:
    paramin = np.concatenate([np.zeros(len(molecules)), res_mo])
    fit = _linear_system(spec, massaxis, shape, molecules, paramin, with_errors=False)
    calc = multispec_parameters(massaxis, molecules, fit.params, shape)
    return float(np.sum((spec - calc) ** 2))


def _simplex_lin_combi(spec, massaxis, shape, molecules, parameters, lb, ub) -> _Fit:
    res_mo, _, _ = fminsearchbnd(
        lambda x: _msd_without_areas(spec, massaxis, shape, molecules, x),
        parameters[-2:], lb[-2:], ub[-2:], maxfunevals=5000, maxiter=5000,
    )
    paramin = np.concatenate([np.zeros(len(molecules)), res_mo])
    fit = _linear_system(spec, massaxis, shape, molecules, paramin, with_errors=False)
    fit.errors = np.full(len(parameters), np.nan)
    return fit


def _simplex(spec, massaxis, shape, molecules, parameters, lb, ub) -> _Fit:
    def msd(p):
        return float(np.sum((spec - multispec_parameters(massaxis, molecules, p, shape)) ** 2))

    params, _, _ = fminsearchbnd(msd, parameters, lb, ub, maxfunevals=5000, maxiter=5000)
    return _Fit(params, np.full(len(parameters), np.nan))


def fit_parameters(method: str, spec: np.ndarray, massaxis: np.ndarray, shape: PeakShape,
                   molecules: Sequence[Molecule], parameters: np.ndarray,
                   lb: np.ndarray, ub: np.ndarray) -> _Fit:
    """Dispatch to one fitting method (``get_fit_params_using_<method>``)."""
    spec = np.asarray(spec, dtype=float)
    if method == "linear_system":
        return _linear_system(spec, massaxis, shape, molecules, parameters)
    if method == "simplex_lin_combi":
        return _simplex_lin_combi(spec, massaxis, shape, molecules, parameters, lb, ub)
    if method == "simplex":
        return _simplex(spec, massaxis, shape, molecules, parameters, lb, ub)
    if method == "linear_system_baseline":
        return _linear_system_baseline(spec, massaxis, shape, molecules, parameters)
    if method in ("genetic", "pattern_search"):
        raise NotImplementedError(f"Fitting method {method!r} is not ported (needs MATLAB toolboxes).")
    raise ValueError(f"Unknown fitting method {method!r}")


def _bounds(parameters: np.ndarray, areaup: float, deltares: float, deltam: float):
    n = len(parameters) - 2
    res, mo = parameters[-2], parameters[-1]
    lb = np.concatenate([np.zeros(n), [res - res * deltares, mo - deltam]])
    ub = np.concatenate([np.full(n, areaup), [res + res * deltares, mo + deltam]])
    return lb, ub


def _limit_points(ind: np.ndarray, maxdatapoints: int) -> np.ndarray:
    """Evenly thin out ``ind`` to ``maxdatapoints`` (MATLAB: ind(round((1:max)*(ndp/max))))."""
    ndp = ind.size
    if ndp <= maxdatapoints:
        return ind
    k = np.arange(1, maxdatapoints + 1)
    return ind[round_half_away(k * (ndp / maxdatapoints)).astype(int) - 1]


# ------------------------------------------------------------------ fitting by ranges


def fit_ranges(peakdata: np.ndarray, ranges: Sequence[FitRange], calibration: Calibration,
               areaup: float = np.inf, deltares: float = 0.0, deltam: float = 0.0,
               method: str = "linear_system",
               progress: ProgressCallback | None = None) -> list[FitRange]:
    """``fitranges``: fit every range on its own. Returns fitted copies of the ranges.

    ``deltares`` is the relative and ``deltam`` the absolute freedom given
    to resolution and mass offset (only used by the simplex methods).
    """
    massaxis = peakdata[:, 0]
    spec = peakdata[:, 1]
    out = copy.deepcopy(list(ranges))
    nmax = max((len(r.molecules) for r in out), default=0)
    if nmax > 200:
        warnings.warn(f"At least one range has more than 200 molecules (max: {nmax}); "
                      "fitting may be slow.", stacklevel=2)

    for i, r in enumerate(out):
        nmol = len(r.molecules)
        ind = find_mass_range2(massaxis, r.molecules, r.resolution, r.massoffset, DATA_WINDOW_SIGMA)
        ind = _limit_points(ind, 50 * nmol)

        parameters = np.zeros(nmol + 2)
        # MATLAB sets every start area to the integral of the selected data.
        dm = np.concatenate([[0.0], np.diff(massaxis[ind])])
        parameters[:nmol] = max(0.0, float(np.sum(spec[ind] * dm)))
        parameters[-2:] = r.resolution, r.massoffset
        lb, ub = _bounds(parameters, areaup, deltares, deltam)

        if ind.size == 0:
            log.warning("Range %d (%.2f-%.2f) has no data points; skipped", i, r.minmass, r.maxmass)
            fit = _Fit(np.concatenate([np.zeros(nmol), parameters[-2:]]), np.full(nmol + 2, np.nan))
        else:
            fit = fit_parameters(method, spec[ind], massaxis[ind], calibration.shape,
                                 r.molecules, parameters, lb, ub)

        for mol, a, e in zip(r.molecules, fit.params[:-2], fit.errors[:-2]):
            mol.area, mol.areaerror = float(a), float(e)
        r.resolution, r.massoffset = float(fit.params[-2]), float(fit.params[-1])
        r.resolutionerror, r.massoffseterror = float(fit.errors[-2]), float(fit.errors[-1])
        if fit.baseline is not None:
            r.extra["baseline"] = fit.baseline
        if progress:
            progress(i + 1, len(out))

    for r in out:  # centres of mass change with the new areas
        r.update_com()
    return out


# ------------------------------------------------------------------ fitting molecule by molecule


def fit_molecules(peakdata: np.ndarray, molecules: Sequence[Molecule], calibration: Calibration,
                  areaup: float = np.inf, deltares: float = 0.0, deltam: float = 0.0,
                  method: str = "linear_system", searchrange: float = DATA_WINDOW_SIGMA,
                  progress: ProgressCallback | None = None) -> list[Molecule]:
    """``fitmolecules``: fit molecules one at a time from low to high mass.

    Each step fits the molecule together with the later molecules that
    overlap it, after subtracting the already fitted earlier molecules.
    Only the current molecule's result is kept for the subtraction.
    """
    massaxis = peakdata[:, 0]
    spec = peakdata[:, 1]
    mols = copy.deepcopy(list(molecules))
    spec_calc = np.zeros_like(spec)
    n = len(mols)
    for i in range(n):
        resolution = float(calibration.resolution(mols[i].com))
        massoffset = float(calibration.massoffset(mols[i].com))
        involved = i + molecules_in_mass_range(mols[i:], mols[i].minmass, mols[i].maxmass,
                                               calibration, searchrange)
        inv_mols = [mols[j] for j in involved]
        ind = find_mass_range2(massaxis, inv_mols, resolution, massoffset, DATA_WINDOW_SIGMA)
        ind = _limit_points(ind, 50 * len(involved))

        parameters = np.array([m.area if m.area != 0 else 0.1 for m in inv_mols]
                              + [resolution, massoffset], dtype=float)
        lb, ub = _bounds(parameters, areaup, deltares, deltam)
        fit = fit_parameters(method, spec[ind] - spec_calc[ind], massaxis[ind], calibration.shape,
                             inv_mols, parameters, lb, ub)

        idx = find_mass_range(massaxis, [mols[i]], resolution, massoffset, 10)
        spec_calc[idx] += multispec_parameters(massaxis[idx], [mols[i]],
                                               fit.params[[0, -2, -1]], calibration.shape)
        for m, a, e in zip(inv_mols, fit.params[:-2], fit.errors[:-2]):
            m.area, m.areaerror = float(a), float(e)
        if progress:
            progress(i + 1, n)
    return mols


# ------------------------------------------------------------------ entry points


def fit_with_calibration(molecules: Sequence[Molecule], peakdata: np.ndarray,
                         calibration: Calibration, mode: str = "ranges",
                         searchrange: float = 1.0, deltam: float = 0.0, deltar: float = 0.0,
                         method: str = "linear_system",
                         progress: ProgressCallback | None = None) -> list[Molecule]:
    """``fitwithcalibration``: fit areas using the calibrated resolution and mass offset.

    ``mode="ranges"`` (MATLAB method 1) groups overlapping molecules and
    fits each group; ``mode="molecules"`` (method 2) fits one molecule at a
    time. Returns fitted copies, in the order of the input.
    """
    if mode == "ranges":
        ranges = find_ranges(molecules, calibration, searchrange)
        fitted = fit_ranges(peakdata, ranges, calibration, np.inf, deltar, deltam, method, progress)
        return [m for r in fitted for m in r.molecules]
    if mode == "molecules":
        return fit_molecules(peakdata, molecules, calibration, np.inf, deltar, deltam, method,
                             progress=progress)
    raise ValueError(f"Unknown mode {mode!r}; use 'ranges' or 'molecules'")


def fit_project(project: Project, settings: Settings | None = None, mode: str = "ranges",
                progress: ProgressCallback | None = None) -> list[Molecule]:
    """Fit all molecules of a project, like the GUI's "Fit" button with "All".

    The spectrum ``project.peakdata`` is already mass-corrected, so the
    mass offset is set to zero for the fit, as in MATLAB. The fitted
    molecules replace ``project.molecules`` and are returned.
    """
    settings = settings or Settings()
    if project.peakdata is None:
        project.update_peakdata()
    cal = copy.copy(project.calibration)
    cal.massoffset_method, cal.massoffset_param = "Flat", 0.0
    fitted = fit_with_calibration(
        project.molecules, project.peakdata, cal, mode=mode,
        searchrange=settings.searchrange, deltam=settings.deltamass,
        deltar=settings.deltaresolution / 100.0, method=settings.fittingmethod_main,
        progress=progress,
    )
    project.molecules = fitted
    project.gui_status["fitted"] = True
    project.gui_status["changed"] = True
    return fitted
