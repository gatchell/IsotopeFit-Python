"""Global fit of all molecules at once, and batch fitting of a folder (port of ``batch_fit.m``).

Unlike the GUI's fit, which handles one range or one molecule at a time,
:func:`fit_global` sets up a single design matrix with one column per
molecule and solves the whole spectrum as one non-negative least-squares
problem:

* every isotope peak gets the peak shape with FWHM = mass / R(mass), using
  the resolution at the *peak's* mass (the GUI fit uses the molecule's
  centre of mass for all its peaks);
* only data points within ±½ FWHM (``searchrange`` × 0.5) of some peak are
  fitted;
* the area errors are the sandwich estimate
  ``sqrt(diag((M'M)⁻¹ M' diag(r²) M (M'M)⁻¹))`` with the residuals r (one
  standard deviation, like the other fits).

MATLAB solves the full sparse matrix with ``lsqnonneg``. Here the matrix is
split into independent blocks of molecules that share data points, and each
block is solved on its own. That gives the same solution, because the blocks
do not interact, and it keeps the dense matrices small.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular
from scipy.optimize import nnls
from scipy.sparse.csgraph import connected_components

from .model import Calibration, Molecule, Project
from .pattern import adapt_peak_shape

ProgressCallback = Callable[[int, int], None]

FWHM_RANGE = 0.5


@dataclass
class GlobalFit:
    areas: np.ndarray
    errors: np.ndarray
    fitmask: np.ndarray  # which points of the spectrum were fitted
    calculated: np.ndarray  # fitted spectrum on the fitted points
    blocks: int  # number of independent blocks solved


def global_design_matrix(massaxis: np.ndarray, molecules: Sequence[Molecule],
                         calibration: Calibration, searchrange: float = 1.0,
                         progress: ProgressCallback | None = None):
    """Sparse design matrix (all points × molecules) and the fit mask of ``batch_fit.m``."""
    massaxis = np.asarray(massaxis, dtype=float)
    n = massaxis.size
    shape = calibration.shape
    fitmask = np.zeros(n, dtype=bool)
    rows, cols, vals = [], [], []
    increasing = massaxis.size < 2 or bool(np.all(np.diff(massaxis) > 0))
    for j, mol in enumerate(molecules):
        pd = np.asarray(mol.peakdata, dtype=float).reshape(-1, 2)
        if pd.shape[0]:
            res = np.atleast_1d(calibration.resolution(pd[:, 0]))
        for (mass, area), R in zip(pd, res if pd.shape[0] else []):
            fwhm = mass / R
            peak = adapt_peak_shape(shape, fwhm, 1.0)
            lo, hi = peak.breaks[0] + mass, peak.breaks[-1] + mass
            half = searchrange * FWHM_RANGE * fwhm
            if increasing:
                ind = np.arange(np.searchsorted(massaxis, lo, side="right"),
                                np.searchsorted(massaxis, hi, side="left"))
                fitmask[np.searchsorted(massaxis, mass - half, side="right"):
                        np.searchsorted(massaxis, mass + half, side="left")] = True
            else:
                ind = np.flatnonzero((lo < massaxis) & (massaxis < hi))
                fitmask[(mass - half < massaxis) & (massaxis < mass + half)] = True
            if ind.size:
                rows.append(ind)
                cols.append(np.full(ind.size, j))
                vals.append(area * peak.to_ppoly()(massaxis[ind] - mass))
        if progress:
            progress(j + 1, len(molecules))
    if rows:
        M = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, len(molecules)))
    else:
        M = sp.csc_matrix((n, len(molecules)))
    return M, fitmask


def _block_errors(M: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Sandwich error estimate of ``batch_fit.m`` through the QR factor of M."""
    m = M.shape[1]
    if M.shape[0] < m:
        return np.full(m, np.nan)
    R = np.linalg.qr(M, mode="r")
    with np.errstate(all="ignore"):
        if np.any(np.abs(np.diag(R)) <= 1e-12 * max(np.abs(np.diag(R)).max(), 1e-300)):
            return np.full(m, np.inf)  # singular; MATLAB's R\I gives Inf/NaN as well
        S = solve_triangular(R, np.eye(m))  # inv(R); inv(M'M) = S S'
        C = S @ S.T
        B = M.T @ (M * (residuals ** 2)[:, None])
        return np.sqrt(np.abs(np.einsum("ij,jk,ki->i", C, B, C)))


def fit_global(peakdata: np.ndarray, molecules: Sequence[Molecule], calibration: Calibration,
               searchrange: float = 1.0, progress: ProgressCallback | None = None) -> GlobalFit:
    """Fit all molecules to ``peakdata`` (already background- and mass-corrected) at once."""
    peakdata = np.asarray(peakdata, dtype=float)
    massaxis, signal = peakdata[:, 0], peakdata[:, 1]
    nmol = len(molecules)

    def stage(offset):
        if progress is None:
            return None
        return lambda i, n: progress(offset + i, 2 * nmol)

    M, fitmask = global_design_matrix(massaxis, molecules, calibration, searchrange, stage(0))
    M = M.tocsr()[fitmask].tocsc()
    y = signal[fitmask]

    areas = np.zeros(nmol)
    errors = np.full(nmol, np.inf)
    # molecules are coupled when their columns share a data point
    G = (M.T @ M) != 0
    nblocks, labels = connected_components(G, directed=False)
    done = 0
    for b in range(nblocks):
        cols = np.flatnonzero(labels == b)
        sub = M[:, cols]
        rows = np.unique(sub.nonzero()[0])
        if rows.size == 0:  # molecule with no fitted points: MATLAB gives area 0, error Inf/NaN
            done += cols.size
            continue
        A = sub[rows].toarray()
        x, _ = nnls(A, y[rows], maxiter=50 * max(cols.size, 1))
        areas[cols] = x
        errors[cols] = _block_errors(A, y[rows] - A @ x)
        done += cols.size
        if progress:
            progress(nmol + done, 2 * nmol)
    return GlobalFit(areas, errors, fitmask, M @ areas, nblocks)


def fit_project_global(project: Project, searchrange: float = 1.0,
                       progress: ProgressCallback | None = None) -> GlobalFit:
    """Global fit of a project; areas and errors are written into its molecules."""
    if project.peakdata is None:
        project.update_peakdata()
    result = fit_global(project.peakdata, project.molecules, project.calibration, searchrange,
                        progress)
    for mol, a, e in zip(project.molecules, result.areas, result.errors):
        mol.area, mol.areaerror = float(a), float(e)
    project.gui_status["fitted"] = True
    project.gui_status["changed"] = True
    return result


def find_ifd_files(folder: str | os.PathLike) -> list[Path]:
    """The ``.ifd`` files directly in ``folder``, sorted by name."""
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() == ".ifd")


def batch_fit(folder: str | os.PathLike, searchrange: float = 1.0, output_folder: str | os.PathLike | None = None,
              log: Callable[[str], None] = print,
              progress: Callable[[int, int], None] | None = None) -> list[Path]:
    """Fit every ``.ifd`` file in ``folder`` with :func:`fit_global` and save it.

    MATLAB overwrites the files in place; pass ``output_folder`` to write the
    fitted files somewhere else instead. Returns the written files.
    """
    from .io import load_ifd, save_ifd

    files = find_ifd_files(folder)
    log(f"Found {len(files)} file(s)")
    out_dir = Path(output_folder) if output_folder is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, f in enumerate(files):
        log(f"Fitting {f.name}...")
        project = load_ifd(f)
        result = fit_project_global(project, searchrange)
        target = (out_dir / f.name) if out_dir is not None else f
        save_ifd(project, target)
        written.append(target)
        log(f"  {len(project.molecules)} molecules, {int(result.fitmask.sum())} points, "
            f"{result.blocks} blocks, total area {result.areas.sum():.6g}")
        if progress:
            progress(i + 1, len(files))
    return written


def main(argv: Sequence[str] | None = None) -> None:
    """Command line: ``isotopefit-batch FOLDER [--out DIR] [--searchrange 1]``."""
    import argparse

    p = argparse.ArgumentParser(prog="isotopefit-batch",
                                description="Fit all molecules of every .ifd file in a folder at once.")
    p.add_argument("folder")
    p.add_argument("--out", help="write the fitted files here instead of overwriting them")
    p.add_argument("--searchrange", type=float, default=1.0,
                   help="fit points within searchrange × ½ FWHM of a peak (default 1)")
    args = p.parse_args(argv)
    batch_fit(args.folder, args.searchrange, args.out)


if __name__ == "__main__":
    main()
