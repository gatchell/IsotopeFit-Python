"""Reading raw spectra from HDF5 files.

Two instrument formats are supported, as in ``IsotopeFit.m`` (``load_h5``):

* ``"clustof"`` -- TOFWERK files: ``/FullSpectra/SumSpectrum`` plus either
  the stored ``/FullSpectra/MassAxis`` or a mass axis rebuilt from the mass
  calibration attributes (port of ``h5cal.m``).
* ``"surftof"`` -- IoniTOF files: ``/SPECdata/AverageSpec`` or the
  individually calibrated ``/SPECdata/Intensities`` with ``/CALdata/Spectrum``.

Note on array order: MATLAB's ``h5read`` returns dimensions in the reverse
order of h5py, so MATLAB's ``cal(1, i)`` is ``cal[i, 0]`` here.
"""

from __future__ import annotations

import ast
import os
import re
from typing import Callable

import numpy as np

from ..model import Calibration, Project


def _attr_str(v) -> str:
    if isinstance(v, np.ndarray):
        v = v.ravel()[0]
    if isinstance(v, bytes):
        v = v.decode("latin-1")
    return str(v).replace("\x00", "").strip()


def _attr_num(v) -> float:
    return float(np.asarray(v).ravel()[0])


# ------------------------------------------------------------ mass calibration (h5cal)

_ALLOWED_FUNCS = {"sqrt": np.sqrt, "exp": np.exp, "log": np.log, "abs": np.abs}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


def _compile_expr(expr: str, variables: set[str]) -> Callable[..., np.ndarray]:
    """Compile a calibration formula safely (only arithmetic and a few functions)."""
    tree = ast.parse(expr.replace("^", "**"), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsupported construct in mass calibration formula: {expr!r}")
        if isinstance(node, ast.Name) and node.id not in variables | set(_ALLOWED_FUNCS):
            raise ValueError(f"Unknown name {node.id!r} in mass calibration formula: {expr!r}")
        if isinstance(node, ast.Call) and not (
            isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS
        ):
            raise ValueError(f"Unsupported function call in mass calibration formula: {expr!r}")
    code = compile(tree, "<mass calibration>", "eval")

    def f(**kw):
        # No builtins are available inside the formula. numpy warnings would
        # need them, so evaluate with floating-point warnings switched off.
        with np.errstate(all="ignore"):
            return eval(code, {"__builtins__": {}}, {**_ALLOWED_FUNCS, **kw})  # noqa: S307

    return f


def mass_axis_from_calibration(formula: str, params: dict[str, float], nsamples: int) -> np.ndarray:
    """Mass of every time bin ``i = 1 .. nsamples`` from a TOFWERK calibration formula.

    ``formula`` is the ``MassCalibration Function`` attribute, e.g.
    ``"i = p1*sqrt(m) + p2"``. Bins are numbered from 1, as in MATLAB
    (``h5cal.m``).

    The usual ``i = p1*sqrt(m) + p2`` form is inverted in closed form. Bins
    before the time zero ``p2`` get negative masses, so the axis increases
    throughout. (MATLAB's symbolic solution, ``((i - p2)/p1)^2``, mirrors
    them to small positive masses, so its axis first runs backwards.)
    Formulas of the form ``m = f(i)`` are evaluated directly; anything else
    is inverted numerically and must be monotonic over the bin range.
    """
    lhs, _, rhs = formula.partition("=")
    lhs, rhs = lhs.strip(), rhs.strip()
    i = np.arange(1, nsamples + 1, dtype=float)
    p = {k: float(v) for k, v in params.items()}
    compact = re.sub(r"\s+", "", rhs)

    if lhs == "m":
        return np.asarray(_compile_expr(rhs, {"i", *p})(i=i, **p), dtype=float)
    if lhs != "i":
        raise ValueError(f"Cannot interpret mass calibration formula: {formula!r}")

    if compact == "p1*sqrt(m)+p2":
        t = (i - p["p2"]) / p["p1"]
        return np.sign(t) * t ** 2

    # Generic case: invert i = f(m) numerically on the masses where f is monotonic.
    f = _compile_expr(rhs, {"m", *p})
    m_grid = np.geomspace(1e-4, 1e5, 200_001)
    i_grid = np.asarray(f(m=m_grid, **p), dtype=float)
    ok = np.isfinite(i_grid)
    m_grid, i_grid = m_grid[ok], i_grid[ok]
    d = np.diff(i_grid)
    if not (np.all(d > 0) or np.all(d < 0)):
        raise ValueError(f"Mass calibration {formula!r} is not monotonic; cannot invert it.")
    order = np.argsort(i_grid)
    m = np.interp(i, i_grid[order], m_grid[order])
    # A few Newton steps polish the interpolated guess.
    for _ in range(5):
        h = 1e-7 * np.maximum(m, 1e-6)
        fm = f(m=m, **p)
        dfm = (f(m=m + h, **p) - fm) / h
        m = np.maximum(m - (fm - i) / dfm, 1e-12)
    return m


def tofwerk_mass_axis(h5file) -> np.ndarray:
    """Port of ``h5cal.m``: rebuild the mass axis from the stored calibration."""
    fs = h5file["/FullSpectra"]
    formula = _attr_str(fs.attrs["MassCalibration Function"])
    nsamples = int(_attr_num(h5file.attrs["NbrSamples"]))
    names = sorted(set(re.findall(r"p[0-9]+", formula)), key=lambda s: int(s[1:]))
    params = {n: _attr_num(fs.attrs[f"MassCalibration {n}"]) for n in names}
    return mass_axis_from_calibration(formula, params, nsamples)


# ------------------------------------------------------------------------- loaders


def _load_clustof(h5, use_mass_axis: bool) -> tuple[np.ndarray, np.ndarray]:
    if use_mass_axis:
        mass = np.asarray(h5["/FullSpectra/MassAxis"], dtype=float).ravel()
    else:
        mass = tofwerk_mass_axis(h5)
    signal = np.asarray(h5["/FullSpectra/SumSpectrum"], dtype=float).ravel()
    return mass, signal


def _load_surftof(h5) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(h5["/SPECdata/AverageSpec"], dtype=float).ravel()
    timebins = np.arange(1, signal.size + 1, dtype=float)
    cal = np.asarray(h5["/CALdata/Spectrum"], dtype=float)  # (n_spectra, 2): a, b
    spectra = np.asarray(h5["/SPECdata/Intensities"], dtype=float)  # (n_spectra, n_bins)
    cal = cal.reshape(-1, 2)
    spectra = spectra.reshape(-1, signal.size)

    if cal.shape[0] != spectra.shape[0]:
        a, b = cal[0]
        return ((timebins - b) / a) ** 2, signal

    # Calibrate each spectrum separately and sum them on the first mass axis.
    a0, b0 = cal[0]
    mass = ((timebins - b0) / a0) ** 2
    total = spectra[0].copy()
    for (a, b), spec in zip(cal[1:], spectra[1:]):
        if a == a0 and b == b0:
            total += spec
        else:
            # MATLAB interp1 (linear) gives NaN outside the range; so does this.
            m_i = ((timebins - b) / a) ** 2
            total += np.interp(mass, m_i, spec, left=np.nan, right=np.nan)
    return mass, total


def load_h5(
    path: str | os.PathLike,
    instrument: str = "clustof",
    use_mass_axis: bool = False,
) -> Project:
    """Load a raw spectrum from an HDF5 file into a new :class:`Project`.

    ``instrument`` is ``"clustof"`` (TOFWERK) or ``"surftof"`` (IoniTOF).
    For CLUSTOF files, ``use_mass_axis=False`` (MATLAB's recommended default)
    rebuilds the mass axis from the stored calibration; ``True`` uses the
    mass axis written at measurement time.
    """
    import h5py

    path = os.fspath(path)
    with h5py.File(path, "r") as h5:
        kind = instrument.lower()
        if kind == "clustof":
            mass, signal = _load_clustof(h5, use_mass_axis)
        elif kind == "surftof":
            mass, signal = _load_surftof(h5)
        else:
            raise ValueError(f"Unknown instrument {instrument!r}; use 'clustof' or 'surftof'.")

    project = Project(
        raw_peakdata=np.column_stack([mass, signal]),
        calibration=Calibration(),
        fileinfo={
            "originalfilename": os.path.splitext(os.path.basename(path))[0],
            "pathname": os.path.dirname(os.path.abspath(path)),
            "filename": "",
            "h5completepath": os.path.abspath(path),
        },
    )
    project.gui_status.update(file_loaded=True, molecules_loaded=False, calibrated=False,
                              bg_corrected=False, drift_corrected=False, changed=True,
                              fitted=False, cs_selected=False)
    project.peakdata = project.raw_peakdata.copy()
    return project
