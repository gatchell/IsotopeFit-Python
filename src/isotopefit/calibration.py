"""Calibration curves (port of ``getcalibrationdata.m`` and helpers)."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator


def calibration_curve(x, y, param, method: str, axis) -> np.ndarray:
    """Evaluate a calibration curve through the points (x, y) on ``axis``.

    Port of ``getcalibrationdata.m``.

    ``method`` (case-insensitive):

    * ``flat``       -- constant ``param``
    * ``polynomial`` -- least-squares polynomial of degree ``param``
    * ``spline``     -- not-a-knot cubic spline (MATLAB ``spline``)
    * ``pchip``      -- shape-preserving cubic (MATLAB ``pchip``)
    * ``spaps``      -- smoothing spline; not ported yet

    Outside the calibrated masses (below the first or above the last point)
    the curve keeps its value at that end. (MATLAB extrapolates the polynomial
    or spline, which for a spline mass offset reaches hundreds of u a few
    hundred u beyond the last calibration point, and can make the resolution
    negative.)
    """
    axis = np.asarray(axis, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = method.lower()
    if m != "flat" and x.size > 1:
        axis = np.clip(axis, x.min(), x.max())

    if m == "flat":
        return np.full(axis.shape, float(param))
    if m == "polynomial":
        p = np.polyfit(x, y, int(param))
        return np.polyval(p, axis)
    if m in ("spline", "pchip"):
        if x.size == 1:  # splinemod/pchipmod: a single point gives a constant
            return np.full(axis.shape, y[0])
        if m == "spline":
            # MATLAB spline: not-a-knot ends; with 2 points it is a straight
            # line, with 3 a parabola. CubicSpline handles both the same way.
            return CubicSpline(x, y, bc_type="not-a-knot")(axis)
        return PchipInterpolator(x, y)(axis)
    if m == "spaps":
        raise NotImplementedError(
            "The 'spaps' calibration method (Curve Fitting Toolbox smoothing spline) "
            "is not ported yet."
        )
    raise ValueError(f"Unknown calibration method: {method!r}")
