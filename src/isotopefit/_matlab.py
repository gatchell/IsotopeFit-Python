"""Small helpers that reproduce MATLAB behaviour exactly where it matters."""

from __future__ import annotations

import numpy as np


def colon(a: float, d: float, b: float) -> np.ndarray:
    """MATLAB's ``a:d:b`` for floating-point arguments.

    ``np.arange`` can end up one element longer or shorter than MATLAB,
    and accumulates rounding error differently. MATLAB computes the count
    with a small tolerance and fills the vector symmetrically from both
    ends, which this reproduces.
    """
    if d == 0 or (d > 0 and a > b) or (d < 0 and a < b):
        return np.empty(0)
    tol = 2.0 * np.finfo(float).eps * max(abs(a), abs(b))
    n = int(np.floor((b - a) / d + tol / abs(d)))  # number of steps
    k = np.arange(n + 1, dtype=float)
    end = a + n * d
    half = n // 2
    out = np.empty(n + 1)
    out[: half + 1] = a + k[: half + 1] * d
    out[half + 1 :] = end - (n - k[half + 1 :]) * d
    return out


def interp1_pchip(x: np.ndarray, y: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """MATLAB's ``interp1(x, y, xi, 'pchip', 'extrap')``.

    Unlike SciPy, MATLAB accepts sample points in any order and sorts them
    first. Repeated sample points are an error in both.
    """
    from scipy.interpolate import PchipInterpolator

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if np.any(np.diff(x) <= 0):
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
        if np.any(np.diff(x) == 0):
            raise ValueError("interp1: the sample points must be distinct.")
    return PchipInterpolator(x, y, extrapolate=True)(np.asarray(xi, dtype=float))


def round_half_away(x):
    """MATLAB's ``round`` (halves away from zero); ``np.round`` rounds halves to even."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def fminsearch(fun, x0, tolx: float = 1e-4, tolfun: float = 1e-4,
               maxfunevals: int | None = None, maxiter: int | None = None):
    """MATLAB's ``fminsearch`` (Nelder-Mead), step for step.

    SciPy's Nelder-Mead is close but differs in details such as the initial
    simplex and tie handling, which can change where a fit ends up. This
    follows MATLAB's algorithm exactly: initial simplex with 5 % steps
    (0.00025 for zero components), reflection 1, expansion 2, contraction
    0.5, shrink 0.5, and MATLAB's stopping tests.

    Returns ``(x, fval, exitflag, n_evaluations)``; exitflag is 1 when the
    tolerances were met and 0 when an evaluation or iteration limit stopped it.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    maxfunevals = 200 * n if maxfunevals is None else maxfunevals
    maxiter = 200 * n if maxiter is None else maxiter
    rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5
    v = np.empty((n + 1, n))
    fv = np.empty(n + 1)
    v[0] = x0
    fv[0] = fun(x0)
    for j in range(n):
        y = x0.copy()
        y[j] = 1.05 * y[j] if y[j] != 0 else 0.00025
        v[j + 1] = y
        fv[j + 1] = fun(y)
    order = np.argsort(fv, kind="stable")
    v, fv = v[order], fv[order]
    itercount, evals = 1, n + 1

    while evals < maxfunevals and itercount < maxiter:
        if (np.max(np.abs(fv[0] - fv[1:])) <= max(tolfun, 10 * np.spacing(abs(fv[0])))
                and np.max(np.abs(v[1:] - v[0])) <= max(tolx, 10 * np.spacing(abs(np.max(v[0]))))):
            return v[0], fv[0], 1, evals
        xbar = v[:n].sum(axis=0) / n
        xr = (1 + rho) * xbar - rho * v[-1]
        fxr = fun(xr)
        evals += 1
        if fxr < fv[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * v[-1]
            fxe = fun(xe)
            evals += 1
            if fxe < fxr:
                v[-1], fv[-1] = xe, fxe
            else:
                v[-1], fv[-1] = xr, fxr
        elif fxr < fv[n - 1]:
            v[-1], fv[-1] = xr, fxr
        else:
            shrink = False
            if fxr < fv[-1]:  # outside contraction
                xc = (1 + psi * rho) * xbar - psi * rho * v[-1]
                fxc = fun(xc)
                evals += 1
                if fxc <= fxr:
                    v[-1], fv[-1] = xc, fxc
                else:
                    shrink = True
            else:  # inside contraction
                xcc = (1 - psi) * xbar + psi * v[-1]
                fxcc = fun(xcc)
                evals += 1
                if fxcc < fv[-1]:
                    v[-1], fv[-1] = xcc, fxcc
                else:
                    shrink = True
            if shrink:
                for j in range(1, n + 1):
                    v[j] = v[0] + sigma * (v[j] - v[0])
                    fv[j] = fun(v[j])
                evals += n
        order = np.argsort(fv, kind="stable")
        v, fv = v[order], fv[order]
        itercount += 1
    return v[0], fv[0], 0, evals


def fminsearchbnd(fun, x0, lb, ub, **options):
    """Port of John D'Errico's ``fminsearchbnd`` (bounded ``fminsearch``).

    Bounded variables are mapped to unbounded ones (square for one-sided
    bounds, sine for two-sided); variables with ``lb == ub`` are held
    fixed and removed from the search. Options go to :func:`fminsearch`.
    Returns ``(x, fval, exitflag)``.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    lb = np.full(n, -np.inf) if lb is None else np.asarray(lb, dtype=float).ravel()
    ub = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=float).ravel()
    cls = np.isfinite(lb).astype(int) + 2 * np.isfinite(ub).astype(int)
    cls[(cls == 3) & (lb == ub)] = 4

    x0u = []
    for i in range(n):
        c = cls[i]
        if c == 1:
            x0u.append(0.0 if x0[i] <= lb[i] else np.sqrt(x0[i] - lb[i]))
        elif c == 2:
            x0u.append(0.0 if x0[i] >= ub[i] else np.sqrt(ub[i] - x0[i]))
        elif c == 3:
            if x0[i] <= lb[i]:
                x0u.append(-np.pi / 2)
            elif x0[i] >= ub[i]:
                x0u.append(np.pi / 2)
            else:
                t = 2 * (x0[i] - lb[i]) / (ub[i] - lb[i]) - 1
                x0u.append(2 * np.pi + np.arcsin(max(-1.0, min(1.0, t))))
        elif c == 0:
            x0u.append(x0[i])

    def xtransform(xu):
        x = np.empty(n)
        k = 0
        for i in range(n):
            c = cls[i]
            if c == 1:
                x[i] = lb[i] + xu[k] ** 2
                k += 1
            elif c == 2:
                x[i] = ub[i] - xu[k] ** 2
                k += 1
            elif c == 3:
                t = (np.sin(xu[k]) + 1) / 2 * (ub[i] - lb[i]) + lb[i]
                x[i] = max(lb[i], min(ub[i], t))
                k += 1
            elif c == 4:
                x[i] = lb[i]
            else:
                x[i] = xu[k]
                k += 1
        return x

    if not x0u:  # everything fixed by the bounds
        x = xtransform(np.empty(0))
        return x, fun(x), 0
    xu, fval, flag, _ = fminsearch(lambda xu: fun(xtransform(xu)), np.array(x0u), **options)
    return xtransform(xu), fval, flag
