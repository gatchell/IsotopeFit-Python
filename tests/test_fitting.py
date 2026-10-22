import copy

import numpy as np
import pytest
from scipy.integrate import quad, trapezoid
from scipy.optimize import nnls

from isotopefit import Calibration, Molecule, PeakShape, Settings, fit_project, load_ifd
from isotopefit._matlab import fminsearch, fminsearchbnd, round_half_away
from isotopefit.fitting import _limit_points, fit_parameters, fit_with_calibration
from isotopefit.molecules import find_mass_range2, find_ranges, init_molecules
from isotopefit.pattern import adapt_peak_shape, multispec, pattern

from conftest import octave_matrix, refdata


# ------------------------------------------------------------------ MATLAB helpers


def test_fminsearch_reproduces_matlab_rosenbrock():
    # MATLAB documentation: fminsearch on Rosenbrock from [-1.2, 1] gives
    # fval = 8.1777e-10 after 159 function evaluations.
    def banana(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    x, fval, flag, evals = fminsearch(banana, [-1.2, 1.0])
    assert flag == 1 and evals == 159
    assert f"{fval:.4e}" == "8.1777e-10"
    np.testing.assert_allclose(x, [1, 1], atol=1e-4)


def test_fminsearchbnd_bounds_and_fixed():
    x, _, _ = fminsearchbnd(lambda x: (x[0] - 3) ** 2 + (x[1] - 1) ** 2, [0.5, 1.0],
                            [0, 1.0], [2, 1.0])
    assert x[1] == 1.0  # lb == ub: held fixed
    assert np.isclose(x[0], 2.0, atol=1e-3)  # pushed against the upper bound


def test_round_half_away():
    np.testing.assert_array_equal(round_half_away([0.5, 1.5, 2.5, -0.5]), [1, 2, 3, -1])


def test_nnls_matches_octave_lsqnonneg():
    C = octave_matrix("nnls_C.txt")
    d = octave_matrix("nnls_d.txt")
    with open(refdata("nnls_x_solution.txt")) as fh:  # decimal commas (written by C#)
        x_ref = np.array([float(s.replace(",", ".")) for s in fh.read().split()])
    x, _ = nnls(C, d, maxiter=10 * C.shape[1])
    np.testing.assert_allclose(x, x_ref, rtol=1e-6, atol=1e-9)


# ------------------------------------------------------------------ patterns


def test_adapted_peak_shape_has_requested_area_and_width():
    g = PeakShape.gaussian()
    p = adapt_peak_shape(g, fwhm=0.05, area=7.0)
    area = sum(quad(lambda x: float(p(np.array([x]))[0]), a, b)[0]
               for a, b in zip(p.breaks[:-1], p.breaks[1:]))
    assert np.isclose(area, 7.0 * area_of(g), rtol=1e-9)
    x = np.linspace(p.breaks[0], p.breaks[-1], 20001)
    y = p(x)
    above = x[y >= y.max() / 2]
    assert np.isclose(above[-1] - above[0], 0.05, rtol=0.02)


def area_of(shape):
    return sum(quad(lambda x: float(shape(np.array([x]))[0]), a, b)[0]
               for a, b in zip(shape.breaks[:-1], shape.breaks[1:]))


def test_pattern_area_and_position():
    mol = Molecule("X", [[100.0, 0.75], [101.0, 0.25]])
    mol.com = 100.25
    x = np.linspace(99, 102.5, 70001)
    y = pattern(mol, 10.0, 2000.0, 0.02, x, PeakShape.gaussian())
    assert np.isclose(trapezoid(y, x), 10.0 * area_of(PeakShape.gaussian()), rtol=1e-4)
    assert np.isclose(x[np.argmax(y)], 100.02, atol=1e-4)  # shifted by the mass offset
    # Unsorted mass axis gives the same values.
    perm = np.random.default_rng(0).permutation(x.size)
    np.testing.assert_allclose(pattern(mol, 10.0, 2000.0, 0.02, x[perm], PeakShape.gaussian()),
                               y[perm])


def test_limit_points_matches_matlab_indexing():
    ind = np.arange(10, 20)  # 10 points
    # MATLAB: ind(round((1:4)*(10/4))) = ind([3 5 8 10]) (1-based; 2.5 and 7.5 round up)
    np.testing.assert_array_equal(_limit_points(ind, 4), ind[[2, 4, 7, 9]])
    np.testing.assert_array_equal(_limit_points(ind, 20), ind)


def test_find_mass_range2_sorted_and_unsorted():
    mol = Molecule("X", [[100.0, 1.0], [102.0, 0.5]])
    mol.com, mol.area = 100.5, 1.0
    x = np.linspace(95, 105, 1001)
    ind = find_mass_range2(x, [mol], 1000.0, 0.0, 3.0)
    sigma = 100.5 / 1000 / (2 * np.sqrt(2 * np.log(2)))
    expected = np.flatnonzero((np.abs(x - 100) <= 3 * sigma) | (np.abs(x - 102) <= 3 * sigma))
    np.testing.assert_array_equal(ind, expected)
    rev = find_mass_range2(x[::-1], [mol], 1000.0, 0.0, 3.0)
    np.testing.assert_array_equal(np.sort(x.size - 1 - rev), expected)


# ------------------------------------------------------------------ fitting


def _synthetic(massoffset=0.0):
    rng = np.random.default_rng(1)
    shape = PeakShape.gaussian()
    cal = Calibration(resolution_param=1500.0)
    mols = [Molecule("A", [[50.0, 0.9], [51.0, 0.1]]),
            Molecule("B", [[50.05, 0.6], [51.05, 0.4]]),  # overlaps A
            Molecule("C", [[80.0, 1.0]])]
    x = np.arange(40, 90, 0.002)
    true = {"A": 100.0, "B": 40.0, "C": 250.0}
    for m in mols:
        m.area = true[m.name]
        m.com = float(np.average(m.peakdata[:, 0], weights=m.peakdata[:, 1]))
    y = multispec(mols, 1500.0, massoffset, x, shape) + rng.normal(0, 1e-3, x.size)
    peakdata = np.column_stack([x, y])
    return peakdata, init_molecules(mols, peakdata), cal, true


@pytest.mark.parametrize("mode", ["ranges", "molecules"])
def test_linear_system_recovers_areas(mode):
    peakdata, mols, cal, true = _synthetic()
    fitted = fit_with_calibration(mols, peakdata, cal, mode=mode, searchrange=1.0)
    for m in fitted:
        assert np.isclose(m.area, true[m.name], rtol=1e-3), (m.name, m.area)
        assert np.isfinite(m.areaerror) and m.areaerror >= 0


def test_simplex_lin_combi_finds_mass_offset():
    peakdata, mols, cal, true = _synthetic(massoffset=0.004)
    fitted = fit_with_calibration(mols, peakdata, cal, searchrange=1.0, deltam=0.01,
                                  method="simplex_lin_combi")
    plain = fit_with_calibration(mols, peakdata, cal, searchrange=1.0)
    for m, p in zip(fitted, plain):
        assert abs(m.area - true[m.name]) <= abs(p.area - true[m.name]) + 1e-6
        assert np.isclose(m.area, true[m.name], rtol=5e-3), (m.name, m.area)


def test_ranges_group_overlapping_molecules():
    peakdata, mols, cal, _ = _synthetic()
    ranges = find_ranges(mols, cal, 1.0)
    assert [[m.name for m in r.molecules] for r in ranges] == [["A", "B"], ["C"]]


def test_fit_testfile_by_ranges_close_to_stored_areas():
    """Fitting by ranges (the GUI default) gives nearly the same areas."""
    project = load_ifd(refdata("testfile.ifd"))
    stored = np.array([m.area for m in project.molecules])
    names = [m.name for m in project.molecules]
    fitted = fit_project(copy.deepcopy(project), Settings())
    assert [m.name for m in fitted] == names
    new = np.array([m.area for m in fitted])
    big = stored > 1e-3 * stored.max()
    rel = np.abs(new - stored)[big] / stored[big]
    # The stored areas come from a molecule-by-molecule fit (see below), so
    # only robust statistics are compared here.
    assert np.median(rel) < 2e-3
    assert np.percentile(rel, 90) < 2e-2
    assert abs(new.sum() / stored.sum() - 1) < 1e-3
    top = np.argsort(-stored)[:20]
    np.testing.assert_allclose(new[top], stored[top], rtol=1e-2)


def test_fit_testfile_molecule_by_molecule_reproduces_matlab():
    """testfile.ifd was fitted molecule by molecule; the port reproduces it."""
    project = load_ifd(refdata("testfile.ifd"))
    stored = np.array([m.area for m in project.molecules])
    stored_err = np.array([m.areaerror for m in project.molecules])
    fitted = fit_project(copy.deepcopy(project), Settings(), mode="molecules")
    new = np.array([m.area for m in fitted])
    err = np.array([m.areaerror for m in fitted])
    nonzero = stored > 0
    rel = np.abs(new - stored)[nonzero] / stored[nonzero]
    # Nearly all molecules agree to rounding error. (Helium sits on the first
    # calibration point; below it the mass offset is now held instead of
    # extrapolated, which moves a few helium fits by about 1e-5.)
    assert np.mean(rel < 1e-9) > 0.85
    # Errors are one standard deviation (MATLAB: 1.96 times that) and use n - m
    # degrees of freedom instead of MATLAB's n, so 1.96 * err is larger than
    # MATLAB's by sqrt(n / (n - m)): a little for wide ranges, more for narrow ones.
    same = (rel < 1e-9) & (stored_err[nonzero] > 0)
    ratio = (1.96 * err[nonzero] / stored_err[nonzero])[same]
    assert np.all(ratio >= 1 - 1e-9) and np.median(ratio) > 1.0001 and np.median(ratio) < 1.2
    assert np.all(new[~nonzero] == 0)
    # The few others sit in nearly degenerate groups where tiny numerical
    # differences move area between molecules; the totals still agree.
    assert abs(new.sum() / stored.sum() - 1) < 1e-4


@pytest.mark.parametrize("method", ["linear_system", "linear_system_baseline"])
def test_errors_are_one_standard_deviation(method):
    """Refit the same overlapping peaks with fresh noise: the scatter of the areas
    matches the reported error (one standard deviation, not a 95 % interval)."""
    shape = PeakShape.gaussian()
    mols = [Molecule(name="A", peakdata=[[100.0, 0.6], [101.0, 0.4]]),
            Molecule(name="B", peakdata=[[101.03, 1.0]])]
    for m in mols:
        m.com = float(np.average(m.peakdata[:, 0], weights=m.peakdata[:, 1]))
    x = np.arange(99.5, 101.6, 0.002)
    params = np.array([1000.0, 500.0, 2500.0, 0.0])
    clean = sum(pattern(m, a, 2500.0, 0.0, x, shape) for m, a in zip(mols, params[:2]))
    rng = np.random.default_rng(3)
    areas, errors = [], []
    for _ in range(400):
        spec = clean + rng.normal(0, 20.0, x.size)
        fit = fit_parameters(method, spec, x, shape, mols, params, None, None)
        areas.append(fit.params[:2])
        errors.append(fit.errors[:2])
    scatter = np.std(areas, axis=0, ddof=1)
    reported = np.mean(errors, axis=0)
    np.testing.assert_allclose(reported, scatter, rtol=0.12)


def test_reduced_solution_equals_direct(monkeypatch):
    """Large fits use the m × m normal-equation form; it gives the same areas and errors."""
    import isotopefit.fitting as fitting

    project = load_ifd(refdata("testfile.ifd"))
    ticks = []
    direct = fit_project(copy.deepcopy(project), Settings(), mode="ranges",
                         progress=lambda i, n: ticks.append((i, n)))
    # progress counts molecules, so it ends at the number of molecules fitted
    assert ticks[-1][0] == ticks[-1][1] == sum(1 for m in direct)
    monkeypatch.setattr(fitting, "REDUCED_SIZE", 0)  # every fit in reduced form
    reduced = fit_project(copy.deepcopy(project), Settings(), mode="ranges")
    a = np.array([m.area for m in direct])
    b = np.array([m.area for m in reduced])
    big = a > 1e-6 * a.max()
    np.testing.assert_allclose(b[big], a[big], rtol=1e-6)
    ea = np.array([m.areaerror for m in direct])
    eb = np.array([m.areaerror for m in reduced])
    ok = np.isfinite(ea) & (ea > 0) & big
    np.testing.assert_allclose(eb[ok], ea[ok], rtol=1e-6)
