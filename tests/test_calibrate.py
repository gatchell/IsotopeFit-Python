import copy
import warnings

import numpy as np
import pytest

from isotopefit import Calibration, Project, Settings, fit_project, load_h5, load_ifd
from isotopefit.background import correct_background, find_background
from isotopefit.calibrate import (calibrate, calibrate_project, calibration_from_ranges,
                                  calibration_ranges, interpolate_parameters, load_calibration,
                                  select_molecules)
from isotopefit.molecules import FitRange
from isotopefit.spectrum import crop, subtract_background

from conftest import refdata


@pytest.fixture(scope="module")
def ref():
    return load_ifd(refdata("testfile.ifd"))


# ------------------------------------------------------------------ background


def test_background_matches_matlab_exactly(ref):
    bg = ref.bgcorrection
    bgm, bgy, start, stop = find_background(ref.raw_peakdata[:, 0], ref.raw_peakdata[:, 1],
                                            bg.ndiv, bg.percent, bg.startmass, bg.endmass)
    np.testing.assert_array_equal(bgm, bg.bgm)
    np.testing.assert_array_equal(bgy, bg.bgy)
    assert (start, stop) == (ref.start, ref.stop)


def test_background_crop_limits():
    m = np.arange(100.0)
    y = np.ones(100)
    bgm, bgy, start, stop = find_background(m, y, ndiv=4, percent=50, startmass=10.5, endmass=60)
    assert (start, stop) == (11, 61)  # first mass >= 10.5 ... first mass >= 60, inclusive
    assert bgm.size == 4 and np.all(bgy == 1)
    assert np.all((bgm >= 11) & (bgm <= 60))


def test_correct_background_on_project(ref):
    proj = Project(raw_peakdata=ref.raw_peakdata, calibration=copy.deepcopy(ref.calibration))
    correct_background(proj, ndiv=50, percent=40)
    np.testing.assert_array_equal(proj.peakdata, ref.peakdata)
    assert proj.gui_status["bg_corrected"] and not proj.gui_status["fitted"]


# ------------------------------------------------------------------ calibration pieces


def test_interpolate_parameters():
    assert interpolate_parameters(Calibration(), 100.0) == (0.0, 3000.0)
    cal = Calibration(comlist=[10, 20, 30], massoffsetlist=[0.1, 0.2, 0.4],
                      resolutionlist=[1000, 2000, 3000])
    assert interpolate_parameters(cal, 5.0) == (0.1, 1000.0)
    assert interpolate_parameters(cal, 30.2) == (0.4, 3000.0)
    mo, res = interpolate_parameters(cal, 20.0)
    assert np.isclose(mo, 0.2) and np.isclose(res, 2000.0)


def test_calibration_ranges_rebuild_stored_ranges(ref):
    cal = ref.calibration
    selected = select_molecules(ref.molecules, cal.namelist)
    assert len(selected) == len(cal.namelist) == 259
    ranges = calibration_ranges(ref.molecules, selected, cal, Settings().searchrange)
    assert len(ranges) == cal.comlist.size == 86
    assert [m.name for r in ranges for m in r.molecules] == cal.namelist


def test_calibration_from_ranges_methods_and_checks():
    ranges = [FitRange([], 0, 0, 0.0, 0.0, com=c, resolution=r, massoffset=o)
              for c, r, o in [(10, 1000, 0.0), (50, 2000, 0.01), (100, 1500, 0.02)]]
    cal, problems = calibration_from_ranges(ranges, Calibration(), np.linspace(1, 120, 500),
                                            massoffset_method="Flat", resolution_method="Flat")
    assert cal.massoffset_param == pytest.approx(0.01) and cal.resolution_param == pytest.approx(1500)
    assert problems == []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cal, problems = calibration_from_ranges(ranges, Calibration(), np.linspace(1, 400, 500),
                                                resolution_method="Polynomial", resolution_param=5)
    assert cal.resolution_param == 2  # capped at npoints - 1
    assert any("too high" in str(x.message) for x in w)
    # The downward parabola would go below 0 by 400, but beyond the last
    # calibration point (100) the curve keeps its end value.
    assert not any("negative" in p for p in problems)
    np.testing.assert_allclose(cal.resolution(np.array([100.0, 400.0])), 1500)
    ranges[2].resolution = -100.0  # negative inside the calibrated range
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal, problems = calibration_from_ranges(ranges, Calibration(), np.linspace(1, 400, 500),
                                                resolution_method="PCHIP")
    assert any("negative" in p for p in problems)


# ------------------------------------------------------------------ against MATLAB


def test_refitting_calibration_reproduces_matlab(ref):
    """Refit the 86 stored calibration ranges starting from the stored calibration."""
    cal = ref.calibration
    peakdata = subtract_background(crop(ref.raw_peakdata, ref.start, ref.stop), ref.bgcorrection)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the stored polynomial goes negative at high masses
        result = calibrate(peakdata, ref.molecules, cal, settings=Settings())
    new = result.calibration
    assert new.namelist == cal.namelist
    assert (new.massoffset_method, new.resolution_method) == (cal.massoffset_method,
                                                              cal.resolution_method)
    mo = np.abs(new.massoffsetlist - cal.massoffsetlist)
    res = np.abs(new.resolutionlist / cal.resolutionlist - 1)
    assert np.median(mo) < 1e-6 and mo.max() < 2e-3
    assert np.median(res) < 1e-4 and res.max() < 0.05
    # MATLAB warns that the polynomial resolution goes negative at high masses;
    # beyond the last calibration point the curve now keeps its end value.
    assert not any("negative" in p for p in result.problems)


def test_pipeline_raw_data_to_areas(ref):
    """New project from raw data -> load calibration -> recalibrate -> fit."""
    proj = Project(raw_peakdata=ref.raw_peakdata)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_calibration(proj, refdata("testfile.ifd"))
        np.testing.assert_array_equal(proj.peakdata, ref.peakdata)
        calibrate_project(proj, settings=Settings())
    fitted = fit_project(proj, Settings(), mode="molecules")
    stored = np.array([m.area for m in ref.molecules])
    new = np.array([m.area for m in fitted])
    big = stored > 1e-3 * stored.max()
    rel = np.abs(new - stored)[big] / stored[big]
    assert np.median(rel) < 5e-3
    assert abs(new.sum() / stored.sum() - 1) < 5e-3


def test_h5_background_smoke():
    pytest.importorskip("h5py")
    proj = load_h5(refdata("clustof.h5"))
    correct_background(proj, ndiv=100, percent=40)
    assert np.all(np.isfinite(proj.peakdata))
    assert proj.bgcorrection.bgm.size == 100
    # Most of a spectrum is background, so the correction brings the median towards zero.
    assert abs(np.median(proj.peakdata[:, 1])) <= abs(np.median(proj.raw_peakdata[:, 1]))
