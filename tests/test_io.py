import numpy as np
import pytest

from isotopefit import PeakShape, load_h5, load_ifd, save_ifd
from isotopefit.io.h5 import mass_axis_from_calibration

from conftest import refdata


@pytest.fixture(scope="module")
def project():
    return load_ifd(refdata("testfile.ifd"))


def test_ifd_contents(project):
    assert project.raw_peakdata.shape == (416375, 2)
    assert (project.start, project.stop) == (0, 416375)
    assert len(project.molecules) == 384
    he = project.molecules[0]
    assert he.name == "[He]"
    assert he.peakdata.shape == (1, 2)  # single isotope still 2-D
    assert he.minind == 9078 - 1  # 1-based in the file, 0-based here
    assert he.rootindex == 0
    assert np.isclose(he.areaerror, 286.7946158331912)  # stored as a sparse matrix
    cal = project.calibration
    assert (cal.massoffset_method, cal.resolution_method) == ("Spline", "Polynomial")
    assert cal.comlist.size == 86 and cal.shape.coefs.shape == (15, 4)
    assert project.bgcorrection.bgm.size == 50
    assert project.peakdata.shape == project.raw_peakdata.shape


def test_ifd_round_trip(project, tmp_path):
    path = tmp_path / "copy.ifd"
    save_ifd(project, path)
    assert path.exists()  # no ".mat" appended
    again = load_ifd(path)
    np.testing.assert_array_equal(again.raw_peakdata, project.raw_peakdata)
    np.testing.assert_array_equal(again.peakdata, project.peakdata)
    assert (again.start, again.stop) == (project.start, project.stop)
    assert len(again.molecules) == len(project.molecules)
    for a, b in zip(again.molecules, project.molecules):
        assert a.name == b.name and a.minind == b.minind and a.rootindex == b.rootindex
        np.testing.assert_array_equal(a.peakdata, b.peakdata)
        assert a.area == b.area
    c1, c2 = again.calibration, project.calibration
    np.testing.assert_array_equal(c1.comlist, c2.comlist)
    np.testing.assert_array_equal(c1.shape.coefs, c2.shape.coefs)
    assert c1.namelist == c2.namelist
    assert again.gui_status == project.gui_status


def test_peakshape_matches_ppval():
    g = PeakShape.gaussian()
    # Max at 0, half maximum at +-0.5 by construction of IsotopeFit's shapes.
    assert np.isclose(g(np.array([0.0]))[0] / 2, g(np.array([0.5]))[0], rtol=0.02)
    # ppval form: coefficient j of piece k multiplies (x - breaks[k])^(order-1-j)
    k, dx = 3, 0.1
    x = g.breaks[k] + dx
    manual = np.polyval(g.coefs[k], dx)
    assert np.isclose(g(np.array([x]))[0], manual)


def test_tofwerk_mass_axis_closed_form():
    m = mass_axis_from_calibration("i = p1*sqrt(m) + p2", {"p1": 2000.0, "p2": 100.0}, 5000)
    i = np.arange(1, 5001)
    t = (i - 100) / 2000
    np.testing.assert_allclose(m, np.sign(t) * t ** 2)
    assert np.all(np.diff(m) > 0)  # bins before time zero get negative masses


def test_tofwerk_mass_axis_numeric_inversion():
    p = {"p1": 2000.0, "p2": 100.0, "p3": 0.49}
    m = mass_axis_from_calibration("i = p1*m^p3 + p2", p, 5000)
    i = np.arange(1, 5001)
    sel = i > 200  # well above time zero
    np.testing.assert_allclose(p["p1"] * m[sel] ** p["p3"] + p["p2"], i[sel], rtol=1e-9)


def test_rejects_unsafe_formula():
    with pytest.raises(ValueError):
        mass_axis_from_calibration("i = __import__('os')", {}, 10)


def test_load_clustof_h5():
    pytest.importorskip("h5py")
    p = load_h5(refdata("clustof.h5"))
    assert p.raw_peakdata.shape[1] == 2 and p.raw_peakdata.shape[0] > 1000
    p2 = load_h5(refdata("clustof.h5"), use_mass_axis=True)
    # The rebuilt and the stored mass axis should agree closely away from time zero.
    sel = p2.raw_peakdata[:, 0] > 10
    np.testing.assert_allclose(p.raw_peakdata[sel, 0], p2.raw_peakdata[sel, 0], rtol=1e-3)
