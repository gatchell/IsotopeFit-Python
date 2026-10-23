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


# ------------------------------------------------------------------ text spectra


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_ascii_tab_separated_with_header(tmp_path):
    from isotopefit.io import load_ascii

    p = _write(tmp_path, "spec.txt", "Mass\tSignal\n(u)\t(counts)\n1.0\t10\n1.5\t20\n2.0\t30\n")
    project = load_ascii(p)
    np.testing.assert_allclose(project.raw_peakdata, [[1.0, 10], [1.5, 20], [2.0, 30]])
    assert project.fileinfo["originalfilename"] == "spec" and project.peakdata is not None


def test_ascii_tofdaq_export_uses_third_column(tmp_path):
    from isotopefit.io import read_ascii_spectrum

    header = "mass spectrum\n=============\n" + "".join(f"info line {i}\n" for i in range(11))
    rows = "".join(f"{m}\t{i}\t{10 * m}\n" for i, m in enumerate((1.0, 2.0, 3.0)))
    data = read_ascii_spectrum(_write(tmp_path, "tofdaq.txt", header + rows))
    np.testing.assert_allclose(data, [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])


@pytest.mark.parametrize("text", [
    "1.0 10\n2.0 20\n3.0 30\n",                      # no header, spaces
    "# comment\n1.0,10\n2.0,20\n3.0,30\n",           # CSV
    "mass;signal\n1,0;10\n2,0;20\n3,0;30\n",         # semicolons, decimal commas
    "m\tI\n1,0\t10\n2,0\t20\n3,0\t30\n",             # tabs, decimal commas
])
def test_ascii_separators(tmp_path, text):
    from isotopefit.io import read_ascii_spectrum

    data = read_ascii_spectrum(_write(tmp_path, "s.dat", text))
    np.testing.assert_allclose(data, [[1.0, 10], [2.0, 20], [3.0, 30]])


def test_ascii_reads_back_the_spectrum_export(tmp_path):
    from isotopefit.io import load_ascii
    from isotopefit.view import export_spectrum

    project = load_ifd(refdata("testfile.ifd"))
    path = tmp_path / "export.txt"
    export_spectrum(project, path, with_fit=True)  # mass, signal, fit
    again = load_ascii(path)
    np.testing.assert_allclose(again.raw_peakdata, project.peakdata, rtol=1e-6)


def test_ascii_rejects_files_without_numbers(tmp_path):
    from isotopefit.io import read_ascii_spectrum

    with pytest.raises(ValueError):
        read_ascii_spectrum(_write(tmp_path, "notes.txt", "just some\nwords here\n"))
