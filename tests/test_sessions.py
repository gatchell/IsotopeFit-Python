import copy

import numpy as np
import pytest

from isotopefit import load_ifd
from isotopefit.calibrate import CalibrationSession
from isotopefit.peakshape import PeakShapeSession, shape_properties
from isotopefit.spectrum import crop, subtract_background

from conftest import refdata
from test_peakshape import load_export


@pytest.fixture(scope="module")
def ref():
    return load_ifd(refdata("testfile.ifd"))


@pytest.fixture()
def session(ref):
    pd = subtract_background(crop(ref.raw_peakdata, ref.start, ref.stop), ref.bgcorrection)
    return CalibrationSession(pd, ref.molecules, ref.calibration)


def test_session_starts_from_stored_calibration(session, ref):
    assert len(session.ranges) == 86
    cal, problems = session.result()
    assert cal.namelist == ref.calibration.namelist
    # Starting values are interpolated at each range's centre of mass, as in MATLAB
    np.testing.assert_allclose(cal.massoffsetlist, ref.calibration.massoffsetlist, atol=1e-4)
    assert (cal.massoffset_method, cal.resolution_method) == ("Spline", "Polynomial")


def test_add_remove_and_lookup(session, ref):
    he30 = next(i for i, m in enumerate(ref.molecules)
                if m.name.startswith("[He]") and i not in session.selected and m.area > 0)
    name = ref.molecules[he30].name
    before = len(session.selected)
    session.add([he30])
    assert len(session.selected) == before + 1
    r = session.range_of(he30)
    assert r is not None
    pos = [m.name for m in session.ranges[r].molecules].index(name)
    assert session.molecule_index(r, pos) == he30
    session.remove([he30])
    assert session.range_of(he30) is None and len(session.selected) == before


def test_edit_fit_and_preview(session):
    r = 10
    old = session.ranges[r].massoffset
    session.step(r, "massoffset", +1)
    assert session.ranges[r].massoffset > old
    session.step(r, "resolution", -1)
    session.fit([r])
    cal, _ = session.result()
    assert cal.massoffsetlist[r] == pytest.approx(old, abs=2e-3)  # back near the optimum
    x, data, total, parts = session.preview(r)
    assert x.size == data.size == total.size and len(parts) == len(session.ranges[r].molecules)
    session.guess_area(r)
    session.set_area(r, 0, 123.0)
    assert session.ranges[r].molecules[0].area == 123.0
    session.massoffset_method = "PCHIP"
    assert session.result()[0].massoffset_method == "PCHIP"


def test_peak_shape_session():
    peakdata, mols, cal, _ = load_export("data_02")
    s = PeakShapeSession(peakdata, mols, cal)
    n = s.x.size
    i = s.add(0.3, 0.4)
    assert s.x.size == n + 1 and s.x[i] == 0.3 and np.all(np.diff(s.x) > 0)
    assert s.y[0] == 0 and s.y[-1] == 0
    assert s.remove([i]) and s.x.size == n
    assert not s.remove(range(n - 2))  # would leave fewer than 3
    j, d = s.nearest(s.x[3], s.y[3])
    assert j == 3 and d == 0
    s.set_fixed(3, True)
    x3 = s.x[3]
    s.fit()
    assert s.x[3] == x3  # fixed knot stayed
    new = s.result()
    assert shape_properties(new.shape)["fwhm"] == pytest.approx(1, abs=1e-3)
