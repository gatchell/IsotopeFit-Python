import copy

import numpy as np
import pytest

from isotopefit import BackgroundCorrection, load_ifd
from isotopefit._matlab import colon, interp1_pchip
from isotopefit.spectrum import crop, guess_area, mass2ind, subtract_background, subtract_mass_offset

from conftest import octave_matrix, refdata


@pytest.fixture(scope="module")
def project():
    return load_ifd(refdata("testfile.ifd"), process=False)


def test_background_subtraction_matches_matlab(project):
    ref = octave_matrix("bgcorrected.txt")
    pd = crop(project.raw_peakdata, project.start, project.stop)
    out = subtract_background(pd, project.bgcorrection)
    np.testing.assert_allclose(out[:, 1], ref, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(out[:, 0], pd[:, 0])  # masses untouched


def test_mass_offset_correction_matches_matlab(project):
    # The reference was generated with a PCHIP mass-offset curve, not the
    # spline stored in the file, so set that explicitly.
    ref = octave_matrix("massAxisCorr.txt")
    cal = copy.copy(project.calibration)
    cal.massoffset_method = "PCHIP"
    out = subtract_mass_offset(project.raw_peakdata, cal)
    # MATLAB extrapolates the offset curve beyond the calibrated masses; the port
    # holds it at its end values there, so compare inside the calibrated range.
    inside = (ref > cal.comlist.min() + 1) & (ref < cal.comlist.max() - 1)
    assert inside.sum() > 100_000
    np.testing.assert_allclose(out[inside, 0], ref[inside], rtol=1e-12, atol=1e-9)
    above = ref > cal.comlist.max() + 1  # offset held at its value at the last point
    np.testing.assert_allclose(
        project.raw_peakdata[above, 0] - out[above, 0],
        cal.massoffset(np.array([cal.comlist.max()]))[0], rtol=1e-6)
    np.testing.assert_array_equal(out[:, 1], project.raw_peakdata[:, 1])  # signal untouched


def test_no_background_points_means_no_change():
    pd = np.column_stack([np.arange(5.0), np.ones(5)])
    np.testing.assert_array_equal(subtract_background(pd, BackgroundCorrection()), pd)


def test_guess_area_and_mass2ind():
    pd = np.column_stack([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, -10.0, 5.0]])
    assert guess_area(pd) == 0.0  # 1 + 2 - 10 < 0 -> clipped
    assert guess_area(pd[:2]) == 1.0
    assert guess_area(pd[:1]) == 0.0
    assert mass2ind(pd[:, 0], 2.2) == 2
    assert mass2ind(pd[:, 0], 1.5) == 1  # ties go to the first, as MATLAB's min


def test_colon_matches_matlab_length():
    # Cases where np.arange gets the length wrong.
    assert colon(0.0, 0.1, 0.3).size == 4
    assert colon(1.0, 0.01, 2.0).size == 101
    x = colon(0.0340, 0.01, 9454.46)
    assert x[0] == 0.0340 and x[-1] <= 9454.46 and np.allclose(np.diff(x), 0.01)
    assert colon(1.0, 1.0, 0.0).size == 0


def test_interp1_sorts_like_matlab():
    x = np.array([0.0, 2.0, 1.0, 3.0])
    y = x**2
    np.testing.assert_allclose(interp1_pchip(x, y, [1.0, 2.0]), [1.0, 4.0])
    with pytest.raises(ValueError):
        interp1_pchip([0.0, 1.0, 1.0], [0.0, 1.0, 2.0], [0.5])
