import numpy as np
import pytest

from isotopefit import load_ifd
from isotopefit.calibration import calibration_curve

from conftest import octave_matrix, refdata


@pytest.fixture(scope="module")
def cal():
    return load_ifd(refdata("testfile.ifd"), process=False).calibration


@pytest.mark.parametrize("method, fname", [("PCHIP", "PCHIPEvaluation.txt"),
                                           ("Spline", "SplineEvaluation.txt")])
def test_interpolated_resolution_matches_matlab(cal, method, fname):
    ref = octave_matrix(fname)  # columns: mass, resolution
    out = calibration_curve(cal.comlist, cal.resolutionlist, 0, method, ref[:, 0])
    np.testing.assert_allclose(out, ref[:, 1], rtol=1e-12)


def test_polynomial_resolution_matches_matlab(cal):
    # Evaluated at comlist. The reference used degree 3 (the file stores 2).
    ref = octave_matrix("PolynomialEvaluation.txt")
    out = calibration_curve(cal.comlist, cal.resolutionlist, 3, "Polynomial", cal.comlist)
    np.testing.assert_allclose(out, ref, rtol=1e-12)


def test_flat_and_single_point():
    x = np.linspace(1, 100, 7)
    assert np.all(calibration_curve([], [], 3000, "Flat", x) == 3000)
    assert np.all(calibration_curve([50.0], [0.2], 0, "pchip", x) == 0.2)
    assert np.all(calibration_curve([50.0], [0.2], 0, "spline", x) == 0.2)


def test_unknown_method():
    with pytest.raises(ValueError):
        calibration_curve([1, 2], [1, 2], 0, "magic", [1.5])


def test_sigma_from_resolution(cal):
    m = np.array([10.0, 100.0, 500.0])
    expected = m / cal.resolution(m) / (2 * np.sqrt(2 * np.log(2)))
    np.testing.assert_allclose(cal.sigma(m), expected)
