import numpy as np
import pytest
import scipy.io as sio

from isotopefit import Calibration, Molecule, PeakShape
from isotopefit.io.ifd import _calibration_from_mat, _molecules_from_mat
from isotopefit.pattern import adapt_peak_shape, multispec
from isotopefit.peakshape import (ConvolutionCore, _msd, convolution_core_for, finalize_shape,
                                  fit_knots, molecule_stems, shape_from_knots, shape_properties,
                                  spline_shape)

from conftest import refdata

SETS = ["data_01", "data_02", "data_03", "data_04", "data_05"]


class _Struct:
    def __init__(self, d):
        self._fieldnames = [k for k in d if not k.startswith("__")]
        for k in self._fieldnames:
            setattr(self, k, d[k])


def load_export(name):
    """A dataset exported by MATLAB's peak_shape_generator."""
    cal = _calibration_from_mat(_Struct(sio.loadmat(refdata(f"{name}/calibration.mat"),
                                                    squeeze_me=True, struct_as_record=False)))
    mols = _molecules_from_mat(sio.loadmat(refdata(f"{name}/molecules.mat"), squeeze_me=True,
                                           struct_as_record=False)["molecules"])
    return (np.loadtxt(refdata(f"{name}/peakdata.txt")), mols, cal,
            np.loadtxt(refdata(f"{name}/molecule_stems.txt")))


@pytest.mark.parametrize("name", SETS)
def test_molecule_stems_match_matlab(name):
    peakdata, mols, _, stems_ref = load_export(name)
    stems = molecule_stems(mols, peakdata[:, 0])
    # The reference was written with save -ascii (8 significant digits).
    np.testing.assert_allclose(stems, stems_ref, rtol=1e-7, atol=1e-7 * stems_ref.max())


@pytest.mark.parametrize("name", SETS)
def test_stored_shapes_are_normalised(name):
    props = shape_properties(load_export(name)[2].shape)
    assert props["area"] == pytest.approx(1, abs=1e-3)
    assert props["xmax"] == pytest.approx(0, abs=1e-3)
    assert props["fwhm"] == pytest.approx(1, abs=1e-3)


def test_spline_shape_zero_outer_pieces():
    s = spline_shape([-2, -1, 0, 1, 2], [0, 0.3, 1, 0.3, 0])
    assert np.all(s.coefs[0] == 0) and np.all(s.coefs[-1] == 0)
    assert s(np.array([0.0]))[0] == pytest.approx(1.0)
    k = shape_from_knots([1.0, -1.0, 0.0], [5.0, 7.0, 1.0])  # unsorted, ends pinned to 0
    np.testing.assert_allclose(k(np.array([-1.0, 0.0, 1.0])), [0, 1, 0], atol=1e-12)


def test_core_recovers_known_shape():
    """Synthetic spectrum made with a known shape: the core reproduces it."""
    shape = PeakShape.gaussian()
    cal = Calibration(resolution_param=2000.0, shape=shape)
    mols = [Molecule("A", [[100.0, 0.7], [101.0, 0.3]]), Molecule("B", [[100.5, 1.0]])]
    for m, a in zip(mols, (50.0, 20.0)):
        m.area = a
        m.com = float(np.average(m.peakdata[:, 0], weights=m.peakdata[:, 1]))
    x = np.arange(99.0, 102.5, 0.001)
    peakdata = np.column_stack([x, multispec(mols, 2000.0, 0.0, x, shape)])
    core = convolution_core_for(peakdata, mols, cal)
    # The normalised core has the FWHM of the shape that made the spectrum (1).
    above = core.masses[core.K >= core.K.max() / 2]
    assert above[-1] - above[0] == pytest.approx(1.0, rel=0.05)


def test_fit_knots_improves_match_and_finalize():
    peakdata, mols, cal, _ = load_export("data_02")
    core = convolution_core_for(peakdata, mols, cal)
    x, y = core.knots_x, core.knots_y
    fixed = np.zeros(x.size, bool)
    fixed[[0, -1]] = True
    sel = slice(np.argmin(abs(core.masses - x[0])), np.argmin(abs(core.masses - x[-1])) + 1)
    before = _msd(np.zeros(2 * (~fixed).sum()), core.masses[sel], core.K[sel], x, y, fixed)
    x2, y2 = fit_knots(core, x, y)
    after = _msd(np.zeros(2 * (~fixed).sum()), core.masses[sel], core.K[sel], x2, y2, fixed)
    assert after < 0.5 * before
    assert np.all(np.abs(x2 - x) <= 0.5 + 1e-9)

    new = finalize_shape(shape_from_knots(x2, y2), cal, core)
    props = shape_properties(new.shape)
    assert props["area"] == pytest.approx(1, abs=1e-6)
    assert props["xmax"] == pytest.approx(0, abs=1e-3)
    assert props["fwhm"] == pytest.approx(1, abs=1e-3)
    # Resolution scaled by the same factor for every point
    ratio = new.resolutionlist / cal.resolutionlist
    assert np.allclose(ratio, ratio[0])
    # Finalising twice changes nothing
    again = finalize_shape(new.shape, new)
    np.testing.assert_allclose(again.shape.coefs, new.shape.coefs, atol=1e-10)
    np.testing.assert_allclose(again.resolutionlist, new.resolutionlist, rtol=1e-9)


def test_finalize_rescales_flat_resolution():
    wide = adapt_peak_shape(PeakShape.gaussian(), 2.0, 1.0)  # FWHM 2
    cal = Calibration(resolution_method="Flat", resolution_param=3000.0, shape=wide)
    new = finalize_shape(wide, cal)
    assert new.resolution_param == pytest.approx(1500.0, rel=1e-3)
