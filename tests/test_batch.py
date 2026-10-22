import shutil

import numpy as np
from scipy.optimize import nnls

from isotopefit import Calibration, Molecule, load_ifd
from isotopefit.batch import batch_fit, find_ifd_files, fit_global, global_design_matrix
from isotopefit.pattern import adapt_peak_shape

from conftest import refdata


def synthetic():
    cal = Calibration(resolution_method="Flat", resolution_param=2000.0)
    mols = [Molecule(name="A", peakdata=[[100.0, 0.7], [101.0, 0.3]]),
            Molecule(name="B", peakdata=[[101.02, 1.0]]),  # overlaps A's second peak
            Molecule(name="C", peakdata=[[150.0, 1.0]])]  # on its own
    x = np.arange(95, 155, 0.002)
    areas = np.array([1000.0, 400.0, 250.0])
    y = np.zeros_like(x)
    for mol, a in zip(mols, areas):
        for m, ab in mol.peakdata:
            pk = adapt_peak_shape(cal.shape, m / 2000.0, a * ab)
            inside = (x > pk.breaks[0] + m) & (x < pk.breaks[-1] + m)
            y[inside] += pk.to_ppoly()(x[inside] - m)
    return np.column_stack([x, y]), mols, cal, areas


def test_global_fit_recovers_exact_areas():
    pd, mols, cal, areas = synthetic()
    r = fit_global(pd, mols, cal)
    np.testing.assert_allclose(r.areas, areas, rtol=1e-9)
    assert r.blocks == 2  # A+B coupled, C alone
    np.testing.assert_allclose(r.errors, 0, atol=1e-6)  # no residuals, no error
    # only points within ±½ FWHM of a peak are fitted
    x = pd[:, 0]
    assert r.fitmask[np.argmin(abs(x - 150))] and not r.fitmask[np.argmin(abs(x - 125))]
    fwhm = 150 / 2000
    near = np.abs(x - 150) < 0.49 * fwhm
    far = (np.abs(x - 150) > 0.51 * fwhm) & (np.abs(x - 150) < 5)
    assert r.fitmask[near].all() and not r.fitmask[far].any()


def test_global_fit_errors_scale_with_noise():
    pd, mols, cal, areas = synthetic()
    rng = np.random.default_rng(1)
    pd[:, 1] += rng.normal(0, 5, pd.shape[0])
    r = fit_global(pd, mols, cal)
    assert np.all(np.abs(r.areas - areas) < 5 * r.errors)
    assert np.all(r.errors > 0)


def test_blocks_equal_one_big_nnls_on_testfile():
    p = load_ifd(refdata("testfile.ifd"))
    r = fit_global(p.peakdata, p.molecules, p.calibration)
    M, mask = global_design_matrix(p.peakdata[:, 0], p.molecules, p.calibration)
    A = M.tocsr()[mask].toarray()
    x, _ = nnls(A, p.peakdata[mask, 1], maxiter=50 * A.shape[1])
    np.testing.assert_allclose(r.areas, x, rtol=1e-9, atol=1e-6)
    assert r.blocks > 100 and np.isfinite(r.errors).all()
    # different method from the molecule-by-molecule fit stored in the file, but close
    stored = np.array([m.area for m in p.molecules])
    assert abs(r.areas.sum() / stored.sum() - 1) < 0.02


def test_batch_fit_folder(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    shutil.copy(refdata("testfile.ifd"), src / "a.ifd")
    (src / "notes.txt").write_text("not an ifd")
    assert [f.name for f in find_ifd_files(src)] == ["a.ifd"]
    log = []
    written = batch_fit(src, output_folder=tmp_path / "out", log=log.append)
    assert [w.name for w in written] == ["a.ifd"] and log[0] == "Found 1 file(s)"
    fitted = load_ifd(tmp_path / "out" / "a.ifd")
    p = load_ifd(src / "a.ifd")  # original left alone
    r = fit_global(p.peakdata, p.molecules, p.calibration)
    np.testing.assert_allclose([m.area for m in fitted.molecules], r.areas, rtol=1e-12)
    np.testing.assert_allclose([m.areaerror for m in fitted.molecules], r.errors, rtol=1e-12)
    assert fitted.gui_status["fitted"]
