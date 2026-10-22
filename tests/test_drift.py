import os

import numpy as np
import pytest

from isotopefit import Project
from isotopefit.drift import (ArraySource, DriftSettings, analyse, apply_correction, corrected_sum,
                              divisions, offset_against)

NB, N = 4000, 12
T = np.arange(NB)
MASS = ((T + 1 - 100) / 300.0) ** 2  # time-of-flight-like axis
CENTRES = np.arange(300, 3900, 150)


def spectrum(drift, rng=None, scale=None):
    s = np.zeros(NB)
    for c in CENTRES:
        s += 1000 * np.exp(-(T - (c + drift(c))) ** 2 / (2 * 2.0 ** 2))
    if rng is not None:
        s = rng.poisson(s * scale) / scale
    return s


def drifting(rate, rng=None, scale=None):
    """Spectra whose peaks move `rate` bins per spectrum at bin 2000, proportional to the bin."""
    return np.array([spectrum(lambda c, w=w: rate * w * c / 2000, rng, scale) for w in range(N)])


def mean_sigma(s):
    out = []
    for c in CENTRES[2:]:
        seg = slice(c - 40, c + 60)
        x, y = T[seg], np.clip(s[seg] - np.median(s), 0, None)
        m = (x * y).sum() / y.sum()
        out.append(np.sqrt(((x - m) ** 2 * y).sum() / y.sum()))
    return np.mean(out)


def settings(**kw):
    return DriftSettings(startmass=MASS[300], ndiv=6, **kw)


def test_offset_sign_and_sub_bin():
    x = np.arange(201)
    s1 = np.exp(-(x - 100) ** 2 / 20)
    for k in (-3, 0, 2, 5):
        s2 = np.exp(-(x - 100 - k) ** 2 / 20)
        assert offset_against(s1, s2)[0] == pytest.approx(k, abs=0.05)
    s2 = np.exp(-(x - 100.4) ** 2 / 20)
    assert offset_against(s1, s2)[0] == pytest.approx(0.4, abs=0.05)  # sub-bin


def test_divisions_like_matlab():
    lo, hi, centre = divisions(MASS, MASS[300], np.inf, 4)
    # MATLAB: s = 301, e = 4000 (1-based); lo = round(s + (e-s)/4*(i-1)) ...
    s, e = 301, 4000
    i = np.arange(1, 5)
    np.testing.assert_array_equal(lo + 1, np.floor(s + (e - s) / 4 * (i - 1) + 0.5))
    np.testing.assert_array_equal(hi + 1, np.floor(s + (e - s) / 4 * i + 0.5))
    np.testing.assert_array_equal(centre + 1, np.floor((lo + hi + 2) / 2 + 0.5))


def test_correction_removes_drift():
    data = drifting(0.6)
    src = ArraySource(data)
    a = analyse(src, MASS, settings())
    # positions follow the true drift relative to the mean, in every division
    true = 0.6 * (np.arange(N) - (N - 1) / 2)[None, :] * (a.centre[:, None]) / 2000
    np.testing.assert_allclose(a.positions, true, atol=0.3)
    corrected = corrected_sum(src, a)
    assert mean_sigma(data.sum(0)) > 3.0
    assert mean_sigma(corrected) == pytest.approx(2.0, abs=0.05)
    assert corrected.sum() == pytest.approx(data.sum(), rel=1e-3)  # all spectra summed


def test_correction_with_noise_and_slow_drift():
    rng = np.random.default_rng(0)
    for rate in (0.15, 0.6):
        data = drifting(rate, rng, 0.05)
        src = ArraySource(data)
        corrected = corrected_sum(src, analyse(src, MASS, settings()))
        assert mean_sigma(corrected) < 2.1


def test_apply_correction_updates_project():
    data = drifting(0.6)
    p = Project(raw_peakdata=np.column_stack([MASS, data.sum(0)]))
    p.update_peakdata()
    src = ArraySource(data)
    corrected = corrected_sum(src, analyse(src, MASS, settings()))
    apply_correction(p, corrected)
    np.testing.assert_array_equal(p.raw_peakdata[:, 1], corrected)
    assert p.gui_status["drift_corrected"] and not p.gui_status["fitted"]
    with pytest.raises(ValueError):
        apply_correction(p, corrected[:-1])
    with pytest.raises(ValueError):
        analyse(src, MASS[:-1], settings())


def test_h5_file(tmp_path):
    h5py = pytest.importorskip("h5py")
    from isotopefit.drift import drift_correct

    data = drifting(0.6)
    # TOFWERK layout as h5py sees it: writes x buffers x segments x bins
    tof = np.repeat(data[:, None, None, :] / 3, 3, axis=1).astype(np.float32)
    path = tmp_path / "drift.h5"
    with h5py.File(path, "w") as f:
        f["/FullSpectra/TofData"] = tof
        f["/FullSpectra/SumSpectrum"] = tof.sum(axis=(0, 1, 2))
    p = Project(raw_peakdata=np.column_stack([MASS, data.sum(0)]))
    a = drift_correct(p, path, settings())
    assert a.n_spectra == N
    assert mean_sigma(p.raw_peakdata[:, 1]) == pytest.approx(2.0, abs=0.05)
    assert os.path.samefile(p.fileinfo["h5completepath"], path)
    # buffers of the first write: three identical buffers, so no drift at all
    p2 = Project(raw_peakdata=np.column_stack([MASS, data[0]]))
    a2 = drift_correct(p2, path, settings(use_buffers=True))
    assert a2.n_spectra == 3 and np.allclose(a2.positions, 0)
