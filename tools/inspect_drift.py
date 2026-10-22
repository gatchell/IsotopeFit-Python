"""Show the layout of a TOFWERK .h5 file and the drift the correction finds in it.

    python tools/inspect_drift.py path/to/file.h5 [startmass] [ndiv]
"""

import sys

import h5py
import numpy as np

from isotopefit import load_h5
from isotopefit.drift import DriftSettings, H5Source, analyse, corrected_sum

path = sys.argv[1]
startmass = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
ndiv = int(sys.argv[3]) if len(sys.argv) > 3 else 4

with h5py.File(path, "r") as f:
    print("Datasets:")
    f.visititems(lambda n, o: print(f"  {n}  {o.shape}  {o.dtype}")
                 if isinstance(o, h5py.Dataset) else None)

project = load_h5(path)
mass = project.raw_peakdata[:, 0]
for buffers in (False, True):
    label = "buffers of write 1" if buffers else "writes"
    with H5Source(path, buffers) as src:
        print(f"\n{label}: {src.n_spectra} spectra of {src.n_bins} bins")
        if src.n_spectra < 2:
            continue
        s = DriftSettings(startmass=startmass, ndiv=ndiv, use_buffers=buffers)
        a = analyse(src, mass, s)
        np.set_printoptions(precision=2, suppress=True, linewidth=150)
        print("division centres (u):", a.coms)
        print("position (bins) of first / middle / last spectrum relative to the mean:")
        for w in (0, a.n_spectra // 2, a.n_spectra - 1):
            print(f"  {w + 1:5d}: {a.positions[:, w]}")
        print("largest |position| per division:", np.abs(a.positions).max(axis=1))
        c = corrected_sum(src, a)
        total = src.sum_spectrum()
        print(f"sum of corrected / SumSpectrum: {c.sum() / total.sum():.6f}")
