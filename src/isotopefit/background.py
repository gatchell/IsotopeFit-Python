"""Background estimation (port of ``find_bg.m`` and the computation in ``bg_correction.m``)."""

from __future__ import annotations

import numpy as np

from ._matlab import round_half_away
from .model import BackgroundCorrection, Project


def find_background(mass: np.ndarray, signal: np.ndarray, ndiv: int = 50, percent: float = 70.0,
                    startmass: float = -np.inf, endmass: float = np.inf):
    """Estimate the background as the mean of the lowest points in equal sections.

    The spectrum between ``startmass`` and ``endmass`` is cut into ``ndiv``
    sections of equal numbers of points. In each, the lowest ``percent`` % of
    the signal values are averaged; that average is placed at the section's
    middle point.

    Returns ``(bgm, bgy, start, stop)``: the background points and the
    Python slice ``start:stop`` of the region used. As in MATLAB, a finite
    ``endmass`` cuts at the first mass >= ``endmass``, inclusive.
    """
    m = np.asarray(mass, dtype=float)
    y = np.asarray(signal, dtype=float)
    ix1 = int(np.argmax(m >= startmass)) if startmass > m[0] else 0
    ix2 = int(np.argmax(m >= endmass)) if endmass < m[-1] else m.size - 1  # inclusive
    m, y = m[ix1:ix2 + 1], y[ix1:ix2 + 1]

    ndiv = int(ndiv)
    step = y.size / ndiv
    nminima = int(round_half_away(step * percent / 100))
    bgm = np.empty(ndiv)
    bgy = np.empty(ndiv)
    for i in range(1, ndiv + 1):  # 1-based like MATLAB, so the index arithmetic matches
        section = np.sort(y[int(np.floor((i - 1) * step)):int(np.floor(i * step))])
        bgm[i - 1] = m[int(np.floor((2 * i - 1) * step / 2))]  # MATLAB: m(floor(..)+1)
        bgy[i - 1] = section[:nminima].mean()
    return bgm, bgy, ix1, ix2 + 1


def correct_background(project: Project, ndiv: int | None = None, percent: float | None = None,
                       startmass: float | None = None, endmass: float | None = None) -> BackgroundCorrection:
    """Estimate the background of ``project.raw_peakdata`` and apply it (menu "Background correction").

    Arguments left as ``None`` keep the project's current settings. The
    cropping (``start``/``stop``) follows ``startmass``/``endmass``, and
    ``project.peakdata`` is recomputed, including the mass calibration.
    """
    bg = project.bgcorrection
    bg.ndiv = bg.ndiv if ndiv is None else int(ndiv)
    bg.percent = bg.percent if percent is None else float(percent)
    bg.startmass = bg.startmass if startmass is None else float(startmass)
    bg.endmass = bg.endmass if endmass is None else float(endmass)
    bg.bgm, bg.bgy, project.start, project.stop = find_background(
        project.raw_peakdata[:, 0], project.raw_peakdata[:, 1],
        bg.ndiv, bg.percent, bg.startmass, bg.endmass)
    project.update_peakdata()
    project.gui_status.update(bg_corrected=True, fitted=False, changed=True)
    return bg
