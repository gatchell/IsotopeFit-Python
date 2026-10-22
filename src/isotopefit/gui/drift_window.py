"""The drift correction window (port of ``driftcorrection.m``)."""

from __future__ import annotations

import contextlib
import os
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QLabel, QLineEdit,
                               QPushButton, QSlider, QSpinBox, QWidget)

from ..drift import (ArraySource, DriftAnalysis, DriftSettings, H5Source, analyse, apply_correction,
                     corrected_sum)
from ..model import Project
from .common import describe_buttons, error_box, find, load_ui, make_plot, run_in_background

COLOURS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
           (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]


def _mass(text: str, default: float) -> float:
    text = text.strip().lower()
    return default if text in ("", "start", "end", "inf") else float(text)


class DriftDialog:
    """Analyse the drift in the project's ``.h5`` file and replace the spectrum by the corrected sum.

    ``source_factory(use_buffers)`` returns a context manager giving a
    spectrum source; by default the project's ``.h5`` file is opened.
    ``exec()`` returns True if the corrected spectrum was applied.
    """

    def __init__(self, parent: QWidget | None, project: Project, h5path: str | None = None,
                 source_factory: Callable | None = None):
        self.project = project
        self.h5path = h5path
        self.source_factory = source_factory or self._h5_factory
        self.d: QDialog = load_ui("drift_dialog.ui", parent)
        d = self.d
        self.start = find(d, QLineEdit, "startEdit")
        self.end = find(d, QLineEdit, "endEdit")
        self.ndiv = find(d, QSpinBox, "ndivSpin")
        self.search = find(d, QSpinBox, "searchSpin")
        self.buffers = find(d, QCheckBox, "buffersCheck")
        self.division = find(d, QComboBox, "divisionCombo")
        self.slider = find(d, QSlider, "spectrumSlider")
        self.spectrum_label = find(d, QLabel, "spectrumLabel")
        self.correct_button = find(d, QPushButton, "correctButton")
        self.buttons = describe_buttons(
            d, "Use corrected spectrum",
            "Replace the spectrum with the drift-corrected sum and close this window.",
            "Keep original spectrum", "Close without changing the spectrum.")
        self.ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.hint = find(d, QLabel, "applyHint")
        self.hint.setVisible(True)
        find(d, QLabel, "fileLabel").setText(
            f"File: {h5path}" if h5path else "Spectra supplied by the caller")

        self.pos_plot = make_plot(find(d, QWidget, "positionPlotHost"), bottom="Spectrum",
                                  left="Position (bins)")
        self.pos_plot.addLegend(offset=(5, 5))
        self.pos_marker = pg.InfiniteLine(angle=90, pen=pg.mkPen((0, 160, 0), width=1.5))
        self.pos_plot.addItem(self.pos_marker)
        self.mass_plot = make_plot(find(d, QWidget, "massPlotHost"), bottom="Mass (u)",
                                   left="Position (bins)")
        self.mass_points = self.mass_plot.plot(pen=None, symbol="o", symbolBrush="r", symbolSize=8)
        self.mass_poly = self.mass_plot.plot(pen=pg.mkPen("k", style=Qt.PenStyle.DashLine))
        self.mass_round = self.mass_plot.plot(pen=pg.mkPen((0, 160, 0), style=Qt.PenStyle.DashLine))
        self.window_plot = make_plot(find(d, QWidget, "windowPlotHost"), bottom="Time bin",
                                     left="Signal")
        self.window_ref = self.window_plot.plot(pen=pg.mkPen((150, 150, 150)), name="mean")
        self.window_cur = self.window_plot.plot(pen=pg.mkPen((30, 90, 200), width=1.5))
        self.compare_plot = make_plot(find(d, QWidget, "comparePlotHost"), bottom="Mass (u)",
                                      left="Signal")
        self.compare_plot.addLegend(offset=(5, 5))
        self.cmp_before = self.compare_plot.plot(pen=pg.mkPen((150, 150, 150)), name="before")
        self.cmp_after = self.compare_plot.plot(pen=pg.mkPen((200, 30, 30)), name="corrected")
        for c in (self.cmp_before, self.cmp_after):
            c.setDownsampling(auto=True, method="peak")
            c.setClipToView(True)

        self.analysis: DriftAnalysis | None = None
        self.corrected: np.ndarray | None = None
        self._pos_curves: list = []

        find(d, QPushButton, "analyseButton").clicked.connect(self.run_analysis)
        self.correct_button.clicked.connect(self.run_correction)
        self.slider.valueChanged.connect(lambda *_: self.show_spectrum())
        self.division.currentIndexChanged.connect(lambda *_: self.show_spectrum())
        self.buttons.accepted.connect(d.accept)
        self.buttons.rejected.connect(d.reject)
        for w in (self.start, self.end):
            w.textChanged.connect(lambda *_: self._invalidate())
        for w in (self.ndiv, self.search):
            w.valueChanged.connect(lambda *_: self._invalidate())
        self.buffers.toggled.connect(lambda *_: self._invalidate())
        self._invalidate()

    # ------------------------------------------------------------------ helpers

    def _h5_factory(self, use_buffers: bool):
        return H5Source(self.h5path, use_buffers)

    def settings(self) -> DriftSettings:
        return DriftSettings(startmass=_mass(self.start.text(), 0.0),
                             endmass=_mass(self.end.text(), np.inf), ndiv=self.ndiv.value(),
                             use_buffers=self.buffers.isChecked(), shiftsearch=self.search.value())

    def _invalidate(self) -> None:
        self.corrected = None
        self.correct_button.setEnabled(self.analysis is not None)
        self.ok.setEnabled(False)
        self.hint.setText("Press Analyse to find the drift, then Correct shift to compute the "
                          "corrected spectrum." if self.analysis is None else
                          "Press Correct shift to compute the corrected spectrum.")

    # ------------------------------------------------------------------ analysis

    def run_analysis(self) -> None:
        try:
            settings = self.settings()
        except ValueError as exc:
            error_box(self.d, "Drift correction", f"Check the masses: {exc}")
            return
        massaxis = self.project.raw_peakdata[:, 0].copy()
        factory = self.source_factory

        def job(progress):
            with contextlib.closing(factory(settings.use_buffers)) as src:
                return analyse(src, massaxis, settings, progress)

        run_in_background(self.d, "Reading the spectra and finding the shifts", job, self._analysed)

    def _analysed(self, analysis: DriftAnalysis) -> None:
        self.analysis = analysis
        self.corrected = None
        n = analysis.n_spectra
        self.slider.blockSignals(True)
        self.slider.setRange(0, n - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.division.blockSignals(True)
        self.division.clear()
        for i, m in enumerate(analysis.coms):
            self.division.addItem(f"{i + 1}  (around {m:.1f} u)")
        self.division.blockSignals(False)
        for c in self._pos_curves:
            self.pos_plot.removeItem(c)
        self.pos_plot.legend.clear()
        self._pos_curves = []
        x = np.arange(n)
        for i, row in enumerate(analysis.positions):
            pen = pg.mkPen(COLOURS[i % len(COLOURS)], width=1.5)
            c = self.pos_plot.plot(x, row, pen=pen, symbol="o", symbolSize=4,
                                   symbolBrush=COLOURS[i % len(COLOURS)], name=f"div {i + 1}")
            self._pos_curves.append(c)
        self.pos_plot.getViewBox().autoRange()
        self.correct_button.setEnabled(True)
        self.ok.setEnabled(False)
        self.hint.setText("Look at the positions; then press Correct shift to compute the "
                          "corrected spectrum.")
        self.show_spectrum()

    def show_spectrum(self) -> None:
        a = self.analysis
        if a is None:
            return
        w = self.slider.value()
        self.spectrum_label.setText(f"Spectrum {w + 1} of {a.n_spectra}")
        self.pos_marker.setValue(w)
        self.mass_points.setData(a.coms, a.positions[:, w])
        m = np.linspace(min(a.coms.min(), 1.0), a.coms.max(), 300)
        poly = np.polyval(a.mass_polys[w], m)
        self.mass_poly.setData(m, poly)
        self.mass_round.setData(m, np.round(poly))
        d = max(self.division.currentIndex(), 0)
        bins = np.arange(a.lo[d], a.hi[d] + 1)
        self.window_ref.setData(bins, a.windows[d].mean(axis=0))
        self.window_cur.setData(bins, a.windows[d][w])

    # ------------------------------------------------------------------ correction

    def run_correction(self) -> None:
        if self.analysis is None:
            return
        analysis = self.analysis
        factory = self.source_factory

        def job(progress):
            with contextlib.closing(factory(analysis.settings.use_buffers)) as src:
                corrected = corrected_sum(src, analysis, progress)
                before = src.sum_spectrum() if hasattr(src, "sum_spectrum") else None
            return corrected, before

        run_in_background(self.d, "Correcting and summing the spectra", job, self._corrected)

    def _corrected(self, out) -> None:
        corrected, before = out
        self.corrected = corrected
        mass = self.project.raw_peakdata[:, 0]
        if before is None or before.size != mass.size:
            before = self.project.raw_peakdata[:, 1]
        self.cmp_before.setData(mass, before)
        self.cmp_after.setData(mass, corrected)
        self.compare_plot.getViewBox().autoRange()
        self.ok.setEnabled(True)
        self.hint.setText("Compare the corrected spectrum (red) with the original (grey). Using "
                          "it replaces the spectrum and redoes the background and mass "
                          "correction; fit again afterwards.")

    def exec(self) -> bool:
        if self.d.exec() != QDialog.DialogCode.Accepted or self.corrected is None:
            return False
        apply_correction(self.project, self.corrected)
        if self.h5path:
            self.project.fileinfo["h5completepath"] = os.path.abspath(self.h5path)
        return True


def array_factory(data: np.ndarray) -> Callable:
    """A ``source_factory`` for spectra already in memory (tests)."""
    return lambda use_buffers: ArraySource(data)
