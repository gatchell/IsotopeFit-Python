"""Background correction and settings dialogs."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QLineEdit,
                               QPushButton, QSpinBox, QWidget)

from .._matlab import interp1_pchip
from ..background import correct_background, find_background
from ..fitting import FITTING_METHODS
from ..model import Project
from ..settings import Settings
from .common import describe_buttons, error_box, find, load_ui, make_plot


def _mass_text(value: float, word: str) -> str:
    return word if not np.isfinite(value) else f"{value:g}"


def _parse_mass(text: str, default: float) -> float:
    text = text.strip().lower()
    if text in ("", "start", "end", "inf", "-inf"):
        return default
    return float(text)


class BackgroundDialog:
    """Port of ``bg_correction.m``: choose sections and percentage, preview, apply."""

    def __init__(self, parent: QWidget, project: Project):
        self.project = project
        self.d: QDialog = load_ui("background_dialog.ui", parent)
        self.ndiv = find(self.d, QSpinBox, "ndivSpin")
        self.percent = find(self.d, QDoubleSpinBox, "percentSpin")
        self.start = find(self.d, QLineEdit, "startEdit")
        self.end = find(self.d, QLineEdit, "endEdit")
        self.plot = make_plot(find(self.d, QWidget, "plotHost"), bottom="Mass (u)", left="Signal")
        bg = project.bgcorrection
        self.ndiv.setValue(int(bg.ndiv))
        self.percent.setValue(float(bg.percent))
        self.start.setText(_mass_text(bg.startmass, "start"))
        self.end.setText(_mass_text(bg.endmass, "end"))
        raw = project.raw_peakdata
        self.raw_curve = self.plot.plot(raw[:, 0], raw[:, 1], pen=pg.mkPen((120, 120, 120)))
        self.raw_curve.setDownsampling(auto=True, method="peak")
        self.raw_curve.setClipToView(True)
        self.bg_curve = self.plot.plot(pen=pg.mkPen("r", width=2))
        self.bg_points = self.plot.plot(pen=None, symbol="o", symbolBrush="r", symbolSize=6)
        find(self.d, QPushButton, "showButton").clicked.connect(self.show_background)
        buttons = describe_buttons(
            self.d, "Subtract background",
            "Subtract the red background curve from the spectrum and close this window.",
            "Close without changes", "Keep the spectrum as it is.",
            hint="Subtracts the red curve from the spectrum in the main window, replacing any "
                 "earlier background correction. Press Show to update the curve after "
                 "changing the settings. If the spectrum is fitted, fit it again afterwards.")
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.d.reject)
        self.show_background()

    def values(self):
        return (self.ndiv.value(), self.percent.value(),
                _parse_mass(self.start.text(), -np.inf), _parse_mass(self.end.text(), np.inf))

    def show_background(self) -> None:
        try:
            ndiv, percent, startmass, endmass = self.values()
            raw = self.project.raw_peakdata
            bgm, bgy, s, e = find_background(raw[:, 0], raw[:, 1], ndiv, percent, startmass, endmass)
        except (ValueError, IndexError) as exc:
            error_box(self.d, "Background correction", str(exc))
            return
        m = raw[s:e, 0]
        self.bg_curve.setData(m, interp1_pchip(bgm, bgy, m))
        self.bg_points.setData(bgm, bgy)

    def _apply(self) -> None:
        try:
            self.values()
        except ValueError as exc:
            error_box(self.d, "Background correction", str(exc))
            return
        self.d.accept()

    def exec(self) -> bool:
        if self.d.exec() != QDialog.DialogCode.Accepted:
            return False
        ndiv, percent, startmass, endmass = self.values()
        correct_background(self.project, ndiv, percent, startmass, endmass)
        return True


class SettingsDialog:
    def __init__(self, parent: QWidget, settings: Settings):
        self.d: QDialog = load_ui("settings_dialog.ui", parent)
        self.searchrange = find(self.d, QDoubleSpinBox, "searchrangeSpin")
        self.deltares = find(self.d, QDoubleSpinBox, "deltaresolutionSpin")
        self.deltamass = find(self.d, QDoubleSpinBox, "deltamassSpin")
        self.minpeakwidth = find(self.d, QDoubleSpinBox, "minpeakwidthSpin")
        self.main_method = find(self.d, QComboBox, "mainMethodCombo")
        self.cal_method = find(self.d, QComboBox, "calMethodCombo")
        for combo in (self.main_method, self.cal_method):
            combo.addItems(FITTING_METHODS)
        self.searchrange.setValue(settings.searchrange)
        self.deltares.setValue(settings.deltaresolution)
        self.deltamass.setValue(settings.deltamass)
        self.minpeakwidth.setValue(settings.minpeakwidth)
        self.main_method.setCurrentText(settings.fittingmethod_main)
        self.cal_method.setCurrentText(settings.fittingmethod_cal)
        buttons = find(self.d, QDialogButtonBox, "buttonBox")
        buttons.accepted.connect(self.d.accept)
        buttons.rejected.connect(self.d.reject)

    def exec(self) -> Settings | None:
        if self.d.exec() != QDialog.DialogCode.Accepted:
            return None
        return Settings(minpeakwidth=self.minpeakwidth.value(),
                        deltaresolution=self.deltares.value(),
                        deltamass=self.deltamass.value(),
                        searchrange=self.searchrange.value(),
                        fittingmethod_main=self.main_method.currentText(),
                        fittingmethod_cal=self.cal_method.currentText())
