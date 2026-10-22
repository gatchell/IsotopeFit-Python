"""The calibration window (port of the GUI part of ``calibrate.m``)."""

from __future__ import annotations

import copy

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QSpinBox, QWidget)

from ..calibrate import CALIBRATION_METHODS, CalibrationSession
from ..model import Calibration, Molecule, Project
from ..settings import Settings
from ..spectrum import crop, subtract_background
from .common import (DiscardGuard, describe_buttons, error_box, find, load_ui, make_plot,
                     run_in_background)

INDEX_ROLE = Qt.ItemDataRole.UserRole


def _method_name(method: str) -> str:
    """Match a stored method name (e.g. 'PChip') to the list entry ('PCHIP')."""
    for m in CALIBRATION_METHODS:
        if m.lower() == method.lower():
            return m
    return CALIBRATION_METHODS[0]


class CalibrationDialog:
    def __init__(self, parent: QWidget, project: Project, settings: Settings):
        peakdata = subtract_background(crop(project.raw_peakdata, project.start, project.stop),
                                       project.bgcorrection)
        self.s = CalibrationSession(peakdata, project.molecules, project.calibration, settings)
        self.d: QDialog = load_ui("calibration_dialog.ui", parent)
        d = self.d
        self.all_list = find(d, QListWidget, "allList")
        self.all_filter = find(d, QLineEdit, "allFilterEdit")
        self.range_list = find(d, QListWidget, "rangeList")
        self.range_mols = find(d, QListWidget, "rangeMoleculeList")
        self.com_label = find(d, QLabel, "comLabel")
        self.mo_spin = find(d, QDoubleSpinBox, "massoffsetSpin")
        self.res_spin = find(d, QDoubleSpinBox, "resolutionSpin")
        self.area_spin = find(d, QDoubleSpinBox, "areaSpin")
        self.fit_mo = find(d, QCheckBox, "fitMassoffsetCheck")
        self.fit_res = find(d, QCheckBox, "fitResolutionCheck")
        self.fit_mo.setToolTip("Let \"Fit this\" and \"Fit all\" change the mass offset")
        self.fit_res.setToolTip("Let \"Fit this\" and \"Fit all\" change the resolution")
        self.fit_result = find(d, QLabel, "fitResultLabel")
        self.mo_method = find(d, QComboBox, "massoffsetMethodCombo")
        self.res_method = find(d, QComboBox, "resolutionMethodCombo")
        self.mo_order = find(d, QSpinBox, "massoffsetOrderSpin")
        self.res_order = find(d, QSpinBox, "resolutionOrderSpin")

        # plots
        self.preview = make_plot(find(d, QWidget, "previewPlotHost"), bottom="Mass (u, uncorrected)",
                                 left="Signal")
        self.preview.getViewBox().setAutoVisible(y=True)
        self.p_data = self.preview.plot(pen=pg.mkPen((110, 110, 110)))
        self.p_total = self.preview.plot(pen=pg.mkPen((200, 30, 30), width=1.5))
        self.p_parts: list = []
        self.mo_plot = make_plot(find(d, QWidget, "massoffsetPlotHost"), bottom="Mass (u)",
                                 left="Mass offset (u)")
        self.res_plot = make_plot(find(d, QWidget, "resolutionPlotHost"), bottom="Mass (u)",
                                  left="Resolution")
        self.curves = {}
        for name, plot in (("mo", self.mo_plot), ("res", self.res_plot)):
            line = plot.plot(pen=pg.mkPen((0, 0, 0), style=Qt.PenStyle.DashLine))
            pts = pg.ScatterPlotItem(size=7, brush=pg.mkBrush(40, 40, 40))
            sel = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(220, 40, 40))
            pts.sigClicked.connect(self._point_clicked)
            plot.cursors.clickable(pts)
            plot.addItem(pts)
            plot.addItem(sel)
            self.curves[name] = (line, pts, sel)

        # method controls
        for combo, method in ((self.mo_method, self.s.massoffset_method),
                              (self.res_method, self.s.resolution_method)):
            combo.addItems(CALIBRATION_METHODS)
            combo.setCurrentText(_method_name(method))
        self.mo_order.setValue(int(self.s.massoffset_param) if self.s.massoffset_method.lower() == "polynomial" else 1)
        self.res_order.setValue(int(self.s.resolution_param) if self.s.resolution_method.lower() == "polynomial" else 2)
        self._methods_changed()

        # molecule list
        for i, m in enumerate(self.s.molecules):
            item = QListWidgetItem(m.name)
            item.setData(INDEX_ROLE, i)
            self.all_list.addItem(item)

        # signals
        self.all_filter.textChanged.connect(self._filter_all)
        self.all_list.itemDoubleClicked.connect(lambda *_: self.add())
        find(d, QPushButton, "addButton").clicked.connect(self.add)
        find(d, QPushButton, "removeButton").clicked.connect(self.remove)
        self.range_list.currentRowChanged.connect(lambda *_: self.show_range())
        self.range_mols.currentRowChanged.connect(lambda *_: self._show_area())
        self.mo_spin.editingFinished.connect(self._params_edited)
        self.res_spin.editingFinished.connect(self._params_edited)
        self.area_spin.editingFinished.connect(self._area_edited)
        find(d, QPushButton, "massoffsetUpButton").clicked.connect(lambda: self._step("massoffset", 1))
        find(d, QPushButton, "massoffsetDownButton").clicked.connect(lambda: self._step("massoffset", -1))
        find(d, QPushButton, "resolutionUpButton").clicked.connect(lambda: self._step("resolution", 1))
        find(d, QPushButton, "resolutionDownButton").clicked.connect(lambda: self._step("resolution", -1))
        find(d, QPushButton, "areaUpButton").clicked.connect(lambda: self._area_step(1.05))
        find(d, QPushButton, "areaDownButton").clicked.connect(lambda: self._area_step(1 / 1.05))
        find(d, QPushButton, "fitThisButton").clicked.connect(self.fit_this)
        find(d, QPushButton, "fitAllButton").clicked.connect(self.fit_all)
        find(d, QPushButton, "guessAreaButton").clicked.connect(self.guess_area)
        for w in (self.mo_method, self.res_method):
            w.currentTextChanged.connect(lambda *_: self._methods_changed())
        for w in (self.mo_order, self.res_order):
            w.valueChanged.connect(lambda *_: self._methods_changed())
        buttons = describe_buttons(
            d, "Apply calibration",
            "Correct the mass axis with these mass offsets, use this resolution curve for "
            "fitting, and close this window.",
            "Discard changes", "Close without changing the calibration.",
            hint="Applying corrects the mass axis of the spectrum with the mass-offset curve and "
                 "uses the resolution curve for all fits. Fit again afterwards to update the "
                 "areas.")
        buttons.accepted.connect(self._try_accept)
        self._guard = DiscardGuard(d, lambda: self._state() != self._initial, "calibration")
        buttons.rejected.connect(self._guard.try_reject)

        self.refresh(select=0)
        self._initial = self._state()

    # ------------------------------------------------------------------ display

    def current(self) -> int:
        r = self.range_list.currentRow()
        return r if 0 <= r < len(self.s.ranges) else -1

    def refresh(self, select: int | None = None) -> None:
        """Rebuild the range list, then show the selected range and the curves."""
        keep = self.current() if select is None else select
        self.range_list.blockSignals(True)
        self.range_list.clear()
        for r in self.s.ranges:
            names = ", ".join(m.name for m in r.molecules)
            self.range_list.addItem(f"{r.com:9.3f}   {names}")
        if self.s.ranges:
            self.range_list.setCurrentRow(min(max(keep, 0), len(self.s.ranges) - 1))
        self.range_list.blockSignals(False)
        self.show_range()

    def show_range(self) -> None:
        r = self.current()
        self.range_mols.blockSignals(True)
        self.range_mols.clear()
        for p in self.p_parts:
            self.preview.removeItem(p)
        self.p_parts = []
        if r < 0:
            self.range_mols.blockSignals(False)
            self.p_data.setData([], [])
            self.p_total.setData([], [])
            self.com_label.setText("–")
            self.update_curves()
            return
        rng = self.s.ranges[r]
        for m in rng.molecules:
            self.range_mols.addItem(m.name)
        self.range_mols.setCurrentRow(0)
        self.range_mols.blockSignals(False)
        self.com_label.setText(f"{rng.com:.4f}")
        for spin, value in ((self.mo_spin, rng.massoffset), (self.res_spin, rng.resolution)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._show_area()
        x, data, total, parts = self.s.preview(r)
        self.p_data.setData(x, data)
        self.p_total.setData(x, total)
        if len(parts) > 1:
            for part in parts:
                self.p_parts.append(self.preview.plot(x, part, pen=pg.mkPen((30, 90, 200), style=Qt.PenStyle.DashLine)))
        if x.size:  # fit both axes to this range, also after manual zooming
            ys = np.concatenate([data, total] + list(parts))
            ymin, ymax = min(float(ys.min()), 0.0), float(ys.max())
            pad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
            self.preview.getViewBox().setRange(xRange=(x[0], x[-1]),
                                               yRange=(ymin - pad, ymax + pad), padding=0.02)
        self.update_curves()

    def _show_area(self) -> None:
        r, j = self.current(), self.range_mols.currentRow()
        self.area_spin.blockSignals(True)
        self.area_spin.setValue(self.s.ranges[r].molecules[j].area if r >= 0 and j >= 0 else 0.0)
        self.area_spin.blockSignals(False)

    def update_curves(self) -> None:
        cal = self.s._points_calibration()
        coms = cal.comlist
        r = self.current()
        if coms.size:
            grid = np.linspace(max(0.0, coms.min() * 0.9), coms.max() * 1.1, 400)
            for name, values, curve in (("mo", cal.massoffsetlist, cal.massoffset),
                                        ("res", cal.resolutionlist, cal.resolution)):
                line, pts, sel = self.curves[name]
                try:
                    line.setData(grid, curve(grid))
                except Exception:  # noqa: BLE001 - e.g. too few points for the method
                    line.setData([], [])
                pts.setData(x=coms, y=values, data=np.arange(coms.size))
                sel.setData(x=coms[[r]] if r >= 0 else [], y=values[[r]] if r >= 0 else [])
        else:
            for line, pts, sel in self.curves.values():
                line.setData([], [])
                pts.setData([])
                sel.setData([])

    def _point_clicked(self, item, points, *args) -> None:
        if len(points):
            self.range_list.setCurrentRow(int(points[0].data()))

    def _filter_all(self, text: str) -> None:
        text = text.lower()
        for i in range(self.all_list.count()):
            item = self.all_list.item(i)
            item.setHidden(text not in item.text().lower())

    # ------------------------------------------------------------------ editing

    def add(self) -> None:
        self._commit_edits()
        idx = [it.data(INDEX_ROLE) for it in self.all_list.selectedItems()]
        if not idx:
            return
        self.s.add(idx)
        self.refresh(select=self.s.range_of(idx[0]) or 0)

    def remove(self) -> None:
        self._commit_edits()
        r, j = self.current(), self.range_mols.currentRow()
        if r < 0 or j < 0:
            return
        self.s.remove([self.s.molecule_index(r, j)])
        self.refresh(select=min(r, max(len(self.s.ranges) - 1, 0)))

    def _params_edited(self) -> None:
        """Typed mass offset or resolution. A spin box also reports when it merely loses focus;
        then its value is the range's value rounded to the shown decimals, and nothing changes."""
        r = self.current()
        if r < 0:
            return
        rng = self.s.ranges[r]
        mo = res = None
        if round(rng.massoffset, self.mo_spin.decimals()) != self.mo_spin.value():
            mo = self.mo_spin.value()
        if round(rng.resolution, self.res_spin.decimals()) != self.res_spin.value():
            res = self.res_spin.value()
        if mo is not None or res is not None:
            self.s.set_range_parameters(r, mo, res)
            self.show_range()

    def _area_edited(self) -> None:
        r, j = self.current(), self.range_mols.currentRow()
        if r < 0 or j < 0:
            return
        area = self.s.ranges[r].molecules[j].area
        if round(area, self.area_spin.decimals()) != self.area_spin.value():  # really edited
            self.s.set_area(r, j, self.area_spin.value())
            self.refresh()

    def _commit_edits(self) -> None:
        """Take over values typed into the boxes before a button acts.

        A spin box reports a typed value when Return is pressed or it loses
        focus, but on macOS clicking a button does not take the focus, so a
        typed resolution would otherwise be ignored (and then overwritten).
        """
        for spin in (self.mo_spin, self.res_spin, self.area_spin):
            spin.blockSignals(True)
            spin.interpretText()
            spin.blockSignals(False)
        self._params_edited()
        self._area_edited()

    def _step(self, what: str, direction: int) -> None:
        self._commit_edits()
        r = self.current()
        if r >= 0:
            self.s.step(r, what, direction)
            self.show_range()

    def _area_step(self, factor: float) -> None:
        self._commit_edits()
        r, j = self.current(), self.range_mols.currentRow()
        if r >= 0 and j >= 0:
            self.s.set_area(r, j, self.s.ranges[r].molecules[j].area * factor)
            self.refresh()

    def _methods_changed(self) -> None:
        mo, res = self.mo_method.currentText(), self.res_method.currentText()
        self.mo_order.setEnabled(mo == "Polynomial")
        self.res_order.setEnabled(res == "Polynomial")
        self.s.massoffset_method, self.s.resolution_method = mo, res
        self.s.massoffset_param = float(self.mo_order.value()) if mo == "Polynomial" else 0.0
        self.s.resolution_param = float(self.res_order.value()) if res == "Polynomial" else 0.0
        if hasattr(self, "range_list"):
            self.update_curves()

    # ------------------------------------------------------------------ fitting

    def fit_this(self) -> None:
        self._commit_edits()
        r = self.current()
        if r < 0:
            return
        fit_mo, fit_res = self.fit_mo.isChecked(), self.fit_res.isChecked()
        before = self.s.ranges[r]
        mo0, res0 = before.massoffset, before.resolution
        self.d.setCursor(Qt.CursorShape.WaitCursor)
        try:
            self.s.fit([r], fit_mo, fit_res)
        finally:
            self.d.unsetCursor()
        after = self.s.ranges[r]
        parts = [f"mass offset {mo0:.5f} → {after.massoffset:.5f} u" if fit_mo
                 else f"mass offset kept at {after.massoffset:.5f} u",
                 f"resolution {res0:.1f} → {after.resolution:.1f}" if fit_res
                 else f"resolution kept at {after.resolution:.1f}"]
        self.fit_result.setText(f"Fitted the areas; {parts[0]}; {parts[1]}.")
        self.refresh()

    def fit_all(self) -> None:
        self._commit_edits()
        if not self.s.ranges:
            return
        if QMessageBox.question(self.d, "Fit all", "Fit all calibration ranges?") != QMessageBox.StandardButton.Yes:
            return
        work = copy.deepcopy(self.s)
        fit_mo, fit_res = self.fit_mo.isChecked(), self.fit_res.isChecked()

        def job(progress):
            work.fit(None, fit_mo, fit_res, progress=progress)
            return work

        def done(session) -> None:
            self.s = session
            what = " and ".join(w for w, on in (("mass offsets", fit_mo), ("resolutions", fit_res)) if on)
            kept = " and ".join(w for w, on in (("mass offsets", fit_mo), ("resolutions", fit_res))
                                if not on)
            text = f"Fitted all {len(session.ranges)} ranges: areas"
            text += f" and {what}" if what else ""
            text += f"; {kept} kept." if kept else "."
            self.fit_result.setText(text)
            self.refresh()

        run_in_background(self.d, "Fitting the calibration ranges", job, done)

    def guess_area(self) -> None:
        self._commit_edits()
        r = self.current()
        if r >= 0:
            self.s.guess_area(r)
            self.refresh()

    # ------------------------------------------------------------------ result

    def _state(self):
        """Everything the user can change, to tell whether there are edits to lose."""
        s = self.s
        ranges = tuple((r.com, r.resolution, r.massoffset, tuple(m.area for m in r.molecules))
                       for r in s.ranges)
        return (ranges, s.massoffset_method, s.massoffset_param, s.resolution_method,
                s.resolution_param)

    def _try_accept(self) -> None:
        """donecalib: warn about a calibration that would cause trouble."""
        self._commit_edits()
        if not self.s.ranges:
            error_box(self.d, "Calibration", "Add at least one calibration molecule.")
            return
        _, problems = self.s.result()
        if problems:
            answer = QMessageBox.question(
                self.d, "Calibration",
                "\n\n".join(problems) + "\n\nContinue with this calibration anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.d.accept()

    def exec(self) -> tuple[Calibration, list[Molecule]] | None:
        if self.d.exec() != QDialog.DialogCode.Accepted:
            return None
        cal, _ = self.s.result()
        return cal, self.s.molecules
