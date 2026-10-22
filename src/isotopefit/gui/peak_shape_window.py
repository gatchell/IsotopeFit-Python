"""The peak shape editor (port of the GUI part of ``peak_shape_generator.m``).

Modelled on the PyQt5 peak shape program in IsotopeFit_Rewrite: plot of the
measured core with the spline and its knots, a table of knots with a
"Fixed" column, right-click to add or remove knots, and a fit button.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton, QStyleFactory,
                               QTableWidget, QTableWidgetItem, QWidget)

from ..model import Calibration
from ..peakshape import PeakShapeSession, shape_properties
from .common import DiscardGuard, describe_buttons, error_box, find, load_ui, make_plot

PICK_PIXELS = 10  # how close a right-click must be to a knot to remove it


class PeakShapeDialog:
    def __init__(self, parent: QWidget, session: PeakShapeSession):
        self.s = session
        self.d: QDialog = load_ui("peak_shape_dialog.ui", parent)
        self.table = find(self.d, QTableWidget, "knotTable")
        self.info = find(self.d, QLabel, "infoLabel")
        self.plot = make_plot(find(self.d, QWidget, "plotHost"),
                              bottom="Mass (FWHM units)", left="Signal")
        self.plot.setMenuEnabled(False)  # right-click adds or removes knots instead
        core = session.core
        self.plot.plot(core.masses, core.K, pen=pg.mkPen((140, 140, 140)))
        self.shape_curve = self.plot.plot(pen=pg.mkPen((0, 0, 0), width=2))
        self.targets: list = []
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        self.plot.cursors.draggable(self._near_knot)
        span = max(abs(session.x[0]), abs(session.x[-1])) * 1.5
        self.plot.setXRange(-span, span)

        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Fixed", "x", "y"])
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._table_edited)
        find(self.d, QPushButton, "removeButton").clicked.connect(self._remove_selected)
        find(self.d, QPushButton, "fitButton").clicked.connect(self._fit)
        buttons = describe_buttons(
            self.d, "Use this peak shape",
            "Normalise the black curve to unit width and area, use it for every molecule, "
            "rescale the resolution to match, and close this window.",
            "Discard changes", "Close and keep the current peak shape.",
            hint="The black curve is normalised to unit width and area and used for every "
                 "molecule. The resolution is rescaled so the peaks keep the width measured "
                 "here. Fit again afterwards to update the areas.")
        buttons.accepted.connect(self._accept)
        self._initial = self._state()
        self._guard = DiscardGuard(self.d, lambda: self._state() != self._initial, "peak shape")
        buttons.rejected.connect(self._guard.try_reject)
        self.refresh()

    def _state(self):
        """The knots, to tell whether there are edits to lose."""
        return self.s.x.tobytes(), self.s.y.tobytes(), self.s.fixed.tobytes()

    # ------------------------------------------------------------------ display

    def refresh(self) -> None:
        s = self.s
        x = np.linspace(s.core.masses[0], s.core.masses[-1], 4000)
        self.shape_curve.setData(x, s.shape(x))
        for t in self.targets:
            self.plot.removeItem(t)
        self.targets = []
        last = s.x.size - 1
        for i, (kx, ky) in enumerate(zip(s.x, s.y)):
            colour = (150, 150, 150) if (s.fixed[i] or i in (0, last)) else (0, 0, 0)
            t = pg.TargetItem(pos=(kx, ky), size=11, symbol="o",
                              pen=pg.mkPen(colour, width=2), brush=pg.mkBrush(None),
                              movable=True)
            t.sigPositionChangeFinished.connect(lambda item, i=i: self._dragged(i, item))
            self.plot.addItem(t)
            self.targets.append(t)
        self._fill_table()
        try:
            p = shape_properties(s.shape)
            self.info.setText(f"Area {p['area']:.4g}, maximum at {p['xmax']:.3f}, FWHM {p['fwhm']:.3f}\n"
                              "(\"Use this peak shape\" normalises area and FWHM to 1 and centres the maximum)")
        except Exception:  # noqa: BLE001
            self.info.setText("")

    def _fill_table(self) -> None:
        """One row per knot: a "Fixed" checkbox, then x and y (editable)."""
        s = self.s
        self.table.blockSignals(True)
        self.table.setRowCount(s.x.size)
        last = s.x.size - 1
        # macOS draws native controls inside table cells as inactive (greyed out) and
        # item check states not at all, so these checkboxes use Qt's own Fusion style.
        if getattr(self, "_box_style", None) is None:
            self._box_style = QStyleFactory.create("Fusion")  # kept: setStyle doesn't own it
        for i in range(s.x.size):
            box = QCheckBox()
            if self._box_style is not None:
                box.setStyle(self._box_style)
            if i in (0, last):  # the end knots are always fixed
                box.setChecked(True)
                box.setEnabled(False)
                box.setToolTip("The first and last knot are always fixed.")
            else:
                box.setChecked(bool(s.fixed[i]))
                box.setToolTip("Fixed knots keep their position when you press Fit.")
                box.toggled.connect(lambda on, i=i: self._set_fixed(i, on))
            cell = QWidget()
            layout = QHBoxLayout(cell)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(box)
            self.table.setCellWidget(i, 0, cell)
            self.table.setItem(i, 1, QTableWidgetItem(f"{s.x[i]:.5g}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{s.y[i]:.5g}"))
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()

    def fixed_box(self, row: int) -> QCheckBox:
        """The "Fixed" checkbox of a table row."""
        return self.table.cellWidget(row, 0).findChild(QCheckBox)

    def _set_fixed(self, index: int, fixed: bool) -> None:
        self.s.set_fixed(index, fixed)
        colour = (150, 150, 150) if fixed else (0, 0, 0)
        self.targets[index].setPen(pg.mkPen(colour, width=2))

    def _later(self) -> None:
        # Rebuilding the knot items from inside one of their own signals is unsafe; defer it.
        QTimer.singleShot(0, self.refresh)

    # ------------------------------------------------------------------ editing

    def _dragged(self, index: int, item) -> None:
        pos = item.pos()
        self.s.move(index, pos.x(), pos.y())
        self._later()

    def _near_knot(self, scene_pos) -> bool:
        """True if the mouse is on a knot (which can be dragged)."""
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos) or self.s.x.size == 0:
            return False
        p = vb.mapSceneToView(scene_pos)
        px, py = vb.viewPixelSize()
        return self.s.nearest(p.x(), p.y(), px, py)[1] < PICK_PIXELS

    def _clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.RightButton:
            return
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        p = vb.mapSceneToView(event.scenePos())
        px, py = vb.viewPixelSize()
        i, dist = self.s.nearest(p.x(), p.y(), px, py)
        if dist < PICK_PIXELS:
            if not self.s.remove([i]):
                self.info.setText("At least three knots are needed.")
        else:
            self.s.add(p.x(), p.y())
        event.accept()
        self._later()

    def _remove_selected(self) -> None:
        self._commit_edits()
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if rows and not self.s.remove(rows):
            self.info.setText("At least three knots are needed.")
            return
        self.refresh()

    def _table_edited(self, item: QTableWidgetItem) -> None:
        i = item.row()
        if item.column() in (1, 2):
            try:
                x = float(self.table.item(i, 1).text())
                y = float(self.table.item(i, 2).text())
            except ValueError:
                self._later()
                return
            self.s.move(i, x, y)
        self._later()

    def _commit_edits(self) -> None:
        """Finish a value being typed into the knot table before a button acts
        (on macOS, clicking a button does not end the edit by itself)."""
        focus = QApplication.focusWidget()
        if focus is not None and self.table.isAncestorOf(focus):
            self.table.setFocus()  # closes the cell editor, which stores the value

    def _accept(self) -> None:
        self._commit_edits()
        self.d.accept()

    def _fit(self) -> None:
        self._commit_edits()
        self.d.setCursor(Qt.CursorShape.WaitCursor)
        try:
            self.s.fit()
        finally:
            self.d.unsetCursor()
        self.refresh()

    # ------------------------------------------------------------------ result

    def exec(self) -> Calibration | None:
        while self.d.exec() == QDialog.DialogCode.Accepted:
            try:
                return self.s.result()
            except ValueError as exc:
                error_box(self.d, "Peak shape", f"The shape cannot be normalised: {exc}")
        return None
