"""Shared GUI helpers: loading Designer files, plot widgets, background work."""

from __future__ import annotations

import traceback
from importlib import resources
from typing import Callable

import pyqtgraph as pg
from PySide6.QtCore import QEvent, QFile, QIODevice, QObject, QPointF, QThread, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen, QPixmap
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QMessageBox, QProgressDialog,
                               QVBoxLayout, QWidget)


def load_ui(name: str, parent: QWidget | None = None) -> QWidget:
    """Load ``gui/ui/<name>`` (made with Qt Designer) and return the top-level widget."""
    loader = QUiLoader()
    with resources.as_file(resources.files(__package__) / "ui" / name) as path:
        f = QFile(str(path))
        if not f.open(QIODevice.OpenModeFlag.ReadOnly):
            raise OSError(f"Cannot open {path}")
        widget = loader.load(f, parent)
        f.close()
    if widget is None:
        raise RuntimeError(f"Loading {name} failed: {loader.errorString()}")
    return widget


def describe_buttons(dialog: QWidget, apply: str, apply_tip: str, discard: str,
                     discard_tip: str, hint: str = "") -> QDialogButtonBox:
    """Replace OK/Cancel with buttons that say what they do, and fill the hint label.

    The dialog's ``buttonBox`` keeps its OK/Cancel roles (so Return and Escape
    still work); only the texts change. ``applyHint`` is an optional label
    next to the buttons describing what applying will change.
    """
    box = find(dialog, QDialogButtonBox, "buttonBox")
    ok = box.button(QDialogButtonBox.StandardButton.Ok)
    cancel = box.button(QDialogButtonBox.StandardButton.Cancel)
    ok.setText(apply)
    ok.setToolTip(apply_tip)
    ok.setDefault(True)
    cancel.setText(discard)
    cancel.setToolTip(discard_tip)
    label = dialog.findChild(QLabel, "applyHint")
    if label is not None:
        label.setText(hint)
        label.setVisible(bool(hint))
    return box


class DiscardGuard(QObject):
    """Ask before a dialog throws away edits: its discard button, Escape and closing the window.

    ``changed()`` tells whether there is anything to lose. Connect the button
    box's ``rejected`` signal to :meth:`try_reject` instead of ``reject``.
    """

    def __init__(self, dialog: QDialog, changed: Callable[[], bool], what: str):
        super().__init__(dialog)
        self.dialog = dialog
        self.changed = changed
        self.what = what
        self._closing = False
        dialog.installEventFilter(self)

    def confirm(self) -> bool:
        """True if there is nothing to lose or the user agrees to discard it."""
        try:
            if not self.changed():
                return True
        except Exception:  # noqa: BLE001 - when in doubt, ask
            pass
        box = QMessageBox(QMessageBox.Icon.Question, self.dialog.windowTitle(),
                          f"Discard the changes to the {self.what}?",
                          parent=self.dialog)
        box.setInformativeText("They have not been applied to the spectrum.")
        discard = box.addButton("Discard changes", QMessageBox.ButtonRole.DestructiveRole)
        keep = box.addButton("Keep editing", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(keep)
        box.exec()
        return box.clickedButton() is discard

    def try_reject(self) -> None:
        if self.confirm():
            self._closing = True
            self.dialog.reject()

    def eventFilter(self, obj, event):  # noqa: N802 - Qt API
        if obj is self.dialog and not self._closing:
            if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                self.try_reject()
                return True
            if event.type() == QEvent.Type.Close and self.dialog.result() == 0:
                if not self.confirm():
                    event.ignore()
                    return True
                self._closing = True
        return False


def find(parent: QObject, cls: type, name: str):
    """Child object by its Designer objectName; fails loudly if the .ui file lacks it."""
    obj = parent.findChild(cls, name)
    if obj is None:
        raise LookupError(f"{cls.__name__} {name!r} not found in the .ui file")
    return obj


def make_plot(host: QWidget, **labels) -> pg.PlotWidget:
    """Put a pyqtgraph plot into a placeholder widget from the .ui file."""
    layout = host.layout()
    if layout is None:
        layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    plot = pg.PlotWidget()
    plot.showGrid(x=True, y=True, alpha=0.2)
    for axis, text in labels.items():
        plot.setLabel(axis, text)
    layout.addWidget(plot)
    plot.cursors = PlotCursor(plot)  # cursors that show what the mouse will do
    return plot


_MAGNIFIER: QCursor | None = None


def magnifier_cursor() -> QCursor:
    """A magnifying glass with a plus sign; the hot spot is the centre of the lens."""
    global _MAGNIFIER
    if _MAGNIFIER is None:
        pix = QPixmap(32, 32)
        pix.fill(QColor(0, 0, 0, 0))
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for colour, extra in ((QColor(255, 255, 255), 2.5), (QColor(0, 0, 0), 0.0)):
            pen = QPen(colour, 2.0 + extra)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawEllipse(QPointF(12, 12), 8, 8)
            p.drawLine(QPointF(8, 12), QPointF(16, 12))
            p.drawLine(QPointF(12, 8), QPointF(12, 16))
            pen.setWidthF(3.5 + extra)
            p.setPen(pen)
            p.drawLine(QPointF(18.5, 18.5), QPointF(27, 27))
        p.end()
        _MAGNIFIER = QCursor(pix, 12, 12)
    return _MAGNIFIER


class PlotCursor(QObject):
    """Mouse cursor of a plot that matches what a click or drag will do.

    * moving the view: an open hand, closed while dragging;
    * box zoom (:meth:`set_box_zoom`): a magnifying glass, a cross while drawing;
    * over a clickable point (:meth:`clickable`): a pointing hand;
    * over something draggable (:meth:`draggable`): a move cursor.
    """

    def __init__(self, plot: pg.PlotWidget):
        super().__init__(plot)
        self.plot = plot
        self.box_zoom = False
        self.pressed = False
        self.hover: Qt.CursorShape | None = None
        self._scatters: list = []
        self._drag_test = None
        plot.viewport().installEventFilter(self)
        plot.scene().sigMouseMoved.connect(self._moved)
        self.apply()

    def set_box_zoom(self, on: bool) -> None:
        self.box_zoom = on
        self.apply()

    def clickable(self, *items) -> None:
        """Show a pointing hand over the points of these scatter or plot-data items."""
        self._scatters += list(items)

    def clear_clickable(self) -> None:
        self._scatters = []

    def draggable(self, test) -> None:
        """Show a move cursor where ``test(scene_pos)`` is true (e.g. near a knot)."""
        self._drag_test = test

    def _over_point(self, pos) -> bool:
        for item in self._scatters:
            scatter = getattr(item, "scatter", item)  # PlotDataItem keeps its points here
            if not scatter.isVisible() or not hasattr(scatter, "pointsAt"):
                continue
            try:
                if len(scatter.pointsAt(scatter.mapFromScene(pos))):
                    return True
            except Exception:  # noqa: BLE001 - e.g. no data yet
                continue
        return False

    def _moved(self, pos) -> None:
        if self.pressed:
            return
        hover = None
        if self._drag_test is not None and self._drag_test(pos):
            hover = Qt.CursorShape.SizeAllCursor
        elif self._scatters and self._over_point(pos):
            hover = Qt.CursorShape.PointingHandCursor
        if hover != self.hover:
            self.hover = hover
            self.apply()

    def apply(self) -> None:
        if self.hover is not None:  # over a point or knot; kept while dragging a knot
            cursor = QCursor(self.hover)
        elif self.box_zoom:
            cursor = QCursor(Qt.CursorShape.CrossCursor) if self.pressed else magnifier_cursor()
        else:
            shape = Qt.CursorShape.ClosedHandCursor if self.pressed else Qt.CursorShape.OpenHandCursor
            cursor = QCursor(shape)
        self.plot.viewport().setCursor(cursor)

    def eventFilter(self, obj, event):  # noqa: N802 - Qt API
        t = event.type()
        if t == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self.pressed = True
            self.apply()
        elif t == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            self.pressed = False
            self.apply()
        return False


def error_box(parent: QWidget | None, title: str, text: str, details: str | None = None) -> None:
    box = QMessageBox(QMessageBox.Icon.Warning, title, text, QMessageBox.StandardButton.Ok, parent)
    if details:
        box.setDetailedText(details)
    box.exec()


_RUNNING: set = set()


class _Worker(QThread):
    progress = Signal(int, int)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable, parent: QObject | None = None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:  # runs in the worker thread
        try:
            result = self._fn(lambda done, total: self.progress.emit(done, total))
        except Exception:  # noqa: BLE001 - reported to the user
            self.failed.emit(traceback.format_exc())
        else:
            self.succeeded.emit(result)


def run_in_background(parent: QWidget, title: str, fn: Callable, on_done: Callable) -> None:
    """Run ``fn(progress)`` in a thread with a progress dialog, then ``on_done(result)``.

    ``fn`` must not touch widgets or data the GUI is showing; give it copies.
    """
    dialog = QProgressDialog(parent)
    dialog.setLabelText(title)
    dialog.setRange(0, 0)  # busy indicator until the first progress report
    dialog.setCancelButton(None)
    dialog.setWindowTitle("IsotopeFit")
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.setMinimumDuration(300)
    worker = _Worker(fn, parent)

    def progress(done: int, total: int) -> None:
        dialog.setMaximum(total)
        dialog.setValue(done)

    def finished_ok(result) -> None:
        dialog.close()
        on_done(result)

    def finished_error(tb: str) -> None:
        dialog.close()
        error_box(parent, "IsotopeFit", f"{title} failed.", tb)

    worker.progress.connect(progress)
    worker.succeeded.connect(finished_ok)
    worker.failed.connect(finished_error)
    _RUNNING.add(worker)  # keep a reference while it runs
    worker.finished.connect(lambda: _RUNNING.discard(worker))
    worker.start()
    dialog.show()
