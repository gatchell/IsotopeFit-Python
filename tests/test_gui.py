"""Smoke test of the GUI without a screen. Skipped unless PySide6 and pyqtgraph are installed."""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from conftest import refdata  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_main_window_opens_file_and_shows_series(app):
    from isotopefit.gui.main_window import MainWindow

    win = MainWindow()
    win.open_path(str(refdata("testfile.ifd")))
    assert win.project is not None and len(win.project.molecules) == 384
    assert not win.w.menuBar().isNativeMenuBar()  # menus inside the window (macOS)
    from PySide6.QtWidgets import QMenu
    menus = [a.menu() for a in win.w.menuBar().actions()]
    assert [m.title().replace("&", "") for m in menus] == ["File", "Molecules", "Calibration", "Data", "View"]
    assert all(isinstance(m, QMenu) and len(m.actions()) > 0 for m in menus)
    assert win.model.rowCount() == 384
    win.filter_edit.setText("TMAl")
    assert 0 < win.proxy.rowCount() < 384
    win.filter_edit.clear()
    # Zoom in by hand somewhere else first: selecting must still fit the y axis.
    win.plot.getViewBox().setRange(xRange=(300, 301), yRange=(0, 1), padding=0)
    win.select_molecule(5)
    assert win.selected_indices() == [5]
    (x0, x1), (y0, y1) = win.plot.getViewBox().viewRange()
    pd = win.project.peakdata
    visible = pd[(pd[:, 0] >= x0) & (pd[:, 0] <= x1), 1]
    assert visible.size and y0 <= visible.min() and y1 >= visible.max()
    assert win.com_label.text() != "–"
    # the species list is filled from the molecule names, most common first,
    # and the series of the first one are shown straight away
    species = [win.species_combo.itemText(i) for i in range(win.species_combo.count())]
    assert species[:2] == ["TMAl", "He"] and win.series and win._series_items
    win.species_combo.setCurrentText("He")
    win.update_series()
    assert any(s.name == "[He]n" for s in win.series)
    # several selected series are plotted together, each with a legend entry
    win.series_list.selectAll()
    lines = [i for i in win._series_items if isinstance(i, type(win.data_curve))]
    assert len(lines) == len(win.series) == len(win.series_plot.plotItem.legend.items)
    win.series_list.clearSelection()
    win.series_list.setCurrentRow(0)
    # Log scale used to crash (segfault) on zero and negative values: draw it for real.
    win.w.show()
    win.set_log_scale(True)
    win.plot.repaint()
    app.processEvents()
    win.overview()
    win.plot.repaint()
    app.processEvents()
    x, y = win.data_curve.getData()
    assert np.all(np.isfinite(y))
    win.set_log_scale(False)
    app.processEvents()
    win.project.gui_status["changed"] = False  # don't ask about saving
    win.w.close()


def test_dialogs_build(app):
    from isotopefit import load_ifd
    from isotopefit.gui.dialogs import BackgroundDialog, SettingsDialog
    from isotopefit.settings import Settings

    project = load_ifd(refdata("testfile.ifd"))
    dlg = BackgroundDialog(None, project)
    assert dlg.values()[:2] == (50, 40.0)
    s = SettingsDialog(None, Settings())
    assert s.main_method.currentText() == "linear_system"


def test_calibration_dialog(app):
    from isotopefit import Settings, load_ifd
    from isotopefit.gui.calibration_window import CalibrationDialog

    project = load_ifd(refdata("testfile.ifd"))
    dlg = CalibrationDialog(None, project, Settings())
    assert dlg.range_list.count() == 86
    dlg.d.show()
    dlg.range_list.setCurrentRow(5)
    assert dlg.current() == 5 and dlg.com_label.text() != "–"
    dlg._step("massoffset", 1)
    dlg.fit_this()
    # with "Fit resolution" off, Fit this leaves the resolution exactly as it was
    dlg._step("resolution", 1)
    res = dlg.s.ranges[5].resolution
    dlg.fit_res.setChecked(False)
    dlg.fit_this()
    assert dlg.s.ranges[5].resolution == res
    assert "resolution kept" in dlg.fit_result.text()
    # a spin box losing focus without an edit must not round the value into the range
    dlg._params_edited()
    assert dlg.s.ranges[5].resolution == res
    # a resolution typed in but not confirmed (no Return, focus still in the box, as
    # after clicking a button on macOS) is used by the next button
    dlg.res_spin.lineEdit().setText("3000.0")
    dlg.fit_this()
    assert dlg.s.ranges[5].resolution == 3000.0
    dlg.fit_res.setChecked(True)
    dlg.mo_method.setCurrentText("PCHIP")
    assert dlg.s.massoffset_method == "PCHIP"
    dlg.all_filter.setText("[He]3")
    assert any(not dlg.all_list.item(i).isHidden() for i in range(dlg.all_list.count()))
    app.processEvents()
    cal, problems = dlg.s.result()
    assert cal.comlist.size == 86 and cal.massoffset_method == "PCHIP"
    dlg.d.reject()  # close() would ask whether to discard the edits


def test_peak_shape_dialog(app):
    from isotopefit.gui.peak_shape_window import PeakShapeDialog
    from isotopefit.peakshape import PeakShapeSession
    from test_peakshape import load_export

    peakdata, mols, cal, _ = load_export("data_02")
    dlg = PeakShapeDialog(None, PeakShapeSession(peakdata, mols, cal))
    dlg.d.show()
    n = dlg.table.rowCount()
    assert n == len(dlg.targets) == dlg.s.x.size
    # every row has a "Fixed" checkbox; the end knots are fixed and locked
    first, last, mid = dlg.fixed_box(0), dlg.fixed_box(n - 1), dlg.fixed_box(n // 2)
    assert first.isChecked() and not first.isEnabled()
    assert last.isChecked() and not last.isEnabled()
    assert mid.isEnabled() and not mid.isChecked()
    mid.setChecked(True)
    assert dlg.s.fixed[n // 2]
    mid.setChecked(False)
    assert not dlg.s.fixed[n // 2]
    dlg.s.add(0.3, 0.4)
    dlg.refresh()
    assert dlg.table.rowCount() == n + 1
    dlg.table.selectRow(dlg.table.rowCount() // 2)
    dlg._remove_selected()
    assert dlg.table.rowCount() == n
    dlg._fit()
    app.processEvents()
    assert dlg.s.result().shape is not None
    dlg.d.reject()  # close() would ask whether to discard the edits


def test_cluster_dialog(app):
    from isotopefit.clusters import ClusterNode
    from isotopefit.gui.cluster_window import ClusterDialog

    dlg = ClusterDialog(None, has_project=True)
    dlg.d.show()
    dlg.formula.setText("He")
    dlg.n.setText("1:5")
    dlg.add(child=False)
    dlg.formula.setText("H2O")
    dlg.name.setText("W")
    dlg.n.setText("[0 1]")
    dlg.add(child=True)  # child of the He node just added and selected
    roots = dlg.nodes()
    assert [r.label for r in roots] == ["He"] and [c.label for c in roots[0].children] == ["W"]
    from isotopefit.clusters import generate
    dlg._set_results(generate(roots))
    assert dlg.result_list.count() == 10
    dlg.result_filter.setText("[W]")
    assert sum(not dlg.result_list.item(i).isHidden() for i in range(10)) == 5
    dlg.result_list.setCurrentRow(3)
    assert dlg.use_button.isEnabled()
    dlg.set_nodes([ClusterNode("Ar", n="1:3")])
    assert dlg.tree.topLevelItemCount() == 1
    dlg.d.close()


def test_global_fit_from_menu(app):
    import time

    from PySide6.QtGui import QAction

    from isotopefit.gui import common
    from isotopefit.gui.common import find
    from isotopefit.gui.main_window import MainWindow

    win = MainWindow()
    action = find(win.w, QAction, "actionGlobalFit")
    assert not action.isEnabled()
    win.open_path(str(refdata("testfile.ifd")))
    assert action.isEnabled() and find(win.w, QAction, "actionBatchFit").isEnabled()
    before = [m.area for m in win.project.molecules]
    action.trigger()
    t0 = time.time()
    while common._RUNNING and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.01)
    for _ in range(5):
        app.processEvents()
    after = [m.area for m in win.project.molecules]
    assert after != before and win.project.gui_status["fitted"]
    assert "Global fit" in win.w.statusBar().currentMessage()
    win.project.gui_status["changed"] = False
    win.w.close()


def _wait(app):
    import time

    from isotopefit.gui import common

    t0 = time.time()
    while common._RUNNING and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.01)
    for _ in range(5):
        app.processEvents()


def test_drift_dialog(app):
    from PySide6.QtCore import QTimer

    from isotopefit import Project
    from isotopefit.gui.drift_window import DriftDialog, array_factory
    from test_drift import MASS, drifting, mean_sigma

    data = drifting(0.6)
    p = Project(raw_peakdata=np.column_stack([MASS, data.sum(0)]))
    p.update_peakdata()
    dlg = DriftDialog(None, p, source_factory=array_factory(data))
    dlg.start.setText(f"{MASS[300]}")
    dlg.ndiv.setValue(6)
    assert not dlg.ok.isEnabled() and not dlg.correct_button.isEnabled()
    dlg.run_analysis()
    _wait(app)
    assert dlg.analysis is not None and dlg.slider.maximum() == 11 and dlg.division.count() == 6
    dlg.slider.setValue(11)
    assert dlg.spectrum_label.text() == "Spectrum 12 of 12"
    dlg.run_correction()
    _wait(app)
    assert dlg.ok.isEnabled()
    QTimer.singleShot(0, dlg.d.accept)
    assert dlg.exec()
    assert mean_sigma(p.raw_peakdata[:, 1]) < 2.1 and p.gui_status["drift_corrected"]


def test_fit_button_computes_curve_in_background(app):
    from PySide6.QtWidgets import QPushButton

    from isotopefit.gui.main_window import MainWindow

    win = MainWindow()
    win.open_path(str(refdata("testfile.ifd")))
    win.select_molecule(5)
    from isotopefit.gui.main_window import FIT_SCOPES

    win.scope_combo.setCurrentIndex(FIT_SCOPES.index("all"))
    assert win.w.findChild(QPushButton, "fitButton").text() == "Fit all molecules"
    win.scope_combo.setCurrentIndex(FIT_SCOPES.index("view"))
    assert win.w.findChild(QPushButton, "fitButton").text() == "Fit molecules in view"
    win.scope_combo.setCurrentIndex(FIT_SCOPES.index("all"))
    win.fit()
    _wait(app)
    x, y = win.fit_curve.getData()
    ex, ey = MainWindow.full_fit_curve(win.project)
    np.testing.assert_array_equal(win._curve_data[win.fit_curve][1], ey)
    assert x is not None and len(x) > 0
    assert win.selected_indices() == [5]  # selection kept after the fit
    win.project.gui_status["changed"] = False
    win.w.close()


class _AnswerBox:
    """Stand-in for the "Discard changes?" question: records it and gives a set answer."""

    def __init__(self, answer):
        from PySide6.QtWidgets import QMessageBox

        outer = self
        self.asked = 0
        self.answer = answer

        class Box(QMessageBox):
            def exec(self):
                outer.asked += 1
                return 0

            def clickedButton(self):
                want = "Discard changes" if outer.answer == "discard" else "Keep editing"
                return next(b for b in self.buttons() if b.text() == want)

        self.cls = Box


def test_calibration_windows_have_descriptive_buttons(app):
    from PySide6.QtWidgets import QDialogButtonBox, QLabel

    from isotopefit import Project, Settings, load_ifd
    from isotopefit.gui.calibration_window import CalibrationDialog
    from isotopefit.gui.dialogs import BackgroundDialog
    from isotopefit.gui.drift_window import DriftDialog, array_factory
    from isotopefit.gui.peak_shape_window import PeakShapeDialog
    from isotopefit.peakshape import PeakShapeSession
    from test_drift import MASS, drifting
    from test_peakshape import load_export

    project = load_ifd(refdata("testfile.ifd"))
    peakdata, mols, cal, _ = load_export("data_02")
    data = drifting(0.6)
    dialogs = {
        "Subtract background": BackgroundDialog(None, project).d,
        "Apply calibration": CalibrationDialog(None, project, Settings()).d,
        "Use this peak shape": PeakShapeDialog(None, PeakShapeSession(peakdata, mols, cal)).d,
        "Use corrected spectrum": DriftDialog(
            None, Project(raw_peakdata=np.column_stack([MASS, data.sum(0)])),
            source_factory=array_factory(data)).d,
    }
    for text, d in dialogs.items():
        box = d.findChild(QDialogButtonBox, "buttonBox")
        assert box.button(QDialogButtonBox.StandardButton.Ok).text() == text
        assert box.button(QDialogButtonBox.StandardButton.Cancel).text() not in ("Cancel", "")
        hint = d.findChild(QLabel, "applyHint")
        assert hint is not None and len(hint.text()) > 20
        d.reject()


def test_discard_guard_asks_only_when_there_are_edits(app, monkeypatch):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    from isotopefit import Settings, load_ifd
    from isotopefit.gui import common
    from isotopefit.gui.calibration_window import CalibrationDialog

    project = load_ifd(refdata("testfile.ifd"))
    dlg = CalibrationDialog(None, project, Settings())
    dlg.d.show()
    keep = _AnswerBox("keep")
    monkeypatch.setattr(common, "QMessageBox", keep.cls)
    dlg._guard.try_reject()  # nothing changed: closes without asking
    assert keep.asked == 0 and not dlg.d.isVisible()

    dlg = CalibrationDialog(None, project, Settings())
    dlg.d.show()
    dlg.range_list.setCurrentRow(5)
    dlg._step("massoffset", 1)
    dlg._guard.try_reject()  # edited: asks; "Keep editing" leaves it open
    assert keep.asked == 1 and dlg.d.isVisible()
    app.sendEvent(dlg.d, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                   Qt.KeyboardModifier.NoModifier))
    assert keep.asked == 2 and dlg.d.isVisible()  # Escape asks too
    discard = _AnswerBox("discard")
    monkeypatch.setattr(common, "QMessageBox", discard.cls)
    dlg.d.close()  # closing the window asks; "Discard changes" closes it
    app.processEvents()
    assert discard.asked == 1 and not dlg.d.isVisible()


def test_plot_controls(app):
    from PySide6.QtWidgets import QCheckBox, QToolButton

    from isotopefit.gui.main_window import MainWindow

    win = MainWindow()
    assert not win.goto_edit.isEnabled()  # nothing to show yet
    win.open_path(str(refdata("testfile.ifd")))
    assert win.goto_edit.isEnabled()
    win.zoom_to(300, 310)
    win.zoom_by(0.5)
    lo, hi = win._view_x()
    assert abs(lo - 302.5) < 1e-6 and abs(hi - 307.5) < 1e-6
    win.pan_by(0.5)
    lo, hi = win._view_x()
    assert abs(lo - 305) < 1e-6 and abs(hi - 310) < 1e-6
    win.goto_edit.setText("100-120")
    win.go_to()
    assert np.allclose(win._view_x(), (100, 120))
    win.overview()  # the whole spectrum, even though the curves only draw what is in view
    lo, hi = win._view_x()
    x = win.project.peakdata[:, 0]
    assert lo <= x.min() and hi >= x.max()
    win.goto_edit.setText("[He]4")
    win.go_to()
    assert [win.project.molecules[i].name for i in win.selected_indices()] == ["[He]4"]
    # the curve checkboxes hide and show curves
    box = win.w.findChild(QCheckBox, "showFitCheck")
    box.setChecked(False)
    assert not win.fit_curve.isVisible()
    box.setChecked(True)
    assert win.fit_curve.isVisible()
    # log button and View menu stay in step
    win.log_button.setChecked(True)
    assert win.log_scale
    win.log_button.setChecked(False)
    from PySide6.QtCore import Qt

    shape = lambda: win.plot.viewport().cursor().shape()  # noqa: E731
    assert shape() == Qt.CursorShape.OpenHandCursor  # dragging moves the view
    win.plot.cursors.pressed = True
    win.plot.cursors.apply()
    assert shape() == Qt.CursorShape.ClosedHandCursor
    win.plot.cursors.pressed = False
    win.w.findChild(QToolButton, "boxZoomButton").setChecked(True)
    assert win.plot.getViewBox().state["mouseMode"] == win.plot.getViewBox().RectMode
    assert shape() == Qt.CursorShape.BitmapCursor  # the magnifying glass
    win.w.findChild(QToolButton, "boxZoomButton").setChecked(False)
    assert shape() == Qt.CursorShape.OpenHandCursor
    # cluster series in log scale, including zero areas
    win.species_combo.setCurrentText("He")
    win.update_series()
    win.set_series_log(True)
    x, y = win._series_items[0].getData()
    assert np.all(np.isfinite(y))
    win.set_series_log(False)
    win.project.gui_status["changed"] = False
    win.w.close()


def test_series_plot_takes_extra_width(app):
    """The plot, not the species column, gets extra width when the window grows.

    Checked through the layout settings: resizing a window is not reliable on
    headless test machines (it is ignored on GitHub's Windows runners).
    """
    from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QWidget

    from isotopefit.gui.main_window import MainWindow

    win = MainWindow()
    panel = win.w.findChild(QHBoxLayout, "seriesPanelLayout")
    assert (panel.stretch(0), panel.stretch(1)) == (0, 1)  # list column fixed, plot column grows
    host = win.w.findChild(QWidget, "seriesPlotHost")
    assert host.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
    list_width = win.series_list.maximumWidth()
    assert win.species_combo.maximumWidth() <= list_width
    win.w.close()


def test_series_axes_follow_points_not_error_bars(app):
    from isotopefit.gui.main_window import MainWindow

    win = MainWindow()
    win.open_path(str(refdata("testfile.ifd")))
    win.species_combo.setCurrentText("He")
    win.update_series()
    row = win.series_list.currentRow()
    s = win.series[row]
    s.errors = s.errors.copy()
    s.errors[len(s.errors) // 2] = 1000 * s.areas.max()  # one point with a huge error
    win.plot_series()
    (x0, x1), (y0, y1) = win.series_plot.getViewBox().viewRange()
    assert y1 < 2 * s.areas.max()  # the axis is set by the points, not that error bar
    assert y1 >= s.areas.max() and x0 <= s.sizes.min() and x1 >= s.sizes.max()
    win.project.gui_status["changed"] = False
    win.w.close()


def test_open_text_spectrum(app, tmp_path):
    from isotopefit.gui.main_window import MainWindow

    path = tmp_path / "spectrum.txt"
    path.write_text("Mass\tSignal\n" + "".join(f"{m:.3f}\t{100 + m}\n" for m in np.arange(1, 50, 0.01)))
    win = MainWindow()
    win.open_path(str(path))
    assert win.project is not None and win.project.raw_peakdata.shape == (4900, 2)
    assert win.path is None  # a new project: Save asks for an .ifd name
    win.project.gui_status["changed"] = False
    win.w.close()
