"""The main window: spectrum, molecule list, cluster series and fitting."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QItemSelectionModel, QObject, QStandardPaths, Qt
from PySide6.QtGui import QAction, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
                               QInputDialog, QListWidgetItem,
                               QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
                               QPushButton, QTableView, QToolButton, QWidget)

from .. import io
from ..batch import batch_fit, find_ifd_files, fit_project_global
from ..calibrate import load_calibration
from ..fitting import fit_project
from ..model import Project
from ..molecules import init_molecules
from ..pattern import pattern
from ..series import cluster_series, export_series, species_in
from ..settings import Settings
from ..view import (calculated_spectrum, parse_goto, export_spectrum, fit_calibration, fit_molecule_subset,
                    molecules_in_view)
from .calibration_window import CalibrationDialog
from .cluster_window import ClusterDialog
from .common import error_box, find, load_ui, make_plot, run_in_background
from .drift_window import DriftDialog
from .dialogs import BackgroundDialog, SettingsDialog
from .peak_shape_window import PeakShapeDialog
from .models import MoleculeTableModel, make_proxy

# Colours of the series plotted together (the first one is the usual black).
SERIES_COLOURS = [(20, 20, 20), (31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189),
                  (255, 127, 14), (140, 86, 75), (227, 119, 194)]

# The entries of the fit drop-downs, in order (see main_window.ui).
FIT_SCOPES = ("selected", "all", "view")
FIT_MODES = ("ranges", "molecules")
FIT_SCOPE_TIPS = (
    "The molecules selected in the list on the right (and any overlapping them)",
    "Every molecule in the list",
    "The molecules visible in the spectrum plot (and any overlapping them)",
)
FIT_MODE_TIPS = (
    "Molecules whose peaks overlap are fitted together, one group at a time. "
    "Fast; the usual choice.",
    "Molecules are fitted one at a time from low to high mass, each together with the "
    "molecules it overlaps, after subtracting those already fitted.",
)
FIT_BUTTON_TEXT = ("Fit selected molecules", "Fit all molecules", "Fit molecules in view")

DATA_PEN = pg.mkPen((110, 110, 110), width=1)
FIT_PEN = pg.mkPen((200, 30, 30), width=1.5)
PATTERN_PEN = pg.mkPen((30, 90, 200), width=1.5, style=Qt.PenStyle.DashLine)


def settings_path() -> Path:
    folder = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation))
    return folder / "config.txt"


class _CloseFilter(QObject):
    """Ask before closing with unsaved changes (the loaded QMainWindow can't be subclassed)."""

    def __init__(self, window: MainWindow):
        super().__init__(window.w)
        self.window = window

    def eventFilter(self, obj, event):  # noqa: N802 - Qt API
        if event.type() == QEvent.Type.Close and not self.window.confirm_discard():
            event.ignore()
            return True
        return False


class _CurveStep:
    """Progress of a fit followed by computing the fitted curve.

    The fit reports (done, total); the bar keeps a last 5 % for the curve, so
    it only reaches the end (and closes) when everything is finished.
    """

    def __init__(self, progress):
        self.progress = progress
        self.total = 1

    def __call__(self, done: int, total: int) -> None:
        self.total = max(int(total), 1)
        self.progress(done, self.total + max(1, self.total // 20))

    def finish(self) -> None:
        end = self.total + max(1, self.total // 20)
        self.progress(end, end)


def _line_pixmap(pen) -> QPixmap:
    """A short line in the pen's colour and style, as a legend entry."""
    pix = QPixmap(24, 12)
    pix.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pix)
    qpen = QPen(pen)
    qpen.setWidthF(2.0)
    painter.setPen(qpen)
    painter.drawLine(2, 6, 22, 6)
    painter.end()
    return pix


class MainWindow:
    def __init__(self):
        self.w: QMainWindow = load_ui("main_window.ui")
        # On macOS a Qt menu bar normally moves to the system menu bar at the top
        # of the screen. When the program is started from a terminal (python -m
        # ..., not an app bundle), macOS often doesn't hand that menu bar to it,
        # so the menus don't open. There the menu bar stays inside the window;
        # the app bundle uses the system menu bar (see app.use_native_menus).
        from .app import use_native_menus
        self.w.menuBar().setNativeMenuBar(use_native_menus())
        self.project: Project | None = None
        self.path: str | None = None
        self.series = []
        self.settings = self._load_settings()
        self.log_scale = False
        self._log_floor = 1e-3
        self._curve_data: dict = {}

        w = self.w
        self.table = find(w, QTableView, "moleculeTable")
        self.filter_edit = find(w, QLineEdit, "filterEdit")
        self.species_combo = find(w, QComboBox, "speciesCombo")
        self.species_combo.lineEdit().setPlaceholderText("e.g. He")
        self.series_list = find(w, QListWidget, "seriesList")
        self.series_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.scope_combo = find(w, QComboBox, "fitScopeCombo")
        self.mode_combo = find(w, QComboBox, "fitModeCombo")
        for combo, tips in ((self.scope_combo, FIT_SCOPE_TIPS), (self.mode_combo, FIT_MODE_TIPS)):
            for i, tip in enumerate(tips):
                combo.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)
        self.scope_combo.setCurrentIndex(FIT_SCOPES.index("all"))
        self.scope_combo.currentIndexChanged.connect(self._fit_controls_changed)
        self.mode_combo.currentIndexChanged.connect(self._fit_controls_changed)
        self.com_label = find(w, QLabel, "comLabel")
        self.res_label = find(w, QLabel, "resolutionLabel")
        self.area_label = find(w, QLabel, "areaLabel")

        # Plots
        self.plot = make_plot(find(w, QWidget, "spectrumPlotHost"), bottom="Mass (u)", left="Signal")
        vb = self.plot.getViewBox()
        vb.setAutoVisible(y=True)  # y follows the visible part when zooming in x
        self.data_curve = self.plot.plot(pen=DATA_PEN, name="data")
        self.fit_curve = self.plot.plot(pen=FIT_PEN, name="fit")
        self.pattern_curve = self.plot.plot(pen=PATTERN_PEN, name="selected")
        for c in (self.data_curve, self.fit_curve):
            c.setDownsampling(auto=True, method="peak")
            c.setClipToView(True)
        self.series_plot = make_plot(find(w, QWidget, "seriesPlotHost"), bottom="n", left="Counts")
        self.series_plot.addLegend(offset=(-5, 5))
        self._series_items: list = []  # line, points, error bars of every plotted series
        self.series_log = False
        self._setup_plot_controls()

        # Molecule table
        self.model = MoleculeTableModel(w)
        self.proxy = make_proxy(self.model, w)
        self.table.setModel(self.proxy)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)
        self.filter_edit.textChanged.connect(self.proxy.setFilterFixedString)

        # Signals
        actions = {
            "actionOpen": self.open_file, "actionSave": self.save, "actionSaveAs": self.save_as,
            "actionSettings": self.edit_settings, "actionQuit": w.close,
            "actionLoadMolecules": self.load_molecules, "actionLoadMoleculeFolder": self.load_molecule_folder,
            "actionBackground": self.background, "actionLoadCalibration": self.load_calibration,
            "actionCalibration": self.calibration, "actionPeakShape": self.peak_shape,
            "actionClusterGenerator": self.cluster_generator,
            "actionExportSeries": self.export_series, "actionExportSpectrum": self.export_spectrum,
            "actionExportFitted": self.export_fitted, "actionOverview": self.overview,
            "actionDriftCorrection": self.drift_correction,
            "actionGlobalFit": self.global_fit, "actionBatchFit": self.batch_fit,
        }
        for name, slot in actions.items():
            find(w, QAction, name).triggered.connect(slot)
        find(w, QAction, "actionLogScale").toggled.connect(self.set_log_scale)
        find(w, QPushButton, "fitButton").clicked.connect(self.fit)
        find(w, QPushButton, "removeButton").clicked.connect(self.remove_selected)
        self.species_combo.activated.connect(lambda *_: self.update_series())
        self.species_combo.lineEdit().returnPressed.connect(self.update_series)
        self.series_list.itemSelectionChanged.connect(self.plot_series)
        self._fit_controls_changed()
        self._close_filter = _CloseFilter(self)
        w.installEventFilter(self._close_filter)
        self._update_actions()

    def show(self) -> None:
        self.w.show()

    # ------------------------------------------------------------------ settings

    def _load_settings(self) -> Settings:
        try:
            return Settings.from_file(settings_path())
        except (OSError, ValueError):
            return Settings()

    def edit_settings(self) -> None:
        new = SettingsDialog(self.w, self.settings).exec()
        if new is not None:
            self.settings = new
            try:
                settings_path().parent.mkdir(parents=True, exist_ok=True)
                new.to_file(settings_path())
            except OSError as exc:
                error_box(self.w, "Settings", f"Could not save the settings: {exc}")

    # ------------------------------------------------------------------ state

    def _status(self, text: str) -> None:
        self.w.statusBar().showMessage(text)

    def _update_actions(self) -> None:
        has = self.project is not None
        has_mols = has and bool(self.project.molecules)
        for name in ("actionSave", "actionSaveAs", "actionBackground", "actionLoadMolecules",
                     "actionLoadMoleculeFolder", "actionLoadCalibration", "actionExportSpectrum",
                     "actionOverview", "actionDriftCorrection"):
            find(self.w, QAction, name).setEnabled(has)
        for name in ("actionExportSeries", "actionExportFitted", "actionCalibration",
                     "actionPeakShape", "actionGlobalFit"):
            find(self.w, QAction, name).setEnabled(has_mols)
        for name in ("wholeButton", "zoomInButton", "zoomOutButton", "panLeftButton",
                     "panRightButton", "fitHeightButton", "logButton", "boxZoomButton",
                     "saveImageButton"):
            find(self.w, QToolButton, name).setEnabled(has)
        self.goto_edit.setEnabled(has)
        find(self.w, QToolButton, "seriesExportButton").setEnabled(has_mols)
        find(self.w, QPushButton, "fitButton").setEnabled(has_mols)
        find(self.w, QPushButton, "removeButton").setEnabled(has_mols)
        title = "IsotopeFit"
        if has:
            name = os.path.basename(self.path) if self.path else self.project.fileinfo.get(
                "originalfilename", "untitled")
            star = " *" if self.project.gui_status.get("changed") else ""
            title = f"{name}{star} – IsotopeFit"
        self.w.setWindowTitle(title)

    def confirm_discard(self) -> bool:
        if self.project is None or not self.project.gui_status.get("changed"):
            return True
        answer = QMessageBox.question(
            self.w, "IsotopeFit", "Save changes before closing?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def set_project(self, project: Project, path: str | None) -> None:
        self.project = project
        self.path = path
        if project.peakdata is None:
            project.update_peakdata()
        self.model.set_molecules(project.molecules)
        self.table.resizeColumnsToContents()
        self._show(self.data_curve, project.peakdata[:, 0], project.peakdata[:, 1])
        self._show(self.pattern_curve, [], [])
        self.update_fit_curve()
        self._refresh_species()  # offer this file's species; keep the chosen one if present
        if self.log_scale:
            self.set_log_scale(True)  # new data, new floor
        self.overview()
        self._update_actions()
        self._status(f"{len(project.molecules)} molecules, {project.peakdata.shape[0]} data points")

    def molecules_changed(self, structure: bool = False, curve=None) -> None:
        """Refresh everything that shows molecule data (``curve``: fit curve computed already)."""
        if structure:
            self.model.set_molecules(self.project.molecules)
        else:
            self.model.refresh()
        self.update_fit_curve(curve)
        if structure:
            self._refresh_species()
        elif self.species_combo.currentText().strip():
            self.update_series(keep_selection=True)
        self._selection_changed()
        self._update_actions()

    # ------------------------------------------------------------------ files

    def open_file(self) -> None:
        if not self.confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self.w, "Open", self._start_dir(),
            "IsotopeFit or HDF5 files (*.ifd *.h5);;IsotopeFit data (*.ifd);;HDF5 spectra (*.h5)")
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        try:
            if path.lower().endswith(".h5"):
                project = self._open_h5(path)
                if project is None:
                    return
                self.set_project(project, None)
            else:
                self.set_project(io.load_ifd(path), path)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            error_box(self.w, "Open", f"Could not open {os.path.basename(path)}:\n{exc}")

    def _open_h5(self, path: str) -> Project | None:
        kind, ok = QInputDialog.getItem(self.w, "Open HDF5 file", "Instrument:",
                                        ["CLUSTOF (TOFWERK)", "SURFTOF (IoniTOF)"], 0, False)
        if not ok:
            return None
        if kind.startswith("CLUSTOF"):
            answer = QMessageBox.question(
                self.w, "Mass axis",
                "Build the mass axis from the calibration stored in the file?\n"
                "(Yes is recommended; No uses the mass axis written during the measurement.)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            return io.load_h5(path, "clustof", use_mass_axis=answer == QMessageBox.StandardButton.No)
        return io.load_h5(path, "surftof")

    def _start_dir(self) -> str:
        return os.path.dirname(self.path) if self.path else (
            self.project.fileinfo.get("pathname", "") if self.project else "")

    def save(self) -> bool:
        if self.project is None:
            return False
        if not self.path:
            return self.save_as()
        try:
            io.save_ifd(self.project, self.path)
        except Exception as exc:  # noqa: BLE001
            error_box(self.w, "Save", f"Could not save:\n{exc}")
            return False
        self.project.gui_status["changed"] = False
        self._update_actions()
        self._status(f"Saved {self.path}")
        return True

    def save_as(self) -> bool:
        if self.project is None:
            return False
        suggestion = os.path.join(self._start_dir(),
                                  self.project.fileinfo.get("originalfilename", "untitled") + ".ifd")
        path, _ = QFileDialog.getSaveFileName(self.w, "Save as", suggestion, "IsotopeFit data (*.ifd)")
        if not path:
            return False
        if not path.lower().endswith(".ifd"):
            path += ".ifd"
        self.path = path
        return self.save()

    # ------------------------------------------------------------------ molecules

    def _use_molecules(self, molecules) -> None:
        self.project.molecules = init_molecules(molecules, self.project.peakdata)
        self.project.gui_status.update(molecules_loaded=True, fitted=False, changed=True)
        self.molecules_changed(structure=True)
        self._status(f"{len(self.project.molecules)} molecules in the mass range")

    def load_molecules(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self.w, "Load molecules", self._start_dir(),
                                              "IsotopeFit files (*.ifm *.ifd)")
        if not path:
            return
        try:
            mols = io.load_ifd(path, process=False).molecules if path.lower().endswith(".ifd") \
                else io.load_ifm(path)
        except Exception as exc:  # noqa: BLE001
            error_box(self.w, "Load molecules", str(exc))
            return
        self._use_molecules(mols)

    def load_molecule_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self.w, "Folder with molecule files", self._start_dir())
        if folder:
            try:
                self._use_molecules(io.load_molecule_folder(folder))
            except Exception as exc:  # noqa: BLE001
                error_box(self.w, "Load molecules", str(exc))

    def cluster_generator(self) -> None:
        """Molecules → Cluster generator: make molecule lists, save them or use them here."""
        from ..molecules import add_molecules

        dlg = ClusterDialog(self.w, has_project=self.project is not None,
                            start_dir=self._start_dir() if self.project else "")
        result = dlg.exec()
        if result is None or self.project is None:
            return
        molecules, mode = result
        if mode == "replace":
            self._use_molecules(molecules)
            return
        before = len(self.project.molecules)
        self.project.molecules = add_molecules(self.project.molecules, molecules,
                                               self.project.peakdata)
        self.project.gui_status.update(molecules_loaded=True, changed=True)
        self.molecules_changed(structure=True)
        self._status(f"Added {len(self.project.molecules) - before} molecules "
                     f"(names already present and molecules outside the mass range skipped).")

    def remove_selected(self) -> None:
        idx = self.selected_indices()
        if not idx:
            return
        if QMessageBox.question(self.w, "Remove molecules",
                                f"Remove {len(idx)} molecule(s)?") != QMessageBox.StandardButton.Yes:
            return
        keep = set(range(len(self.project.molecules))) - set(idx)
        self.project.molecules = [self.project.molecules[i] for i in sorted(keep)]
        self.project.gui_status["changed"] = True
        self.molecules_changed(structure=True)

    # ------------------------------------------------------------------ selection and plots

    def selected_indices(self) -> list[int]:
        rows = {self.proxy.mapToSource(i).row() for i in self.table.selectionModel().selectedRows()}
        return sorted(rows)

    def select_molecule(self, index: int) -> None:
        proxy_index = self.proxy.mapFromSource(self.model.index(index, 0))
        if not proxy_index.isValid():  # hidden by the filter
            self.filter_edit.clear()
            proxy_index = self.proxy.mapFromSource(self.model.index(index, 0))
        flags = QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows
        self.table.selectionModel().select(proxy_index, flags)
        self.table.scrollTo(proxy_index)

    def _selection_changed(self, *args) -> None:
        idx = self.selected_indices()
        if self.project is None or len(idx) != 1:
            self._show(self.pattern_curve, [], [])
            for label in (self.com_label, self.res_label, self.area_label):
                label.setText("–")
            return
        m = self.project.molecules[idx[0]]
        cal = self.project.calibration
        res = float(cal.resolution(m.com))
        self.com_label.setText(f"{m.com:.4f}")
        self.res_label.setText(f"{res:.0f}")
        err = f" ± {m.areaerror:.3g}" if np.isfinite(m.areaerror) else ""
        self.area_label.setText(f"{m.area:.4g}{err}")
        sigma = float(fit_calibration(self.project).sigma(m.com))
        lo, hi = m.minmass - 12 * sigma, m.maxmass + 12 * sigma
        pd = self.project.peakdata
        x = pd[(pd[:, 0] >= lo) & (pd[:, 0] <= hi), 0]
        self._show(self.pattern_curve, x, pattern(m, m.area, res, 0.0, x, cal.shape))
        self.zoom_to(lo, hi)

    @staticmethod
    def full_fit_curve(project: Project):
        """The calculated spectrum of all molecules over the whole spectrum."""
        pd = project.peakdata
        return calculated_spectrum(project, pd[:, 0].min(), pd[:, 0].max(),
                                   indices=range(len(project.molecules)))

    def update_fit_curve(self, curve=None) -> None:
        p = self.project
        if p is None or not p.molecules:
            self._show(self.fit_curve, [], [])
            return
        x, y = curve if curve is not None else self.full_fit_curve(p)
        self._show(self.fit_curve, x, y)

    # Log scale: pyqtgraph crashes (segfault in Qt's path drawing) on the -inf
    # and NaN that log10 gives for zero and negative values, which every
    # background-corrected spectrum has. So in log mode the curves get a copy
    # with those values raised to a small positive floor.

    def _show(self, curve, x, y) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self._curve_data[curve] = (x, y)
        if self.log_scale and y.size:
            y = np.where(np.isfinite(y) & (y > self._log_floor), y, self._log_floor)
        curve.setData(x, y)

    def set_log_scale(self, on: bool) -> None:
        if on and self.project is not None:
            y = self.project.peakdata[:, 1]
            positive = y[np.isfinite(y) & (y > 0)]
            self._log_floor = float(positive.min()) / 10 if positive.size else 1e-3
        self.log_scale = on
        self.plot.setLogMode(y=False)  # switch only after the data are safe for log10
        for curve, (x, y) in list(self._curve_data.items()):
            self._show(curve, x, y)
        self.plot.setLogMode(y=on)

    def zoom_to(self, lo: float, hi: float) -> None:
        """Show masses lo..hi with the y axis fitted to what is visible.

        pyqtgraph's automatic y range stops following once the user has zoomed
        with the mouse, so the y range is computed here from the data, fit and
        pattern curves inside lo..hi (in log mode, from their log10 values).
        """
        ys = []
        for x, y in self._curve_data.values():
            if x.size:
                sel = (x >= lo) & (x <= hi)
                ys.append(y[sel])
        y = np.concatenate(ys) if ys else np.empty(0)
        y = y[np.isfinite(y)]
        vb = self.plot.getViewBox()
        if y.size == 0:
            vb.setXRange(lo, hi, padding=0)
            return
        if self.log_scale:
            y = np.log10(np.maximum(y, self._log_floor))
        ymin, ymax = float(y.min()), float(y.max())
        if not self.log_scale:
            ymin = min(ymin, 0.0)  # keep the baseline in view
        pad = 0.05 * (ymax - ymin) if ymax > ymin else max(abs(ymax), 1.0) * 0.05
        vb.setRange(xRange=(lo, hi), yRange=(ymin - pad, ymax + pad), padding=0)

    def overview(self) -> None:
        """Show the whole spectrum.

        Computed from the data: the curves only draw what is in view (to stay
        fast), so pyqtgraph's own auto-range would only see the visible part.
        """
        if self.project is None or self.project.peakdata is None or not self.project.peakdata.size:
            self.plot.getViewBox().autoRange()
            return
        x = self.project.peakdata[:, 0]
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        pad = 0.01 * (hi - lo)
        self.zoom_to(lo - pad, hi + pad)

    # ------------------------------------------------------------------ plot controls

    def _setup_plot_controls(self) -> None:
        w = self.w

        def button(name):
            return find(w, QToolButton, name)

        button("wholeButton").clicked.connect(self.overview)
        button("zoomInButton").clicked.connect(lambda: self.zoom_by(0.5))
        button("zoomOutButton").clicked.connect(lambda: self.zoom_by(2.0))
        button("panLeftButton").clicked.connect(lambda: self.pan_by(-0.5))
        button("panRightButton").clicked.connect(lambda: self.pan_by(0.5))
        button("fitHeightButton").clicked.connect(self.fit_height)
        self.goto_edit = find(w, QLineEdit, "gotoEdit")
        self.goto_edit.returnPressed.connect(self.go_to)
        # The log button and the View menu entry stay in step.
        self.log_button = button("logButton")
        log_action = find(w, QAction, "actionLogScale")
        self.log_button.toggled.connect(log_action.setChecked)
        log_action.toggled.connect(self.log_button.setChecked)
        button("boxZoomButton").toggled.connect(self.set_box_zoom)
        button("saveImageButton").clicked.connect(self.save_image)
        for name, curve, pen in (("showDataCheck", self.data_curve, DATA_PEN),
                                 ("showFitCheck", self.fit_curve, FIT_PEN),
                                 ("showPatternCheck", self.pattern_curve, PATTERN_PEN)):
            box = find(w, QCheckBox, name)
            box.toggled.connect(curve.setVisible)
            # A short line in the curve's colour next to the box makes it the legend.
            # (A checkbox's own icon is drawn over its text on macOS, so a label is used.)
            line = QLabel()
            line.setPixmap(_line_pixmap(pen))
            line.setToolTip(box.toolTip())
            controls = w.findChild(QHBoxLayout, "spectrumControls")
            controls.insertWidget(controls.indexOf(box) + 1, line)
            controls.insertSpacing(controls.indexOf(line) + 1, 8)
        self.cursor_label = find(w, QLabel, "cursorLabel")
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)

        self.series_errors_check = find(w, QCheckBox, "seriesErrorsCheck")
        self.series_errors_check.toggled.connect(self._show_series_errors)
        button("seriesLogButton").toggled.connect(self.set_series_log)
        button("seriesFitButton").clicked.connect(self.fit_series_view)
        button("seriesExportButton").clicked.connect(self.export_series)

    def _view_x(self) -> tuple[float, float]:
        lo, hi = self.plot.getViewBox().viewRange()[0]
        return float(lo), float(hi)

    def zoom_by(self, factor: float) -> None:
        """Show ``factor`` times the current mass range, around its centre."""
        lo, hi = self._view_x()
        c, half = (lo + hi) / 2, (hi - lo) / 2 * factor
        self.zoom_to(c - half, c + half)

    def pan_by(self, fraction: float) -> None:
        """Move the view by ``fraction`` of its width (negative: to lower masses)."""
        lo, hi = self._view_x()
        d = (hi - lo) * fraction
        self.zoom_to(lo + d, hi + d)

    def fit_height(self) -> None:
        self.zoom_to(*self._view_x())

    def go_to(self) -> None:
        if self.project is None:
            return
        target = parse_goto(self.goto_edit.text(), self.project.molecules)
        if target is None:
            self._status(f"Nothing found for {self.goto_edit.text().strip()!r}: type a mass, a "
                         "range such as 300-320, or part of a molecule name.")
        elif target[0] == "range":
            self.zoom_to(target[1], target[2])
        else:
            self.select_molecule(target[1])

    def set_box_zoom(self, on: bool) -> None:
        mode = pg.ViewBox.RectMode if on else pg.ViewBox.PanMode
        self.plot.getViewBox().setMouseMode(mode)
        self.plot.cursors.set_box_zoom(on)

    def save_image(self) -> None:
        import pyqtgraph.exporters as exporters

        path, chosen = QFileDialog.getSaveFileName(
            self.w, "Save plot", self._start_dir(), "PNG image (*.png);;SVG drawing (*.svg)")
        if not path:
            return
        svg = path.lower().endswith(".svg") or (chosen.startswith("SVG")
                                                 and not path.lower().endswith(".png"))
        if not os.path.splitext(path)[1]:
            path += ".svg" if svg else ".png"
        try:
            if svg:
                exporters.SVGExporter(self.plot.plotItem).export(path)
            else:
                ex = exporters.ImageExporter(self.plot.plotItem)
                ex.parameters()["width"] = max(1600, int(self.plot.width() * 2))
                ex.export(path)
        except Exception as exc:  # noqa: BLE001
            error_box(self.w, "Save image", f"Could not save the image: {exc}")
            return
        self._status(f"Plot saved to {os.path.basename(path)}.")

    def _mouse_moved(self, pos) -> None:
        vb = self.plot.getViewBox()
        if self.project is None or not vb.sceneBoundingRect().contains(pos):
            self.cursor_label.setText(" ")
            return
        p = vb.mapSceneToView(pos)
        y = 10 ** p.y() if self.log_scale else p.y()
        self.cursor_label.setText(f"Mass {p.x():.3f} u    Signal {y:.4g}")

    def set_series_log(self, on: bool) -> None:
        self.series_log = on
        self.series_plot.getAxis("left").setLogMode(on)
        self.plot_series()

    # ------------------------------------------------------------------ cluster series

    def _refresh_species(self) -> None:
        """Offer the species found in the molecule names; keep or pick one, show its series."""
        current = self.species_combo.currentText().strip()
        names = species_in(self.project.molecules) if self.project is not None else []
        self.species_combo.blockSignals(True)
        self.species_combo.clear()
        self.species_combo.addItems(names)
        if current:
            self.species_combo.setCurrentText(current)
        elif names:
            self.species_combo.setCurrentIndex(0)  # the most common species
        else:
            self.species_combo.setCurrentText("")
        self.species_combo.blockSignals(False)
        self.update_series(keep_selection=bool(current))

    def update_series(self, keep_selection: bool = False) -> None:
        species = self.species_combo.currentText().strip().strip("[]")
        if self.project is None or not species:
            self.series = []
            self.series_list.clear()
            self.plot_series()
            return
        old = {self.series[r].name for r in self._selected_series_rows()} if keep_selection else set()
        self.series = cluster_series(self.project.molecules, species, self.project.peakdata)
        self.series_list.blockSignals(True)
        self.series_list.clear()
        for s in self.series:
            n = len(s.sizes)
            item = QListWidgetItem(f"{s.name}  ·  {n} size{'s' if n != 1 else ''}")
            item.setToolTip(f"{s.name} for n = {int(s.sizes.min())} … {int(s.sizes.max())}")
            self.series_list.addItem(item)
        for row, s in enumerate(self.series):
            if s.name in old:
                self.series_list.item(row).setSelected(True)
        if not self._selected_series_rows() and self.series:  # select the longest series
            best = max(range(len(self.series)), key=lambda r: len(self.series[r].sizes))
            self.series_list.setCurrentRow(best)
        self.series_list.blockSignals(False)
        if not self.series:
            self._status(f"No molecule name contains [{species}].")
        self.plot_series()

    def _selected_series_rows(self) -> list[int]:
        return sorted(self.series_list.row(i) for i in self.series_list.selectedItems())

    def _series_xy(self, s):
        """x, y and error-bar extents of a series, in log10 units when the axis is logarithmic."""
        err = np.where(np.isfinite(s.errors), s.errors, 0.0)
        x = s.sizes.astype(float)
        if not self.series_log:
            return x, s.areas, err, err
        # Plotted as log10 values with a log axis (pyqtgraph's own log mode does
        # not handle error bars, and zeros would crash it).
        pos = s.areas[s.areas > 0]
        floor = float(pos.min()) / 10 if pos.size else 1.0
        y = np.log10(np.maximum(s.areas, floor))
        top = np.log10(np.maximum(s.areas + err, floor)) - y
        bottom = y - np.log10(np.maximum(s.areas - err, floor))
        return x, y, top, bottom

    def plot_series(self) -> None:
        """Plot every selected series, each in its own colour, with a legend."""
        for item in self._series_items:
            self.series_plot.removeItem(item)
        self._series_items = []
        self.series_plot.cursors.clear_clickable()
        legend = self.series_plot.plotItem.legend
        if legend is not None:
            legend.clear()
        rows = self._selected_series_rows()
        if not rows:
            self.series_plot.setTitle("")
            return
        for k, row in enumerate(rows):
            s = self.series[row]
            colour = SERIES_COLOURS[k % len(SERIES_COLOURS)]
            x, y, top, bottom = self._series_xy(s)
            line = self.series_plot.plot(x, y, name=s.name,
                                         pen=pg.mkPen(colour, style=Qt.PenStyle.DashLine))
            points = pg.ScatterPlotItem(x=x, y=y, data=s.indices, size=8, pen=None,
                                        brush=pg.mkBrush(colour))
            points.sigClicked.connect(self._series_point_clicked)
            errors = pg.ErrorBarItem(x=x, y=y, top=top, bottom=bottom, pen=pg.mkPen(colour))
            errors.setVisible(self.series_errors_check.isChecked())
            self.series_plot.addItem(errors)
            self.series_plot.addItem(points)
            self.series_plot.cursors.clickable(points)
            self._series_items += [line, points, errors]
        self.series_plot.setTitle(self.series[rows[0]].name if len(rows) == 1
                                  else f"{len(rows)} series")
        self.fit_series_view()

    def fit_series_view(self) -> None:
        """Fit the axes to the data points, not to the error bars.

        pyqtgraph's auto-range would include the full error bars, so a single
        point with a large error would squash all the others.
        """
        xs, ys = [], []
        for item in self._series_items:
            if isinstance(item, pg.PlotDataItem):
                x, y = item.getData()
                if x is not None and len(x):
                    xs.append(np.asarray(x, dtype=float))
                    ys.append(np.asarray(y, dtype=float))
        vb = self.series_plot.getViewBox()
        if not xs:
            vb.autoRange()
            return
        x, y = np.concatenate(xs), np.concatenate(ys)
        y = y[np.isfinite(y)]
        xlo, xhi = float(np.nanmin(x)) - 0.5, float(np.nanmax(x)) + 0.5  # n is a whole number
        ylo, yhi = (float(y.min()), float(y.max())) if y.size else (0.0, 1.0)
        if not self.series_log:
            ylo = min(ylo, 0.0)  # keep zero in view for counts
        pad = 0.05 * (yhi - ylo) if yhi > ylo else max(abs(yhi), 1.0) * 0.05
        vb.setRange(xRange=(xlo, xhi), yRange=(ylo - pad, yhi + pad), padding=0)

    def _show_series_errors(self, on: bool) -> None:
        for item in self._series_items:
            if isinstance(item, pg.ErrorBarItem):
                item.setVisible(on)

    def _series_point_clicked(self, item, points, *args) -> None:
        if len(points):
            self.select_molecule(int(points[0].data()))

    # ------------------------------------------------------------------ fitting

    def _fit_controls_changed(self, *args) -> None:
        """The Fit button says what it will fit; the tooltips explain the choices."""
        i, j = self.scope_combo.currentIndex(), self.mode_combo.currentIndex()
        button = find(self.w, QPushButton, "fitButton")
        button.setText(FIT_BUTTON_TEXT[i])
        button.setToolTip(f"{FIT_SCOPE_TIPS[i]}.\n{FIT_MODE_TIPS[j]}")
        self.scope_combo.setToolTip(FIT_SCOPE_TIPS[i])
        self.mode_combo.setToolTip(FIT_MODE_TIPS[j])

    def fit(self) -> None:
        if self.project is None or not self.project.molecules:
            return
        scope = FIT_SCOPES[self.scope_combo.currentIndex()]
        mode = FIT_MODES[self.mode_combo.currentIndex()]
        settings = self.settings
        work = copy.deepcopy(self.project)
        if scope == "all":
            def job(progress):
                step = _CurveStep(progress)
                fit_project(work, settings, mode=mode, progress=step)
                curve = self.full_fit_curve(work)
                step.finish()
                return work.molecules, curve
        else:
            if scope == "selected":
                idx = self.selected_indices()
                if not idx:
                    self._status("Select molecules in the list first, or choose All or "
                                 "In the current view under Molecules.")
                    return
            else:
                lo, hi = self.plot.getViewBox().viewRange()[0]
                idx = molecules_in_view(self.project, lo, hi, settings.searchrange).tolist()
                if not idx:
                    self._status("There are no molecules in the current view.")
                    return

            def job(progress):
                step = _CurveStep(progress)
                fit_molecule_subset(work, idx, settings, mode=mode, progress=step)
                curve = self.full_fit_curve(work)
                step.finish()
                return work.molecules, curve

        def done(out) -> None:
            # The fit curve is computed in the background too; here only show it.
            molecules, curve = out
            selected = self.selected_indices()
            self.project.molecules = molecules
            self.project.gui_status.update(fitted=True, changed=True)
            self.model.set_molecules(molecules)
            self.molecules_changed(curve=curve)
            if selected:
                self.select_molecule(selected[0])
            what = {"selected": "the selected molecules", "all": "all molecules",
                    "view": "the molecules in view"}[scope]
            how = "overlapping groups together" if mode == "ranges" else "one molecule at a time"
            self._status(f"Fitted {what}, {how} ({settings.fittingmethod_main}).")

        run_in_background(self.w, "Fitting", job, done)

    def global_fit(self) -> None:
        """Fit every molecule at once (``batch_fit.m`` on the open project)."""
        if self.project is None or not self.project.molecules:
            return
        work = copy.deepcopy(self.project)
        searchrange = self.settings.searchrange

        def job(progress):
            step = _CurveStep(progress)
            result = fit_project_global(work, searchrange, step)
            curve = self.full_fit_curve(work)
            step.finish()
            return work.molecules, result, curve

        def done(out) -> None:
            molecules, result, curve = out
            selected = self.selected_indices()
            self.project.molecules = molecules
            self.project.gui_status.update(fitted=True, changed=True)
            self.model.set_molecules(molecules)
            self.molecules_changed(curve=curve)
            if selected:
                self.select_molecule(selected[0])
            self._status(f"Global fit: {len(molecules)} molecules on {int(result.fitmask.sum())} "
                         f"points in {result.blocks} independent groups.")

        run_in_background(self.w, "Fitting all molecules at once", job, done)

    def batch_fit(self) -> None:
        """Fit every .ifd file in a folder with the global fit."""
        folder = QFileDialog.getExistingDirectory(self.w, "Folder with .ifd files", self._start_dir())
        if not folder:
            return
        files = find_ifd_files(folder)
        if not files:
            error_box(self.w, "Batch fit", "There are no .ifd files in this folder.")
            return
        box = QMessageBox(QMessageBox.Icon.Question, "Batch fit",
                          f"Fit all molecules of {len(files)} file(s) at once and save them.\n\n"
                          "Overwrite the files (as MATLAB does), or save the fitted files to "
                          "another folder?", parent=self.w)
        overwrite = box.addButton("Overwrite", QMessageBox.ButtonRole.DestructiveRole)
        elsewhere = box.addButton("Save to folder…", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is overwrite:
            out = None
        elif box.clickedButton() is elsewhere:
            out = QFileDialog.getExistingDirectory(self.w, "Save fitted files to", folder)
            if not out:
                return
        else:
            return
        searchrange = self.settings.searchrange
        lines: list[str] = []

        def job(progress):
            return batch_fit(folder, searchrange, out, log=lines.append, progress=progress)

        def done(written) -> None:
            QMessageBox.information(self.w, "Batch fit", f"Fitted and saved {len(written)} file(s).\n\n"
                                    + "\n".join(lines[1:]))
            if self.path and out is None and Path(self.path) in {Path(p) for p in written}:
                self._status("The open file was refitted on disk; open it again to see the result.")

        run_in_background(self.w, "Batch fitting", job, done)

    # ------------------------------------------------------------------ calibration

    def background(self) -> None:
        if self.project is not None and BackgroundDialog(self.w, self.project).exec():
            was_fitted = self.project.gui_status.get("fitted")
            self.project.gui_status.update(bg_corrected=True, fitted=False, changed=True)
            self.set_project(self.project, self.path)
            self._status("Background subtracted." + (" Fit again to update the areas."
                                                     if was_fitted else ""))

    def drift_correction(self) -> None:
        if self.project is None:
            return
        path = self.project.fileinfo.get("h5completepath", "")
        if not path or not os.path.isfile(path):
            why = "was not found" if path else "is not known"
            answer = QMessageBox.question(self.w, "Drift correction",
                                          f"The original .h5 file {why}. Select it?")
            if answer != QMessageBox.StandardButton.Yes:
                return
            path, _ = QFileDialog.getOpenFileName(self.w, "Original .h5 file", self._start_dir(),
                                                  "HDF5 data (*.h5)")
            if not path:
                return
        try:
            dialog = DriftDialog(self.w, self.project, path)
        except Exception as exc:  # noqa: BLE001
            error_box(self.w, "Drift correction", str(exc))
            return
        if dialog.exec():
            self.set_project(self.project, self.path)
            self._update_actions()
            self._status("Drift corrected; background and mass correction redone. Fit again.")

    def load_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self.w, "Load calibration and molecules",
                                              self._start_dir(), "IsotopeFit data (*.ifd)")
        if not path:
            return
        try:
            load_calibration(self.project, path)
        except Exception as exc:  # noqa: BLE001
            error_box(self.w, "Load calibration", str(exc))
            return
        self.set_project(self.project, self.path)
        self.project.gui_status["changed"] = True
        self._update_actions()

    def calibration(self) -> None:
        if self.project is None or not self.project.molecules:
            return
        try:
            dialog = CalibrationDialog(self.w, self.project, self.settings)
        except Exception as exc:  # noqa: BLE001
            error_box(self.w, "Calibration", f"Could not open the calibration: {exc}")
            return
        result = dialog.exec()
        if result is None:
            return
        cal, molecules = result
        self.project.calibration = cal
        self.project.molecules = molecules
        self.project.update_peakdata()
        self.project.gui_status.update(calibrated=True, fitted=False, changed=True)
        self.set_project(self.project, self.path)
        self._status(f"Calibration applied: {len(cal.comlist)} ranges, "
                     f"{cal.massoffset_method} mass offset, {cal.resolution_method} resolution. "
                     "Fit again to update the areas.")

    def peak_shape(self) -> None:
        """Measure the peak shape in the current view (menu "Edit peak shape")."""
        from ..peakshape import PeakShapeSession
        from ..spectrum import mass2ind

        p = self.project
        if p is None or not p.molecules:
            return
        lo, hi = self.plot.getViewBox().viewRange()[0]
        idx = molecules_in_view(p, lo, hi, self.settings.searchrange)
        if idx.size == 0:
            error_box(self.w, "Peak shape", "Zoom in on a few well-fitted peaks first; "
                      "there are no molecules in the current view.")
            return
        if not p.gui_status.get("fitted"):
            answer = QMessageBox.question(
                self.w, "Peak shape",
                "The spectrum is not fitted. Fit the molecules in the current view first?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Cancel:
                return
            if answer == QMessageBox.StandardButton.Yes:
                fit_molecule_subset(p, idx.tolist(), self.settings)
                self.molecules_changed()
        i0, i1 = mass2ind(p.peakdata[:, 0], lo), mass2ind(p.peakdata[:, 0], hi)
        try:
            session = PeakShapeSession(p.peakdata[i0:i1 + 1], [p.molecules[i] for i in idx],
                                       p.calibration)
        except Exception as exc:  # noqa: BLE001
            error_box(self.w, "Peak shape", f"Could not measure the peak shape here: {exc}")
            return
        new = PeakShapeDialog(self.w, session).exec()
        if new is None:
            return
        p.calibration = new
        p.gui_status.update(fitted=False, changed=True)
        self.update_fit_curve()
        self._selection_changed()
        self._update_actions()
        self._status("New peak shape in use. Fit again to update the areas.")

    # ------------------------------------------------------------------ export

    def _save_name(self, title: str, suffix: str) -> str | None:
        base = os.path.splitext(self.path)[0] if self.path else os.path.join(
            self._start_dir(), self.project.fileinfo.get("originalfilename", "untitled"))
        path, _ = QFileDialog.getSaveFileName(self.w, title, base + suffix, "Text files (*.txt);;All files (*)")
        return path or None

    def export_series(self) -> None:
        rows = self._selected_series_rows()
        if not rows:
            self._status("Choose a species and select one or more series first.")
            return
        chosen = [self.series[r] for r in sorted(rows)]
        tag = "".join("_" + s.name for s in chosen).replace("[", "").replace("]", "")
        path = self._save_name("Export cluster series", f"{tag}.txt")
        if path:
            export_series(chosen, path)
            self._status(f"Exported {len(chosen)} series to {path}")

    def export_spectrum(self) -> None:
        path = self._save_name("Export calibrated mass spectrum", "_massspec.txt")
        if path:
            export_spectrum(self.project, path)

    def export_fitted(self) -> None:
        path = self._save_name("Export fitted mass spectrum", "_fitted.txt")
        if path:
            export_spectrum(self.project, path, with_fit=True)
