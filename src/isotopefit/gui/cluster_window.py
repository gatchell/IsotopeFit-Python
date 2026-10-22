"""The cluster generator window (port of ``cluster_generator.m``)."""

from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QPushButton, QSpinBox, QTreeWidget,
                               QTreeWidgetItem, QWidget)

from .. import io
from ..clusters import ClusterNode, families, generate, load_icg, save_icg
from ..isotopes import parse_molecule
from ..model import Molecule
from .common import error_box, find, load_ui, make_plot, run_in_background

NODE_ROLE = Qt.ItemDataRole.UserRole
INDEX_ROLE = Qt.ItemDataRole.UserRole


class ClusterDialog:
    """Edit a cluster definition, generate the molecules, save them or hand them to the project.

    After ``exec()``, ``result`` is ``None`` or ``(molecules, mode)`` with
    mode ``"replace"`` or ``"append"``.
    """

    def __init__(self, parent: QWidget | None, has_project: bool = False, start_dir: str = ""):
        self.d: QDialog = load_ui("cluster_dialog.ui", parent)
        d = self.d
        self.start_dir = start_dir
        self.tree = find(d, QTreeWidget, "tree")
        self.formula = find(d, QLineEdit, "formulaEdit")
        self.name = find(d, QLineEdit, "nameEdit")
        self.n = find(d, QLineEdit, "nEdit")
        self.charge = find(d, QSpinBox, "chargeSpin")
        self.threshold = find(d, QLineEdit, "thresholdEdit")
        self.accuracy = find(d, QLineEdit, "accuracyEdit")
        self.result_label = find(d, QLabel, "resultLabel")
        self.result_list = find(d, QListWidget, "resultList")
        self.result_filter = find(d, QLineEdit, "resultFilterEdit")
        self.plot = make_plot(find(d, QWidget, "patternPlotHost"), bottom="Mass (u)",
                              left="Abundance")
        self.stems = pg.BarGraphItem(x=[], height=[], width=0.05, brush=pg.mkBrush(30, 90, 200))
        self.plot.addItem(self.stems)
        self.molecules: list[Molecule] = []
        self.result: tuple[list[Molecule], str] | None = None

        find(d, QPushButton, "addTopButton").clicked.connect(lambda: self.add(child=False))
        find(d, QPushButton, "addChildButton").clicked.connect(lambda: self.add(child=True))
        find(d, QPushButton, "updateButton").clicked.connect(self.update_selected)
        find(d, QPushButton, "removeButton").clicked.connect(self.remove_selected)
        find(d, QPushButton, "openButton").clicked.connect(self.open_definition)
        find(d, QPushButton, "saveButton").clicked.connect(self.save_definition)
        find(d, QPushButton, "generateButton").clicked.connect(self.generate)
        find(d, QPushButton, "saveIfmButton").clicked.connect(self.save_ifm)
        self.use_button = find(d, QPushButton, "useButton")
        self.append_button = find(d, QPushButton, "appendButton")
        self.use_button.clicked.connect(lambda: self._finish("replace"))
        self.append_button.clicked.connect(lambda: self._finish("append"))
        self.has_project = has_project
        self.tree.currentItemChanged.connect(lambda *_: self._selected())
        self.result_list.currentItemChanged.connect(lambda *_: self._show_pattern())
        self.result_filter.textChanged.connect(self._filter_results)
        self.formula.textChanged.connect(lambda *_: self._formula_preview())
        find(d, QDialogButtonBox, "buttonBox").rejected.connect(d.reject)
        self.tree.header().setStretchLastSection(False)
        self._set_results([])

    # ------------------------------------------------------------------ tree <-> nodes

    def _form_node(self) -> ClusterNode | None:
        formula = self.formula.text().strip()
        if not formula:
            error_box(self.d, "Cluster generator", "Enter a sum formula.")
            return None
        try:
            node = ClusterNode(formula=formula, name=self.name.text().strip(),
                               n=self.n.text().strip() or "1", charge=self.charge.value(),
                               threshold=float(self.threshold.text()),
                               mass_accuracy=float(self.accuracy.text()))
            node.multiplicities()
            parse_molecule(formula, node.mass_accuracy, node.threshold)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            error_box(self.d, "Cluster generator", str(exc))
            return None
        return node

    def _item_for(self, node: ClusterNode, top: bool) -> QTreeWidgetItem:
        item = QTreeWidgetItem()
        self._fill_item(item, node, top)
        return item

    def _fill_item(self, item: QTreeWidgetItem, node: ClusterNode, top: bool) -> None:
        values = [node.label, node.formula, node.n]
        values += [str(node.charge), f"{node.threshold:g}", f"{node.mass_accuracy:g}"] if top else ["", "", ""]
        for col, v in enumerate(values):
            item.setText(col, v)
        item.setData(0, NODE_ROLE, node)

    def nodes(self) -> list[ClusterNode]:
        """The definition shown in the tree, as ClusterNode objects."""
        def build(item: QTreeWidgetItem) -> ClusterNode:
            src: ClusterNode = item.data(0, NODE_ROLE)
            node = ClusterNode(src.formula, src.name, src.n, src.charge, src.threshold,
                               src.mass_accuracy)
            node.children = [build(item.child(i)) for i in range(item.childCount())]
            return node
        return [build(self.tree.topLevelItem(i)) for i in range(self.tree.topLevelItemCount())]

    def set_nodes(self, roots: list[ClusterNode]) -> None:
        self.tree.clear()

        def add(parent, node: ClusterNode, top: bool) -> None:
            item = self._item_for(node, top)
            if parent is None:
                self.tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
            for c in node.children:
                add(item, c, False)

        for r in roots:
            add(None, r, True)
        self.tree.expandAll()
        for col in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(col)

    # ------------------------------------------------------------------ editing

    def add(self, child: bool) -> None:
        node = self._form_node()
        if node is None:
            return
        parent = self.tree.currentItem() if child else None
        if child and parent is None:
            error_box(self.d, "Cluster generator", "Select the species to add to first.")
            return
        item = self._item_for(node, top=parent is None)
        if parent is None:
            self.tree.addTopLevelItem(item)
        else:
            parent.addChild(item)
            parent.setExpanded(True)
        self.tree.setCurrentItem(item)
        for col in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(col)

    def update_selected(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            return
        node = self._form_node()
        if node is not None:
            self._fill_item(item, node, top=item.parent() is None)

    def remove_selected(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            return
        parent = item.parent()
        if parent is None:
            self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(item))
        else:
            parent.removeChild(item)

    def _selected(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            return
        node: ClusterNode = item.data(0, NODE_ROLE)
        top = item.parent() is None
        self.formula.setText(node.formula)
        self.name.setText(node.name if node.name != node.formula else "")
        self.n.setText(node.n)
        # charge, threshold and accuracy belong to the family: edit them on the top node
        top_item = item
        while top_item.parent() is not None:
            top_item = top_item.parent()
        top_node: ClusterNode = top_item.data(0, NODE_ROLE)
        self.charge.setValue(int(top_node.charge))
        self.threshold.setText(f"{top_node.threshold:g}")
        self.accuracy.setText(f"{top_node.mass_accuracy:g}")
        for w in (self.charge, self.threshold, self.accuracy):
            w.setEnabled(top)
        count = sum(f.count() for f in self._safe_families())
        self.result_label.setText(f"The definition gives up to {count} molecules.")

    def _safe_families(self):
        try:
            return families(self.nodes())
        except ValueError:
            return []

    def _formula_preview(self) -> None:
        """Show the pattern of the formula being typed (one molecule)."""
        try:
            d = parse_molecule(self.formula.text().strip(), float(self.accuracy.text()),
                               float(self.threshold.text()))
        except Exception:  # noqa: BLE001 - incomplete input while typing
            return
        self._plot_stems(d, self.formula.text().strip())

    # ------------------------------------------------------------------ generating

    def generate(self) -> None:
        roots = self.nodes()
        if not roots:
            error_box(self.d, "Cluster generator", "Add at least one species.")
            return
        try:
            fams = families(roots)
        except ValueError as exc:
            error_box(self.d, "Cluster generator", str(exc))
            return
        total = sum(f.count() for f in fams)
        run_in_background(self.d, f"Generating up to {total} molecules",
                          lambda progress: generate(roots, progress), self._set_results)

    def _set_results(self, molecules: list[Molecule]) -> None:
        self.molecules = molecules
        self.result_list.clear()
        for i, m in enumerate(molecules):
            item = QListWidgetItem(m.name)
            item.setData(INDEX_ROLE, i)
            self.result_list.addItem(item)
        if molecules:
            self.result_label.setText(f"{len(molecules)} molecules")
            self.result_list.setCurrentRow(0)
        enabled = bool(molecules)
        find(self.d, QPushButton, "saveIfmButton").setEnabled(enabled)
        self.use_button.setEnabled(enabled and self.has_project)
        self.append_button.setEnabled(enabled and self.has_project)
        self._filter_results(self.result_filter.text())

    def _filter_results(self, text: str) -> None:
        text = text.lower()
        for i in range(self.result_list.count()):
            item = self.result_list.item(i)
            item.setHidden(text not in item.text().lower())

    def _show_pattern(self) -> None:
        item = self.result_list.currentItem()
        if item is not None:
            m = self.molecules[item.data(INDEX_ROLE)]
            self._plot_stems(m.peakdata, m.name)

    def _plot_stems(self, peakdata: np.ndarray, title: str) -> None:
        pd = np.asarray(peakdata, dtype=float).reshape(-1, 2)
        span = pd[:, 0].max() - pd[:, 0].min() if pd.shape[0] > 1 else 1.0
        self.stems.setOpts(x=pd[:, 0], height=pd[:, 1], width=max(span / 300, 0.02))
        self.plot.setTitle(title)
        self.plot.getViewBox().autoRange(padding=0.1)

    # ------------------------------------------------------------------ files

    def open_definition(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self.d, "Open cluster definition", self.start_dir,
                                              "Cluster definitions (*.icg)")
        if path:
            try:
                self.set_nodes(load_icg(path))
            except Exception as exc:  # noqa: BLE001
                error_box(self.d, "Cluster generator", f"Could not open {os.path.basename(path)}: {exc}")

    def save_definition(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self.d, "Save cluster definition", self.start_dir,
                                              "Cluster definitions (*.icg)")
        if path:
            if not path.lower().endswith(".icg"):
                path += ".icg"
            save_icg(self.nodes(), path)

    def save_ifm(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self.d, "Save molecules", self.start_dir,
                                              "IsotopeFit molecules (*.ifm)")
        if path:
            if not path.lower().endswith(".ifm"):
                path += ".ifm"
            io.save_ifm(self.molecules, path)
            self.result_label.setText(f"{len(self.molecules)} molecules saved to {os.path.basename(path)}")

    def _finish(self, mode: str) -> None:
        self.result = (self.molecules, mode)
        self.d.accept()

    def exec(self) -> tuple[list[Molecule], str] | None:
        self.d.exec()
        return self.result
