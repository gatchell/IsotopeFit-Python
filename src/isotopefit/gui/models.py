"""Qt item models."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt

from ..model import Molecule

SORT_ROLE = Qt.ItemDataRole.UserRole


class MoleculeTableModel(QAbstractTableModel):
    """Molecules with name, centre of mass, area and area error."""

    HEADERS = ("Name", "Mass", "Area", "Error")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._molecules: Sequence[Molecule] = []

    def set_molecules(self, molecules: Sequence[Molecule]) -> None:
        self.beginResetModel()
        self._molecules = molecules
        self.endResetModel()

    def refresh(self) -> None:
        """Values changed (e.g. after a fit) but not the list itself."""
        if self._molecules:
            self.dataChanged.emit(self.index(0, 0),
                                  self.index(len(self._molecules) - 1, len(self.HEADERS) - 1))

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._molecules)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        m = self._molecules[index.row()]
        col = index.column()
        values = (m.name, m.com, m.area, m.areaerror)
        if role == SORT_ROLE:
            v = values[col]
            return v if col == 0 else (float(v) if np.isfinite(v) else -np.inf)
        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return m.name
            v = values[col]
            if not np.isfinite(v):
                return "–"
            return f"{v:.4f}" if col == 1 else f"{v:.4g}"
        if role == Qt.ItemDataRole.TextAlignmentRole and col > 0:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None


def make_proxy(model: MoleculeTableModel, parent=None) -> QSortFilterProxyModel:
    proxy = QSortFilterProxyModel(parent)
    proxy.setSourceModel(model)
    proxy.setSortRole(SORT_ROLE)
    proxy.setFilterKeyColumn(0)
    proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    return proxy
