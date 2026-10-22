"""Find out why menus don't drop down on this Mac.

Run:  python tools/diagnose_menus.py
One small window opens. Try each numbered item and note which ones show a
drop-down list, then close the window and send the printed versions.
"""

import platform
import sys

import PySide6
from PySide6.QtCore import Qt, qVersion
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QComboBox, QLabel, QMainWindow, QMenu, QMenuBar,
                               QPushButton, QToolButton, QVBoxLayout, QWidget)

print("Python", sys.version.split()[0], "| PySide6", PySide6.__version__, "| Qt", qVersion(),
      "| macOS", platform.mac_ver()[0])

QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, True)
app = QApplication(sys.argv)

win = QMainWindow()
win.setWindowTitle("Menu test")
bar = QMenuBar(win)
m1 = bar.addMenu("1 Menu bar")
for t in ("Alpha", "Beta", "Gamma"):
    m1.addAction(QAction(t, win, triggered=lambda checked=False, t=t: print("menu bar:", t)))
win.setMenuBar(bar)

central = QWidget()
lay = QVBoxLayout(central)
lay.addWidget(QLabel("Click each control and note whether a list drops down:"))

combo = QComboBox()
combo.addItems(["2 Combo box: first", "second", "third"])
lay.addWidget(combo)

popup = QMenu(win)
for t in ("One", "Two", "Three"):
    popup.addAction(t)
btn = QPushButton("3 Button that opens a menu with exec()")
btn.clicked.connect(lambda: popup.exec(btn.mapToGlobal(btn.rect().bottomLeft())))
lay.addWidget(btn)

tool = QToolButton()
tool.setText("4 Tool button with menu")
tool.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
tool.setMenu(popup)
lay.addWidget(tool)

win.setCentralWidget(central)
win.resize(420, 220)
win.show()
win.raise_()
win.activateWindow()
sys.exit(app.exec())
