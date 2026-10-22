"""Start the IsotopeFit GUI: ``isotopefit-gui [file.ifd|file.h5]`` or ``python -m isotopefit.gui``."""

from __future__ import annotations

import os
import sys


def use_native_menus() -> bool:
    """System menu bar in the macOS app bundle, menus inside the window otherwise.

    ``ISOTOPEFIT_NATIVE_MENUS=1`` or ``=0`` overrides this.
    """
    env = os.environ.get("ISOTOPEFIT_NATIVE_MENUS")
    if env is not None:
        return env.strip() not in ("", "0", "false", "no")
    return sys.platform == "darwin" and bool(getattr(sys, "frozen", False))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    try:
        import pyqtgraph as pg
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        print(f"The GUI needs PySide6 and pyqtgraph ({exc}).\n"
              'Install them with:  pip install -e ".[gui]"', file=sys.stderr)
        return 1

    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QIcon

    from .main_window import MainWindow

    # Started from a terminal, keep menu bars inside the windows. On macOS, Qt
    # otherwise moves them to the system menu bar, which misbehaves for programs
    # started from a terminal; and switching a menu bar from native to in-window
    # after it has been built can leave its menus unable to open. This must be
    # set before QApplication exists. The app bundle uses the system menu bar.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar,
                              not use_native_menus())

    class Application(QApplication):
        """Opens files double-clicked in Finder (macOS sends them as events, not arguments)."""

        window = None
        pending: list[str] = []

        def event(self, e):  # noqa: N802 - Qt API
            if e.type() == QEvent.Type.FileOpen:
                path = e.file()
                if self.window is None:
                    self.pending.append(path)
                elif self.window.confirm_discard():
                    self.window.open_path(path)
                return True
            return super().event(e)

    app = Application(argv)
    app.setApplicationName("IsotopeFit")
    app.setOrganizationName("IsotopeFit")
    pg.setConfigOptions(background="w", foreground="k", antialias=False)
    from importlib import resources
    icon = resources.files(__package__) / "icon.png"
    if icon.is_file():
        app.setWindowIcon(QIcon(str(icon)))
    window = MainWindow()
    app.window = window
    window.show()
    # Bring the window to the front and make it the active one. On macOS, a
    # program started from a terminal is not always activated, and pop-ups
    # (menus, combo boxes) then may not appear until it is.
    window.w.raise_()
    window.w.activateWindow()
    files = [a for a in argv[1:] if not a.startswith("-psn_")] + app.pending
    if files:
        window.open_path(files[-1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
