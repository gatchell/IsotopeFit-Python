# PyInstaller configuration for the macOS app and the Windows program.
#   macOS:    bash packaging/build_mac.sh        -> dist/IsotopeFit.app
#   Windows:  python -m PyInstaller packaging/IsotopeFit.spec   -> dist/IsotopeFit/IsotopeFit.exe
# (run from the project folder, with the virtual environment active). GitHub Actions
# builds both: see .github/workflows/build.yml.

import os
import re
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
with open(os.path.join(ROOT, "src", "isotopefit", "__init__.py")) as f:
    VERSION = re.search(r'__version__ = "([^"]+)"', f.read()).group(1)
SHORT_VERSION = re.match(r"[0-9.]+", VERSION).group(0).rstrip(".")

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[os.path.join(ROOT, "src")],
    datas=collect_data_files("isotopefit"),  # .ui layouts, isotope tables, peak shape, icon
    # SciPy imports parts of its bundled array-API helpers (scipy._external) dynamically,
    # which PyInstaller cannot see; newer SciPy versions moved them there.
    hiddenimports=["h5py", "scipy.io.matlab", "PySide6.QtUiTools"]
    + collect_submodules("scipy._external")
    + collect_submodules("scipy._lib.array_api_compat")
    + collect_submodules("scipy._lib.array_api_extra"),
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "PyQt5", "PyQt6", "PySide2"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="IsotopeFit",
    console=False,
    argv_emulation=False,
    upx=False,
    # Windows: PyInstaller turns the PNG into an .ico (needs Pillow). macOS uses the .icns below.
    icon=os.path.join(SPECPATH, "icon.png") if sys.platform == "win32" else None,
)

coll = COLLECT(exe, a.binaries, a.datas, name="IsotopeFit", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="IsotopeFit.app",
        icon=os.path.join(SPECPATH, "IsotopeFit.icns"),
        bundle_identifier="io.github.isotopefit",
        version=SHORT_VERSION,
        info_plist={
            "CFBundleName": "IsotopeFit",
            "CFBundleDisplayName": "IsotopeFit",
            "CFBundleShortVersionString": SHORT_VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "CFBundleDocumentTypes": [
                {
                    "CFBundleTypeName": "IsotopeFit project",
                    "CFBundleTypeExtensions": ["ifd"],
                    "CFBundleTypeRole": "Editor",
                    "LSHandlerRank": "Owner",
                },
                {
                    "CFBundleTypeName": "HDF5 mass spectrum",
                    "CFBundleTypeExtensions": ["h5"],
                    "CFBundleTypeRole": "Viewer",
                    "LSHandlerRank": "Alternate",
                },
            ],
        },
    )
