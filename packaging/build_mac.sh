#!/usr/bin/env bash
# Build dist/IsotopeFit.app. Run from anywhere, with the virtual environment active:
#     bash packaging/build_mac.sh
# Add --install to copy the app to /Applications afterwards.
# The Python used: $PYTHON if set, else the active virtual environment's, else the
# first of python3.14, python3, python found on the PATH.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname)" != "Darwin" ]]; then
    echo "This builds a macOS app; run it on a Mac." >&2
    exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
    PY="$(command -v "$PYTHON" || echo "$PYTHON")"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PY="$VIRTUAL_ENV/bin/python"
else
    PY=""
    for candidate in python3.14 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PY="$(command -v "$candidate")"
            break
        fi
    done
fi
if [[ -z "$PY" ]]; then
    echo "No Python found. Activate the virtual environment, or run: PYTHON=/path/to/python bash $0" >&2
    exit 1
fi
echo "== Python: $PY ($("$PY" --version))"
"$PY" -c "import isotopefit, PySide6, pyqtgraph, h5py" || {
    echo 'Install the program first:  pip install -e ".[gui]"' >&2
    exit 1
}
"$PY" -m pip install --quiet --upgrade "pyinstaller>=6.16" pyinstaller-hooks-contrib

echo "== Icon"
ICONSET=build/IsotopeFit.iconset
rm -rf "$ICONSET"
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
    sips -z $size $size packaging/icon.png --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -z $double $double packaging/icon.png --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o packaging/IsotopeFit.icns

echo "== App"
"$PY" -m PyInstaller --noconfirm --clean --distpath dist --workpath build/pyinstaller \
    packaging/IsotopeFit.spec

APP=dist/IsotopeFit.app
echo "== Built $APP ($(du -sh "$APP" | cut -f1))"

if [[ "${1:-}" == "--install" ]]; then
    rm -rf /Applications/IsotopeFit.app
    cp -R "$APP" /Applications/
    echo "== Copied to /Applications"
fi
