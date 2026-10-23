"""Reading a spectrum from a text (ASCII) file (port of ``load_ascii`` in ``IsotopeFit.m``).

Two kinds of file are recognised, as in MATLAB:

* TOFDAQ text exports, which start with ``mass spectrum`` and a line of
  ``=``; their masses are in the first column and the signal in the third;
* anything else: masses in the first column, the signal in the second.

MATLAB expects tabs and always skips the first two lines. Here header lines
are recognised by their content (a line is data when its first columns are
numbers), and tabs, semicolons, commas or spaces between the columns all
work. Numbers may use a decimal comma when the columns are separated by tabs
or semicolons. Lines starting with ``#`` or ``%`` are ignored. The spectrum
exports of this program (Data menu) can be read back this way.
"""

from __future__ import annotations

import io
import os
import re

import numpy as np

from ..model import Calibration, Project

_TOFDAQ = re.compile(r"^\s*mass\s+spectrum\s+=+", re.I)


def _delimiter(line: str) -> str | None:
    """The column separator of a data line (None: any whitespace)."""
    if "\t" in line:
        return "\t"
    if ";" in line:
        return ";"
    if "," in line:
        return ","
    return None


def _numbers(line: str, delimiter: str | None, decimal_comma: bool) -> list[float] | None:
    if decimal_comma:
        line = line.replace(",", ".")
    parts = [p for p in (line.split(delimiter) if delimiter else line.split()) if p.strip()]
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def read_ascii_spectrum(path: str | os.PathLike) -> np.ndarray:
    """The (n, 2) [mass, signal] array of a text file; see the module description."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    lines = text.splitlines()
    head = "\n".join(lines[:3])
    tofdaq = bool(_TOFDAQ.match(" ".join(head.split())))
    column = 2 if tofdaq else 1

    # The first line whose first columns are numbers starts the data.
    start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s[0] in "#%":
            continue
        delimiter = _delimiter(s)
        decimal_comma = delimiter in ("\t", ";") and "," in s
        values = _numbers(s, delimiter, decimal_comma)
        if values is not None and len(values) > column:
            start = i
            break
    if start is None:
        raise ValueError("No columns of numbers found. Expected masses in the first column "
                         "and the signal in the second (third for TOFDAQ exports).")

    body = "\n".join(lines[start:])
    if decimal_comma:
        body = body.replace(",", ".")
    data = np.loadtxt(io.StringIO(body), delimiter=delimiter, comments=("#", "%"),
                      usecols=(0, column), ndmin=2)
    data = data[np.all(np.isfinite(data), axis=1)]
    if data.shape[0] < 2:
        raise ValueError("The file holds fewer than two data points.")
    return data


def load_ascii(path: str | os.PathLike) -> Project:
    """Load a spectrum from a text file into a new :class:`Project` (no calibration yet)."""
    path = os.fspath(path)
    peakdata = read_ascii_spectrum(path)
    project = Project(
        raw_peakdata=peakdata,
        calibration=Calibration(),
        fileinfo={
            "originalfilename": os.path.splitext(os.path.basename(path))[0],
            "pathname": os.path.dirname(os.path.abspath(path)),
            "filename": "",
        },
    )
    project.gui_status.update(file_loaded=True, molecules_loaded=False, calibrated=False,
                              bg_corrected=False, drift_corrected=False, changed=True,
                              fitted=False, cs_selected=False)
    project.peakdata = project.raw_peakdata.copy()
    return project
