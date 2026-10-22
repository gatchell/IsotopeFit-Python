"""Reading and writing IsotopeFit project files (``.ifd``).

An ``.ifd`` file is a MATLAB ``.mat`` file holding a single struct ``data``
with the fields ``raw_peakdata``, ``startind``, ``endind``, ``molecules``,
``calibration``, ``bgcorrectiondata``, ``guistatusvector`` and ``fileinfo``.
Port of ``load_ifd.m`` and the save part of ``IsotopeFit.m``, including
the compatibility fixes for old files.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from ..model import (
    DEFAULT_GUI_STATUS,
    GUI_STATUS_FLAGS,
    BackgroundCorrection,
    Calibration,
    Molecule,
    PeakShape,
    Project,
)

_MOLECULE_FIELDS = ("name", "peakdata", "minmass", "maxmass", "com", "minind", "maxind",
                    "area", "areaerror", "rootindex")
_INDEX_FIELDS = ("minind", "maxind", "rootindex")


# --------------------------------------------------------------------------- loading


def _scalar(v: Any, default: float = np.nan) -> float:
    """MATLAB scalar (possibly sparse or empty) -> float."""
    if sp.issparse(v):
        v = v.toarray()
    v = np.asarray(v, dtype=float).ravel()
    return float(v[0]) if v.size else default


def _fields(s) -> dict[str, Any]:
    return {f: getattr(s, f) for f in s._fieldnames}


def _strlist(v) -> list[str]:
    if isinstance(v, str):
        return [v]
    return [str(x) for x in np.asarray(v, dtype=object).ravel()]


def _molecule_from_mat(s) -> Molecule:
    f = _fields(s)
    m = Molecule(name=str(f.pop("name", "")), peakdata=f.pop("peakdata"))
    for key in ("minmass", "maxmass", "com", "area", "areaerror"):
        if key in f:
            setattr(m, key, _scalar(f.pop(key), getattr(m, key)))
    for key in _INDEX_FIELDS:
        if key in f:
            v = _scalar(f.pop(key))
            setattr(m, key, int(v) - 1 if np.isfinite(v) else -1)
    m.extra = f
    return m


def _molecules_from_mat(v) -> list[Molecule]:
    """``convert_molecule_datatype``: handles both struct arrays and old cell arrays."""
    if v is None:
        return []
    if hasattr(v, "_fieldnames"):  # a single molecule gets squeezed to a scalar struct
        return [_molecule_from_mat(v)]
    out = []
    for item in np.asarray(v, dtype=object).ravel():
        if hasattr(item, "_fieldnames"):
            out.append(_molecule_from_mat(item))
        else:  # old file: cell of 1x1 structs
            out.extend(_molecules_from_mat(item))
    return out


def _calibration_from_mat(c) -> Calibration:
    f = _fields(c)
    shape = f.get("shape")
    if shape is not None and hasattr(shape, "breaks"):
        peakshape = PeakShape(shape.breaks, shape.coefs)
    else:  # old file without custom peak shapes: Gaussian default
        peakshape = PeakShape.gaussian()
    return Calibration(
        comlist=f.get("comlist", []),
        massoffsetlist=f.get("massoffsetlist", []),
        resolutionlist=f.get("resolutionlist", []),
        massoffset_method=str(f.get("massoffsetmethode", "Flat")),
        resolution_method=str(f.get("resolutionmethode", "Flat")),
        massoffset_param=_scalar(f.get("massoffsetparam", 0), 0.0),
        resolution_param=_scalar(f.get("resolutionparam", 3000), 3000.0),
        namelist=_strlist(f.get("namelist", [])),
        shape=peakshape,
    )


def _bg_from_mat(b) -> BackgroundCorrection:
    f = _fields(b)
    return BackgroundCorrection(
        startmass=_scalar(f.get("startmass", -np.inf), -np.inf),
        endmass=_scalar(f.get("endmass", np.inf), np.inf),
        ndiv=int(_scalar(f.get("ndiv", 50), 50)),
        percent=_scalar(f.get("percent", 70), 70.0),
        bgm=f.get("bgm", []),  # missing in old files: no correction
        bgy=f.get("bgy", []),
    )


def load_ifd(path: str | os.PathLike, process: bool = True) -> Project:
    """Load an ``.ifd`` project file.

    With ``process=True`` (default) the processed spectrum
    ``Project.peakdata`` is computed as MATLAB does on load: crop, subtract
    the background, correct the mass axis.
    """
    try:
        mat = sio.loadmat(
            os.fspath(path), appendmat=False, squeeze_me=True, struct_as_record=False
        )
    except NotImplementedError as exc:  # MATLAB v7.3 files are HDF5
        raise NotImplementedError(
            f"{path} is a MATLAB v7.3 (HDF5) file, which is not supported yet."
        ) from exc
    data = mat["data"]
    d = _fields(data)

    status = np.asarray(d.get("guistatusvector", []), dtype=int).ravel()
    if status.size != len(GUI_STATUS_FLAGS):  # old file
        status = np.array(DEFAULT_GUI_STATUS)

    fileinfo = {k: str(v) for k, v in _fields(d["fileinfo"]).items()} if "fileinfo" in d else {}
    fileinfo["filename"] = os.path.basename(os.fspath(path))
    fileinfo["pathname"] = os.path.dirname(os.path.abspath(os.fspath(path)))
    fileinfo.setdefault("originalfilename", os.path.splitext(fileinfo["filename"])[0])

    project = Project(
        raw_peakdata=np.asarray(d["raw_peakdata"], dtype=float),
        start=int(_scalar(d.get("startind", 1))) - 1,
        stop=int(_scalar(d.get("endind", len(d["raw_peakdata"])))),
        molecules=_molecules_from_mat(d.get("molecules")),
        calibration=_calibration_from_mat(d["calibration"]) if "calibration" in d else Calibration(),
        bgcorrection=_bg_from_mat(d["bgcorrectiondata"]) if "bgcorrectiondata" in d
        else BackgroundCorrection(),
        gui_status=dict(zip(GUI_STATUS_FLAGS, map(bool, status))),
        fileinfo=fileinfo,
    )
    if process:
        project.update_peakdata()
    return project


# --------------------------------------------------------------------------- saving


def _mat_array(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    return a.reshape(1, -1) if a.ndim == 1 else a


def _cell(strings: list[str]) -> np.ndarray:
    c = np.empty((len(strings), 1), dtype=object)
    for i, s in enumerate(strings):
        c[i, 0] = s
    return c


def _molecules_to_mat(molecules: list[Molecule]) -> np.ndarray:
    extra_keys: list[str] = []
    for m in molecules:
        extra_keys += [k for k in m.extra if k not in extra_keys]
    names = list(_MOLECULE_FIELDS) + extra_keys
    arr = np.empty((1, len(molecules)), dtype=[(n, object) for n in names])
    for i, m in enumerate(molecules):
        rec = {
            "name": m.name,
            "peakdata": np.asarray(m.peakdata, dtype=float),
            "minmass": m.minmass,
            "maxmass": m.maxmass,
            "com": m.com,
            "area": m.area,
            "areaerror": m.areaerror,
        }
        for key in _INDEX_FIELDS:
            rec[key] = float(getattr(m, key) + 1)  # back to 1-based
        for key in extra_keys:
            rec[key] = m.extra.get(key, np.zeros((0, 0)))
        arr[0, i] = tuple(rec[n] for n in names)
    return arr


def save_ifd(project: Project, path: str | os.PathLike) -> None:
    """Write a project to an ``.ifd`` file that the MATLAB version can open."""
    cal = project.calibration
    bg = project.bgcorrection
    fileinfo = dict(project.fileinfo)
    fileinfo["filename"] = os.path.basename(os.fspath(path))
    fileinfo["pathname"] = os.path.dirname(os.path.abspath(os.fspath(path))) + os.sep
    data = {
        "raw_peakdata": np.asarray(project.raw_peakdata, dtype=float),
        "startind": float(project.start + 1),
        "endind": float(project.stop),
        "molecules": _molecules_to_mat(project.molecules),
        "calibration": {
            "comlist": _mat_array(cal.comlist),
            "massoffsetlist": _mat_array(cal.massoffsetlist),
            "resolutionlist": _mat_array(cal.resolutionlist),
            "massoffsetmethode": cal.massoffset_method,
            "resolutionmethode": cal.resolution_method,
            "massoffsetparam": float(cal.massoffset_param),
            "resolutionparam": float(cal.resolution_param),
            "namelist": _cell(cal.namelist),
            "shape": {
                "form": "pp",
                "breaks": _mat_array(cal.shape.breaks),
                "coefs": np.asarray(cal.shape.coefs, dtype=float),
                "pieces": float(cal.shape.coefs.shape[0]),
                "order": float(cal.shape.order),
                "dim": 1.0,
            },
        },
        "bgcorrectiondata": {
            "startmass": float(bg.startmass),
            "endmass": float(bg.endmass),
            "ndiv": float(bg.ndiv),
            "percent": float(bg.percent),
            "bgm": _mat_array(bg.bgm),
            "bgy": _mat_array(bg.bgy),
        },
        "guistatusvector": np.array(
            [[float(project.gui_status.get(k, False)) for k in GUI_STATUS_FLAGS]]
        ),
        "fileinfo": fileinfo,
    }
    sio.savemat(
        os.fspath(path), {"data": data}, appendmat=False, do_compression=True, oned_as="row"
    )
