"""Molecule lists: ``.ifm`` files and folders of pattern text files.

An ``.ifm`` file (written by the cluster generator) is a MATLAB ``.mat``
file with ``data.molecules`` (``name``, ``peakdata``) and ``data.namelist``.
A molecule text file holds two columns, mass and abundance; the file name
without ``.txt`` is the molecule name. Ports of ``load_molecules_from_ifm.m``,
``load_molecules_from_folder.m`` and the saving part of ``generate_cluster_ifm.m``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.io as sio

from ..isotopes import renorm
from ..model import Molecule
from .ifd import _molecules_from_mat


def load_ifm(path: str | os.PathLike) -> list[Molecule]:
    """Molecules stored in an ``.ifm`` file (names and patterns only)."""
    mat = sio.loadmat(os.fspath(path), appendmat=False, squeeze_me=True, struct_as_record=False)
    return _molecules_from_mat(mat["data"].molecules)


def save_ifm(molecules: Sequence[Molecule], path: str | os.PathLike) -> None:
    """Write molecules to an ``.ifm`` file that the MATLAB version can read."""
    arr = np.empty((1, len(molecules)), dtype=[("peakdata", object), ("name", object)])
    for i, m in enumerate(molecules):
        arr[0, i] = (np.asarray(m.peakdata, dtype=float), m.name)
    namelist = np.empty((1, len(molecules)), dtype=object)
    for i, m in enumerate(molecules):
        namelist[0, i] = m.name
    sio.savemat(os.fspath(path), {"data": {"molecules": arr, "namelist": namelist}},
                appendmat=False, do_compression=True)


def load_molecule_files(paths: Iterable[str | os.PathLike]) -> list[Molecule]:
    """Molecules from two-column text files; abundances are normalised to sum 1."""
    out = []
    for p in map(Path, paths):
        out.append(Molecule(name=p.stem, peakdata=renorm(np.loadtxt(p, ndmin=2))))
    return out


def load_molecule_folder(folder: str | os.PathLike, pattern: str = "*.txt") -> list[Molecule]:
    """All molecule text files in a folder, sorted by file name."""
    return load_molecule_files(sorted(Path(folder).glob(pattern)))
