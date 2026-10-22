"""Cluster definitions for the cluster generator (port of the logic in ``cluster_generator.m``).

A definition is a tree. Each top-level node is a species with its
multiplicities, e.g. He with n = 1:50; child nodes add further species, e.g.
H2O with n = [0 1]. Every path from a top-level node down to a leaf is one
family of mixed clusters, generated with :func:`isotopefit.isotopes.generate_clusters`.
Charge, intensity threshold and mass accuracy are set on the top-level node
and apply to its whole family.

Definitions are stored in ``.icg`` files (MATLAB ``.mat`` format), which the
MATLAB cluster generator can also read.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import scipy.io as sio

from .isotopes import generate_clusters
from .model import Molecule


def parse_n(text: str) -> list[int]:
    """Multiplicities written as in MATLAB: ``"1:10"``, ``"0:2:20"``, ``"[0 1 2]"``, ``"[1,2,4]"``, ``"5"``.

    Ranges and single numbers can be combined, e.g. ``"[0:3, 7]"``.
    Order is kept; duplicates are dropped.
    """
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s.strip():
        raise ValueError("No multiplicities given")
    out: list[int] = []
    for token in re.split(r"[,\s]+", s.strip()):
        if not token:
            continue
        parts = token.split(":")
        try:
            nums = [int(float(p)) for p in parts]
        except ValueError:
            raise ValueError(f"Cannot read multiplicities {text!r}") from None
        if len(parts) == 1:
            values = nums
        elif len(parts) == 2:
            values = list(range(nums[0], nums[1] + 1))
        elif len(parts) == 3:
            start, step, stop = nums
            if step == 0:
                raise ValueError(f"Step 0 in {text!r}")
            values = list(range(start, stop + (1 if step > 0 else -1), step))
        else:
            raise ValueError(f"Cannot read multiplicities {text!r}")
        if any(v < 0 for v in values):
            raise ValueError(f"Negative multiplicity in {text!r}")
        out += [v for v in values if v not in out]
    if not out:
        raise ValueError(f"{text!r} gives no multiplicities")
    return out


@dataclass
class ClusterNode:
    """One species in a cluster definition tree."""

    formula: str  # sum formula, e.g. "H2O"
    name: str = ""  # name used in molecule names, e.g. "W"; defaults to the formula
    n: str = "1"  # multiplicities, MATLAB syntax (see parse_n)
    charge: int = 1  # only used on top-level nodes
    threshold: float = 5e-4  # relative intensity below which peaks are dropped (top level)
    mass_accuracy: float = 1e-3  # peaks closer than this are merged (top level)
    children: list[ClusterNode] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.name or self.formula

    def multiplicities(self) -> list[int]:
        return parse_n(self.n)


@dataclass
class Family:
    formulas: list[str]
    names: list[str]
    nlists: list[list[int]]
    charge: int
    threshold: float
    mass_accuracy: float

    def count(self) -> int:
        """Upper bound for the number of molecules (the empty cluster is not counted)."""
        total = int(np.prod([len(n) for n in self.nlists]))
        return total - (1 if all(0 in n for n in self.nlists) else 0)


def families(roots: Sequence[ClusterNode]) -> list[Family]:
    """Every top-level-to-leaf path as one family (``get_all_nodes``)."""
    out: list[Family] = []

    def walk(node: ClusterNode, path: list[ClusterNode], top: ClusterNode) -> None:
        path = path + [node]
        if not node.children:
            out.append(Family([p.formula for p in path], [p.label for p in path],
                              [p.multiplicities() for p in path],
                              int(top.charge), float(top.threshold), float(top.mass_accuracy)))
            return
        for child in node.children:
            walk(child, path, top)

    for root in roots:
        walk(root, [], root)
    return out


def generate(roots: Sequence[ClusterNode],
             progress: Callable[[int, int], None] | None = None) -> list[Molecule]:
    """All molecules of a definition, as MATLAB's "Generate ifm" writes them.

    Families are generated in tree order; a name already produced by an
    earlier family is not generated again.
    """
    fams = families(roots)
    seen: list[str] = []
    out: list[Molecule] = []
    for i, f in enumerate(fams):
        mols = generate_clusters(f.formulas, f.nlists, f.mass_accuracy, f.threshold, f.names,
                                 f.charge, existing=seen)
        out += mols
        seen += [m.name for m in mols]
        if progress:
            progress(i + 1, len(fams))
    return out


# ------------------------------------------------------------------ .icg files


def _ud(node: ClusterNode) -> dict:
    return {"sumf": node.formula, "name": node.label, "n": node.n,
            "charge": float(node.charge), "th": float(node.threshold),
            "mapprox": float(node.mass_accuracy)}


def _to_struct(ud: dict, children: Sequence[ClusterNode]):
    """MATLAB struct with fields ud and children (a struct array, or [] for a leaf)."""
    if not children:
        kids = np.zeros((0, 0))
    else:
        kids = np.empty((1, len(children)), dtype=[("ud", object), ("children", object)])
        for i, c in enumerate(children):
            s = _to_struct(_ud(c), c.children)
            kids[0, i] = (s["ud"], s["children"])
    return {"ud": ud, "children": kids}


def save_icg(roots: Sequence[ClusterNode], path: str | os.PathLike) -> None:
    """Save a definition so that the MATLAB cluster generator can open it too."""
    root_ud = {"sumf": "", "name": "root", "n": "1", "charge": 1.0, "th": 5e-4, "mapprox": 1e-3}
    sio.savemat(os.fspath(path), {"data": _to_struct(root_ud, roots)}, appendmat=False,
                do_compression=True)


def _get(s, key, default):
    v = getattr(s, key, default)
    if isinstance(v, np.ndarray):
        v = v.ravel()[0] if v.size else default
    return v


def _from_struct(s, inherited: ClusterNode | None) -> ClusterNode:
    ud = s.ud
    node = ClusterNode(
        formula=str(_get(ud, "sumf", "")), name=str(_get(ud, "name", "")), n=str(_get(ud, "n", "1")),
        charge=int(float(_get(ud, "charge", inherited.charge if inherited else 1))),
        threshold=float(_get(ud, "th", inherited.threshold if inherited else 5e-4)),
        mass_accuracy=float(_get(ud, "mapprox", inherited.mass_accuracy if inherited else 1e-3)),
    )
    node.children = [_from_struct(c, node) for c in _children(s)]
    return node


def _children(s) -> list:
    kids = getattr(s, "children", None)
    if kids is None:
        return []
    if hasattr(kids, "_fieldnames"):
        return [kids]
    return [k for k in np.asarray(kids, dtype=object).ravel() if hasattr(k, "_fieldnames")]


def load_icg(path: str | os.PathLike) -> list[ClusterNode]:
    """Top-level nodes of a definition saved by the MATLAB or Python cluster generator."""
    data = sio.loadmat(os.fspath(path), appendmat=False, squeeze_me=True,
                       struct_as_record=False)["data"]
    roots = []
    for c in _children(data):
        top = _from_struct(c, None)
        roots.append(top)
    return roots
