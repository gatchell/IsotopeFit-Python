"""Cluster series: areas of [X]n against n (port of ``sortmolecules`` and the series export).

Given a species such as ``"He"``, every molecule name is split into the
multiplicity of ``[He]`` and the rest; molecules with the same rest form a
series, e.g. ``[He]n[H2O]``. Areas are divided by the local bin width of the
spectrum, which turns them into approximate numbers of counts.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model import Molecule


@dataclass
class ClusterSeries:
    name: str  # e.g. "[He]n[H2O]"
    sizes: np.ndarray  # n
    areas: np.ndarray  # area / bin width
    errors: np.ndarray
    indices: np.ndarray  # positions in the molecule list


def _split(name: str, tag: str) -> tuple[int, str]:
    """(n, rest of the name) as in ``sortmolecules``; n = 0 when the species is absent."""
    padded = name + "["
    ix = padded.find(tag)
    if ix < 0:
        return 0, padded
    j = ix + len(tag)
    k = padded.index("[", j)
    num = padded[j:k]
    n = int(num) if num else 1
    return n, padded.replace(tag + num, "")


def species_in(molecules: Sequence[Molecule]) -> list[str]:
    """The species named in brackets in the molecule names, most common first.

    ``[He]4[H2O]`` contains ``He`` and ``H2O``. Ties are sorted by name.
    """
    counts = Counter()
    for m in molecules:
        counts.update(set(re.findall(r"\[([^\[\]]+)\]", m.name)))
    return [name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def cluster_series(molecules: Sequence[Molecule], species: str,
                   peakdata: np.ndarray) -> list[ClusterSeries]:
    """All series of ``species`` among ``molecules``, sorted by name."""
    tag = f"[{species}]"
    nbins = peakdata.shape[0]
    table: dict[str, dict[int, tuple[float, float, int]]] = {}
    for i, m in enumerate(molecules):
        n, rest = _split(m.name, tag)
        lo = max(0, m.minind - 10)
        hi = min(nbins - 1, m.maxind + 10)
        width = (peakdata[lo, 0] - peakdata[hi, 0]) / (lo - hi) if hi != lo else 1.0
        table.setdefault(rest, {})[n] = (m.area / width, m.areaerror / width, i)

    out = []
    for rest, entries in table.items():
        sizes = np.array(sorted(entries))
        vals = np.array([entries[n] for n in sizes], dtype=float).reshape(-1, 3)
        out.append(ClusterSeries(name=f"{tag}n{rest[:-1]}", sizes=sizes, areas=vals[:, 0],
                                 errors=vals[:, 1], indices=vals[:, 2].astype(int)))
    out.sort(key=lambda s: s.name)
    return out


def export_series(series: Sequence[ClusterSeries], path: str | os.PathLike) -> None:
    """Write series as a tab-separated table like the MATLAB export.

    One row per cluster size from 0 to the largest size; for each series an
    area and an error column; missing entries are ``--``.
    """
    nmax = max((int(s.sizes.max()) for s in series if s.sizes.size), default=0)
    lookup = [dict(zip(s.sizes.tolist(), zip(s.areas, s.errors))) for s in series]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t" + "".join(f"{s.name}\t(error)\t" for s in series) + "\n")
        for n in range(nmax + 1):
            fh.write(f"{n}\t")
            for d in lookup:
                if n in d:
                    fh.write(f"{d[n][0]:e}\t{d[n][1]:e}\t")
                else:
                    fh.write("--\t--\t")
            fh.write("\n")
