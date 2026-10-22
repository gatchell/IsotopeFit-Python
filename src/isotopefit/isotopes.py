"""Isotope patterns of atoms, molecules and clusters.

Port of the ``IsotopeDistribution`` folder. A distribution is an (n, 2)
array of [mass, abundance]. Building a pattern means repeatedly
convolving distributions, then merging peaks closer than
``minmassdistance`` and dropping peaks below ``th`` times the largest
one. The order of these steps matters for the result, so it follows
MATLAB exactly.
"""

from __future__ import annotations

import heapq
import itertools
from functools import lru_cache
from importlib import resources

import numpy as np

from .model import Molecule

ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Uub Uut "
    "Uuq Uup Uuh Uus Uuo"
).split()


# ------------------------------------------------------------------ basic operations


def renorm(d: np.ndarray) -> np.ndarray:
    """Scale abundances to sum to 1."""
    d = np.array(d, dtype=float).reshape(-1, 2)
    d[:, 1] /= d[:, 1].sum()
    return d


def convolute(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """All combinations of two distributions (masses add, abundances multiply).

    Same ordering as MATLAB's ``kron``: the second distribution varies fastest.
    """
    d1 = np.asarray(d1, dtype=float).reshape(-1, 2)
    d2 = np.asarray(d2, dtype=float).reshape(-1, 2)
    return np.column_stack([
        np.add.outer(d1[:, 0], d2[:, 0]).ravel(),
        np.multiply.outer(d1[:, 1], d2[:, 1]).ravel(),
    ])


def approx_p(d: np.ndarray, th: float) -> np.ndarray:
    """Keep peaks with abundance >= th * maximum."""
    d = np.asarray(d, dtype=float).reshape(-1, 2)
    return d[d[:, 1] >= th * d[:, 1].max()]


def approx_masses(d: np.ndarray, minmassdistance: float) -> np.ndarray:
    """Merge neighbouring peaks closer than ``minmassdistance``.

    Port of ``approx_masses.m``. The closest pair is merged first into one
    peak at the abundance-weighted mean mass, and so on until no gap is at
    most ``minmassdistance``. Ties go to the lowest mass.

    After a merge both neighbouring gaps are recomputed. (MATLAB forgets the
    gap to the left of the merged peak, which has moved, and so sometimes
    merges peaks that are further apart than ``minmassdistance``.) MATLAB
    rescans all gaps after every merge (quadratic time); a heap is faster.
    """
    d = np.asarray(d, dtype=float).reshape(-1, 2)
    order = np.argsort(d[:, 0], kind="stable")
    mass = d[order, 0].copy()
    p = d[order, 1].copy()
    n = mass.size
    if n < 2:
        return np.column_stack([mass, p])

    nxt = np.arange(1, n + 1)  # n means "none"
    prv = np.arange(-1, n - 1)
    alive = np.ones(n, dtype=bool)
    gap = np.diff(mass)  # gap[k]: stored gap between k and nxt[k]
    version = np.zeros(n, dtype=np.int64)
    heap = [(gap[k], k, 0) for k in range(n - 1)]
    heapq.heapify(heap)

    while heap:
        g, k, ver = heap[0]
        if not alive[k] or ver != version[k] or nxt[k] >= n:
            heapq.heappop(heap)
            continue
        if g > minmassdistance:
            break
        heapq.heappop(heap)
        r = nxt[k]
        psum = p[k] + p[r]
        mass[k] = (mass[k] * p[k] + mass[r] * p[r]) / psum
        p[k] = psum
        alive[r] = False
        nxt[k] = nxt[r]
        if nxt[k] < n:
            prv[nxt[k]] = k
            version[k] += 1
            heapq.heappush(heap, (mass[nxt[k]] - mass[k], k, version[k]))
        left = prv[k]
        if left >= 0:  # the merged peak moved: its left gap changed too
            version[left] += 1
            heapq.heappush(heap, (mass[k] - mass[left], left, version[left]))
    return np.column_stack([mass[alive], p[alive]])


def _reduce(d: np.ndarray, minmassdistance: float, th: float) -> np.ndarray:
    return approx_p(approx_masses(d, minmassdistance), th)


# ------------------------------------------------------------------ atoms and molecules


@lru_cache(maxsize=None)
def _element_table(symbol: str) -> np.ndarray:
    path = resources.files(__package__) / "data" / "atoms" / f"{symbol}.txt"
    if not path.is_file():
        raise ValueError(f"No isotope table for element {symbol!r}")
    text = path.read_text()
    if not text.strip():
        raise ValueError(f"The isotope table for {symbol!r} is empty")
    return renorm(np.loadtxt(text.splitlines(), ndmin=2))


def element_table(symbol: str) -> np.ndarray:
    """Normalised isotope table of an element (masses, abundances summing to 1).

    Besides real elements, the tables include ``D`` (deuterium), ``Cth``
    (pure 13C) and ``X`` (a test element), as in the MATLAB version.
    """
    return _element_table(symbol).copy()


def available_elements() -> list[str]:
    folder = resources.files(__package__) / "data" / "atoms"
    return sorted(p.name[:-4] for p in folder.iterdir() if p.name.endswith(".txt"))


def atomic_distribution(element: str, n: int, minmassdistance: float, th: float) -> np.ndarray:
    """Pattern of ``n`` atoms of one element (``atomic_distribution.m``)."""
    d = element_table(element)
    out = d
    for _ in range(int(n) - 1):
        out = _reduce(convolute(out, d), minmassdistance, th)
    return out


def self_convolute(d: np.ndarray, n: int, minmassdistance: float, th: float) -> np.ndarray:
    """Pattern of ``n`` copies of ``d`` (``self_convolute.m``)."""
    out = d
    for _ in range(int(n) - 1):
        out = _reduce(convolute(out, d), minmassdistance, th)
    return out


def _tokenize(formula: str):
    """Yield ('el', symbol, n) and ('group', subformula, n) like ``parse_molecule.m``.

    Characters other than element symbols, digits after them and brackets
    are skipped, as in MATLAB.
    """
    s, i, n = formula, 0, len(formula)

    def number(j: int) -> tuple[int, int]:
        k = j
        while k < n and s[k].isdigit():
            k += 1
        return (int(s[j:k]) if k > j else 1), k

    while i < n:
        c = s[i]
        if "A" <= c <= "Z":
            j = i + 1
            while j < n and "a" <= s[j] <= "z":
                j += 1
            count, k = number(j)
            yield "el", s[i:j], count
            i = k
        elif c == "(":
            depth, j = 1, i
            while depth > 0:
                j += 1
                if j >= n:
                    raise ValueError(f"Unbalanced parentheses in {formula!r}")
                depth += {"(": 1, ")": -1}.get(s[j], 0)
            count, k = number(j + 1)
            yield "group", s[i + 1 : j], count
            i = k
        else:
            i += 1


def parse_molecule(formula: str, minmassdistance: float, th: float) -> np.ndarray:
    """Isotope pattern of a sum formula such as ``"C60"``, ``"H2O"`` or ``"(H2O)3Na"``.

    Port of ``parse_molecule.m``.
    """
    parts = []
    for kind, what, count in _tokenize(formula):
        if kind == "el":
            parts.append(atomic_distribution(what, count, minmassdistance, th))
        else:
            parts.append(self_convolute(parse_molecule(what, minmassdistance, th),
                                        count, minmassdistance, th))
    if not parts:
        raise ValueError(f"No elements found in formula {formula!r}")
    out = parts[0]
    for d in parts[1:]:
        out = _reduce(convolute(out, d), minmassdistance, th)
    return out


def sum_formula_counts(formula: str) -> dict[str, int]:
    """Number of atoms per element (``sum_formula2cid.m``, as a dict)."""
    counts: dict[str, int] = {}
    for kind, what, count in _tokenize(formula):
        sub = {what: 1} if kind == "el" else sum_formula_counts(what)
        for el, k in sub.items():
            counts[el] = counts.get(el, 0) + k * count
    return counts


# ------------------------------------------------------------------ clusters


def cluster_name(names: list[str], numbers: list[int]) -> str:
    """``[C60]2[H2O]`` style name; species with number 0 are left out."""
    out = ""
    for name, k in zip(names, numbers):
        if k > 0:
            out += f"[{name}]" + (str(k) if k > 1 else "")
    return out


def generate_clusters(
    formulas: list[str],
    nlists: list[list[int]],
    minmassdistance: float = 1e-3,
    th: float = 1e-6,
    names: list[str] | None = None,
    charge: int = 1,
    existing: list[str] | None = None,
) -> list[Molecule]:
    """All mixed clusters of the given species (port of ``generate_cluster_ifm.m``).

    ``nlists[i]`` lists the multimer numbers of species ``i``, e.g.
    ``generate_clusters(["He", "H2O"], [range(1, 51), [0, 1]])`` gives
    He1..He50 and He1..He50 with one water. Combinations come in MATLAB's
    order (the first species varies fastest). Names already in
    ``existing`` and the empty cluster are skipped. Masses are divided by
    ``charge``.

    The returned molecules only carry ``name`` and ``peakdata``; call
    :func:`isotopefit.molecules.init_molecules` to place them in a spectrum.
    """
    names = list(formulas) if names is None else list(names)
    nlists = [list(nl) for nl in nlists]
    seen = set(existing or [])

    # Multimer patterns of each species, built up from the monomer.
    multimers = []
    for formula, nl in zip(formulas, nlists):
        mono = parse_molecule(formula, minmassdistance, th)
        pats = {1: mono}
        for k in range(2, max(nl) + 1):
            pats[k] = _reduce(convolute(pats[k - 1], mono), minmassdistance, th)
        multimers.append(pats)

    out = []
    # itertools.product varies the last factor fastest; MATLAB varies the first.
    for combo in itertools.product(*reversed(nlists)):
        numbers = list(reversed(combo))
        name = cluster_name(names, numbers)
        if not name or name in seen:
            continue
        d = np.array([[0.0, 1.0]])
        for pats, k in zip(multimers, numbers):
            if k > 0:
                d = _reduce(convolute(d, pats[k]), minmassdistance, th)
        d = d.copy()
        d[:, 0] /= charge
        out.append(Molecule(name=name, peakdata=d))
        seen.add(name)
    return out
