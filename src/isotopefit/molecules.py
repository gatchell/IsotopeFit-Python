"""Placing molecules in a spectrum and grouping them into fit ranges.

Ports of ``init_molecule_properties.m``, ``findranges.m``,
``calccomofranges.m``, ``calccomofmolecules.m``, ``findmassrange.m``,
``findmassrange2.m`` and ``molecules_in_massrange_with_sigma.m``.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .model import Calibration, Molecule
from .spectrum import guess_area, increasing_from, mass2ind

log = logging.getLogger(__name__)


def init_molecules(molecules: Sequence[Molecule], peakdata: np.ndarray) -> list[Molecule]:
    """``init_molecule_properties``: prepare molecules for a spectrum.

    Returns copies with ``minmass``, ``maxmass``, ``com``, ``minind``,
    ``maxind``, a first ``area`` guess (integral of the data under the
    pattern) and ``areaerror = NaN``. Molecules starting above the last
    mass of the spectrum are dropped. The rest are sorted by their lowest
    mass and get ``rootindex`` 0, 1, 2, ...
    """
    massaxis = peakdata[:, 0]
    last_mass = peakdata[-1, 0]
    kept = []
    for mol in molecules:
        m = copy.deepcopy(mol)
        masses, peaks = m.peakdata[:, 0], m.peakdata[:, 1]
        m.minmass = float(masses.min())
        m.maxmass = float(masses.max())
        if m.minmass >= last_mass:
            log.info("Molecule %s out of range", m.name)
            continue
        m.com = float(np.sum(masses * peaks) / np.sum(peaks))
        m.minind = mass2ind(massaxis, m.minmass)
        m.maxind = mass2ind(massaxis, m.maxmass)
        if m.maxind - m.minind <= 1:  # too narrow to integrate
            m.area = 0.0
        else:  # MATLAB: guessarea(peakdata(minind:maxind-1, :))
            m.area = guess_area(peakdata[m.minind : m.maxind])
        m.areaerror = np.nan
        kept.append(m)
    log.info("%d molecules out of mass range, %d loaded",
             len(molecules) - len(kept), len(kept))
    kept.sort(key=lambda m: m.minmass)  # Python's sort is stable, like MATLAB's
    for i, m in enumerate(kept):
        m.rootindex = i
    return kept


def com_of_molecules(molecules: Sequence[Molecule]) -> float:
    """``calccomofmolecules``: area-weighted centre of mass (plain mean if all areas are 0)."""
    areas = np.array([m.area for m in molecules], dtype=float)
    coms = np.array([m.com for m in molecules], dtype=float)
    if areas.sum() == 0:
        return float(coms.mean())
    return float(np.sum(coms * areas) / areas.sum())


@dataclass
class FitRange:
    """A group of molecules whose patterns overlap and are fitted together."""

    molecules: list[Molecule]
    minind: int
    maxind: int
    minmass: float
    maxmass: float
    com: float = np.nan
    resolution: float = np.nan
    resolutionerror: float = np.nan
    massoffset: float = np.nan
    massoffseterror: float = np.nan
    extra: dict = field(default_factory=dict)

    def update_com(self) -> None:
        """``calccomofranges`` for this range."""
        self.com = com_of_molecules(self.molecules)


def find_ranges(molecules: Sequence[Molecule], calibration: Calibration,
                searchrange: float) -> list[FitRange]:
    """``findranges``: group molecules (sorted by mass) into overlapping ranges.

    A molecule joins the current range when its lowest mass minus
    ``searchrange`` sigma reaches the range's highest mass plus
    ``searchrange`` sigma of the previous molecule. Each range gets the
    calibrated resolution and mass offset at its centre of mass.
    """
    ranges: list[FitRange] = []
    for i, mol in enumerate(molecules):
        if i > 0:
            r = ranges[-1]
            mass_minus = searchrange * float(calibration.sigma(mol.com))
            mass_plus = searchrange * float(calibration.sigma(molecules[i - 1].com))
            if mol.minmass - mass_minus <= r.maxmass + mass_plus:
                if r.maxind < mol.maxind:
                    r.maxind, r.maxmass = mol.maxind, mol.maxmass
                if r.minind > mol.minind:
                    r.minind, r.minmass = mol.minind, mol.minmass
                r.molecules.append(mol)
                continue
        ranges.append(FitRange([mol], mol.minind, mol.maxind, mol.minmass, mol.maxmass))
    for r in ranges:
        r.update_com()
        r.resolution = float(calibration.resolution(r.com))
        r.massoffset = float(calibration.massoffset(r.com))
    return ranges


def _sigma(com: float, resolution: float) -> float:
    return com / resolution / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def find_mass_range(massaxis: np.ndarray, molecules: Sequence[Molecule], resolution: float,
                    massoffset: float, searchrange: float) -> np.ndarray:
    """``findmassrange``: contiguous indices from the first to the last molecule ± searchrange·sigma."""
    sigma = _sigma(com_of_molecules(molecules), resolution)
    minind = mass2ind(massaxis, molecules[0].minmass + massoffset - searchrange * sigma)
    maxind = mass2ind(massaxis, molecules[-1].maxmass + massoffset + searchrange * sigma)
    return np.arange(minind, maxind + 1)


def find_mass_range2(massaxis: np.ndarray, molecules: Sequence[Molecule], resolution: float,
                     massoffset: float, searchrange: float) -> np.ndarray:
    """``findmassrange2``: indices within ±searchrange·sigma of any isotopologue peak."""
    massaxis = np.asarray(massaxis, dtype=float)
    sigma = _sigma(com_of_molecules(molecules), resolution)
    peaks = np.concatenate([m.peakdata[:, 0] for m in molecules]) + massoffset
    lo, hi = peaks - searchrange * sigma, peaks + searchrange * sigma
    k = increasing_from(massaxis)
    if k > massaxis.size // 2:  # no long increasing part: test every point
        mask = np.zeros(massaxis.size, dtype=bool)
        for a, b in zip(lo, hi):
            mask |= (massaxis >= a) & (massaxis <= b)
        return np.flatnonzero(mask)
    tail = massaxis[k:]
    i0 = np.searchsorted(tail, lo, side="left")
    i1 = np.searchsorted(tail, hi, side="right")
    keep = i1 > i0
    parts = []
    if k:  # the few points before the increasing part
        head = massaxis[:k]
        hmask = np.zeros(k, dtype=bool)
        for a, b in zip(lo, hi):
            hmask |= (head >= a) & (head <= b)
        parts.append(np.flatnonzero(hmask))
    if keep.any():
        i0, i1 = i0[keep], i1[keep]
        order = np.argsort(i0, kind="stable")
        i0, i1 = i0[order], np.maximum.accumulate(i1[order])
        # merge overlapping [i0, i1) intervals and list their indices
        new = np.concatenate([[True], i0[1:] > i1[:-1]])
        starts = i0[new]
        ends = i1[np.concatenate([np.flatnonzero(new)[1:] - 1, [i0.size - 1]])]
        parts += [np.arange(a, b) + k for a, b in zip(starts, ends)]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.intp)


def molecules_in_mass_range(molecules: Sequence[Molecule], minmass: float, maxmass: float,
                            calibration: Calibration, searchrange: float) -> np.ndarray:
    """``molecules_in_massrange_with_sigma``: indices of molecules overlapping [minmass, maxmass].

    Each molecule's extent is widened by ``searchrange`` sigma on both sides.
    """
    mins = np.array([m.minmass for m in molecules], dtype=float)
    maxs = np.array([m.maxmass for m in molecules], dtype=float)
    mins = mins - searchrange * calibration.sigma(mins)
    maxs = maxs + searchrange * calibration.sigma(maxs)
    inside = ((mins >= minmass) & (mins <= maxmass)) | ((maxs >= minmass) & (maxs <= maxmass)) \
        | ((mins <= minmass) & (maxs >= maxmass))
    return np.flatnonzero(inside)


def add_molecules(existing: Sequence[Molecule], new: Sequence[Molecule],
                  peakdata: np.ndarray) -> list[Molecule]:
    """Add molecules to a list without touching the existing ones (their areas are kept).

    New molecules whose name is already present are skipped; the others are
    prepared with :func:`init_molecules`. The result is sorted by lowest mass
    and renumbered (``rootindex``).
    """
    names = {m.name for m in existing}
    fresh = init_molecules([m for m in new if m.name not in names], peakdata)
    out = list(existing) + fresh
    out.sort(key=lambda m: m.minmass)
    for i, m in enumerate(out):
        m.rootindex = i
    return out
