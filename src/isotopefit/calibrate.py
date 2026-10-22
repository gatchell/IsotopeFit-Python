"""Mass-offset and resolution calibration (the computations behind ``calibrate.m``).

In the MATLAB GUI the user picks calibration molecules, fits them, and
chooses how mass offset and resolution are interpolated over the mass
axis. Here the same steps are functions:

1. :func:`calibration_ranges` -- group the chosen molecules into ranges and
   give each range starting values (``findranges`` + ``addrangeparameters``).
2. :func:`fit_calibration_ranges` -- fit area, mass offset and resolution
   of each range (the "Fit all" / "Fit this" buttons).
3. :func:`calibration_from_ranges` -- turn the fitted ranges into a
   :class:`~isotopefit.model.Calibration` (``ranges2list`` +
   ``updatepolynomials``) and check it (``donecalib``).

:func:`calibrate` runs all three; :func:`calibrate_project` does it on a
project and recomputes its corrected spectrum.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .fitting import ProgressCallback, fit_ranges
from .model import Calibration, Molecule, Project
from .molecules import FitRange, find_ranges
from .settings import Settings
from .spectrum import crop, subtract_background

#: Freedom given to resolution (relative) and mass offset (absolute) by the
#: calibration window's fit buttons. The MATLAB comment says "per cent", but
#: 0.5 is used as a factor, i.e. +-50 % of the resolution.
CAL_DELTARES = 0.5
CAL_DELTAM = 0.5


def interpolate_parameters(calibration: Calibration, com: float) -> tuple[float, float]:
    """``parameterinterpolation``: starting mass offset and resolution at ``com``.

    PCHIP through the existing calibration points; constant beyond 0.1 mass
    units outside them; 0 and 3000 if there are no points yet.
    """
    c = calibration
    if c.comlist.size == 0:
        return 0.0, 3000.0
    if com <= c.comlist[0] - 0.1:
        return float(c.massoffsetlist[0]), float(c.resolutionlist[0])
    if com >= c.comlist[-1] + 0.1:
        return float(c.massoffsetlist[-1]), float(c.resolutionlist[-1])
    from .calibration import calibration_curve

    return (float(calibration_curve(c.comlist, c.massoffsetlist, 0, "pchip", com)),
            float(calibration_curve(c.comlist, c.resolutionlist, 0, "pchip", com)))


def select_molecules(molecules: Sequence[Molecule], names: Iterable[str] | None) -> list[int]:
    """Indices (sorted) of the molecules whose names are in ``names``."""
    wanted = set(names or [])
    missing = wanted - {m.name for m in molecules}
    if missing:
        warnings.warn(f"Calibration molecules not in the molecule list: {sorted(missing)}",
                      stacklevel=2)
    return [i for i, m in enumerate(molecules) if m.name in wanted]


def calibration_ranges(molecules: Sequence[Molecule], selected: Sequence[int],
                       calibration: Calibration, searchrange: float = 1.0) -> list[FitRange]:
    """Ranges of the selected molecules with interpolated starting parameters."""
    chosen = [molecules[i] for i in sorted(selected)]
    if not chosen:
        return []
    ranges = find_ranges(chosen, calibration, searchrange)
    for r in ranges:
        r.massoffset, r.resolution = interpolate_parameters(calibration, r.com)
    return ranges


def fit_calibration_ranges(peakdata: np.ndarray, ranges: Sequence[FitRange],
                           calibration: Calibration, method: str = "simplex_lin_combi",
                           fit_massoffset: bool = True, fit_resolution: bool = True,
                           deltares: float = CAL_DELTARES, deltam: float = CAL_DELTAM,
                           progress: ProgressCallback | None = None) -> list[FitRange]:
    """Fit areas, and within the bounds mass offset and resolution, of every range.

    ``peakdata`` must be background-corrected but *not* mass-corrected,
    because the fitted mass offsets are relative to the raw mass axis.
    Mass offset and resolution only move with a method that fits them
    (``simplex_lin_combi`` or ``simplex``).
    """
    return fit_ranges(peakdata, ranges, calibration, np.inf,
                      deltares if fit_resolution else 0.0,
                      deltam if fit_massoffset else 0.0, method, progress)


@dataclass
class CalibrationResult:
    calibration: Calibration
    molecules: list[Molecule]
    ranges: list[FitRange]
    problems: list[str] = field(default_factory=list)


def _method_param(method: str, values: np.ndarray, param: float, npoints: int) -> float:
    """``updatepolynomials``: the parameter that goes with a method."""
    m = method.lower()
    if m == "flat":
        return float(np.mean(values)) if values.size else float(param)
    if m == "polynomial":
        order = int(param)
        if order >= npoints:  # MATLAB: "polynomial order too high!"
            warnings.warn(f"Polynomial order {order} too high for {npoints} points; "
                          f"using {npoints - 1}.", stacklevel=3)
            order = npoints - 1
        return float(order)
    if m in ("spline", "pchip"):
        return 0.0
    return float(param)


def calibration_from_ranges(ranges: Sequence[FitRange], base: Calibration,
                            massaxis: np.ndarray | None = None,
                            massoffset_method: str | None = None, massoffset_param: float | None = None,
                            resolution_method: str | None = None, resolution_param: float | None = None,
                            ) -> tuple[Calibration, list[str]]:
    """Build a calibration from fitted ranges and check it on ``massaxis``.

    Methods and parameters default to those of ``base``. Returns the new
    calibration and a list of problems the MATLAB window would have warned
    about (negative resolution; a mass-offset curve steep enough to make
    the corrected mass axis non-monotonic).
    """
    cal = copy.deepcopy(base)
    cal.comlist = np.array([r.com for r in ranges], dtype=float)
    cal.massoffsetlist = np.array([r.massoffset for r in ranges], dtype=float)
    cal.resolutionlist = np.array([r.resolution for r in ranges], dtype=float)
    cal.namelist = [m.name for r in ranges for m in r.molecules]
    if massoffset_method is not None:
        cal.massoffset_method = massoffset_method
    if resolution_method is not None:
        cal.resolution_method = resolution_method
    n = cal.comlist.size
    cal.massoffset_param = _method_param(
        cal.massoffset_method, cal.massoffsetlist,
        cal.massoffset_param if massoffset_param is None else massoffset_param, n)
    cal.resolution_param = _method_param(
        cal.resolution_method, cal.resolutionlist,
        cal.resolution_param if resolution_param is None else resolution_param, n)

    problems = []
    if massaxis is not None and n:
        res = cal.resolution(massaxis)
        offset = cal.massoffset(massaxis)
        if np.any(res < 0):
            problems.append("Resolution becomes negative within the mass range; change the "
                            "method or add calibration molecules.")
        elif np.any(np.diff(massaxis) - np.diff(offset) < 0):
            problems.append("The mass-offset curve is steeper than the mass axis, so the "
                            "corrected mass axis would not be monotonic.")
    return cal, problems


def calibrate(peakdata: np.ndarray, molecules: Sequence[Molecule], calibration: Calibration,
              names: Iterable[str] | None = None, settings: Settings | None = None,
              fit: bool = True, fit_massoffset: bool = True, fit_resolution: bool = True,
              massoffset_method: str | None = None, massoffset_param: float | None = None,
              resolution_method: str | None = None, resolution_param: float | None = None,
              progress: ProgressCallback | None = None) -> CalibrationResult:
    """Calibrate on the molecules named in ``names`` (default: ``calibration.namelist``).

    ``peakdata`` is background-corrected but not mass-corrected. The
    fitted areas are written to copies of the calibration molecules; the
    other molecules are returned unchanged. With ``fit=False`` the ranges
    keep their interpolated starting values (useful to re-evaluate a stored
    calibration with a different interpolation method).
    """
    settings = settings or Settings()
    names = calibration.namelist if names is None else list(names)
    mols = copy.deepcopy(list(molecules))
    selected = select_molecules(mols, names)
    ranges = calibration_ranges(mols, selected, calibration, settings.searchrange)
    if fit and ranges:
        ranges = fit_calibration_ranges(peakdata, ranges, calibration, settings.fittingmethod_cal,
                                        fit_massoffset, fit_resolution, progress=progress)
        # updatemolecules: ranges keep the molecules in the order they were selected.
        fitted = [m for r in ranges for m in r.molecules]
        for i, fm in zip(sorted(selected), fitted):
            mols[i].area, mols[i].areaerror = fm.area, fm.areaerror
    cal, problems = calibration_from_ranges(ranges, calibration, peakdata[:, 0],
                                            massoffset_method, massoffset_param,
                                            resolution_method, resolution_param)
    for p in problems:
        warnings.warn(p, stacklevel=2)
    return CalibrationResult(cal, mols, ranges, problems)


def calibrate_project(project: Project, names: Iterable[str] | None = None,
                      settings: Settings | None = None, **kwargs) -> CalibrationResult:
    """Calibrate a project (menu "Calibration") and apply the result.

    Uses the cropped, background-corrected spectrum, stores the new
    calibration and molecule areas in the project and recomputes
    ``project.peakdata`` with the new mass offset. Keyword arguments go to
    :func:`calibrate`.
    """
    peakdata = subtract_background(crop(project.raw_peakdata, project.start, project.stop),
                                   project.bgcorrection)
    result = calibrate(peakdata, project.molecules, project.calibration, names, settings, **kwargs)
    project.calibration = result.calibration
    project.molecules = result.molecules
    project.update_peakdata()
    project.gui_status.update(calibrated=True, fitted=False, changed=True)
    return result


def load_calibration(project: Project, path) -> Project:
    """Menu "Load calibration": take background, molecules and calibration from an ``.ifd`` file.

    The molecules are placed in this project's spectrum again (out-of-range
    ones dropped, indices and area guesses recomputed). The cropping of this
    project is kept. Port of ``menuloadcalibration`` in ``IsotopeFit.m``.
    """
    from .io.ifd import load_ifd
    from .molecules import init_molecules

    source = load_ifd(path, process=False)
    if project.peakdata is None:
        project.update_peakdata()
    if source.raw_peakdata[-1, 0] < project.peakdata[-1, 0]:
        warnings.warn("This spectrum reaches higher masses than the one the molecules were "
                      "loaded for; molecules at higher masses may be missing.", stacklevel=2)
    project.bgcorrection = source.bgcorrection
    project.molecules = init_molecules(source.molecules, project.peakdata)
    project.calibration = source.calibration
    project.update_peakdata()
    project.gui_status.update(molecules_loaded=True, calibrated=True, changed=True)
    return project


CALIBRATION_METHODS = ("Flat", "Polynomial", "Spline", "PCHIP")


class CalibrationSession:
    """State of the calibration window (``calibrate.m``), without any GUI.

    Holds the calibration molecules grouped into ranges, each with its own
    mass offset and resolution. Every change updates the calibration points
    (``ranges2list``), so ranges added later start from values interpolated
    between the points fitted so far.

    ``peakdata`` is background-corrected but not mass-corrected.
    """

    def __init__(self, peakdata: np.ndarray, molecules: Sequence[Molecule],
                 calibration: Calibration, settings: Settings | None = None):
        self.peakdata = np.asarray(peakdata, dtype=float)
        self.molecules = copy.deepcopy(list(molecules))
        self.base = copy.deepcopy(calibration)
        self.settings = settings or Settings()
        self.massoffset_method = calibration.massoffset_method
        self.massoffset_param = calibration.massoffset_param
        self.resolution_method = calibration.resolution_method
        self.resolution_param = calibration.resolution_param
        # Points used for interpolating starting values; start with the stored ones.
        self._points = copy.deepcopy(calibration)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.selected = select_molecules(self.molecules, calibration.namelist)
        self.ranges: list[FitRange] = []
        self._rebuild()

    # ------------------------------------------------------------------ selection

    def _rebuild(self) -> None:
        """findranges + addrangeparameters + ranges2list (after adding or removing)."""
        self.ranges = calibration_ranges(self.molecules, self.selected, self._points,
                                         self.settings.searchrange)
        self._update_points()

    def _update_points(self) -> None:
        self._points.comlist = np.array([r.com for r in self.ranges], dtype=float)
        self._points.massoffsetlist = np.array([r.massoffset for r in self.ranges], dtype=float)
        self._points.resolutionlist = np.array([r.resolution for r in self.ranges], dtype=float)

    def add(self, indices: Iterable[int]) -> None:
        self.selected = sorted(set(self.selected) | set(int(i) for i in indices))
        self._rebuild()

    def remove(self, indices: Iterable[int]) -> None:
        self.selected = sorted(set(self.selected) - set(int(i) for i in indices))
        self._rebuild()

    def range_of(self, molecule_index: int) -> int | None:
        """Index of the range holding a molecule (``memberofrange``)."""
        name = self.molecules[molecule_index].name
        for i, r in enumerate(self.ranges):
            if any(m.name == name for m in r.molecules):
                return i
        return None

    def molecule_index(self, range_index: int, position: int) -> int:
        """Position in the molecule list of the n-th molecule of a range."""
        name = self.ranges[range_index].molecules[position].name
        return next(i for i in self.selected if self.molecules[i].name == name)

    # ------------------------------------------------------------------ editing and fitting

    def _write_back(self, ranges: Sequence[FitRange]) -> None:
        """updatemolecules: copy fitted areas into the molecule list."""
        by_name = {m.name: m for r in ranges for m in r.molecules}
        for i in self.selected:
            m = by_name.get(self.molecules[i].name)
            if m is not None:
                self.molecules[i].area, self.molecules[i].areaerror = m.area, m.areaerror

    def fit(self, range_indices: Iterable[int] | None = None, fit_massoffset: bool = True,
            fit_resolution: bool = True, progress: ProgressCallback | None = None) -> None:
        """"Fit this" / "Fit all": the calibration fitting method within the bounds."""
        idx = list(range(len(self.ranges))) if range_indices is None else list(range_indices)
        fitted = fit_calibration_ranges(self.peakdata, [self.ranges[i] for i in idx], self._points,
                                        self.settings.fittingmethod_cal, fit_massoffset,
                                        fit_resolution, progress=progress)
        for i, r in zip(idx, fitted):
            self.ranges[i] = r
        self._write_back(fitted)
        self._update_points()

    def guess_area(self, range_index: int) -> None:
        """"Guess area": areas only, by linear least squares, keeping offset and resolution."""
        fitted = fit_ranges(self.peakdata, [self.ranges[range_index]], self._points, np.inf,
                            0.0, 0.0, "linear_system")
        self.ranges[range_index] = fitted[0]
        self._write_back(fitted)
        self._update_points()

    def set_range_parameters(self, range_index: int, massoffset: float | None = None,
                             resolution: float | None = None) -> None:
        r = self.ranges[range_index]
        if massoffset is not None:
            r.massoffset = float(massoffset)
        if resolution is not None:
            r.resolution = float(resolution)
        self._update_points()

    def set_area(self, range_index: int, position: int, area: float) -> None:
        r = self.ranges[range_index]
        r.molecules[position].area = float(area)
        self.molecules[self.molecule_index(range_index, position)].area = float(area)
        r.update_com()  # calccomofranges
        self._update_points()

    def step(self, range_index: int, what: str, direction: int) -> None:
        """The arrow buttons: mass offset by 0.1 sigma, resolution by 5 %."""
        r = self.ranges[range_index]
        if what == "massoffset":
            r.massoffset += direction * 0.1 * float(self._points_calibration().sigma(r.com))
        elif what == "resolution":
            r.resolution *= 1 + direction * 0.05
        else:
            raise ValueError(what)
        self._update_points()

    # ------------------------------------------------------------------ results

    def _points_calibration(self) -> Calibration:
        cal, _ = calibration_from_ranges(self.ranges, self.base, None,
                                         self.massoffset_method, self.massoffset_param,
                                         self.resolution_method, self.resolution_param)
        return cal

    def result(self) -> tuple[Calibration, list[str]]:
        """The calibration as it would be saved, and the problems to warn about."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return calibration_from_ranges(self.ranges, self.base, self.peakdata[:, 0],
                                           self.massoffset_method, self.massoffset_param,
                                           self.resolution_method, self.resolution_param)

    def preview(self, range_index: int, width: float = 6.0):
        """Data and calculated signal around a range (raw mass axis).

        Returns ``(x, data, total, per_molecule)``; ``width`` is in sigma.
        """
        from .pattern import pattern

        r = self.ranges[range_index]
        sigma = r.com / r.resolution / (2 * np.sqrt(2 * np.log(2)))
        lo = r.minmass + r.massoffset - width * sigma
        hi = r.maxmass + r.massoffset + width * sigma
        sel = (self.peakdata[:, 0] >= lo) & (self.peakdata[:, 0] <= hi)
        x = self.peakdata[sel, 0]
        parts = [pattern(m, m.area, r.resolution, r.massoffset, x, self.base.shape)
                 for m in r.molecules]
        total = np.sum(parts, axis=0) if parts else np.zeros_like(x)
        return x, self.peakdata[sel, 1], total, parts
