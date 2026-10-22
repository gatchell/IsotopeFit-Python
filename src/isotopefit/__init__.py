"""IsotopeFit: evaluation of mass spectra with a focus on cluster series.

Python port of the MATLAB program by S. Ralser, J. Postler and M. Harnisch
(University of Innsbruck). See Ralser et al., Int. J. Mass Spectrom. (2015),
doi:10.1016/j.ijms.2015.01.004.
"""

from .batch import batch_fit, fit_global, fit_project_global
from .background import correct_background, find_background
from .calibrate import calibrate, calibrate_project, load_calibration
from .fitting import fit_project, fit_with_calibration
from .io import load_h5, load_ifd, load_ifm, load_molecule_folder, save_ifd, save_ifm
from .isotopes import generate_clusters, parse_molecule
from .model import BackgroundCorrection, Calibration, Molecule, PeakShape, Project
from .molecules import find_ranges, init_molecules
from .settings import Settings

__version__ = "0.1.0.dev0"

__all__ = [
    "BackgroundCorrection",
    "Calibration",
    "Molecule",
    "PeakShape",
    "Project",
    "Settings",
    "batch_fit",
    "calibrate",
    "calibrate_project",
    "correct_background",
    "find_background",
    "find_ranges",
    "fit_global",
    "fit_project",
    "fit_project_global",
    "fit_with_calibration",
    "generate_clusters",
    "init_molecules",
    "load_h5",
    "load_ifd",
    "load_calibration",
    "load_ifm",
    "load_molecule_folder",
    "parse_molecule",
    "save_ifd",
    "save_ifm",
]
