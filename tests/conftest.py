"""Shared test helpers.

Reference data comes from the earlier C# port (IsotopeFitSharp), whose test
files were exported from MATLAB/Octave, from IsotopeFitPySharp, and from the
peak shape generator exports in IsotopeFit_Rewrite/PeakShapeFitting. By
default they are looked up next to this project in the Nextcloud folder;
set ISOTOPEFIT_TESTDATA to a folder holding the same files to override.
Tests whose files are missing are skipped, not failed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [
    Path(os.environ["ISOTOPEFIT_TESTDATA"]) if "ISOTOPEFIT_TESTDATA" in os.environ else None,
    ROOT / "tests" / "data",  # files that come with the repository (testfile.ifd, Camphor.icg)
    ROOT.parent / "IsotopeFitSharp" / "IsotopeFitLib.Tests" / "TestData",
    ROOT.parent / "IsotopeFitPySharp" / "IfdTestFiles",
    ROOT.parent / "IsotopeFit_Rewrite" / "PeakShapeFitting",  # peak shape generator exports
]


def refdata(name: str) -> Path:
    """Path of a reference file, or skip the test if it isn't available."""
    for d in SEARCH_DIRS:
        if d is not None and (d / name).is_file():
            return d / name
    pytest.skip(f"reference file {name} not found")


def octave_matrix(name: str) -> np.ndarray:
    """Load a matrix saved in Octave's text format."""
    return np.loadtxt(refdata(name), comments="#")
