import copy

import numpy as np
import pytest

from isotopefit import Settings, load_ifd
from isotopefit.series import _split, cluster_series, export_series
from isotopefit.view import (calculated_spectrum, export_spectrum, fit_calibration,
                             fit_molecule_subset, involved_molecules, molecules_in_view)

from conftest import refdata


@pytest.fixture(scope="module")
def project():
    return load_ifd(refdata("testfile.ifd"))


def test_split_names_like_matlab():
    assert _split("[He]12[H2O]", "[He]") == (12, "[H2O][")
    assert _split("[He][H2O]", "[He]") == (1, "[H2O][")
    assert _split("[He]", "[He]") == (1, "[")
    assert _split("[Ar]2", "[He]") == (0, "[Ar]2[")


def test_cluster_series(project, tmp_path):
    series = cluster_series(project.molecules, "He", project.peakdata)
    he = [s for s in series if s.name == "[He]n"][0]
    assert len(he.sizes) == 150 and np.all(np.diff(he.sizes) > 0)
    m = project.molecules[he.indices[0]]
    assert m.name == ("[He]" if he.sizes[0] == 1 else f"[He]{he.sizes[0]}")
    # areas are divided by the local bin width (> 0), so they are larger than the raw areas
    assert np.all(he.areas >= np.array([project.molecules[i].area for i in he.indices]))
    path = tmp_path / "series.txt"
    export_series([he], path)
    lines = path.read_text().splitlines()
    assert lines[0] == "\t[He]n\t(error)\t"
    assert lines[1].startswith("0\t--\t--")  # no He0
    assert len(lines) == 1 + he.sizes.max() + 1


def test_calculated_spectrum_matches_data_where_fitted(project):
    x, y = calculated_spectrum(project, 150, 160)
    data = project.peakdata[(project.peakdata[:, 0] >= 150) & (project.peakdata[:, 0] <= 160), 1]
    assert x.size == data.size
    # A good fit explains most of the signal
    assert np.sum((data - y) ** 2) < 0.2 * np.sum(data ** 2)


def test_involved_and_subset_fit(project):
    cal = fit_calibration(project)
    assert list(involved_molecules(project.molecules, [100], cal, 1.0)) == [100]
    blocks = involved_molecules(project.molecules, [100, 101, 200], cal, 1.0)
    assert {100, 101, 200} <= set(blocks.tolist())
    m = project.molecules[100]
    inview = molecules_in_view(project, m.com - 0.5, m.com + 0.5)
    assert 100 in inview
    work = copy.deepcopy(project)
    idx = fit_molecule_subset(work, [100], Settings())
    assert list(idx) == [100]
    assert work.molecules[100].area == pytest.approx(project.molecules[100].area, rel=1e-6)


def test_export_spectrum(project, tmp_path):
    path = tmp_path / "spec.txt"
    export_spectrum(project, path, with_fit=True)
    data = np.loadtxt(path, skiprows=1)
    assert data.shape == (project.peakdata.shape[0], 3)


def test_parse_goto():
    from isotopefit import Molecule
    from isotopefit.view import parse_goto

    mols = [Molecule(name=n, peakdata=[[1.0, 1.0]]) for n in ("[He]", "[He]4", "[TMAl][H]")]
    assert parse_goto("300-320", mols) == ("range", 300.0, 320.0)
    assert parse_goto("320 – 300", mols) == ("range", 300.0, 320.0)
    assert parse_goto("300 to 320 u", mols) == ("range", 300.0, 320.0)
    kind, lo, hi = parse_goto("300", mols)
    assert kind == "range" and lo < 300 < hi and hi - lo <= 4
    assert parse_goto("He", mols) == ("molecule", 0)  # exact, brackets optional
    assert parse_goto("[he]4", mols) == ("molecule", 1)
    assert parse_goto("tmal", mols) == ("molecule", 2)  # part of a name
    assert parse_goto("xyz", mols) is None and parse_goto("  ", mols) is None


def test_species_in_names():
    from isotopefit import Molecule
    from isotopefit.series import species_in

    mols = [Molecule(name=n, peakdata=[[1.0, 1.0]])
            for n in ("[He]", "[He]2", "[He]2[H2O]", "[TMAl]", "[TMAl - 1H][He]")]
    assert species_in(mols) == ["He", "H2O", "TMAl", "TMAl - 1H"]  # ties by name
