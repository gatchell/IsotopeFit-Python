from pathlib import Path

import numpy as np
import pytest

from isotopefit.clusters import ClusterNode, families, generate, load_icg, parse_n, save_icg
from isotopefit.isotopes import generate_clusters


def test_parse_n_matlab_syntax():
    assert parse_n("1:5") == [1, 2, 3, 4, 5]
    assert parse_n("0:2:6") == [0, 2, 4, 6]
    assert parse_n("[0 1 2]") == [0, 1, 2]
    assert parse_n("[1,2,4]") == [1, 2, 4]
    assert parse_n("7") == [7]
    assert parse_n("[0:2, 5, 2]") == [0, 1, 2, 5]
    for bad in ("", "a:b", "1:2:3:4", "-1:2", "[]", "1:0:5"):
        with pytest.raises(ValueError):
            parse_n(bad)


def tree():
    he = ClusterNode("He", n="1:20", threshold=1e-3, mass_accuracy=1e-3)
    he.children = [ClusterNode("H2O", name="W", n="[0 1]"), ClusterNode("Ar", n="1")]
    c60 = ClusterNode("C60", n="1:2", charge=2, threshold=1e-2)
    return [he, c60]


def test_families_are_paths_to_leaves():
    fams = families(tree())
    assert [f.names for f in fams] == [["He", "W"], ["He", "Ar"], ["C60"]]
    assert fams[0].nlists == [list(range(1, 21)), [0, 1]]
    assert fams[2].charge == 2 and fams[2].threshold == 1e-2
    assert fams[1].threshold == 1e-3  # children use the top-level settings
    assert fams[0].count() == 40


def test_generate_matches_generate_clusters_and_skips_duplicates():
    mols = generate(tree())
    names = [m.name for m in mols]
    assert len(names) == len(set(names))
    # [He]n without water comes from the first family only
    assert names[:3] == ["[He]", "[He]2", "[He]3"]
    assert "[He]5[Ar]" in names and "[He]5[W]" in names
    ref = generate_clusters(["He", "H2O"], [range(1, 21), [0, 1]], 1e-3, 1e-3, ["He", "W"])
    by = {m.name: m for m in mols}
    for m in ref:
        np.testing.assert_array_equal(by[m.name].peakdata, m.peakdata)
    c60 = by["[C60]2"].peakdata
    assert c60[0, 0] == pytest.approx(720.0, abs=0.5)  # charge 2: 1440 / 2


def test_icg_round_trip(tmp_path):
    path = tmp_path / "def.icg"
    save_icg(tree(), path)
    again = load_icg(path)
    assert [n.label for n in again] == ["He", "C60"]
    assert [c.label for c in again[0].children] == ["W", "Ar"]
    assert again[0].children[0].formula == "H2O" and again[0].children[0].n == "[0 1]"
    assert again[1].charge == 2 and again[1].threshold == pytest.approx(1e-2)
    assert [f.names for f in families(again)] == [f.names for f in families(tree())]


def test_add_molecules_keeps_existing_areas():
    from isotopefit import load_ifd
    from isotopefit.molecules import add_molecules

    from conftest import refdata

    p = load_ifd(refdata("testfile.ifd"))
    areas = {m.name: m.area for m in p.molecules}
    new = generate([ClusterNode("Ar", n="1:3"), ClusterNode("He", n="1:2")])
    out = add_molecules(p.molecules, new, p.peakdata)
    names = [m.name for m in out]
    assert len(names) == len(set(names))
    assert "[Ar]2" in names and names.count("[He]") == 1  # [He] existed already
    assert all(m.area == areas[m.name] for m in out if m.name in areas)
    assert [m.rootindex for m in out] == list(range(len(out)))
    assert all(a.minmass <= b.minmass for a, b in zip(out, out[1:]))


MATLAB_ICG = Path(__file__).parent / "data" / "Camphor.icg"


def test_load_matlab_icg(tmp_path):
    """A definition saved by the MATLAB cluster generator (2018)."""
    roots = load_icg(MATLAB_ICG)
    assert [r.label for r in roots] == ["He", "O", "Camphor", "N", "CO2"]
    camphor = roots[2]
    assert (camphor.formula, camphor.n) == ("C10H16O", "0:50")
    assert [(c.label, c.formula, c.n) for c in camphor.children] == [
        ("H", "H", "0:10"), ("Camphor-H", "C10H15O", "0:10")]
    assert roots[1].children[0].n == "0:2"  # single child stored as a struct, not an array
    fams = families(roots)
    assert [f.names for f in fams] == [["He"], ["O", "H"], ["Camphor", "H"],
                                       ["Camphor", "Camphor-H"], ["N"], ["CO2"]]
    assert all((f.charge, f.threshold, f.mass_accuracy) == (1, 5e-4, 1e-3) for f in fams)
    mols = generate(roots)
    names = [m.name for m in mols]
    # the pure camphor clusters of the second camphor family are not generated twice
    assert len(names) == len(set(names)) == 200 + 6 + 560 + 510 + 2 + 1
    save_icg(roots, tmp_path / "again.icg")
    assert load_icg(tmp_path / "again.icg") == roots
