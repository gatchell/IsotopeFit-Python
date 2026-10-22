import re

import numpy as np
import pytest

from isotopefit import load_ifd
from isotopefit.isotopes import (approx_masses, approx_p, available_elements, convolute,
                                 element_table, generate_clusters, parse_molecule,
                                 sum_formula_counts)

from conftest import refdata

# Abbreviations used in testfile.ifd (trimethylaluminium experiments).
ALIASES = {"TMAl": "AlC3H9", "DMAl - 1H": "AlC2H5", "TMAl - 1H": "AlC3H8",
           "DMAl + N": "AlC2H6N", "DMAl": "AlC2H6"}


def approx_masses_reference(d, minmassdistance):
    """Straightforward version: merge the closest pair, recompute every gap, repeat."""
    order = np.argsort(d[:, 0], kind="stable")
    mass, p = list(d[order, 0]), list(d[order, 1])
    while len(mass) > 1:
        gaps = np.diff(mass)
        ix = int(np.argmin(gaps))
        if gaps[ix] > minmassdistance:
            break
        psum = p[ix] + p[ix + 1]
        mass[ix] = (mass[ix] * p[ix] + mass[ix + 1] * p[ix + 1]) / psum
        p[ix] = psum
        del mass[ix + 1], p[ix + 1]
    return np.column_stack([mass, p])


def test_element_tables():
    assert {"H", "C", "O", "He", "Al", "D", "Cth"} <= set(available_elements())
    for el in ("H", "C", "O", "Ar", "Xe"):
        t = element_table(el)
        assert np.isclose(t[:, 1].sum(), 1.0)
    with pytest.raises(ValueError):
        element_table("Po")  # no stable isotopes; empty table in the MATLAB data
    tl = element_table("Tl")  # the MATLAB data had a copy of lead here
    np.testing.assert_allclose(tl, [[202.97234, 0.2952], [204.97443, 0.7048]], atol=1e-4)
    with pytest.raises(ValueError):
        element_table("Zz")


def test_convolute_order_matches_kron():
    a = np.array([[1.0, 0.5], [2.0, 0.5]])
    b = np.array([[10.0, 0.1], [20.0, 0.9]])
    np.testing.assert_allclose(convolute(a, b),
                               [[11, 0.05], [21, 0.45], [12, 0.05], [22, 0.45]])


@pytest.mark.parametrize("seed", range(20))
def test_approx_masses_equals_simple_loop(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(2, 300)
    d = np.column_stack([rng.uniform(0, 5, n), rng.uniform(0, 1, n)])
    for mmd in (1e-3, 1e-2, 0.05):
        np.testing.assert_array_equal(approx_masses(d, mmd), approx_masses_reference(d, mmd))


def test_approx_masses_updates_left_gap():
    # After merging 1.0 and 1.02 the merged peak sits at 1.01, so its gap to
    # 0.97 is 0.04 > 0.035 and the peaks stay apart. (MATLAB used the old gap
    # of 0.03 and merged once more.)
    d = np.array([[0.97, 1.0], [1.0, 1.0], [1.02, 1.0]])
    np.testing.assert_allclose(approx_masses(d, 0.035), [[0.97, 1.0], [1.01, 2.0]])


def test_approx_p():
    d = np.array([[1.0, 1.0], [2.0, 1e-4], [3.0, 1e-2]])
    np.testing.assert_array_equal(approx_p(d, 1e-3), d[[0, 2]])


def test_parse_molecule_water():
    d = parse_molecule("H2O", 1e-3, 1e-6)
    mono = d[np.argmax(d[:, 1])]
    assert np.isclose(mono[0], 2 * 1.0078250322 + 15.994914620)
    assert np.isclose(d[:, 1].sum(), 1.0)
    # Brackets and multipliers
    np.testing.assert_allclose(parse_molecule("(H2O)2", 1e-3, 1e-6)[0, 0], 2 * mono[0])


def test_sum_formula_counts():
    assert sum_formula_counts("(H2O)3Na") == {"H": 6, "O": 3, "Na": 1}
    assert sum_formula_counts("C60") == {"C": 60}
    assert sum_formula_counts("Al(CH3)3") == {"Al": 1, "C": 3, "H": 9}


def test_cluster_names_and_order():
    mols = generate_clusters(["He", "H2O"], [[1, 2, 3], [0, 1]], 1e-3, 1e-3)
    assert [m.name for m in mols] == ["[He]", "[He]2", "[He]3",
                                      "[He][H2O]", "[He]2[H2O]", "[He]3[H2O]"]
    mols = generate_clusters(["He"], [[0, 1, 2]], 1e-3, 1e-3, names=["X"], existing=["[X]"])
    assert [m.name for m in mols] == ["[X]2"]
    half = generate_clusters(["He"], [[2]], 1e-3, 1e-3, charge=2)[0]
    np.testing.assert_allclose(half.peakdata[0, 0], 4.0026032541)


def _from_name(name, mmd=1e-3, th=1e-3):
    parts = re.findall(r"\[([^\]]+)\](\d*)", name)
    formulas = [ALIASES.get(s, s) for s, _ in parts]
    numbers = [[int(k) if k else 1] for _, k in parts]
    return generate_clusters(formulas, numbers, mmd, th)[0].peakdata


@pytest.mark.parametrize("name", [
    "[He]", "[He]4", "[He]38", "[CH2][H]2", "[O][H]2", "[DMAl][N2]", "[DMAl + N]2",
    "[TMAl][DMAl + N]2", "[DMAl][N2][TMAl]2[H]", "[TMAl]4[DMAl - 1H][H]",
    "[TMAl]3[TMAl - 1H]2", "[TMAl]5[DMAl - 1H]",
])
def test_patterns_identical_to_matlab(name):
    """Patterns stored in testfile.ifd (made by the MATLAB cluster generator)."""
    project = load_ifd(refdata("testfile.ifd"), process=False)
    stored = {m.name: m.peakdata for m in project.molecules}[name]
    np.testing.assert_array_equal(_from_name(name), stored)


def test_all_helium_clusters_identical_to_matlab():
    project = load_ifd(refdata("testfile.ifd"), process=False)
    he = [m for m in project.molecules if re.fullmatch(r"\[He\]\d*", m.name)]
    assert len(he) == 150
    for m in he:
        np.testing.assert_array_equal(_from_name(m.name), m.peakdata)
