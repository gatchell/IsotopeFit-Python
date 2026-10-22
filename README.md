# IsotopeFit (Python)

A pure-Python port of [IsotopeFit](https://github.com/nano-bio/IsotopeFit), the MATLAB
program by Stefan Ralser, Johannes Postler and Martina Harnisch (University of
Innsbruck) for evaluating mass spectra, especially for extracting cluster series.
Theory: Ralser et al., *Int. J. Mass Spectrom.* (2015), doi:10.1016/j.ijms.2015.01.004.

The MATLAB original sits next to this folder in `../IsotopeFit-master` and is the reference
for behaviour.

## Setup

Requires Python 3.10 or newer.

```bash
cd IsotopeFit_Python
python3 -m venv ~/.venvs/isotopefit      # keep the environment out of Nextcloud
source ~/.venvs/isotopefit/bin/activate        # fish: source ~/.venvs/isotopefit/bin/activate.fish
pip install -e ".[dev]"
pytest
```

The environment goes outside the synced folder because a virtual environment holds
thousands of files that Nextcloud would otherwise sync.

## The GUI

```bash
pip install -e ".[gui,dev]"      # adds PySide6 and pyqtgraph
isotopefit-gui                   # or: isotopefit-gui measurement.ifd
```

The main window shows the spectrum with the fitted curve (red) and the selected molecule's
pattern (blue), the molecule list with a filter, and below the spectrum the cluster series: pick a species
from the list (it offers the species found in the molecule names, most common first) or type one
and press Enter; select one or more series to plot them together, and click a point to jump to
that molecule. Fit with the
Fit box under the molecule list: choose which molecules (selected in the list, all, or those in
the current view) and the method (overlapping groups together, or one molecule at a time); the
button says what it will fit, and fits run in the background. Menus: open `.ifd`/`.h5`, save, load molecules, background correction, load a
calibration, settings, and the exports.

Above the spectrum: Whole spectrum, Zoom in/out, ◀ ▶ to step through the masses, Fit height (fit
the vertical axis to what is visible), a Go to box (a mass such as `300`, a range such as
`300-320`, or a molecule name such as `[He]4`), checkboxes that show or hide the data, fit and
selected-molecule curves (they double as the legend), Log scale, Box zoom (drag a box instead of
moving the view) and Save image (PNG or SVG). The mass and signal under the mouse are shown below
the plot. Above the cluster series: error bars on/off, Log scale, Fit view and Export. The mouse
still works as before: scroll to zoom, drag to move, right-click for more. The cursor shows what a
drag will do: an open hand moves the view, a magnifying glass draws a zoom box, a pointing hand
marks points you can click, and a move cursor marks peak-shape knots you can drag.

*Molecules → Cluster generator* builds molecule lists: add a species (sum formula, optional short
name, multiplicities such as `1:50` or `[0 1]`), add further species to it to make mixed clusters,
and press Generate. Every path from a top-level species to a leaf in the tree is one family, as in
the MATLAB cluster generator. Charge, intensity threshold and mass accuracy are set on the
top-level species. Save the molecules as `.ifm`, or use them in (or add them to) the open
project; definitions are saved as `.icg` files compatible with the MATLAB version.

*Calibration → Mass and resolution calibration* opens the calibration window: add molecules from
the left list, pick a range, fit it ("Fit this", "Fit all") or adjust mass offset, resolution and
areas by hand with the arrow buttons, and choose how the points are interpolated (Flat,
Polynomial, Spline, PCHIP); click a point in the two lower plots to jump to that range.
*Calibration → Edit peak shape* measures the peak shape from the peaks in the current view: drag
knots, right-click to add or remove them, fix knots in the table, and "Fit"; "Use this peak shape" normalises the
shape and rescales the resolution.

*Calibration → Drift correction* reads the individual spectra (writes, or the buffers of the
first write) from the project's `.h5` file, finds how far the peaks move during the measurement in
a few mass divisions, and sums the spectra again with each one moved back. Press Analyse, look at
the positions over time and against mass (slider: spectrum), press "Correct shift" to see the
corrected spectrum next to the original, and "Use corrected spectrum". Background and mass correction are then
redone; fit again afterwards. `tools/inspect_drift.py file.h5` prints the same analysis in the
terminal.

*Data → Fit all molecules at once* fits the whole spectrum as one problem, as MATLAB's
`batch_fit` does; *Data → Batch fit a folder* does that for every `.ifd` file in a folder and saves
them (overwriting, or into another folder). See "Global fit and batch fitting" below.

The calibration windows (background, mass and resolution, peak shape, drift) say on their
buttons and in a line next to them what applying will change. The calibration and peak-shape
windows ask before discarding edits, whether you click "Discard changes", press Escape or close
the window.

The window layouts are Qt Designer files in `src/isotopefit/gui/ui/`. Open them with
`pyside6-designer` (installed with PySide6) and rearrange freely; keep the object names, because
the code looks widgets up by name. Plots go into the empty placeholder widgets named `...PlotHost`.

## Mac app

```bash
bash packaging/build_mac.sh            # builds dist/IsotopeFit.app
bash packaging/build_mac.sh --install  # ... and copies it to /Applications
```

Run it from the project folder with the virtual environment active (it uses that Python and
installs PyInstaller into it). The app contains Python and all libraries, so it runs on Macs
without any of this installed, as long as they have the same processor type (Apple silicon or
Intel) as the Mac that built it. In the app, the menus are in the macOS menu bar; double-clicking
an `.ifd` file in Finder opens it (use "Open With" the first time if another program claims
`.ifd`). Rebuild the app after changing the code. The app is not signed with a developer
certificate, so on another Mac the first start needs right-click → Open.

## Windows program and GitHub builds

GitHub builds the Windows program and the macOS app on its own machines
(`.github/workflows/build.yml`), after running the tests:

* After every push, the builds can be downloaded from the **Actions** tab: open the latest run
  and download `IsotopeFit-Windows` or `IsotopeFit-macOS` under *Artifacts*.
* To publish a version, tag it and push the tag, e.g. `git tag v0.1.0 && git push --tags`.
  The zipped builds are then attached to a release on the repository's **Releases** page.

On Windows, unzip `IsotopeFit-windows.zip` and run `IsotopeFit\IsotopeFit.exe`. The program is
not signed, so the first start shows "Windows protected your PC": click *More info* → *Run
anyway*. The macOS build is for Apple silicon Macs; build on an Intel Mac with
`packaging/build_mac.sh` if needed.

To build the Windows program on a Windows PC yourself: `pip install -e ".[gui]" pyinstaller
pillow`, then `python -m PyInstaller packaging/IsotopeFit.spec`; the program is in
`dist\IsotopeFit`.

## Quick use

```python
from isotopefit import load_ifd, load_h5, save_ifd

project = load_ifd("measurement.ifd")       # old MATLAB project files load directly
project.peakdata                            # cropped, background-subtracted, mass-corrected
project.molecules[0].name, project.calibration.resolution([100.0])

raw = load_h5("DataFile.h5", instrument="clustof")   # or "surftof"
save_ifd(project, "copy.ifd")               # readable by the MATLAB version
```

Fitting, and building a molecule list:

```python
from isotopefit import Settings, fit_project, generate_clusters, init_molecules

fit_project(project, Settings())                     # like "Fit" -> "All" in the GUI
fit_project(project, Settings(), mode="molecules")   # the GUI's second fit mode
[(m.name, m.area, m.areaerror) for m in project.molecules[:5]]

# He1..He50, each also with one water
mols = generate_clusters(["He", "H2O"], [range(1, 51), [0, 1]], minmassdistance=1e-3, th=1e-3)
project.molecules = init_molecules(mols, project.peakdata)
```

From a raw spectrum to areas:

```python
from isotopefit import (Project, Settings, calibrate_project, correct_background,
                        fit_project, load_calibration, load_h5)

project = load_h5("DataFile.h5")
correct_background(project, ndiv=50, percent=40)
load_calibration(project, "earlier_measurement.ifd")  # molecules, peak shape, calibration
result = calibrate_project(project, settings=Settings())  # refit on the stored calibration molecules
print(result.problems)                                    # warnings the MATLAB window would show
fit_project(project, Settings())
```

Calibration is interactive in MATLAB, and the hard part, choosing molecules and starting values
that put each fit on the right peak, still needs a person (and later the GUI). Starting from
nothing, the fits can lock onto a neighbouring peak one mass unit away. Starting from an existing
calibration, as above, works reliably.

## Global fit and batch fitting

```bash
isotopefit-batch path/to/folder                 # fits every .ifd in the folder and overwrites it
isotopefit-batch path/to/folder --out fitted/   # writes the fitted files to fitted/ instead
```

```python
from isotopefit import fit_project_global
result = fit_project_global(project)        # areas and errors go into project.molecules
```

This is MATLAB's `batch_fit.m`, which differs from the GUI's fit:

* All molecules are fitted together in one non-negative least-squares problem, instead of range
  by range or molecule by molecule. Resolution and mass offset are not adjusted.
* Each isotope peak gets the width belonging to its own mass (mass / R(mass)); the GUI fit uses
  the molecule's centre of mass for all its peaks.
* Only points within ±½ FWHM of a peak are fitted (× the search range setting in the GUI).
* The error is a sandwich estimate from the residuals, √diag((MᵀM)⁻¹ Mᵀ diag(r²) M (MᵀM)⁻¹),
  instead of the GUI fit's √((MᵀM)⁻¹ · RSS / (n − m)). Both are one standard deviation.

MATLAB solves one large sparse system. The port splits it into the groups of molecules that
share data points and solves each group separately, which gives the same result (checked to
10⁻¹² against one big solve on `testfile.ifd`) and takes about a second for 384 molecules.
On `testfile.ifd` the total area comes out within 0.6 % of the stored molecule-by-molecule fit.
MATLAB's batch fit saved its internal state rather than a regular `.ifd`; the port saves a
normal `.ifd`.

## Layout

```
src/isotopefit/
  model.py        data classes: Project, Molecule, Calibration, BackgroundCorrection, PeakShape
  calibration.py  mass offset / resolution curves (getcalibrationdata.m)
  spectrum.py     crop, background subtraction, mass-axis correction, small helpers
  _matlab.py      exact MATLAB behaviour where NumPy differs (a:d:b, interp1 sorting)
  isotopes.py     isotope patterns and cluster generation (IsotopeDistribution/)
  pattern.py      peak shape scaling and calculated spectra (pattern_func.m, multispec*.m)
  molecules.py    placing molecules in a spectrum, fit ranges (init_molecule_properties.m, findranges.m)
  fitting.py      fit methods and entry points (fitwithcalibration.m, fitranges.m, fitmolecules.m)
  drift.py        drift correction from the individual spectra in the .h5 file (driftcorrection.m)
  batch.py        global fit of all molecules, batch fitting of a folder (batch_fit.m)
  background.py   background estimation (find_bg.m, bg_correction.m)
  calibrate.py    mass-offset and resolution calibration (calibrate.m without the window)
  peakshape.py    measuring the peak shape from the spectrum (peak_shape_generator.m without the window)
  clusters.py     cluster definitions (tree of species), .icg files
  series.py       cluster series [X]n and their export (sortmolecules in IsotopeFit.m)
  view.py         calculated spectra, fitting parts of a spectrum, spectrum export
  settings.py     program settings (config.txt)
  gui/            PySide6 + pyqtgraph user interface; layouts in gui/ui/*.ui
  io/ifd.py       .ifd project files, load and save (load_ifd.m, save_file)
  io/molecules.py .ifm molecule files and folders of pattern text files
  io/h5.py        TOFWERK (CLUSTOF) and IoniTOF (SURFTOF) HDF5 spectra (load_h5, h5cal.m)
  data/           default Gaussian peak shape and the element isotope tables, copied from MATLAB
packaging/        macOS app and Windows program: PyInstaller spec, Mac build script, icon
.github/          GitHub Actions: tests and builds for Windows and macOS
tests/            checks against reference data from the earlier C# port
```

## Conventions

* **0-based indices everywhere.** Conversion to MATLAB's 1-based indices happens only when
  reading or writing `.ifd` files.
* **No GUI code in the library.** Where MATLAB opens dialogs or wait bars in the middle of a
  computation, the Python functions take arguments or report progress through callbacks.
* **Faithful first, fixes second.** Numerical behaviour follows MATLAB, including its quirks,
  so results can be compared. Deliberate deviations are listed below.

## Reference tests

The tests look for reference files in `../IsotopeFitSharp/IsotopeFitLib.Tests/TestData` and
`../IsotopeFitPySharp/IfdTestFiles` (override with `ISOTOPEFIT_TESTDATA`). They were
exported from MATLAB/Octave for the earlier C# port. Missing files mean skipped tests.

Verified so far against `testfile.ifd`:

| Step | Agreement |
|---|---|
| Background subtraction (416 375 points) | within 5·10⁻¹⁰ of signals up to 2.5·10⁶ |
| Mass-axis correction (PCHIP offset curve), inside the calibrated masses | within 2·10⁻¹¹ |
| Resolution curve: PCHIP, spline, polynomial | to rounding error (~10⁻¹⁶ relative) |
| Isotope patterns of the 371 stored molecules the generator can rebuild | 224 bit-for-bit identical with the default settings, 127 more with other thresholds |
| Molecule-by-molecule fit, 384 molecules | 326 of the 371 non-zero areas identical to ~10⁻¹³; total area within 10⁻⁴ (errors differ on purpose, see below) |
| Fit by ranges (GUI default) vs stored areas | median difference 0.07 % |
| Background estimation (50 points) | identical |
| Refitting the 86 calibration ranges from the stored calibration | median mass-offset difference 1.5·10⁻⁸, resolution 1.6·10⁻⁶ (relative); all within 7·10⁻⁴ u and 3 % |
| Raw data → load calibration → recalibrate → fit | areas within 0.1 % (median) of the stored ones, total within 0.04 % |
| Molecule combs for the peak shape (5 exported data sets) | identical within the 8 digits MATLAB saved |
| `fminsearch` on MATLAB's Rosenbrock example | same minimum (8.1777·10⁻¹⁰) after the same 159 evaluations |
| Non-negative least squares | matches Octave's `lsqnonneg` on a 200×200 test problem |

The remaining 20 stored patterns were generated with settings I could not reconstruct (the
file mixes molecules made at different times); 13 more use abbreviations the test cannot
translate. Most molecules whose fitted areas differ sit in nearly degenerate groups, where tiny
numerical differences move area between neighbours; a few helium clusters differ by ~10⁻⁵
because the mass offset below the first calibration point is no longer extrapolated.

Two reference files were generated with settings different from those stored in
`testfile.ifd` (PCHIP mass offset instead of spline; polynomial degree 3 instead of 2).
The tests set these explicitly.

## Porting status

| Area | MATLAB source | Status |
|---|---|---|
| Data model, `.ifd` load/save | `load_ifd.m`, `IsotopeFit.m` | done |
| HDF5 import | `IsotopeFit.m` (`load_h5`), `h5cal.m` | done, tested on `clustof.h5` |
| Calibration curves | `getcalibrationdata.m` and helpers | done, except `spaps` |
| Background, mass offset | `subtractbg.m`, `subtractmassoffset.m` | done |
| Background estimation | `bg_correction.m`, `find_bg.m` | done |
| Calibration | `calibrate.m`, `parameterinterpolation.m`, "Load calibration" | done |
| Isotope patterns, cluster generation | `IsotopeDistribution/` | done |
| Cluster generator window, `.icg` definitions | `cluster_generator.m` | first version |
| `.ifm` files, molecule folders | `load_molecules_from_*.m`, `generate_cluster_ifm.m` | done |
| Peak shapes, pattern evaluation | `pattern_func.m`, `peak_width_adaption.m` | done |
| Peak shape measurement | `peak_shape_generator.m`, `get_convolution_core.m`, `create_molecule_stems.m` | done |
| Fitting: linear system, simplex, simplex_lin_combi, baseline | `fitwithcalibration.m`, `fitranges.m`, `fitmolecules.m`, `get_fit_params_using_*.m` | done |
| Alternative error estimates | `get_fitting_error(s).m` | not needed: unused in MATLAB except by the crashing `simplex` errors and the toolbox methods |
| Drift correction | `driftcorrection.m` | done, with MATLAB's errors fixed; tested on synthetic drift; file layout checked on `clustof.h5` (100 writes × 6 buffers, drift below about 1 bin) |
| Global fit, batch fitting | `batch_fit.m` | done (GUI and `isotopefit-batch`) |
| GUI: main window, cluster series, background dialog, settings | `IsotopeFit.m`, `bg_correction.m`, `settingswindow.m` | first version |
| GUI: calibration window, peak-shape editor | `calibrate.m`, `peak_shape_generator.m` | first version |
| Exports: cluster series, calibrated and fitted spectrum | `IsotopeFit.m` | done |

Not planned unless needed: genetic-algorithm and pattern-search fitting methods (need
MATLAB toolboxes, not used by the default settings), files in `deprecated/`.

## Differences from the MATLAB version

The port reproduced MATLAB exactly first (that is what the reference tests above check). These
MATLAB errors are now corrected, so results differ from MATLAB where they apply:

* **Error estimates** are one standard deviation (the standard error of the fitted area). MATLAB's
  fits reported 1.96 times that, a 95 % interval, while its batch fit reported one standard
  deviation, so the two did not compare. MATLAB also divided the residual sum of squares by the
  number of data points *n* instead of the degrees of freedom *n − m* (*m* molecules in the fit).
  Compared with MATLAB's GUI fit, errors are now smaller by a factor 1.96/√(n/(n−m)), about 1.94
  for `testfile.ifd` (the degrees-of-freedom correction adds about 1 %, up to 10 % for the
  narrowest ranges).
* **Calibration curves beyond the calibrated masses** were extrapolated. For a spline mass offset
  this reaches 0.7 u at 1000 u, 130 u at 1500 u and 300 000 u at the end of `testfile.ifd`
  (last calibration point 893 u), so everything above ~1000 u sat at nonsense masses; a
  polynomial resolution went negative above ~1800 u. Every curve now keeps its end value outside
  the calibrated range.
* **Merging close isotope peaks** (`approx_masses.m`) did not update the gap to the left of a
  merged peak, so peaks slightly further apart than the mass accuracy could be merged. None of the
  stored patterns in `testfile.ifd` is affected.
* **Thallium** (`Atoms/Tl.txt`) was a copy of lead. It now holds ²⁰³Tl (29.52 %) and ²⁰⁵Tl
  (70.48 %). `Po.txt` stays empty: polonium has no stable isotopes.
* **HDF5 mass axis before time zero**: MATLAB's formula mirrored the first time bins (before the
  ions can arrive) to small positive masses, so the mass axis first ran backwards. These bins now
  get negative masses and the axis increases throughout. `.ifd` files saved by MATLAB keep their
  axis; the program handles both.
* **Drift correction** (`driftcorrection.m`) made peaks broader instead of narrower in tests with
  a known drift: its shift had the wrong sign, the polynomial of spectrum i+1 was applied to
  spectrum i, the last spectrum was not summed, whole-bin steps between consecutive spectra hid
  slow drifts, and the time-domain fit used the first bin of each division instead of its
  centre. The port correlates every spectrum with the sum of all spectra, to a fraction of a bin.
  The drift data MATLAB stored in `calibration.dc` is not written to `.ifd` files.
* The `simplex` fitting method crashes in MATLAB when it estimates errors (`get_fitting_errors.m`
  calls `pattern_func` without the peak shape). The port fits but returns no errors.
* In the calibration window, choosing Spline or PCHIP for the *resolution* reset the *mass-offset*
  parameter to 0 (`updatepolynomials` in `calibrate.m`). The port sets the resolution parameter.
* After a new peak shape is accepted, MATLAB rescales the resolution *list* but not a Flat
  resolution *parameter*, so with a Flat resolution the peaks change width. The port rescales both.
* `parse_molecule.m` hangs on characters other than element symbols, digits and brackets. The
  port skips them.

Kept as in MATLAB: the calibration fit buttons let the resolution move by ±50 %, although the code
comment says 0.5 per cent (`deltares = 0.5`); see `CAL_DELTARES` in `calibrate.py`.

## License

BSD 3-clause, same as the original; see `LICENSE.md`.
