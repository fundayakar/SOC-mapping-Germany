# SOC-mapping-Germany — reproducibility bundle v2.0.0

**Separating neighbourhood information from spatial arrangement in soil organic carbon prediction across Germany**

This version unifies the real-patch RF–MLP–CNN benchmark, separately retrained CNN information controls, coordinate metadata audit and paired nominal–GPS extraction experiment. The measured target is LUCAS 2018 topsoil SOC (0–20 cm). Both the 2018 and 2022 predictor conditions predict that same target; this is not an SOC-change or external temporal-validation experiment.

The old Zenodo record [10.5281/zenodo.20480364](https://doi.org/10.5281/zenodo.20480364), version v1.1.0, contains the earlier point-RF study. It does **not** archive the current patch, CNN-control or GPS experiments. The version-specific DOI for this bundle is pending until the new Zenodo version is published.

## What is included

| Directory / entry | Purpose |
|---|---|
| `experiments/main/` | Unchanged source models, harmonized GEE extraction and target-scale diagnostics; historical main-pipeline summaries |
| `experiments/cnn_controls/` | Unchanged same-capacity control fitting, matched fresh original CNN, shuffle controls and exploratory loss-gate source/aggregates |
| `experiments/gps/` | Unchanged paired extraction/refit and analysis source; frozen-prediction bootstrap aggregates |
| `experiments/coordinates/` | Coordinate-audit source and aggregate audit records |
| `data/predictors/` | Six real, unlabelled EO patch arrays: main 663-point inputs for each year, and nominal/GPS 662-point inputs for each year |
| `data/roles/` | Frozen outer and epoch-selection memberships for the 663- and 662-point cohorts |
| `attach_labels.py` | Bind independently obtained, licensed SOC labels by POINT_ID with exact label-hash checks |
| `train_release.py` | Portable local refitting entry point using the unchanged fitting functions and frozen roles |
| `report_release.py`, `results/` | Regenerate the canonical benchmark/control/GPS result index from published aggregate scores |
| `validate_release.py`, `validation/` | Integrity, predictor, cohort, role and recorded checkpoint-portability checks |
| `SOURCE_CODE_PROVENANCE.json`, `RELEASE_MANIFEST.json` | Unchanged scientific-source hashes and complete release-file hashes |

The public bundle deliberately excludes original SOC records, labelled export CSVs, point-level observed values/residuals, original fitted neural weights and notebook outputs. See `DATA_ACCESS.md`. The author's separately retained licensed working archive contains those experiment records. Published code and aggregates can be inspected without any account; exact refitting requires the independently licensed labels. The EO arrays and roles require no Earth Engine account for local use.

## Environment

The original CPU experiments used Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, PyTorch 2.14.1+cpu, matplotlib 3.10.8 and joblib 1.5.3. The coordinate audit additionally uses pyproj 3.7.2.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python validate_release.py --output ../release_integrity.json
python report_release.py --output ../rebuilt_soc_results
```

`requirements.txt` pins the recorded Linux CPU environment. On another platform, install a compatible PyTorch build from its official distribution and record the changed environment. Matching version numbers alone do not guarantee bitwise-identical float32 arithmetic or identical retraining trajectories across CPUs.

## Attach licensed labels and refit

Obtain the original LUCAS 2018 soil data directly from ESDAC under its conditions. Do not use an old GitHub/Zenodo copy as a substitute for an independent licence. Select the 0–20 cm SOC rows; the label attachment accepts `POINT_ID`, `OC` in g/kg and `Depth` by default. Column names and CSV delimiter can be configured explicitly.

```bash
python attach_labels.py --labels /path/to/licensed/LUCAS2018_SOIL.csv --output ../local_labelled_arrays
python train_release.py --data ../local_labelled_arrays --output ../new_soc_smoke --suite all --smoke --workers 4
python train_release.py --data ../local_labelled_arrays --output ../new_soc_full --suite all --workers 4
```

The first command checks all six arrays against the study's exact SOC-label array hashes; missing IDs, duplicated samples, mismatched values and negative/nonfinite SOC stop execution. Depth is not guessed from row position. If a source file lacks a depth column, the user must provide the already verified 0–20 cm records.

`--smoke` trains 15 complete model-fold fits for the 2018/log/seed0/fold0 condition across the three suites, records partial fold scores and explicitly does not claim full-suite completion. The full run comprises 640 main benchmark fits, 360 additional CNN controls and 1,280 paired GPS fits. The 160 fresh original CNN fits within the main benchmark are the reference for the controls, giving the same 520 original/control fits as the completed information-control experiment. RF fitted forests are not archived; their full settings, inputs and deterministic seeds are retained for refitting.

Run any suite separately with `--suite main`, `--suite controls` or `--suite gps`. Outputs must be outside the immutable release directory. Existing outputs are reused only under an identical design fingerprint. A changed design requires a new output directory. `oof_predictions.csv` in a new local run contains licensed observed SOC and must not be uploaded publicly with this release.

The new portable entry point handles paths and local label attachment; the original scientific fitting functions are unchanged. The old component drivers and validators remain under `experiments/` for provenance. Their historical relative paths, cached-run fingerprints and requirements for original private exports/checkpoints mean they are not the public entry point. Do not run an old driver against a partially reconstructed historical output directory.

## Experimental design and result conventions

- Main cohort: 663 common points selected using completeness of the earlier five point-predictor configurations across both years. Seasonal-feature coverage influenced that initial selection. This is not all German LUCAS sites.
- Predictors: fourteen raster channels, 17×17 patches on a common 20 m EPSG:3035 grid, plus three native-grid ERA5-Land climate scalars. The validity mask is a separate CNN channel and is constant for this cohort.
- Spatial design: 100 km nominal blocks, five repetitions of five folds (seeds 0–4); nominal 20 km buffered version uses the first three repetitions. The epoch-selection memberships are training-only and frozen.
- Fixed RF: 800 trees, min leaf 2, all features, bootstrap, random seed 42. MLP: 1,121 parameters. CNN: 8,449 parameters. Raw SOC and log1p(SOC) training are separate; physical predictions are nonnegative and have no label-based upper cap.
- CNN information controls: original_fresh, centre-tiled, joint multiband shuffle with centre retained (seed 314159), and a second shuffle (271828) on repetition 0. Each variant is separately trained with the same architecture, initialization and original-training normalization. No GPS centre-tiled/shuffle controls were trained.
- GPS sensitivity: 662 usable points in both conditions, excluding 45782870 from both. Nominal and GPS windows are genuine separate GEE extractions; shared pixels/masks match exactly. The original outer and inner memberships are restricted to the same 662 points, never recomputed from GPS positions. The nominal buffer minimum is 20,000.118 m; the GPS minimum is 19,979.032 m, so it is not a GPS-redesigned 20 km buffer.

**Canonical main benchmark:** `results/main_scores_by_repeat.csv` uses centerRF, contextRF, centerMLP and **original_fresh** as patchCNN. `results/main_summary.csv` has 32 conditions. The old main-pipeline patchCNN in `experiments/main/` is a historical reference and is not the primary CNN used in the manuscript. Main summaries average pooled OOF metrics computed separately per repetition. Their SD and win counts are descriptive; repetitions share observations and are not independent confidence-interval replicates.

**GPS:** `results/gps_summary.csv` has 32 conditions. Its RMSE is sqrt(mean squared error across points and repetitions), not mean repetition-specific RMSE. Intervals use 2,000 paired resamples of 47 nominal blocks, seed 20261010, conditional on fixed OOF predictions with no model refitting or multiplicity correction. A zero-containing interval is not an equivalence test. The CNN-versus-MLP comparison changes architecture/capacity as well as context.

**Exploratory supplementary analyses:** target retransformation/smearing and conditional loss-gate scores/source are retained separately. The gate aggregates condition on earlier A/B/C expert fits. The public portable runner covers the main benchmark, information controls and GPS experiment; it does not refit the smearing/gate pilots. Those pilots' frozen expert calibration/test inputs and original checkpoints are in the author's licensed archive. They are not a new-algorithm or independent-validation claim.

## Validation of this bundle

The recorded main-pipeline cohort/fold audit passed for 640 model-fold records and 84,864 predictions. The GPS analysis passed for 1,280 fits, 169,472 OOF rows and 84,736 paired rows. The coordinate audit was rerun and matched all 778 source points to the official Eurostat records.

All 1,480 archived neural checkpoints (520 CNN original/controls, 640 GPS, 320 historical main) were replayed on the packaging CPU. 960 matched the historical strict `1e-8` tolerance. All passed the explicitly reported portability tolerance `atol=1e-4`, `rtol=1e-6`; the largest absolute forward-pass difference was 0.000106686 g/kg. Original predictions and manuscript numbers were preserved. `validation/weight_portability.json` records these results; original component validation files describe their original audit environment. The strict original CNN validator remains unchanged and can fail on another CPU at sub-milligram-per-kilogram float32 differences.

Fresh local training may diverge beyond forward-pass rounding through epoch selection. Use the saved published aggregates as the reported-study reference and label new runs by their environment and fingerprint. The smoke test validates the delivered entry point and frozen roles; it is not a new full experiment.

## Licence and attribution

Author-created code, documentation and aggregate results retain the CC BY 4.0 licence used for the existing Zenodo software record. Third-party data remain subject to their original conditions; this licence does not grant redistribution of LUCAS soil-property records. See `LICENSE.md` and `DATA_ACCESS.md`.

The LUCAS 2018 topsoil dataset was supplied by the European Commission through ESDAC, managed by the Joint Research Centre. Source documentation: Fernández-Ugalde et al. (2022), *LUCAS 2018 Soil Module. Presentation of dataset and results*, EUR 31144 EN, doi:10.2760/215013. EO sources are Copernicus Sentinel-1/2 and ERA5-Land, with SRTM terrain; dataset identifiers and processing are recorded in the included GEE scripts.
