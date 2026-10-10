# New-version metadata

Create a new version of the existing SOC-mapping-Germany record, rather than relabelling its v1.1.0 files as current.

- Title: SOC-mapping-Germany: Separating neighbourhood information from spatial arrangement in soil organic carbon prediction across Germany
- Creator: Funda Yakar; ORCID 0000-0002-7082-3956
- Resource type: Software
- Version: 2.0.0
- Licence for author-created code/documentation/aggregated results: CC BY 4.0; third-party data conditions remain applicable.
- Repository: https://github.com/fundayakar/SOC-mapping-Germany
- Previous version: https://doi.org/10.5281/zenodo.20480364
- Existing concept DOI: https://doi.org/10.5281/zenodo.20480363
- File to upload: SOC_reproducibility_v2.0.0_public.zip
- Never upload: SOC_reproducibility_v2.0.0_licensed_working.zip

Description:

This reproducibility bundle accompanies the Germany LUCAS SOC study separating neighbourhood information from spatial arrangement. It unifies the real-patch RF–MLP–CNN benchmark, separately retrained same-capacity CNN controls, coordinate metadata audit and paired nominal–GPS extraction/refit. Both 2018 and 2022 predictors use the same measured LUCAS 2018 SOC target. The public archive includes unchanged scientific source, a portable refitting driver, six genuine unlabelled EO predictor arrays, frozen outer/inner memberships, aggregate results, source provenance and validation records. LUCAS soil-property labels must be obtained independently from ESDAC under its licence and attached locally by POINT_ID; source SOC records, point-level residuals and original fitted checkpoints are excluded from the public archive. The prior v1.1.0 DOI contains only the earlier point-RF analysis. Main results use the fresh matched CNN reference. GPS results concern 662 paired points and fixed-prediction block-bootstrap intervals; they do not constitute an equivalence test or a GPS-redesigned validation scheme. Exploratory smearing and loss-gate source/aggregates are separate supplementary diagnostics and are not refitted by the public runner.

After publication, record the actual version-specific DOI and immutable GitHub commit in the manuscript Data availability statement. Do not cite the old version DOI as if it archived this bundle.
