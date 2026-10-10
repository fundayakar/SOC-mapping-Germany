# Data access and redistribution boundary

Official LUCAS 2018 soil access: https://esdac.jrc.ec.europa.eu/content/lucas-2018-topsoil-data

Conditions checked 2026-10-10 on the official dataset page. Its notification specifies a personal, non-transferable licence with no sublicensing, limits access to the licensee and authorized organizational staff, and requires aggregation for published results. Accordingly, this public bundle does not redistribute source SOC values, label-bearing arrays/exports, point-level residuals, original checkpoints or notebook output. An open licence on an author's Zenodo record does not override the source-data conditions.

Request the dataset directly from ESDAC, read its conditions and use `attach_labels.py` locally. The script does not register, accept terms or download soil data on your behalf. It attaches labels to the released genuine EO predictor arrays by POINT_ID and verifies the original full-array hashes. Keep the newly labelled arrays and local prediction outputs in your licensed environment.

Eurostat coordinate source: https://ec.europa.eu/eurostat/cache/lucas/DE_2018_20200213.CSV

The predictor arrays retain the nominal spatial-design coordinates, patch centres and recorded GPS metadata needed for the paired experiment. These are survey/EO metadata, not soil measurements or private landowner records. No landowner or private person is identified. Coordinate provenance and finite GPS precision are documented in the audit source. No map locating individual soil samples is included in the public bundle.

The full working archive is retained for the author's own reproduction and is not a public upload candidate. Original expert calibration/test inputs for supplementary smearing/gate pilots likewise remain in that licensed archive. Aggregate pilot results and unchanged analysis source are public, with this restriction stated explicitly.

Any proposal to make the original labelled data/checkpoints public requires a separate verified permission from the data owner. Do not infer that permission from this bundle's CC BY notice or the old v1.1.0 record.
