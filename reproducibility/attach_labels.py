"""Attach independently licensed LUCAS labels by POINT_ID; never downloads them."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
def main(a):
    d=pd.read_csv(a.labels,dtype={a.id_column:str},sep=a.sep,low_memory=False)
    if a.depth_column in d:
        keep=d[a.depth_column].astype(str).str.strip().isin(['0-20 cm','0-20','0–20','0–20 cm'])
        d=d[keep]
    if d[a.id_column].duplicated().any():raise ValueError('Duplicate IDs after depth filter; resolve sample identity before attachment.')
    y=pd.to_numeric(d.set_index(a.id_column)[a.value_column],errors='raise')
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Output must be new or empty.')
    out.mkdir(parents=True,exist_ok=True);manifest=json.loads((ROOT/'data/predictor_manifest.json').read_text())
    for record in manifest:
        src=ROOT/f"data/predictors/{record['name']}.npz"
        assert hashlib.sha256(src.read_bytes()).hexdigest()==record['predictors_sha256']
        arrays=dict(np.load(src,allow_pickle=False));ids=arrays['point_id']
        missing=set(ids)-set(y.index)
        if missing:raise ValueError(f'Missing {len(missing)} required labels in {record["name"]}.')
        values=y.loc[ids].to_numpy(dtype=record['label_dtype'])
        if not np.isfinite(values).all() or (values<0).any():raise ValueError('SOC must be finite and nonnegative, in g/kg.')
        assert hashlib.sha256(values.tobytes()).hexdigest()==record['label_array_sha256'], 'Labels differ from the original study. Use a separate scientific design; do not claim exact reproduction.'
        arrays['y']=values;np.savez_compressed(out/f"{record['name']}.npz",**arrays)
    (out/'label_attachment.json').write_text(json.dumps({'labels_source_sha256':hashlib.sha256(Path(a.labels).read_bytes()).hexdigest(),'arrays':6,'matched_original_labels':True},indent=2)+'\n')
    print('Six labelled arrays created locally; all original label hashes matched.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--labels',required=True);p.add_argument('--output',required=True)
    p.add_argument('--id-column',default='POINT_ID');p.add_argument('--value-column',default='OC');p.add_argument('--depth-column',default='Depth');p.add_argument('--sep',default=',')
    main(p.parse_args())
