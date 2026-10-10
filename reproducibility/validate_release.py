"""Verify file hashes, published predictors, frozen roles and canonical scores."""
from pathlib import Path
import argparse, hashlib,json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(a):
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    for r in manifest['files']:assert (ROOT/r['path']).is_file() and sha(ROOT/r['path'])==r['sha256'],r['path']
    assert {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['RELEASE_MANIFEST.json']}=={r['path'] for r in manifest['files']},'Unexpected files in release tree.'
    for r in json.loads((ROOT/'SOURCE_CODE_PROVENANCE.json').read_text()):assert sha(ROOT/r['file'])==r['sha256']
    arrays={}
    for r in json.loads((ROOT/'data/predictor_manifest.json').read_text()):
        p=ROOT/f'data/predictors/{r["name"]}.npz';assert sha(p)==r['predictors_sha256']
        d=dict(np.load(p,allow_pickle=False));arrays[r['name']]=d
        assert 'y' not in d and d['X'].shape==(r['n'],14,17,17) and d['valid'].shape==(r['n'],1,17,17)
        assert np.isfinite(d['X']).all() and np.isfinite(d['climate']).all() and np.all(d['valid']==1)
    min_buffer=1e20
    for suite in ['main','gps']:
        roles=pd.read_csv(ROOT/f'data/roles/{suite}_roles.csv',dtype={'point_id':str});assert not roles.duplicated().any()
        d=arrays['main_2018' if suite=='main' else 'nominal_2018'];ids={p:i for i,p in enumerate(d['point_id'])}
        for (protocol,seed,fold),g in roles.groupby(['protocol','seed','fold']):
            r={name:set(x.point_id) for name,x in g.groupby('role')}
            assert not r['train']&r['test'] and not r['epoch_train']&r['epoch_valid'] and r['epoch_train']|r['epoch_valid']<=r['train']
            for aa,bb in [('train','test'),('epoch_train','epoch_valid')]:
                x=d['xy'][[ids[p] for p in sorted(r[aa])]];y=d['xy'][[ids[p] for p in sorted(r[bb])]]
                assert not set(map(tuple,np.floor(x/100000).astype(int)))&set(map(tuple,np.floor(y/100000).astype(int)))
                distance=float(cKDTree(x).query(y)[0].min())
                if protocol.endswith('buffer20'):assert distance>20000;min_buffer=min(min_buffer,distance)
        assert roles[['protocol','seed','fold']].drop_duplicates().shape[0]==40
        test=roles.query("role=='test'");assert test.groupby(['protocol','seed']).point_id.nunique().eq(len(ids)).all()
    for year in [2018,2022]:
        nom,gps=arrays[f'nominal_{year}'],arrays[f'gps_{year}'];assert np.array_equal(nom['point_id'],gps['point_id']) and np.array_equal(nom['climate'],gps['climate'])
        index={p:i for i,p in enumerate(arrays[f'main_{year}']['point_id'])}
        rows=[index[p] for p in nom['point_id']]
        for key in ['X','valid','climate','xy','patch_center_xy']:np.testing.assert_array_equal(nom[key],arrays[f'main_{year}'][key][rows])
    main_scores=pd.read_csv(ROOT/'results/main_scores_by_repeat.csv');assert len(main_scores)==128
    assert set(main_scores.model)=={'centerRF','contextRF','centerMLP','patchCNN'}
    assert len(pd.read_csv(ROOT/'results/main_summary.csv'))==32 and len(pd.read_csv(ROOT/'results/gps_summary.csv'))==32
    # Public CSVs cannot contain source labels or point-level residuals.
    for p in ROOT.rglob('*.csv'):
        cols=set(pd.read_csv(p,nrows=0).columns)
        assert not cols&{'OC','actual_SOC','y','error_nominal','error_gps','nominal_SE','gps_SE'},p
    report={'all_passed':True,'manifest_files':len(manifest['files']),'predictor_arrays':6,'source_labels_public':False,'frozen_role_groups_per_cohort':40,'minimum_nominal_buffer_m':min_buffer,'main_repeat_rows':128,'main_conditions':32,'gps_conditions':32}
    if a.output:Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');main(p.parse_args())
