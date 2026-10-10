from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parent
folds=pd.read_csv(ROOT/'stage2/results/fold_assignments.csv',dtype={'point_id':str}).query("cohort=='global_0_20'")
source=pd.read_csv(ROOT/'stage2/input/exports/SOC_unified_2018.csv',dtype={'point_id':str})
source=source[source.Depth.eq('0-20 cm')].set_index('point_id').OC
# Depth strings are recorded by the source export, never inferred from row position.
if len(source)==0:
    source=pd.read_csv(ROOT/'stage2/input/exports/SOC_unified_2018.csv',dtype={'point_id':str})
    depth=source.Depth.astype(str).unique().tolist()
    assert len(depth)==1 and depth[0] in ['0-20','0–20','0-20 cm'],depth
    source=source.set_index('point_id').OC
source=pd.to_numeric(source,errors='coerce')
fit_count=0;weight_count=0;prediction_count=0;min_buffer=1e20
for folder,protocol,repeats in [('results','block_100',5),('results_buffer20','block_100_buffer20',3)]:
    for year in [2018,2022]:
        out=ROOT/'cnn'/folder/str(year)
        d=dict(np.load(ROOT/f'cnn/data/patches_{year}.npz',allow_pickle=False))
        assert len(d['point_id'])==663
        np.testing.assert_allclose(d['y'],source.loc[d['point_id']].to_numpy(),atol=1e-10,rtol=0)
        ids={p:i for i,p in enumerate(d['point_id'])}
        p=pd.read_csv(out/'oof_predictions.csv',dtype={'point_id':str})
        assert np.isfinite(p[['pred_SOC','actual_SOC']]).all().all() and p.pred_SOC.ge(0).all()
        np.testing.assert_allclose(p.actual_SOC,source.loc[p.point_id].to_numpy(),atol=1e-10,rtol=0)
        assert p.protocol.eq(protocol).all()
        for (model,target,seed,fold),g in p.groupby(['model','target','seed','fold']):
            f=folds[(folds.protocol==protocol)&(folds.seed==seed)&(folds.fold==fold)]
            test=f[f.role.eq('test')].point_id
            train=f[f.role.eq('train')].point_id
            assert set(g.point_id)==set(test) and len(g)==len(test)
            assert not set(train)&set(test)
            if protocol.endswith('buffer20'):
                dist=cKDTree(d['xy'][[ids[x] for x in train]]).query(d['xy'][[ids[x] for x in test]])[0].min()
                assert dist>20000
                min_buffer=min(min_buffer,float(dist))
            result=json.loads((out/f'{model}_{target}_s{seed}_f{fold}.json').read_text())
            assert result['fingerprint']==json.loads((out/'design.json').read_text())['fingerprint']
            if model in ['centerMLP','patchCNN']:
                assert (out/f'{model}_{target}_s{seed}_f{fold}.pt').stat().st_size>0
                weight_count+=1
            fit_count+=1
        assert p.groupby(['model','target','seed']).size().eq(663).all()
        assert p.groupby(['model','target','seed']).point_id.nunique().eq(663).all()
        assert len(p)==663*repeats*8
        prediction_count+=len(p)
assert fit_count==640 and weight_count==320 and prediction_count==84864
nb=json.loads((ROOT/'cnn/SOC_PyTorch_walkthrough.ipynb').read_text())
code=[c for c in nb['cells'] if c['cell_type']=='code']
assert len(code)==7 and [c['execution_count'] for c in code]==list(range(1,8))
assert not any(o['output_type']=='error' for c in code for o in c['outputs'])
status=dict(all_passed=True,model_fold_results=fit_count,neural_weights=weight_count,heldout_predictions=prediction_count,min_buffer_train_test_distance_m=min_buffer,actual_target_and_fold_membership_checked=True,notebook_code_cells_executed=len(code))
(ROOT/'stage3_validation.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status,indent=2))
