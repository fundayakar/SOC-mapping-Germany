"""Audit spatial roles, source labels, saved expert replay and calibration-only gates."""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
import joblib
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

HERE=Path(__file__).resolve().parent

def main(root):
    sys.path.insert(0,str(root/'cnn'));sys.path.insert(0,str(root))
    import train_patch_models as t
    import rf_compare as stage2
    t.torch.set_num_threads(1)
    allroles=[]; count=0; maxdelta=0; min_buffer=1e20;scaler_checks=0
    for out in sorted((HERE/'pilot').iterdir()):
        if not out.is_dir():continue
        s=json.loads((out/'completion.json').read_text());year=s['year'];target=s['target'];protocol=s['protocol']
        d=dict(np.load(root/f'cnn/data/patches_{year}.npz',allow_pickle=False));ids={p:i for i,p in enumerate(d['point_id'])}
        role=pd.read_csv(out/'roles.csv',dtype={'point_id':str})
        sets={r:np.array([ids[p] for p in role[role.role.eq(r)].point_id]) for r in role.role.unique()}
        a,b,c=sets['expert_train'],sets['gate_calibration'],sets['outer_test']
        assert not(set(a)&set(b) or set(a)&set(c) or set(b)&set(c))
        assert set(sets['epoch_train'])|set(sets['epoch_validation'])<=set(a)
        assert not set(sets['epoch_train'])&set(sets['epoch_validation'])
        blocks=np.floor(d['xy']/100000).astype(int);blocks=np.array([f'{x}_{y}' for x,y in blocks])
        for aa,bb in [(a,b),(a,c),(b,c),(sets['epoch_train'],sets['epoch_validation'])]:
            assert not set(blocks[aa])&set(blocks[bb])
            distance=float(cKDTree(d['xy'][aa]).query(d['xy'][bb])[0].min())
            assert distance>np.sqrt(2)*340
            if protocol.endswith('buffer20'):assert distance>20000;min_buffer=min(min_buffer,distance)
        r=pd.read_csv(out/'predictions.csv',dtype={'point_id':str});np.testing.assert_array_equal(r.point_id,d['point_id'][c]);np.testing.assert_array_equal(r.actual_SOC,d['y'][c])
        rf_path=out/'contextRF.joblib'
        if rf_path.is_file():
            rf=joblib.load(rf_path)
        else:
            # Compact deliverables omit large intermediate forests; rebuild
            # deterministically on the recorded expert-only cohort for replay.
            fit_y=np.log1p(d['y'][a]) if target=='log' else d['y'][a]
            rf=RandomForestRegressor(**stage2.RF).fit(t.rf_features(d,'contextRF')[a],fit_y)
        ck=t.torch.load(out/'patchCNN.pt',map_location='cpu',weights_only=False)
        cnn=t.PatchCNN();cnn.load_state_dict(ck['state_dict'])
        # Source raster/climate means and SDs were float32; preserve their dtype
        # when reading JSON-style lists, rather than altering inference arithmetic.
        norm={k:np.asarray(v,dtype=np.float32) if k in ['mean','sd','cmean','csd'] else float(v) for k,v in ck['normalization'].items()}
        expected=t.normalization(d,a,target,'patchCNN')
        for k in expected:np.testing.assert_allclose(norm[k],expected[k],atol=1e-12,rtol=1e-12)
        idx=np.concatenate([b,c]);pr=rf.predict(t.rf_features(d,'contextRF')[idx]);pr=np.maximum(0,np.expm1(pr) if target=='log' else pr)
        pc=t.neural_predict(cnn,norm,d,idx,target,'cpu')
        np.testing.assert_allclose(pr[len(b):],r.contextRF,atol=1e-9,rtol=1e-10);np.testing.assert_allclose(pc[len(b):],r.patchCNN,atol=1e-8,rtol=1e-8)
        maxdelta=max(maxdelta,float(np.max(abs(pc[len(b):]-r.patchCNN))))
        pd.DataFrame(dict(point_id=d['point_id'][b],actual_SOC=d['y'][b],pred_RF=pr[:len(b)],pred_CNN=pc[:len(b)])).to_csv(out/'calibration_predictions.csv',index=False)
        rz=np.log1p(pr) if target=='log' else pr;cz=np.log1p(pc) if target=='log' else pc;dy=cz-rz
        y=np.log1p(d['y'][b]) if target=='log' else d['y'][b]
        weight=float(np.clip(np.sum(dy[:len(b)]*(y-rz[:len(b)]))/max(np.sum(dy[:len(b)]**2),1e-12),0,1))
        np.testing.assert_allclose(weight,s['constant_CNN_weight'],atol=1e-12)
        f=pd.read_csv(HERE/f'inputs/features_{year}.csv',dtype={'point_id':str}).set_index('point_id').loc[d['point_id'][idx]]
        f['RF_prediction_target_scale']=rz;f['CNN_prediction_target_scale']=cz;f['CNN_minus_RF_target_scale']=dy
        weights=pd.read_csv(out/'gate_weights.csv',dtype={'point_id':str})
        for path in out.glob('gate_*.joblib'):
            gate_name=path.stem.removeprefix('gate_');g=joblib.load(path);features=g['features']
            assert not any(name in features for name in ['actual_SOC','OC','SOC','point_id','x3035','y3035','landcover_observed_2018'])
            values=f[features].to_numpy();scaler=g['model'].named_steps['standardscaler']
            np.testing.assert_allclose(scaler.mean_,values[:len(b)].mean(axis=0),rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(scaler.var_,values[:len(b)].var(axis=0),rtol=1e-10,atol=1e-10)
            labels=(abs(cz[:len(b)]-y)<abs(rz[:len(b)]-y)).astype(int)
            recalibrated=clone(g['model']).fit(values[:len(b)],labels)
            np.testing.assert_allclose(recalibrated.named_steps['logisticregression'].coef_,g['model'].named_steps['logisticregression'].coef_,rtol=1e-7,atol=1e-7)
            np.testing.assert_allclose(recalibrated.named_steps['logisticregression'].intercept_,g['model'].named_steps['logisticregression'].intercept_,rtol=1e-7,atol=1e-7)
            wg=weights[weights.gate.eq(gate_name)]
            np.testing.assert_array_equal(wg.point_id,d['point_id'][c])
            prob=g['model'].predict_proba(values[len(b):])[:,1]
            np.testing.assert_allclose(prob,wg.CNN_weight,rtol=1e-10,atol=1e-10)
            pred=rz[len(b):]+prob*dy[len(b):];pred=np.maximum(0,np.expm1(pred) if target=='log' else pred)
            np.testing.assert_allclose(pred,r['soft_gate_'+gate_name],rtol=1e-8,atol=1e-8)
            scaler_checks+=1
        # Demonstrate that held-out labels are unused by the CNN prediction path.
        modified=dict(d);modified['y']=d['y'].copy();modified['y'][c]=d['y'][c][::-1]+9999
        invariant=t.neural_predict(cnn,norm,modified,idx,target,'cpu')
        np.testing.assert_array_equal(invariant,pc)
        role=role.assign(year=year,target=target,protocol=protocol,fold=s['fold']);allroles.append(role);count+=1
    assert count==40 and scaler_checks==200
    pd.concat(allroles,ignore_index=True).to_csv(HERE/'results/honest_pilot_all_roles.csv',index=False)
    result=dict(all_passed=True,expert_pairs=40,RF_models=40,CNN_models=40,gates_checked=200,
        outer_test_unique_predictions=663*8,strategies=13,expert_normalizations_training_only=True,
        gate_scalers_calibration_only=True,gate_classifier_calibration_only_refit_verified=True,spatial_disjoint_roles_checked=True,min_buffer_role_distance_m=min_buffer,
        saved_expert_and_gate_predictions_replayed=True,max_CNN_prediction_replay_delta=maxdelta,
        CNN_test_label_perturbation_invariant=True,not_independent_external_validation=True)
    (HERE/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage3-root',type=Path,default=HERE.parent/'target_transform');a=ap.parse_args();main(a.stage3_root.resolve())
