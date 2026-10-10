"""Audit transformed inputs, shared normalization, spatial roles and saved predictions."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold
import joblib
import run_ablations as a
import loss_aligned_gate as gate

HERE=Path(__file__).resolve().parent

def main(root):
    t=a.imports(root);design=json.loads((HERE/'design.json').read_text());fresh=json.loads((HERE/'matched_design.json').read_text())
    cache={};transform_stats=[]
    for year in [2018,2022]:
        orig=dict(np.load(root/f'cnn/data/patches_{year}.npz',allow_pickle=False));assert orig['X'].shape==(663,14,17,17) and np.all(orig['valid']==1)
        idx=orig['bands'].tolist().index('NDVI')
        for variant in ['original_fresh','center_tiled','shuffle_314159','shuffle_271828']:
            d=orig if variant=='original_fresh' else a.transformed(orig,variant)
            np.testing.assert_array_equal(d['X'][:,:,8,8],orig['X'][:,:,8,8])
            np.testing.assert_array_equal(d['climate'],orig['climate']);np.testing.assert_array_equal(d['y'],orig['y'])
            if variant=='center_tiled':np.testing.assert_array_equal(d['X'],np.broadcast_to(orig['X'][:,:,8:9,8:9],orig['X'].shape))
            if variant.startswith('shuffle'):
                seed=int(variant.split('_')[1]);m=a.permutation(orig['point_id'],seed)
                np.testing.assert_array_equal(np.sort(m,axis=1),np.broadcast_to(np.arange(289),(663,289)))
                assert (m[:,144]==144).all()
                stored=dict(np.load(HERE/f'permutations/seed_{seed}.npz',allow_pickle=False))
                ix={p:i for i,p in enumerate(stored['point_id'])};expected=stored['mapping'][[ix[p] for p in orig['point_id']]]
                np.testing.assert_array_equal(m,expected)
                restored=np.take_along_axis(d['X'].reshape(663,14,289),np.argsort(m,axis=1)[:,None,:],axis=2).reshape(orig['X'].shape)
                np.testing.assert_array_equal(restored,orig['X'])
            ndvi=d['X'][:,idx]
            transform_stats.append(dict(year=year,variant=variant,NDVI_mean_neighbor_difference=float((abs(np.diff(ndvi,axis=1)).mean()+abs(np.diff(ndvi,axis=2)).mean())/2)))
            cache[(year,variant)]=d
    pd.DataFrame(transform_stats).to_csv(HERE/'results/transform_geometry.csv',index=False)
    folds=pd.read_csv(root/'stage2/results/fold_assignments.csv',dtype={'point_id':str}).query("cohort=='global_0_20'")
    paths=sorted((HERE/'runs').glob('*.json'))+sorted((HERE/'fresh_originals').glob('*.json'));assert len(paths)==520
    count=0;max_replay=0.;minimum_buffer=1e20;roles=[]
    for path in paths:
        r=json.loads(path.read_text());year=r['year'];variant=r['variant'];target=r['target'];protocol=r['protocol'];seed=r['seed'];fold=r['fold']
        assert r['fingerprint']==(fresh['fingerprint'] if variant=='original_fresh' else design['fingerprint'])
        d=cache[(year,variant)];orig=cache[(year,'original_fresh')];ids={p:i for i,p in enumerate(d['point_id'])}
        g=folds[(folds.protocol==protocol)&(folds.seed==seed)&(folds.fold==fold)]
        tr=np.array(sorted(ids[p] for p in g[g.role.eq('train')].point_id));te=np.array(sorted(ids[p] for p in g[g.role.eq('test')].point_id))
        assert r['point_id']==d['point_id'][te].tolist() and r['expert_train_ids']==d['point_id'][tr].tolist()
        np.testing.assert_array_equal(r['actual_SOC'],d['y'][te])
        bxy=np.floor(d['xy']/100000).astype(int);blocks=np.array([f'{x}_{y}' for x,y in bxy]);aa,bb=next(GroupKFold(3).split(tr,groups=blocks[tr]));it=tr[aa];iv=tr[bb]
        if protocol.endswith('buffer20'):it=it[cKDTree(d['xy'][iv]).query(d['xy'][it])[0]>20000]
        assert r['epoch_train_ids']==d['point_id'][it].tolist() and r['epoch_validation_ids']==d['point_id'][iv].tolist()
        for aa,bb in [(tr,te),(it,iv)]:
            assert not set(blocks[aa])&set(blocks[bb]) and not set(aa)&set(bb)
            distance=float(cKDTree(d['xy'][aa]).query(d['xy'][bb])[0].min())
            if protocol.endswith('buffer20'):assert distance>20000;minimum_buffer=min(minimum_buffer,distance)
        ck=t.torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False)
        norm={k:np.asarray(v,dtype=ck['normalization_dtypes'][k]) if isinstance(v,list) else float(v) for k,v in ck['normalization'].items()}
        expected=t.normalization(orig,tr,target,'patchCNN')
        for k in expected:np.testing.assert_allclose(norm[k],expected[k],atol=1e-12,rtol=1e-12)
        net=t.PatchCNN();net.load_state_dict(ck['state_dict']);assert sum(p.numel() for p in net.parameters())==ck['parameter_count']==8449
        assert ck['net_seed']==42+100*seed+fold and ck['epochs']==r['selected_epochs']
        pred=t.neural_predict(net,norm,d,te,target,'cpu');np.testing.assert_allclose(pred,r['pred_SOC'],rtol=1e-8,atol=1e-8)
        max_replay=max(max_replay,float(np.max(abs(pred-np.asarray(r['pred_SOC'])))))
        # All control branches receive exactly the original normalized point vector.
        point=t.tensors(d,te,norm,target)[1].numpy();point_orig=t.tensors(orig,te,norm,target)[1].numpy();np.testing.assert_array_equal(point,point_orig)
        if seed==0 and fold==0:
            modified=dict(d);modified['y']=d['y'].copy();modified['y'][te]=d['y'][te][::-1]+9999
            np.testing.assert_array_equal(t.neural_predict(net,norm,modified,te,target,'cpu'),pred)
        for name,idx in [('outer_train',tr),('outer_test',te),('epoch_train',it),('epoch_validation',iv)]:
            roles.append(pd.DataFrame(dict(year=year,target=target,protocol=protocol,seed=seed,fold=fold,variant=variant,role=name,point_id=d['point_id'][idx])))
        count+=1
    assert count==520
    pd.concat(roles,ignore_index=True).to_csv(HERE/'results/all_roles.csv',index=False)
    audit=dict(all_passed=True,new_CNN_model_folds=520,weights_replayed=520,max_prediction_replay_delta=max_replay,
        shared_point_branch_exact=True,normalization_original_training_only=True,parameter_count_each=8449,
        transformations_invertible_multiband_shuffle=True,labels_not_used_for_permutations=True,spatial_folds_checked=True,
        minimum_buffer_inner_or_outer_distance_m=minimum_buffer,test_label_perturbation_invariant=True)
    (HERE/'validation.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2),flush=True)
    validate_gates(root)

def validate_gates(root):
    out=HERE/'loss_gate';p=pd.read_csv(out/'predictions.csv',dtype={'point_id':str});choices=pd.read_csv(out/'selected_hyperparameters.csv');hyper=pd.read_csv(out/'inner_cv.csv');roles=pd.read_csv(out/'inner_roles.csv',dtype={'point_id':str})
    assert len(p)==2652 and len(choices)==40 and len(hyper)==240
    checked=0
    for (year,protocol,fold),test in p.groupby(['year','protocol','fold']):
        name=f'{protocol}_{year}_f{fold}';cal=pd.read_csv(out/f'inputs/{name}_calibration.csv',dtype={'point_id':str})
        assert not set(test.point_id)&set(cal.point_id)
        d=dict(np.load(root/f'cnn/data/patches_{year}.npz',allow_pickle=False));ix={q:i for i,q in enumerate(d['point_id'])}
        np.testing.assert_array_equal(cal.actual_SOC,d['y'][[ix[q] for q in cal.point_id]])
        np.testing.assert_array_equal(test.actual_SOC,d['y'][[ix[q] for q in test.point_id]])
        x=gate.features(cal.pred_RF.to_numpy(),cal.pred_CNN.to_numpy());xt=gate.features(test.contextRF.to_numpy(),test.patchCNN.to_numpy())
        for loss in ['MAE','MSE']:
            choice=choices.query('year==@year and protocol==@protocol and fold==@fold and objective==@loss').iloc[0]
            for family,key,value in [('direct','selected_lambda','lambda_'),('winner_probability','selected_C','C')]:
                h=hyper.query('year==@year and protocol==@protocol and fold==@fold and objective==@loss and family==@family')
                ordered=h.sort_values(['inner_pooled_loss','hyperparameter']);assert ordered.iloc[0].hyperparameter==choice[key]
                kind='direct' if family=='direct' else 'winner';m=joblib.load(out/f'models/{name}_{kind}_{loss}.joblib')
                np.testing.assert_allclose(m['scaler'].mean_,x.mean(axis=0),atol=1e-10,rtol=1e-10);assert m[value]==choice[key]
                w=gate.weight(m,xt) if family=='direct' else gate.winner_weight(m,xt)
                pred=test.contextRF+w*(test.patchCNN-test.contextRF)
                column='direct_'+loss if family=='direct' else 'winner_tuned_'+loss
                np.testing.assert_allclose(pred,test[column],atol=1e-8,rtol=1e-8);checked+=1
        for inner,g in roles.query('year==@year and protocol==@protocol and fold==@fold').groupby('inner'):
            tr=g[g.role.eq('gate_inner_train')].point_id.tolist();va=g[g.role.eq('gate_inner_validation')].point_id.tolist()
            assert set(tr)|set(va)<=set(cal.point_id) and not set(tr)&set(va)
            if protocol.endswith('buffer20'):assert cKDTree(d['xy'][[ix[q] for q in tr]]).query(d['xy'][[ix[q] for q in va]])[0].min()>20000
    assert checked==80
    v=json.loads((out/'validation.json').read_text());v['inner_hyperparam_fits']=len(hyper)*3
    v['inner_fit_counts_per_family']={'direct':360,'winner_probability':360};v['saved_gate_predictions_replayed']=checked;v['inner_selected_hyperparameters_verified']=True
    (out/'validation.json').write_text(json.dumps(v,indent=2));print('Loss gate audit passed; 80 final models, 720 calibration-inner fits.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage3-root',type=Path,default=HERE.parent/'target_transform');x=ap.parse_args();main(x.stage3_root.resolve())
