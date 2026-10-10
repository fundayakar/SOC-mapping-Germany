"""Exploratory, honest three-way spatial expert/gate pilot; outer seed=0 only.
Experts fit A, gates/stacking fit disjoint calibration B, predictions evaluated C.
This is a smaller-training regime than stage 3, not a nested-cross-fit final method.
"""
from pathlib import Path
import argparse, concurrent.futures as cf, hashlib, json, sys, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, balanced_accuracy_score
import joblib

HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def setup_imports(root):
    sys.path.insert(0,str(root/'cnn'));sys.path.insert(0,str(root))
    import train_patch_models as trainer
    import rf_compare as stage2
    trainer.torch.set_num_threads(1)
    return trainer,stage2

def main_job(root,year,target,protocol,fold):
    trainer,stage2=setup_imports(root);start=time.time()
    out=HERE/'pilot'/f'{protocol}_{year}_{target}_f{fold}';out.mkdir(parents=True,exist_ok=True)
    design=json.loads((HERE/'pilot_design.json').read_text());fingerprint=design['fingerprint']
    done=out/'completion.json'
    if done.exists():
        assert json.loads(done.read_text())['fingerprint']==fingerprint
        return out.name,0
    d=dict(np.load(root/f'cnn/data/patches_{year}.npz',allow_pickle=False))
    f=pd.read_csv(HERE/f'inputs/features_{year}.csv',dtype={'point_id':str}).set_index('point_id').loc[d['point_id']]
    folds=pd.read_csv(root/'stage2/results/fold_assignments.csv',dtype={'point_id':str})
    g=folds.query("cohort=='global_0_20' and protocol==@protocol and seed==0 and fold==@fold")
    ids={p:i for i,p in enumerate(d['point_id'])}
    tr=np.array(sorted(ids[p] for p in g[g.role.eq('train')].point_id));te=np.array(sorted(ids[p] for p in g[g.role.eq('test')].point_id))
    xy=d['xy']; bxy=np.floor(xy/100000).astype(int);blocks=np.array([f'{a}_{b}' for a,b in bxy])
    a,b=next(GroupKFold(3).split(tr,groups=blocks[tr]));expert=tr[a];cal=tr[b]
    if protocol.endswith('buffer20'):expert=expert[cKDTree(xy[cal]).query(xy[expert])[0]>20000]
    a,b=next(GroupKFold(3).split(expert,groups=blocks[expert]));it=expert[a];iv=expert[b]
    if protocol.endswith('buffer20'):it=it[cKDTree(xy[iv]).query(xy[it])[0]>20000]
    assert min(len(expert),len(cal),len(te),len(it),len(iv))>=20
    assert not(set(expert)&set(cal) or set(expert)&set(te) or set(cal)&set(te))
    assert not(set(blocks[expert])&set(blocks[cal]) or set(blocks[expert])&set(blocks[te]) or set(blocks[cal])&set(blocks[te]))
    distances={name:float(cKDTree(xy[a]).query(xy[b])[0].min()) for name,a,b in [('expert_cal',expert,cal),('expert_test',expert,te),('cal_test',cal,te),('epoch_train_valid',it,iv)]}
    if protocol.endswith('buffer20'):assert min(distances.values())>20000
    else:assert min(distances.values())>np.sqrt(2)*340
    y=np.log1p(d['y']) if target=='log' else d['y']
    X=trainer.rf_features(d,'contextRF');rf=RandomForestRegressor(**stage2.RF).fit(X[expert],y[expert])
    joblib.dump(rf,out/'contextRF.joblib')
    seed=1042+fold # Fixed for both years/targets/protocols; no test-tuned seed.
    _,_,epoch,curve=trainer.network_fit(d,it,iv,target,'patchCNN',seed,'cpu',100,15,True)
    cnn,norm,_,_=trainer.network_fit(d,expert,None,target,'patchCNN',seed,'cpu',epoch,15,False)
    trainer.torch.save(dict(state_dict=cnn.state_dict(),normalization={k:np.asarray(v).tolist() for k,v in norm.items()},epochs=epoch,fingerprint=fingerprint),out/'patchCNN.pt')
    all_eval=np.concatenate([cal,te]);ncal=len(cal)
    pred_rf=rf.predict(X[all_eval]); pred_rf=np.maximum(0,np.expm1(pred_rf) if target=='log' else pred_rf)
    pred_cnn=trainer.neural_predict(cnn,norm,d,all_eval,target,'cpu')
    r=np.log1p(pred_rf) if target=='log' else pred_rf;c=np.log1p(pred_cnn) if target=='log' else pred_cnn
    delta=c-r;observed=y[cal]
    winner=(abs(c[:ncal]-observed)<abs(r[:ncal]-observed)).astype(int)
    test_winner=(abs(c[ncal:]-y[te])<abs(r[ncal:]-y[te])).astype(int)
    env=[s+'_center' for s in d['bands']]+d['climate_names'].tolist()
    hetero=[s+'_std' for s in d['bands']]+[s+'_center_mean_abs' for s in d['bands']]+['NDVI_neighbor_difference']
    aenv=f.iloc[all_eval][env].to_numpy();ah=f.iloc[all_eval][hetero].to_numpy()
    apred=np.column_stack([r,c,delta])
    gate_X={'environment_only':aenv,'heterogeneity_only':ah,'predictions_only':apred,
            'environment_and_heterogeneity':np.column_stack([aenv,ah]),
            'all_with_predictions':np.column_stack([aenv,ah,apred])}
    names={'environment_only':env,'heterogeneity_only':hetero,'predictions_only':['RF_prediction_target_scale','CNN_prediction_target_scale','CNN_minus_RF_target_scale'],
        'environment_and_heterogeneity':env+hetero,'all_with_predictions':env+hetero+['RF_prediction_target_scale','CNN_prediction_target_scale','CNN_minus_RF_target_scale']}
    constant=float(np.clip(np.sum(delta[:ncal]*(observed-r[:ncal]))/max(np.sum(delta[:ncal]**2),1e-12),0,1))
    cnn_single=bool(np.mean((c[:ncal]-observed)**2)<np.mean((r[:ncal]-observed)**2))
    strategies={'contextRF':r[ncal:],'patchCNN':c[ncal:],'fixed_50_target_scale':(r[ncal:]+c[ncal:])/2,
                'calibrated_constant_blend':r[ncal:]+constant*delta[ncal:],
                'calibration_best_single':c[ncal:] if cnn_single else r[ncal:]}
    stacking=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(apred[:ncal,:2],observed)
    strategies['standard_stacking_ridge']=stacking.predict(apred[ncal:,:2]);joblib.dump(stacking,out/'standard_stacking_ridge.joblib')
    gate_scores=[];weights=[]
    for name,v in gate_X.items():
        assert np.isfinite(v).all()
        if len(np.unique(winner))<2:
            probs=np.full(len(te),float(winner[0]));gate=None
        else:
            gate=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=42))
            gate.fit(v[:ncal],winner);probs=gate.predict_proba(v[ncal:])[:,1]
            joblib.dump(dict(model=gate,features=names[name]),out/f'gate_{name}.joblib')
        strategies['soft_gate_'+name]=r[ncal:]+probs*delta[ncal:]
        if name=='all_with_predictions':strategies['hard_gate_'+name]=np.where(probs>=.5,c[ncal:],r[ncal:])
        truth=test_winner
        gate_scores.append(dict(gate=name,year=year,target=target,protocol=protocol,fold=fold,n_test=len(te),
            n_cal=len(cal),calibration_CNN_win_fraction=float(winner.mean()),test_CNN_win_fraction=float(truth.mean()),
            AUC=float(roc_auc_score(truth,probs)) if len(np.unique(truth))==2 else None,
            Brier=float(brier_score_loss(truth,probs)),constant_Brier=float(brier_score_loss(truth,np.full(len(te),winner.mean()))),
            balanced_accuracy=float(balanced_accuracy_score(truth,probs>=.5))))
        weights.append(pd.DataFrame(dict(point_id=d['point_id'][te],gate=name,CNN_weight=probs,true_winner_CNN=truth)))
    physical={name:np.maximum(0,np.expm1(v) if target=='log' else v) for name,v in strategies.items()}
    physical['fixed_50_physical_scale']=(pred_rf[ncal:]+pred_cnn[ncal:])/2
    assert all(np.isfinite(v).all() for v in physical.values())
    rows=pd.DataFrame(dict(point_id=d['point_id'][te],actual_SOC=d['y'][te],year=year,target=target,protocol=protocol,fold=fold,**physical))
    rows.to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(gate_scores).to_csv(out/'gate_scores.csv',index=False)
    pd.concat(weights,ignore_index=True).to_csv(out/'gate_weights.csv',index=False)
    roles=[]
    for name,idx in [('expert_train',expert),('gate_calibration',cal),('outer_test',te),('epoch_train',it),('epoch_validation',iv)]:
        roles.append(pd.DataFrame(dict(point_id=d['point_id'][idx],role=name,block100=blocks[idx])))
    pd.concat(roles,ignore_index=True).to_csv(out/'roles.csv',index=False)
    done.write_text(json.dumps(dict(fingerprint=fingerprint,year=year,target=target,protocol=protocol,fold=fold,seed=0,
        n_outer_train=len(tr),n_expert=len(expert),n_calibration=len(cal),n_test=len(te),n_epoch_train=len(it),n_epoch_valid=len(iv),
        distances_m=distances,epochs=epoch,inner_curve=curve,constant_CNN_weight=constant,
        calibration_best_single='patchCNN' if cnn_single else 'contextRF',
        elapsed_seconds=time.time()-start),indent=2))
    return out.name,time.time()-start

def metrics(y,p):
    e=p-y;le=np.log1p(p)-np.log1p(y)
    high=y>=150;low=~high
    return dict(n=len(y),RMSE=float(np.sqrt(np.mean(e**2))),MAE=float(np.mean(abs(e))),log_RMSE=float(np.sqrt(np.mean(le**2))),
        raw_R2=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)),log_R2=float(1-np.sum(le**2)/np.sum((np.log1p(y)-np.log1p(y).mean())**2)),
        low_RMSE=float(np.sqrt(np.mean(e[low]**2))),high_bias=float(e[high].mean()))

def main(root,workers):
    (HERE/'pilot').mkdir(exist_ok=True)
    design=dict(name='honest_three_way_pilot',exploratory=True,outer_seed=0,outer_folds=5,years=[2018,2022],targets=['log','raw'],protocols=['block_100','block_100_buffer20'],
        split='First GroupKFold(3) split of each outer training set: ~2/3 expert A, ~1/3 calibration B. Epoch selection: first GroupKFold(3) inside A. Test C unchanged.',
        buffer='20km A-B and inner epoch train-validation; original buffered outer training ensures A-C and B-C >20km.',
        expert_refit='NO expert refit on A+B after learning gate. Both test and calibration predictions come from exactly the same experts fitted on A.',
        gate='StandardScaler fitted on calibration B only; LogisticRegression C=0.1, no tuning, no class weighting. Winner uses target-space absolute error. Soft weight is predicted CNN-win probability.',
        gate_features='Center+climate 17, heterogeneity 29, model predictions 3; five predefined feature sets. No true SOC, coordinates, observed landcover or test error as predictors.',
        constant='Calibration-only optimal convex blend minimizing target-scale MSE; clipped to [0,1].',
        stacking='Calibration-only StandardScaler + Ridge(alpha=10) on two expert predictions; no tuning.',
        caution='One outer repeat, one country, same targets as previous experiments. Gate winner classification and SOC mixture utility are different objectives. This is not a novelty claim.',
        hashes={str(p.relative_to(root)):sha(p) for p in [root/'cnn/train_patch_models.py',root/'cnn/prepare_patches.py',root/'target_compare.py',root/'stage2/rf_compare.py',root/'stage2/results/fold_assignments.csv',root/'cnn/data/patches_2018.npz',root/'cnn/data/patches_2022.npz']},
        code_hash=sha(__file__))
    design['fingerprint']=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    path=HERE/'pilot_design.json'
    if path.exists():assert json.loads(path.read_text())['fingerprint']==design['fingerprint'],'Use a separate folder after any design change.'
    path.write_text(json.dumps(design,indent=2))
    jobs=[(root,y,t,p,f) for y in [2018,2022] for t in ['log','raw'] for p in ['block_100','block_100_buffer20'] for f in range(5)]
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(main_job,*j) for j in jobs]
        for i,f in enumerate(cf.as_completed(fs),1):
            name,seconds=f.result();print(f'Completed {i}/40 {name} ({seconds:.1f}s)',flush=True)
    paths=sorted((HERE/'pilot').glob('*/predictions.csv'));assert len(paths)==40
    predictions=pd.concat([pd.read_csv(p,dtype={'point_id':str}) for p in paths],ignore_index=True)
    assert len(predictions)==663*8 and not predictions.duplicated(['point_id','year','target','protocol']).any()
    strategies=[c for c in predictions if c not in ['point_id','actual_SOC','year','target','protocol','fold']]
    scores=[];fold_scores=[]
    for k,g in predictions.groupby(['year','target','protocol']):
        assert len(g)==663 and g.point_id.nunique()==663
        for name in strategies:scores.append(dict(zip(['year','target','protocol'],k),strategy=name,**metrics(g.actual_SOC.to_numpy(),g[name].to_numpy())))
        for fold,h in g.groupby('fold'):
            for name in strategies:fold_scores.append(dict(zip(['year','target','protocol'],k),fold=int(fold),strategy=name,**metrics(h.actual_SOC.to_numpy(),h[name].to_numpy())))
    predictions.to_csv(HERE/'results/honest_pilot_predictions.csv',index=False)
    pd.DataFrame(scores).to_csv(HERE/'results/honest_pilot_scores.csv',index=False)
    pd.DataFrame(fold_scores).to_csv(HERE/'results/honest_pilot_fold_scores.csv',index=False)
    gates=pd.concat([pd.read_csv(p) for p in sorted((HERE/'pilot').glob('*/gate_scores.csv'))],ignore_index=True)
    gates.to_csv(HERE/'results/honest_gate_fold_scores.csv',index=False)
    weights=[]
    for p in sorted((HERE/'pilot').glob('*/gate_weights.csv')):
        status=json.loads((p.parent/'completion.json').read_text());a=pd.read_csv(p,dtype={'point_id':str})
        for name in ['year','target','protocol','fold']:a[name]=status[name]
        weights.append(a)
    weights=pd.concat(weights,ignore_index=True);gw=[]
    for k,g in weights.groupby(['year','target','protocol','gate']):
        truth=g.true_winner_CNN.to_numpy();prob=g.CNN_weight.to_numpy()
        baseline=g.merge(gates[['year','target','protocol','gate','fold','calibration_CNN_win_fraction']],on=['year','target','protocol','gate','fold'],validate='many_to_one').calibration_CNN_win_fraction.to_numpy()
        gw.append(dict(zip(['year','target','protocol','gate'],k),n=len(g),AUC=roc_auc_score(truth,prob),Brier=brier_score_loss(truth,prob),
            constant_Brier=brier_score_loss(truth,baseline),balanced_accuracy=balanced_accuracy_score(truth,prob>=.5),
            CNN_winner_fraction=float(truth.mean())))
    pd.DataFrame(gw).to_csv(HERE/'results/honest_gate_pooled_scores.csv',index=False)
    print('Completed honest pilot: 40 expert pairs; 8 × 663 outer-test points; 13 strategies.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage3-root',type=Path,default=HERE.parent/'target_transform');ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();main(args.stage3_root.resolve(),args.workers)
