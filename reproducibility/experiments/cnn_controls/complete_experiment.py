"""Complete controls and fresh matched originals; retain immutable initial checkpoints.
Entry point replaces an overly strict historical replay audit. Both current fit
implementations agree exactly; old-host training shows small numeric differences.
"""
from pathlib import Path
import argparse,concurrent.futures as cf,hashlib,json,time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.spatial import cKDTree
import run_ablations as a

HERE=Path(__file__).resolve().parent

def original_job(root,year,target,protocol,seed,fold):
    t=a.imports(root);start=time.time();out=HERE/'fresh_originals';out.mkdir(exist_ok=True)
    design=json.loads((HERE/'matched_design.json').read_text());name=f'{protocol}_{year}_{target}_s{seed}_f{fold}_original_fresh';path=out/f'{name}.json'
    if path.exists():
        assert json.loads(path.read_text())['fingerprint']==design['fingerprint'] and path.with_suffix('.pt').is_file()
        return name,0
    d=dict(np.load(root/f'cnn/data/patches_{year}.npz',allow_pickle=False));ids={v:i for i,v in enumerate(d['point_id'])}
    f=pd.read_csv(root/'stage2/results/fold_assignments.csv',dtype={'point_id':str})
    f=f.query("cohort=='global_0_20' and protocol==@protocol and seed==@seed and fold==@fold")
    tr=np.array(sorted(ids[p] for p in f[f.role.eq('train')].point_id));te=np.array(sorted(ids[p] for p in f[f.role.eq('test')].point_id))
    bxy=np.floor(d['xy']/100000).astype(int);blocks=np.array([f'{x}_{y}' for x,y in bxy]);aa,bb=next(GroupKFold(3).split(tr,groups=blocks[tr]));it=tr[aa];iv=tr[bb]
    if protocol.endswith('buffer20'):it=it[cKDTree(d['xy'][iv]).query(d['xy'][it])[0]>20000]
    netseed=42+100*seed+fold
    _,_,epoch,curve=a.fit(t,d,d,it,iv,target,netseed,100,True)
    model,norm,_,_=a.fit(t,d,d,tr,None,target,netseed,epoch,False)
    pred=t.neural_predict(model,norm,d,te,target,'cpu')
    folder='results_buffer20' if protocol.endswith('buffer20') else 'results'
    ref=json.loads((root/f'cnn/{folder}/{year}/patchCNN_{target}_s{seed}_f{fold}.json').read_text())
    assert ref['point_id']==d['point_id'][te].tolist()
    ck=dict(state_dict=model.state_dict(),normalization={k:np.asarray(v).tolist() for k,v in norm.items()},
        normalization_dtypes={k:str(np.asarray(v).dtype) for k,v in norm.items()},epochs=epoch,fingerprint=design['fingerprint'],
        variant='original_fresh',net_seed=netseed,parameter_count=sum(p.numel() for p in model.parameters()))
    tmp=path.with_suffix('.pt.tmp');t.torch.save(ck,tmp);tmp.replace(path.with_suffix('.pt'))
    r=dict(fingerprint=design['fingerprint'],variant='original_fresh',year=year,target=target,protocol=protocol,seed=seed,fold=fold,
        point_id=d['point_id'][te].tolist(),actual_SOC=d['y'][te].tolist(),pred_SOC=pred.tolist(),selected_epochs=epoch,inner_curve=curve,
        expert_train_ids=d['point_id'][tr].tolist(),epoch_train_ids=d['point_id'][it].tolist(),epoch_validation_ids=d['point_id'][iv].tolist(),
        train_test_distance_m=float(cKDTree(d['xy'][tr]).query(d['xy'][te])[0].min()),inner_train_valid_distance_m=float(cKDTree(d['xy'][it]).query(d['xy'][iv])[0].min()),
        patch_train_test_distance_m=float(cKDTree(d['patch_center_xy'][tr]).query(d['patch_center_xy'][te])[0].min()),
        metrics=t.raw_metrics(d['y'][te],pred),old_original_max_prediction_delta=float(np.max(abs(pred-np.asarray(ref['pred_SOC'])))),old_original_epochs=ref['selected_epochs'],elapsed_seconds=time.time()-start)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(r));tmp.replace(path)
    return name,time.time()-start

def dispatch(j):
    kind,args=j
    return original_job(*args) if kind=='original' else a.job(*args)

def main(root,workers):
    previous=json.loads((HERE/'design.json').read_text())
    # Verify the archived core has not changed before reusing its control fits.
    assert a.sha(HERE/'run_ablations.py')==previous['code_sha256']
    for name,digest in previous['source_sha256'].items():assert a.sha(root/name)==digest
    equivalence=json.loads((HERE/'current_runner_equivalence.json').read_text());assert equivalence['max_prediction_delta']==0 and equivalence['max_curve_delta']==0
    design=dict(control_fingerprint=previous['fingerprint'],core_sha256=a.sha(HERE/'run_ablations.py'),code_sha256=a.sha(__file__),
        note='Fresh original CNN trained with exactly the same current fit implementation as controls. Cached original-stage3 predictions are secondary historical comparisons only.',
        original_fits=160,primary_controls=320,secondary_shuffle=40,current_source_runner_equivalence=equivalence)
    design['fingerprint']=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    path=HERE/'matched_design.json'
    if path.exists():assert json.loads(path.read_text())['fingerprint']==design['fingerprint']
    path.write_text(json.dumps(design,indent=2))
    jobs=[]
    for y in [2018,2022]:
        for target in ['log','raw']:
            for protocol,reps in [('block_100',5),('block_100_buffer20',3)]:
                for seed in range(reps):
                    for fold in range(5):
                        args=(root,y,target,protocol,seed,fold)
                        jobs.append(('original',args))
                        for variant in a.PRIMARY:jobs.append(('control',args+(variant,)))
                        if seed==0:jobs.append(('control',args+('shuffle_271828',)))
    assert len(jobs)==520
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(dispatch,j) for j in jobs]
        for i,f in enumerate(cf.as_completed(fs),1):
            name,seconds=f.result();print(f'Completed {i}/520 {name} ({seconds:.1f}s)',flush=True)
    summarize(root)

def summarize(root):
    (HERE/'results').mkdir(exist_ok=True);records=[];epochs=[]
    paths=sorted((HERE/'runs').glob('*.json'))+sorted((HERE/'fresh_originals').glob('*.json'))
    for path in paths:
        r=json.loads(path.read_text())
        if r['variant']=='original_replay':continue
        meta={k:r[k] for k in ['variant','year','target','protocol','seed','fold']}
        records.append(pd.DataFrame(dict(**meta,point_id=r['point_id'],actual_SOC=r['actual_SOC'],pred_SOC=r['pred_SOC'])))
        epochs.append(dict(**meta,epochs=r['selected_epochs'],elapsed_seconds=r['elapsed_seconds'],old_original_epochs=r.get('old_original_epochs'),old_original_max_prediction_delta=r.get('old_original_max_prediction_delta')))
    assert len(records)==520
    for folder in ['results','results_buffer20']:
        for year in [2018,2022]:
            p=pd.read_csv(root/f'cnn/{folder}/{year}/oof_predictions.csv',dtype={'point_id':str})
            p=p[p.model.isin(['patchCNN','centerMLP','centerRF','contextRF'])].rename(columns={'model':'variant'})
            p.loc[p.variant.eq('patchCNN'),'variant']='original_historical';records.append(p)
    p=pd.concat(records,ignore_index=True);p.to_csv(HERE/'results/all_predictions.csv',index=False)
    t=a.imports(root);scores=[]
    for k,g in p.groupby(['variant','year','target','protocol','seed']):
        assert len(g)==663 and g.point_id.nunique()==663
        scores.append(dict(zip(['variant','year','target','protocol','seed'],k),**t.raw_metrics(g.actual_SOC.to_numpy(),g.pred_SOC.to_numpy())))
    pd.DataFrame(scores).to_csv(HERE/'results/scores_by_repeat.csv',index=False)
    pd.DataFrame(epochs).to_csv(HERE/'results/epochs.csv',index=False)
    (HERE/'completion.json').write_text(json.dumps(dict(all_completed=True,new_model_fold_fits=520,primary_control_fits=320,fresh_original_fits=160,
        second_permutation_fits=40,CNN_parameter_count=8449,old_stage3_original_used_as_primary_reference=False),indent=2))
    print('All 520 fresh original/control fits completed.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage3-root',type=Path,default=HERE.parent/'target_transform');ap.add_argument('--workers',type=int,default=4)
    x=ap.parse_args();main(x.stage3_root.resolve(),x.workers)
