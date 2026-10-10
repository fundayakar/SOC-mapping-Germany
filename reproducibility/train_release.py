"""Portable entry to unchanged fit functions, with frozen published memberships.

All new runs have separate fingerprinted outputs; archived source drivers and
their cached-run guards are preserved for provenance, not used as this entry.
"""
from pathlib import Path
import argparse, concurrent.futures as cf, hashlib, json, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parent
MAIN=ROOT/'experiments/main'
sys.path[:0]=[str(MAIN/'cnn'),str(MAIN),str(MAIN/'stage2'),str(ROOT/'experiments/cnn_controls')]
import train_patch_models as t
import rf_compare
import run_ablations as controls

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fit_job(spec,data_root,out,fingerprint):
    suite,center,year,protocol,seed,fold,target,model=spec
    t.torch.set_num_threads(1)
    name=f'{suite}_{center}_{year}_{protocol}_s{seed}_f{fold}_{target}_{model}'
    dest=out/'runs'/f'{name}.json'
    if dest.exists():
        r=json.loads(dest.read_text());assert r['fingerprint']==fingerprint
        if not model.endswith('RF'):assert dest.with_suffix('.pt').exists()
        return name,'cached'
    orig=dict(np.load(data_root/f'{center}_{year}.npz',allow_pickle=False))
    d=controls.transformed(orig,model) if suite=='controls' else orig
    roles=pd.read_csv(ROOT/f'data/roles/{"gps" if suite=="gps" else "main"}_roles.csv',dtype={'point_id':str})
    roles=roles.query('protocol==@protocol and seed==@seed and fold==@fold')
    ix={v:i for i,v in enumerate(d['point_id'])}
    indices={role:np.array(sorted(ix[p] for p in g.point_id)) for role,g in roles.groupby('role')}
    tr,te,it,iv=[indices[k] for k in ['train','test','epoch_train','epoch_valid']]
    assert not set(tr)&set(te) and not set(it)&set(iv) and set(it)|set(iv)<=set(tr)
    block=np.floor(d['xy']/100000).astype(int);blocks=np.array([f'{x}_{y}' for x,y in block])
    assert not set(blocks[tr])&set(blocks[te]) and not set(blocks[it])&set(blocks[iv])
    if protocol.endswith('buffer20'):
        assert min(cKDTree(d['xy'][tr]).query(d['xy'][te])[0].min(),cKDTree(d['xy'][it]).query(d['xy'][iv])[0].min())>20000
    epoch=None;curve=[];started=time.time()
    if model.endswith('RF'):
        X=t.rf_features(d,model);y=np.log1p(d['y']) if target=='log' else d['y']
        pred=RandomForestRegressor(**rf_compare.RF).fit(X[tr],y[tr]).predict(X[te])
        pred=np.maximum(0,np.expm1(pred) if target=='log' else pred)
    else:
        net_seed=42+100*seed+fold
        if suite=='controls' or (suite=='main' and model=='patchCNN'):
            _,_,epoch,curve=controls.fit(t,orig,d,it,iv,target,net_seed,100,True)
            network,norm,_,_=controls.fit(t,orig,d,tr,None,target,net_seed,epoch,False)
        else:
            _,_,epoch,curve=t.network_fit(d,it,iv,target,model,net_seed,'cpu',100,15,True)
            network,norm,_,_=t.network_fit(d,tr,None,target,model,net_seed,'cpu',epoch,15,False)
        pred=t.neural_predict(network,norm,d,te,target,'cpu')
        t.torch.save({'state_dict':network.state_dict(),'normalization':{k:np.asarray(v).tolist() for k,v in norm.items()},'normalization_dtypes':{k:str(np.asarray(v).dtype) for k,v in norm.items()},'epochs':epoch,'fingerprint':fingerprint},dest.with_suffix('.pt'))
    record=dict(suite=suite,center=center,year=year,protocol=protocol,seed=seed,fold=fold,target=target,model=model,fingerprint=fingerprint,
        point_id=d['point_id'][te].tolist(),actual_SOC=d['y'][te].tolist(),pred_SOC=pred.tolist(),selected_epochs=epoch,inner_curve=curve,
        metrics=t.raw_metrics(d['y'][te],pred),elapsed_seconds=time.time()-started)
    dest.write_text(json.dumps(record));return name,'fitted'

def specs(suites):
    jobs=[]
    for suite in suites:
      for center in (['nominal','gps'] if suite=='gps' else ['main']):
       for year in [2018,2022]:
        for protocol,reps in [('block_100',5),('block_100_buffer20',3)]:
         for seed in range(reps):
          for fold in range(5):
           for target in ['log','raw']:
            models=['centerRF','contextRF','centerMLP','patchCNN'] if suite!='controls' else ['center_tiled','shuffle_314159']+(['shuffle_271828'] if seed==0 else [])
            jobs.extend((suite,center,year,protocol,seed,fold,target,m) for m in models)
    return jobs

def summarize(out,jobs,fingerprint,smoke=False):
    records=[]
    for spec in jobs:
        suite,center,year,protocol,seed,fold,target,model=spec
        r=json.loads((out/'runs'/f'{suite}_{center}_{year}_{protocol}_s{seed}_f{fold}_{target}_{model}.json').read_text());assert r['fingerprint']==fingerprint
        records.append(pd.DataFrame({**{k:r[k] for k in ['suite','center','year','protocol','seed','fold','target','model']},'point_id':r['point_id'],'actual_SOC':r['actual_SOC'],'pred_SOC':r['pred_SOC']}))
    p=pd.concat(records,ignore_index=True);p.to_csv(out/'oof_predictions.csv',index=False)
    rows=[]
    keys=['suite','center','year','protocol','seed','target','model']
    for key,g in p.groupby(keys):
        assert g.point_id.nunique()==len(g)
        if not smoke:assert len(g)==(662 if key[0]=='gps' else 663)
        rows.append(dict(zip(keys,key),**t.raw_metrics(g.actual_SOC.to_numpy(),g.pred_SOC.to_numpy())))
    pd.DataFrame(rows).to_csv(out/('partial_fold_scores.csv' if smoke else 'scores_by_repeat.csv'),index=False)
    (out/'completion.json').write_text(json.dumps({'all_completed':not smoke,'smoke_completed':smoke,'expected_fits':len(jobs),'completed_fits':len(records),'OOF_rows':len(p),'fingerprint':fingerprint},indent=2)+'\n')

def main(a):
    suites=['main','controls','gps'] if a.suite=='all' else [a.suite]
    jobs=specs(suites)
    # Smoke tests fit complete chosen folds, but never claim suite completion.
    if a.smoke:jobs=[j for j in jobs if j[2]==2018 and j[3]=='block_100' and j[4]==0 and j[5]==0 and j[6]=='log']
    data=Path(a.data).resolve();out=Path(a.output).resolve()
    if ROOT==out or ROOT in out.parents:raise ValueError('Keep new outputs outside the immutable release tree.')
    code=[Path(__file__),MAIN/'cnn/train_patch_models.py',MAIN/'target_compare.py',MAIN/'stage2/rf_compare.py',MAIN/'cnn/prepare_patches.py',ROOT/'experiments/cnn_controls/run_ablations.py']
    files=code+list((ROOT/'data/roles').glob('*.csv'))+sorted(data.glob('*.npz'))
    design={'suite':a.suite,'smoke':a.smoke,'torch_version':t.torch.__version__,'threads':1,'fits':len(jobs),'sha256':{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name:sha(p) for p in files}}
    fingerprint=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest();design['fingerprint']=fingerprint
    out.mkdir(parents=True,exist_ok=True);(out/'runs').mkdir(exist_ok=True)
    if (out/'design.json').exists():assert json.loads((out/'design.json').read_text())==design,'Changed design: use another output directory.'
    else:
        if any((out/'runs').iterdir()):raise ValueError('Unfingerprinted prior runs present.')
        (out/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(fit_job,j,data,out,fingerprint) for j in jobs]
        for i,f in enumerate(cf.as_completed(fs),1):print(i,len(jobs),*f.result(),flush=True)
    summarize(out,jobs,fingerprint,a.smoke)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--suite',choices=['main','controls','gps','all'],default='all');p.add_argument('--workers',type=int,default=4);p.add_argument('--smoke',action='store_true');main(p.parse_args())
