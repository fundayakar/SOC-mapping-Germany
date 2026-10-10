"""Paired nominal/GPS extraction comparison with frozen outer AND inner roles.

Never falls back to shifted/padded imagery. Refit both centers on the same 662
points. Original model implementation is preserved under vendor/.
"""
from pathlib import Path
import argparse, concurrent.futures as cf, hashlib, json, os, sys, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'vendor/cnn'), str(ROOT/'vendor'), str(ROOT/'vendor/stage2')]
import train_patch_models as t
import rf_compare as stage2

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, obj):
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, allow_nan=False)); tmp.replace(path)

def make_design():
    paths = [ROOT/'inputs/frozen_roles_662.csv', ROOT/'inputs/gps_crosswalk_662.csv',
             ROOT/'run_comparison.py', ROOT/'prepare_gps.py', ROOT/'GEE_SOC_GPS_patches.js',
             ROOT/'inputs/nominal_replay_reference.csv', ROOT/'inputs/SOC_GPS_patch17_2018.csv',
             ROOT/'inputs/SOC_GPS_patch17_2022.csv'] + sorted((ROOT/'vendor').rglob('*.py'))
    paths += sorted((ROOT/'data').glob('*.npz'))
    design = dict(n=662, years=[2018,2022], centers=['nominal','gps'],
        targets=['raw','log'], models=['centerRF','contextRF','centerMLP','patchCNN'],
        protocol_repeats={'block_100':5,'block_100_buffer20':3},
        outer_inner='Original 663-point outer/inner memberships, restricted to the same 662 GPS-usable points; never recalculated from GPS.',
        preparation='Same 17x17, 14 raster bands, common validity mask, 3 native-grid climate values; exact new GPS-centered GEE extraction.',
        RF=stage2.RF, torch_version=t.torch.__version__, threads=1,
        neural='Original architecture; Adam .001, weight decay .0001, batch32; max100 epochs, patience15, train-only normalization and epoch selection.',
        seeds='RF42; neural42+100*repeat+fold; same for both centers.',
        coordinate_interpretation='GPS observations have finite precision; extraction sensitivity, not a causal decomposition or exact soil-center validation.',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    design['fingerprint']=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    dest=ROOT/'results/design.json'
    if dest.exists(): assert json.loads(dest.read_text())==design, 'Inputs changed: use a new experiment directory.'
    else: atomic_json(dest,design)
    return design

def job(spec):
    center,year,protocol,seed,fold,target,model=spec
    t.torch.set_num_threads(1)
    name=f'{center}_{year}_{protocol}_s{seed}_f{fold}_{target}_{model}'
    path=ROOT/f'results/runs/{name}.json'
    design=json.loads((ROOT/'results/design.json').read_text())
    if path.exists():
        record=json.loads(path.read_text()); assert record['fingerprint']==design['fingerprint']
        return name,record['metrics']['raw_rmse'],True
    start=time.time();d=dict(np.load(ROOT/f'data/{center}_{year}.npz',allow_pickle=False))
    assert d['X'].shape==(662,14,17,17) and (d['valid'][:,0,8,8]==1).all()
    assert np.isfinite(d['X']).all() and np.isfinite(d['climate']).all()
    roles=pd.read_csv(ROOT/'inputs/frozen_roles_662.csv',dtype={'point_id':str})
    g=roles.query('protocol==@protocol and seed==@seed and fold==@fold')
    idx={p:i for i,p in enumerate(d['point_id'])}
    ix={role:np.array(sorted(idx[p] for p in a.point_id)) for role,a in g.groupby('role')}
    tr,te,it,iv=(ix[k] for k in ['train','test','epoch_train','epoch_valid'])
    assert not set(tr)&set(te) and set(it)<=set(tr) and set(iv)<=set(tr) and not set(it)&set(iv)
    bxy=np.floor(d['xy']/100000).astype(int);blocks=np.array([f'{x}_{y}' for x,y in bxy])
    assert not set(blocks[tr])&set(blocks[te]) and not set(blocks[it])&set(blocks[iv])
    nominal_distance=float(cKDTree(d['xy'][tr]).query(d['xy'][te])[0].min())
    inner_distance=float(cKDTree(d['xy'][it]).query(d['xy'][iv])[0].min())
    if protocol.endswith('buffer20'): assert min(nominal_distance,inner_distance)>20000
    assert cKDTree(d['patch_center_xy'][tr]).query(d['patch_center_xy'][te])[0].min()>np.sqrt(2)*340
    epoch=None;curve=[]
    if model.endswith('RF'):
        X=t.rf_features(d,model);y=np.log1p(d['y']) if target=='log' else d['y']
        rf=RandomForestRegressor(**stage2.RF).fit(X[tr],y[tr]);pred=rf.predict(X[te])
        if target=='log': pred=np.expm1(pred)
        pred=np.maximum(0,pred)
        # RF reproducibility uses frozen design and code; large RF binaries omitted.
    else:
        netseed=42+100*seed+fold
        _,_,epoch,curve=t.network_fit(d,it,iv,target,model,netseed,'cpu',100,15,True)
        network,norm,_,_=t.network_fit(d,tr,None,target,model,netseed,'cpu',epoch,15,False)
        pred=t.neural_predict(network,norm,d,te,target,'cpu')
        checkpoint=dict(state_dict=network.state_dict(),normalization={k:np.asarray(v).tolist() for k,v in norm.items()},
            normalization_dtypes={k:str(np.asarray(v).dtype) for k,v in norm.items()},epochs=epoch,fingerprint=design['fingerprint'])
        tmp=path.with_suffix('.pt.tmp');t.torch.save(checkpoint,tmp);tmp.replace(path.with_suffix('.pt'))
    rec=dict(center=center,year=year,protocol=protocol,seed=seed,fold=fold,target=target,model=model,
        point_id=d['point_id'][te].tolist(),actual_SOC=d['y'][te].tolist(),pred_SOC=pred.tolist(),
        selected_epochs=epoch,inner_curve=curve,n_train=len(tr),n_test=len(te),
        nominal_train_test_distance_m=nominal_distance,nominal_inner_distance_m=inner_distance,
        fingerprint=design['fingerprint'],metrics=t.raw_metrics(d['y'][te],pred),elapsed_seconds=time.time()-start)
    atomic_json(path,rec)
    return name,rec['metrics']['raw_rmse'],False

def summarize(design):
    paths=sorted((ROOT/'results/runs').glob('*.json'));rows=[]
    for p in paths:
        r=json.loads(p.read_text());assert r['fingerprint']==design['fingerprint']
        meta={k:r[k] for k in ['center','year','protocol','seed','fold','target','model']}
        rows.append(pd.DataFrame(dict(**meta,point_id=r['point_id'],actual_SOC=r['actual_SOC'],pred_SOC=r['pred_SOC'])))
    predictions=pd.concat(rows,ignore_index=True);predictions.to_csv(ROOT/'results/oof_predictions.csv',index=False)
    scores=[]
    for key,g in predictions.groupby(['center','year','protocol','seed','target','model']):
        if len(g)!=662 or g.point_id.nunique()!=662: continue
        scores.append(dict(zip(['center','year','protocol','seed','target','model'],key),**t.raw_metrics(g.actual_SOC.to_numpy(),g.pred_SOC.to_numpy())))
    scores=pd.DataFrame(scores);scores.to_csv(ROOT/'results/pooled_oof_by_repeat.csv',index=False)
    keys=['year','protocol','seed','target','model'];a=scores.query("center=='nominal'").set_index(keys);b=scores.query("center=='gps'").set_index(keys)
    common=a.index.intersection(b.index);paired=a.loc[common,[]].copy()
    for m in ['raw_r2','log_r2','raw_rmse','raw_mae','raw_bias','ccc','below150_raw_rmse','ge150_raw_bias']:
        paired[f'nominal_{m}']=a.loc[common,m];paired[f'gps_{m}']=b.loc[common,m];paired[f'delta_{m}']=b.loc[common,m]-a.loc[common,m]
    paired.reset_index().to_csv(ROOT/'results/paired_deltas_by_repeat.csv',index=False)
    complete=len(paths)==1280
    atomic_json(ROOT/'results/completion.json',dict(all_completed=complete,completed_fits=len(paths),expected_fits=1280,
        pooled_groups=len(scores),expected_pooled_groups=256,fingerprint=design['fingerprint']))
    print('Completion:',len(paths),'/1280 fits;',len(scores),'/256 pooled groups',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--centers',nargs='+',default=['nominal','gps']);p.add_argument('--summary-only',action='store_true');a=p.parse_args()
    (ROOT/'results/runs').mkdir(parents=True,exist_ok=True);design=make_design()
    if a.summary_only: summarize(design);return
    jobs=[]
    for center in a.centers:
      for year in [2018,2022]:
       if not (ROOT/f'data/{center}_{year}.npz').exists(): raise FileNotFoundError(f'Missing real {center} data, {year}')
       for protocol,repeats in design['protocol_repeats'].items():
        for seed in range(repeats):
         for fold in range(5):
          for target in ['raw','log']:
           for model in design['models']:jobs.append((center,year,protocol,seed,fold,target,model))
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(job,j):j for j in jobs}
        for i,f in enumerate(cf.as_completed(futures),1):
            name,rmse,cached=f.result()
            print(f'{i}/{len(jobs)} {name} RMSE={rmse:.4f} cached={cached}',flush=True)
    summarize(design)

if __name__=='__main__': main()
