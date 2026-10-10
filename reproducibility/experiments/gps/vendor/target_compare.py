"""Diagnostic follow-up: target scale and training-only spatial cross-fit smearing.
Uses unchanged outer folds from the stage-2 package; not a new external test.
"""
import argparse,concurrent.futures as cf,hashlib,json,sys,time,platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy,sklearn
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score,mean_absolute_error

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage2'))
import rf_compare as stage2
DATA={}

def raw_metrics(y,p):
    assert np.isfinite(p).all() and (p>=0).all()
    e=p-y;high=y>=150;low=~high;sse=e**2
    den=np.var(y)+np.var(p)+(y.mean()-p.mean())**2
    return dict(n=len(y),raw_r2=float(r2_score(y,p)),log_r2=float(r2_score(np.log1p(y),np.log1p(p))),
        raw_rmse=float(np.sqrt(sse.mean())),raw_mae=float(mean_absolute_error(y,p)),raw_bias=float(e.mean()),
        ccc=float(2*np.mean((y-y.mean())*(p-p.mean()))/den),
        below150_raw_r2=float(r2_score(y[low],p[low])),below150_raw_rmse=float(np.sqrt(sse[low].mean())),
        below150_raw_mae=float(abs(e[low]).mean()),below150_raw_bias=float(e[low].mean()),
        n_ge150=int(high.sum()),ge150_raw_bias=float(e[high].mean()) if high.any() else None,
        ge150_raw_mae=float(abs(e[high]).mean()) if high.any() else None,
        ge150_observed_mean=float(y[high].mean()) if high.any() else None,
        ge150_predicted_mean=float(p[high].mean()) if high.any() else None,
        ge150_sse_fraction=float(sse[high].sum()/sse.sum()))

def setup(data,out,design_hash):
    d=data['global_0_20'];ix={v:i for i,v in enumerate(d['ids'])}
    f=pd.read_csv(ROOT/'stage2/results/fold_assignments.csv',dtype={'point_id':str})
    f=f.query("cohort=='global_0_20' and protocol in ['block_100','block_100_buffer20']")
    jobs=[];diagnostics=[];inner_assignments=[]
    bxy=np.floor(d['xy']/100000).astype(int);blocks=np.array([f'{x}_{y}' for x,y in bxy])
    for (protocol,seed,fold),g in f.groupby(['protocol','seed','fold']):
        tr=sorted(ix[p] for p in g[g.role.eq('train')].point_id)
        te=sorted(ix[p] for p in g[g.role.eq('test')].point_id)
        tr=np.array(tr);te=np.array(te)
        assert not set(tr)&set(te) and not set(blocks[tr])&set(blocks[te])
        outer_distance=cKDTree(d['xy'][tr]).query(d['xy'][te])[0].min()/1000
        if protocol.endswith('buffer20'):assert outer_distance>20
        inner=[]
        for inner_fold,(a,b) in enumerate(GroupKFold(3).split(tr,groups=blocks[tr])):
            train=tr[a];valid=tr[b]
            if protocol.endswith('buffer20'):
                keep=cKDTree(d['xy'][valid]).query(d['xy'][train])[0]>20000
                train=train[keep]
            assert min(len(train),len(valid))>=20 and not set(blocks[train])&set(blocks[valid])
            assert set(train)<=set(tr) and set(valid)<=set(tr) and not set(valid)&set(te)
            distance=cKDTree(d['xy'][train]).query(d['xy'][valid])[0].min()/1000
            if protocol.endswith('buffer20'):assert distance>20
            inner.append(dict(train=train.tolist(),valid=valid.tolist()))
            diagnostics.append(dict(protocol=protocol,seed=int(seed),outer_fold=int(fold),inner_fold=inner_fold,n_train=len(train),n_valid=len(valid),minimum_training_distance_km=float(distance)))
            for role,indices in [('train',train),('valid',valid)]:
                inner_assignments.extend(dict(protocol=protocol,seed=int(seed),outer_fold=int(fold),inner_fold=inner_fold,role=role,point_id=d['ids'][i]) for i in indices)
        assert sorted(i for s in inner for i in s['valid'])==tr.tolist()
        models=['M1','M6'] if protocol.endswith('buffer20') else d['names']
        for model in models:
            for year in [2018,2022]:
                job=dict(protocol=protocol,seed=int(seed),fold=int(fold),model=model,year=year,train=tr.tolist(),test=te.tolist(),inner=inner,design_hash=design_hash)
                job['id']=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:24]
                jobs.append(job)
    pd.DataFrame(diagnostics).to_csv(out/'inner_fold_diagnostics.csv',index=False)
    pd.DataFrame(inner_assignments).to_csv(out/'inner_fold_assignments.csv',index=False)
    assert len(jobs)==310
    return jobs

def initialize(data):
    global DATA
    DATA=data['global_0_20']

def fit(job):
    d=DATA;X=d['X'][(job['model'],job['year'])];z=d['y'];y=np.expm1(z)
    tr=np.array(job['train']);te=np.array(job['test']);start=time.time()
    calibration=[]
    for inner_fold,s in enumerate(job['inner']):
        it=np.array(s['train']);iv=np.array(s['valid'])
        pred=RandomForestRegressor(**stage2.RF).fit(X[it],z[it]).predict(X[iv])
        calibration.extend(dict(point_id=d['ids'][i],inner_fold=inner_fold,true_log=float(z[i]),pred_log=float(p)) for i,p in zip(iv,pred))
    errors=np.array([r['true_log']-r['pred_log'] for r in calibration])
    factor=float(np.exp(errors).mean())
    assert np.isfinite(factor) and factor>0
    pz=RandomForestRegressor(**stage2.RF).fit(X[tr],z[tr]).predict(X[te])
    pr=RandomForestRegressor(**stage2.RF).fit(X[tr],y[tr]).predict(X[te])
    predictions={'log_naive':np.expm1(pz),'log_spatial_smear':np.maximum(0,np.exp(pz)*factor-1),'raw_SOC_RF':pr}
    return dict(job={k:v for k,v in job.items() if k not in ['train','test','inner']},n_train=len(tr),
        smearing_factor=factor,calibration=calibration,point_id=d['ids'][te].tolist(),actual_SOC=y[te].tolist(),
        predictions={k:v.tolist() for k,v in predictions.items()},metrics={k:raw_metrics(y[te],v) for k,v in predictions.items()},
        elapsed_seconds=time.time()-start)

def summarize(jobs,out):
    frames=[];factors=[]
    for job in jobs:
        r=json.loads((out/'checkpoints'/f'{job["id"]}.json').read_text())
        m={k:job[k] for k in ['protocol','seed','fold','model','year']}
        factors.append(dict(**m,n_train=r['n_train'],factor=r['smearing_factor']))
        for method,p in r['predictions'].items():
            frames.append(pd.DataFrame(dict(**m,method=method,point_id=r['point_id'],actual_SOC=r['actual_SOC'],pred_SOC=p)))
    p=pd.concat(frames,ignore_index=True);p.to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(factors).to_csv(out/'smearing_factors.csv',index=False)
    # Verify the original baseline is numerically reproduced for every shared prediction.
    old=pd.read_csv(ROOT/'stage2/results/oof_predictions.csv',dtype={'point_id':str}).query("cohort=='global_0_20'")
    keys=['protocol','seed','fold','model','year','point_id']
    check=p[p.method.eq('log_naive')].merge(old,on=keys,validate='one_to_one')
    assert len(check)==len(p[p.method.eq('log_naive')])
    max_delta=float(np.max(np.abs(check.pred_SOC-np.expm1(check.pred_log))))
    assert max_delta<1e-10,(max_delta,'Baseline changed')
    pooled=[];keys=['protocol','model','year','method','seed']
    for key,g in p.groupby(keys):
        assert len(g)==663 and g.point_id.nunique()==663
        pooled.append(dict(zip(keys,key),**raw_metrics(g.actual_SOC.to_numpy(),g.pred_SOC.to_numpy())))
    pooled=pd.DataFrame(pooled);pooled.to_csv(out/'pooled_oof_by_repeat.csv',index=False)
    measures=[k for k in pooled.columns if k not in keys+['n','n_ge150']]
    s=pooled.groupby(keys[:-1])[measures].agg(['mean','std','min','max']).reset_index()
    s.columns=['_'.join(c).rstrip('_') for c in s.columns];s.to_csv(out/'performance_summary.csv',index=False)
    pairs=[]
    for method in ['log_spatial_smear','raw_SOC_RF']:
        k=['protocol','model','year','seed']
        j=pooled[pooled.method.eq(method)].merge(pooled[pooled.method.eq('log_naive')],on=k,suffixes=('_new','_naive'),validate='one_to_one')
        for _,r in j.iterrows():
            row={x:r[x] for x in k};row['method']=method
            for metric in measures:row['delta_'+metric]=r[metric+'_new']-r[metric+'_naive']
            pairs.append(row)
    pd.DataFrame(pairs).to_csv(out/'paired_changes_vs_naive.csv',index=False)
    return max_delta

def main(workers):
    out=ROOT/'results';out.mkdir(exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    data,features,tables=stage2.load(ROOT/'stage2/input')
    files=sorted((ROOT/'stage2/input').rglob('*.csv'))+[ROOT/'stage2/results/fold_assignments.csv',Path(__file__),ROOT/'stage2/rf_compare.py']
    design=dict(RF=stage2.RF,features=features,n_points=663,depth='0-20 cm',target='2018 SOC concentration g/kg',
        methods=['log_naive','log_spatial_smear','raw_SOC_RF'],
        primary='unchanged stage-2 100km 5x5 spatial outer folds; five models x two predictor years',
        sensitivity='unchanged stage-2 100km+20km buffered 3x5 outer folds; M1/M6 x two years',
        smearing='S = mean(exp(z - inner held-out zhat)); corrected = max(0, exp(outer zhat)*S - 1). Three grouped 100km inner folds restricted to outer training; buffered protocol also removes inner training points <=20km from inner validation.',
        selection='Diagnostic follow-up motivated by inspected stage-2 held-out errors; not independent confirmation. No test-set-based calibration, clipping threshold selection, hyperparameter tuning or organic-soil exclusion.',
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    design_hash=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest();design['design_hash']=design_hash
    path=out/'design.json'
    if path.exists():assert json.loads(path.read_text())['design_hash']==design_hash,'Design changed; use new output directory'
    path.write_text(json.dumps(design,indent=2))
    jobs=setup(data,out,design_hash);pending=[j for j in jobs if not (out/'checkpoints'/f'{j["id"]}.json').exists()]
    print(f'{len(jobs)} outer jobs, {len(pending)} pending; five RF fits/job, {len(jobs)*5} total fits',flush=True)
    with cf.ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(data,)) as pool:
        fs={pool.submit(fit,j):j for j in pending}
        for i,f in enumerate(cf.as_completed(fs),1):
            job=fs[f];r=f.result();path=out/'checkpoints'/f'{job["id"]}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(r));tmp.replace(path)
            if i%10==0 or i==len(pending):print(f'Completed {i}/{len(pending)} outer jobs',flush=True)
    delta=summarize(jobs,out)
    (out/'completion.json').write_text(json.dumps(dict(all_completed=True,outer_jobs=len(jobs),RF_fits=len(jobs)*5,baseline_reproduction_max_abs_error=delta,cnn_trained=False),indent=2))
    print(pd.read_csv(out/'performance_summary.csv').query("protocol=='block_100' and model=='M6'")[['year','method','raw_r2_mean','raw_rmse_mean','raw_mae_mean','log_r2_mean','ge150_raw_bias_mean']].round(4).to_string(index=False),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=6);main(a.parse_args().workers)
