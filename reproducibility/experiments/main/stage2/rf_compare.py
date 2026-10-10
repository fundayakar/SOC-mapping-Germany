"""Paired RF predictor-year comparison using REAL harmonized GEE exports.

Run: python rf_compare.py --input input --output results --workers 6
Fixed RF settings; no tuning or test-set-selected feature choices. Checkpoints
are keyed by the complete design, input hashes, cohort, split and model.
"""
import argparse, concurrent.futures as cf, hashlib, json, os, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

RF=dict(n_estimators=800,min_samples_leaf=2,min_samples_split=2,
        max_features=1.0,bootstrap=True,criterion='squared_error',random_state=42,n_jobs=1)
DATA={}
COHORT_ID_SHA256={
    'global_0_20':'e4c0d5e372a55a15b7b7aa51787f4162ab6bfe708161b43f3f00ba17d59da8d4',
    'core_0_20':'93d4612968042c3456b3bdcd48476ae196a721e4c0a19cac5a66f2e74e82be10',
    'all_depths_global':'375e3f463df46b4044489063ceba6a34d35ad9d638bdf3974e2d47af5b410c07'}

def metrics(y,p):
    raw=np.expm1(y);prediction=np.expm1(p)
    rmse=float(np.sqrt(mean_squared_error(raw,prediction)))
    den=np.var(raw)+np.var(prediction)+(raw.mean()-prediction.mean())**2
    out=dict(log_r2=float(r2_score(y,p)),raw_r2=float(r2_score(raw,prediction)),
        log_rmse=float(np.sqrt(mean_squared_error(y,p))),raw_rmse=rmse,
        raw_mae=float(mean_absolute_error(raw,prediction)),raw_bias=float((prediction-raw).mean()),
        ccc=float(2*np.mean((raw-raw.mean())*(prediction-prediction.mean()))/den),
        rpiq=float((np.percentile(raw,75)-np.percentile(raw,25))/rmse),n_test=len(y))
    low=raw<150;high=~low
    out.update(n_SOC_below150=int(low.sum()),n_SOC_ge150=int(high.sum()))
    if low.sum()>2:
        out.update(below150_raw_r2=float(r2_score(raw[low],prediction[low])),
            below150_raw_rmse=float(np.sqrt(mean_squared_error(raw[low],prediction[low]))))
    if high.sum():out['ge150_raw_bias']=float((prediction[high]-raw[high]).mean())
    return out

def load(input_dir):
    root=Path(input_dir)
    design=json.loads((root/'point_qc/design.json').read_text())
    features=design['features']
    tables={y:pd.read_csv(root/f'exports/SOC_unified_{y}.csv',dtype={'point_id':str}).set_index('point_id',drop=False) for y in [2018,2022]}
    for d in tables.values():d['OC']=pd.to_numeric(d.OC.astype(str).str.replace(',','.',regex=False),errors='coerce')
    grid=pd.read_csv(root/'projected_points_3035.csv',dtype={'point_id':str}).set_index('point_id')
    global_ids=pd.read_csv(root/'point_qc/global_paired_ids.csv',dtype=str).point_id.to_numpy()
    all_ids=pd.read_csv(root/'point_qc_all_depths/global_paired_ids.csv',dtype=str).point_id.to_numpy()
    core_ids=pd.read_csv(root/'point_qc/year_pair_eligible_ids_M6.csv',dtype=str).point_id.to_numpy()
    cohorts={
        'global_0_20':(global_ids,list(features)),
        'core_0_20':(core_ids,['M1','M6']),
        'all_depths_global':(all_ids,['M1','M6'])}
    data={}
    for cohort,(ids,names) in cohorts.items():
        assert len(ids)==len(set(ids)) and set(ids)<=set(grid.index)
        assert hashlib.sha256('\n'.join(ids).encode()).hexdigest()==COHORT_ID_SHA256[cohort], 'Cohort membership/order changed; use a new design and output directory'
        a,b=tables[2018].loc[ids],tables[2022].loc[ids]
        assert np.array_equal(a.OC.to_numpy(),b.OC.to_numpy())
        assert np.allclose(a[['survey_lon','survey_lat']],b[['survey_lon','survey_lat']],atol=1e-9,rtol=0)
        assert np.allclose(a[['survey_lon','survey_lat']],grid.loc[ids,['survey_lon','survey_lat']],atol=1e-9,rtol=0)
        if cohort!='all_depths_global':assert a.Depth.eq('0-20 cm').all()
        xy=grid.loc[ids,['x_3035','y_3035']].to_numpy(float)
        cells=a.era_grid_x.astype(str)+'_'+a.era_grid_y.astype(str)
        data[cohort]=dict(ids=ids,y=np.log1p(a.OC.to_numpy(float)),xy=xy,cells=cells.to_numpy(),
            X={(name,y):tables[y].loc[ids,features[name]].to_numpy(float) for name in names for y in [2018,2022]},names=names)
        assert np.isfinite(data[cohort]['y']).all()
        for X in data[cohort]['X'].values():assert np.isfinite(X).all()
    assert len(global_ids)==663 and len(core_ids)==757 and len(all_ids)==677
    return data,features,tables

def splits(d,protocol,seeds):
    n=len(d['ids'])
    if protocol=='random5':
        for seed in seeds:
            for fold,(train,test) in enumerate(KFold(5,shuffle=True,random_state=seed).split(np.arange(n))):
                yield seed,fold,train,test,None
        return
    km=int(protocol.split('_')[1]);buffer_km=20 if protocol.endswith('buffer20') else 0
    bxy=np.floor(d['xy']/(km*1000)).astype(int)
    blocks=np.array([f'{x}_{y}' for x,y in bxy]);unique=np.unique(blocks)
    for seed in seeds:
        assignment=dict(zip(unique,np.random.RandomState(seed).randint(0,5,len(unique))))
        fold_ids=np.array([assignment[b] for b in blocks])
        for fold in range(5):
            train=np.flatnonzero(fold_ids!=fold);test=np.flatnonzero(fold_ids==fold)
            if buffer_km:
                distance=cKDTree(d['xy'][test]).query(d['xy'][train])[0]
                train=train[distance>buffer_km*1000]
            assert min(len(train),len(test))>=20
            assert not set(blocks[train])&set(blocks[test])
            yield seed,fold,train,test,blocks

def setup_jobs(data,out,design_hash):
    jobs=[];rows=[];assignments=[]
    for cohort,d in data.items():
        specs=[('random5',range(5)),('block_100',range(5))] if cohort!='all_depths_global' else [('block_100',range(3))]
        if cohort=='global_0_20':specs += [('block_50',range(3)),('block_150',range(3)),('block_100_buffer20',range(3))]
        for protocol,seeds in specs:
            models=['M1','M6'] if protocol.endswith('buffer20') else d['names']
            for seed,fold,train,test,blocks in splits(d,protocol,seeds):
                distances=cKDTree(d['xy'][train]).query(d['xy'][test])[0]/1000
                sharing=np.isin(d['cells'][test],d['cells'][train])
                meta=dict(cohort=cohort,protocol=protocol,seed=seed,fold=fold,
                    n_train=len(train),n_test=len(test),nn_train_km_min=float(distances.min()),
                    nn_train_km_median=float(np.median(distances)),nn_train_km_p90=float(np.percentile(distances,90)),
                    n_test_sharing_native_climate_cell=int(sharing.sum()))
                if blocks is not None:meta.update(n_train_blocks=len(np.unique(blocks[train])),n_test_blocks=len(np.unique(blocks[test])))
                rows.append(meta)
                for role,indices in [('train',train),('test',test)]:
                    assignments.extend(dict(cohort=cohort,protocol=protocol,seed=seed,fold=fold,role=role,point_id=d['ids'][i]) for i in indices)
                for model in models:
                    for year in [2018,2022]:
                        job=dict(meta,model=model,year=year,train=train.tolist(),test=test.tolist(),design_hash=design_hash)
                        job['id']=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:24]
                        jobs.append(job)
    pd.DataFrame(rows).to_csv(out/'fold_diagnostics.csv',index=False)
    pd.DataFrame(assignments).to_csv(out/'fold_assignments.csv',index=False)
    return jobs

def initialize(data):
    global DATA
    DATA=data

def fit(job):
    d=DATA[job['cohort']];X=d['X'][(job['model'],job['year'])];tr=np.array(job['train']);te=np.array(job['test'])
    started=time.time()
    model=RandomForestRegressor(**RF).fit(X[tr],d['y'][tr])
    p=model.predict(X[te]);dummy=np.full(len(te),d['y'][tr].mean())
    return dict(job={k:v for k,v in job.items() if k not in ['train','test']},metrics=metrics(d['y'][te],p),
        point_id=d['ids'][te].tolist(),true_log=d['y'][te].tolist(),pred_log=p.tolist(),dummy_log=dummy.tolist(),
        elapsed_seconds=time.time()-started)

def summarize(jobs,out):
    records=[];scores=[]
    for j in jobs:
        r=json.loads((out/'checkpoints'/f'{j["id"]}.json').read_text());m=r['job']
        scores.append(dict(m,**r['metrics'],elapsed_seconds=r['elapsed_seconds']))
        meta={k:m[k] for k in ['cohort','protocol','seed','fold','model','year']}
        records.append(pd.DataFrame(dict(point_id=r['point_id'],true_log=r['true_log'],pred_log=r['pred_log'],dummy_log=r['dummy_log'],**meta)))
    pred=pd.concat(records,ignore_index=True)
    pred.to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'fold_scores.csv',index=False)
    pooled=[];baseline=[]
    keys=['cohort','protocol','model','year','seed']
    for key,g in pred.groupby(keys):
        assert not g.point_id.duplicated().any()
        pooled.append(dict(zip(keys,key),**metrics(g.true_log.to_numpy(),g.pred_log.to_numpy())))
        baseline.append(dict(zip(keys,key),**metrics(g.true_log.to_numpy(),g.dummy_log.to_numpy())))
    pooled=pd.DataFrame(pooled);pooled.to_csv(out/'pooled_oof_by_repeat.csv',index=False)
    pd.DataFrame(baseline).to_csv(out/'training_mean_baselines.csv',index=False)
    measures=['log_r2','raw_r2','raw_rmse','raw_mae','raw_bias','ccc','rpiq','below150_raw_r2','ge150_raw_bias']
    summary=pooled.groupby(keys[:-1])[measures].agg(['mean','std','min','max']).reset_index()
    summary.columns=['_'.join(c).rstrip('_') for c in summary.columns]
    summary.to_csv(out/'performance_summary.csv',index=False)
    year_pairs=pooled[pooled.year.eq(2018)].merge(pooled[pooled.year.eq(2022)],on=['cohort','protocol','model','seed'],suffixes=('_2018','_2022'),validate='one_to_one')
    year_cols=['cohort','protocol','model','seed']
    year_delta=year_pairs[year_cols].copy()
    for metric in measures:year_delta[metric+'_delta_2018_minus_2022']=year_pairs[metric+'_2018']-year_pairs[metric+'_2022']
    year_delta.to_csv(out/'paired_year_changes_by_repeat.csv',index=False)
    year_summary=year_delta.groupby(year_cols[:-1]).agg({c:['mean','std','min','max'] for c in year_delta.columns if c.endswith('minus_2022')}).reset_index()
    year_summary.columns=['_'.join(c).rstrip('_') for c in year_summary.columns]
    year_summary.to_csv(out/'paired_year_changes_summary.csv',index=False)
    effect=[]
    for model in ['M3','M4','M5_additive','M6']:
        p=pooled[pooled.model.eq(model)].merge(pooled[pooled.model.eq('M1')],on=['cohort','protocol','year','seed'],suffixes=('_model','_M1'),validate='one_to_one')
        for _,r in p.iterrows():
            row={k:r[k] for k in ['cohort','protocol','year','seed']};row['model']=model
            for metric in measures:row[metric+'_delta_vs_M1']=r[metric+'_model']-r[metric+'_M1']
            effect.append(row)
    pd.DataFrame(effect).to_csv(out/'paired_model_changes_vs_M1.csv',index=False)
    return pooled,summary

def main(input_dir,output,workers):
    root=Path(input_dir);out=Path(output);out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    data,features,tables=load(root)
    files=[root/f'exports/SOC_unified_{y}.csv' for y in [2018,2022]]+[root/'projected_points_3035.csv',root/'point_qc/design.json',root/'point_qc_all_depths/design.json']
    design=dict(RF=RF,features=features,input_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        target='log1p SOC concentration; expm1 back-transform without smearing',cohorts={k:len(v['ids']) for k,v in data.items()},
        software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        sampling='Matched 2018 targets; new harmonized predictors from 2018 vs 2022; not SOC change or independent temporal validation.',
        tuning='Fixed historical RF settings. No hyperparameter optimization or algorithm novelty claim.',
        spatial_grid='GEE transforms of the source points into EPSG:3035; north/east grid origin 0. Block size does not guarantee that distance from training.',
        primary='global_0_20 n=663, random5 and block_100: 5 repetitions of 5 folds; same split for all models and years',
        sensitivity='50/150 km: 3 repetitions; core_0_20 n=757 M1/M6: 5 repetitions; all_depths n=677 M1/M6: block100 3 repetitions; buffer20: M1/M6 3 repetitions')
    design_hash=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    design['design_hash']=design_hash
    existing=out/'design.json'
    if existing.exists():assert json.loads(existing.read_text())['design_hash']==design_hash,'Output belongs to another design'
    existing.write_text(json.dumps(design,indent=2),encoding='utf-8')
    jobs=setup_jobs(data,out,design_hash);pending=[j for j in jobs if not (out/'checkpoints'/f'{j["id"]}.json').exists()]
    print(f'RF fits: {len(jobs)} total, {len(pending)} pending; cohorts {design["cohorts"]}',flush=True)
    with cf.ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(data,)) as pool:
        futures={pool.submit(fit,j):j for j in pending}
        for i,future in enumerate(cf.as_completed(futures),1):
            j=futures[future];r=future.result();p=out/'checkpoints'/f'{j["id"]}.json'
            tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r),encoding='utf-8');tmp.replace(p)
            if i%20==0 or i==len(pending):print(f'Completed {i}/{len(pending)} pending fits; last {j["cohort"]} {j["protocol"]} {j["model"]} {j["year"]}',flush=True)
    pooled,summary=summarize(jobs,out)
    (out/'completion.json').write_text(json.dumps(dict(n_fits=len(jobs),all_completed=True,design_hash=design_hash,cnn_trained=False),indent=2))
    print(summary.query("cohort=='global_0_20' and protocol=='block_100'")[[c for c in summary if c in ['cohort','protocol','model','year','log_r2_mean','raw_r2_mean','raw_rmse_mean']]].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default='input');p.add_argument('--output',default='results');p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();main(a.input,a.output,a.workers)
