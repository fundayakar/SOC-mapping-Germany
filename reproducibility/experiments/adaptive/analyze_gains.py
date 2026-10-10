"""Point-level, paired exploratory diagnostics; never estimates a learned gate's accuracy.
Requires the completed stage-3 package. Gains > 0 favour patchCNN over contextRF.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
FEATURES=['NDVI','VV','VH','elev','slope']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def extract_features(d):
    x=d['X'].astype('float64'); b=d['bands'].tolist()
    assert x.shape==(663,14,17,17) and np.all(d['valid']==1)
    out={'point_id':d['point_id'],'x3035':d['xy'][:,0],'y3035':d['xy'][:,1]}
    for i,name in enumerate(b):
        out[name+'_center']=x[:,i,8,8]
        out[name+'_std']=x[:,i].std(axis=(1,2))
        out[name+'_center_mean_abs']=abs(x[:,i,8,8]-x[:,i].mean(axis=(1,2)))
    ndvi=x[:,b.index('NDVI')]
    out['NDVI_neighbor_difference']=(abs(np.diff(ndvi,axis=1)).mean(axis=(1,2))+abs(np.diff(ndvi,axis=2)).mean(axis=(1,2)))/2
    for i,name in enumerate(d['climate_names']): out[name]=d['climate'][:,i]
    return pd.DataFrame(out)

def main(root):
    (HERE/'inputs').mkdir(exist_ok=True); (HERE/'results').mkdir(exist_ok=True)
    source=pd.read_csv(root/'stage2/input/exports/SOC_unified_2018.csv',dtype={'point_id':str})
    assert not source.point_id.duplicated().any()
    land=source.set_index('point_id').LC0_Desc.fillna('Unknown')
    records=[]; manifest={}
    for year in [2018,2022]:
        path=root/f'cnn/data/patches_{year}.npz'; d=dict(np.load(path,allow_pickle=False))
        f=extract_features(d); f['landcover_observed_2018']=land.loc[f.point_id].to_numpy()
        f.to_csv(HERE/f'inputs/features_{year}.csv',index=False);manifest[str(path.relative_to(root))]=sha(path)
        for folder,protocol,reps in [('results','block_100',5),('results_buffer20','block_100_buffer20',3)]:
            path=root/f'cnn/{folder}/{year}/oof_predictions.csv'
            p=pd.read_csv(path,dtype={'point_id':str}); manifest[str(path.relative_to(root))]=sha(path)
            p=p[p.model.isin(['contextRF','patchCNN'])].copy()
            keys=['point_id','target','year','protocol','seed','fold']
            assert not p.duplicated(keys+['model']).any()
            r=p[p.model.eq('contextRF')].drop(columns='model').rename(columns={'pred_SOC':'pred_RF'})
            c=p[p.model.eq('patchCNN')].drop(columns='model').rename(columns={'pred_SOC':'pred_CNN','actual_SOC':'actual_CNN'})
            q=r.merge(c,on=keys,validate='one_to_one')
            np.testing.assert_array_equal(q.actual_SOC,q.actual_CNN);q=q.drop(columns='actual_CNN')
            assert len(q)==663*reps*2 and q.groupby(['target','seed']).point_id.nunique().eq(663).all()
            np.testing.assert_allclose(q.actual_SOC,source.set_index('point_id').loc[q.point_id,'OC'].astype(float),rtol=0,atol=1e-10)
            for model in ['RF','CNN']:
                e=q[f'pred_{model}']-q.actual_SOC
                le=np.log1p(q[f'pred_{model}'])-np.log1p(q.actual_SOC)
                q[f'{model}_AE']=abs(e);q[f'{model}_SE']=e**2
                q[f'{model}_logAE']=abs(le);q[f'{model}_logSE']=le**2
            for loss in ['AE','SE','logAE','logSE']:
                q['gain_'+loss]=q[f'RF_{loss}']-q[f'CNN_{loss}']
            records.append(q)
    paired=pd.concat(records,ignore_index=True);assert len(paired)==21216
    paired.to_csv(HERE/'inputs/paired_predictions.csv',index=False)
    keys=['protocol','year','target','point_id']; losses=['AE','SE','logAE','logSE']
    point=paired.groupby(keys,as_index=False).agg(actual_SOC=('actual_SOC','first'),n_repeats=('seed','size'),
        **{f'mean_{k}':(k,'mean') for k in [f'{m}_{l}' for m in ['RF','CNN'] for l in losses]+['gain_'+l for l in losses]},
        CNN_AE_win_fraction=('gain_AE',lambda v:float((v>0).mean())),
        CNN_logAE_win_fraction=('gain_logAE',lambda v:float((v>0).mean())))
    merged=[]
    for year,g in point.groupby('year'):
        f=pd.read_csv(HERE/f'inputs/features_{year}.csv',dtype={'point_id':str})
        merged.append(g.merge(f,on='point_id',validate='many_to_one'))
    point=pd.concat(merged,ignore_index=True)
    point['SOC_group']=pd.cut(point.actual_SOC,[-np.inf,30,60,150,np.inf],right=False,labels=['<30','30–<60','60–<150','≥150'])
    point.to_csv(HERE/'results/point_gains.csv',index=False)
    summaries=[]; group_summaries=[]; associations=[]; quantiles=[]
    interesting=['NDVI_std','NDVI_center_mean_abs','NDVI_neighbor_difference','VV_std','VH_std','elev_std','slope_std','NDVI_center','slope_center','elev_center','sm_annual']
    for key,g in point.groupby(['protocol','year','target']):
        meta=dict(zip(['protocol','year','target'],key)); total=g.mean_gain_SE.sum();high=g.actual_SOC.ge(150)
        s=dict(**meta,n=len(g),mean_gain_AE=g.mean_gain_AE.mean(),mean_gain_logSE=g.mean_gain_logSE.mean(),
            CNN_mean_AE_better_fraction=(g.mean_gain_AE>0).mean(),CNN_mean_logAE_better_fraction=(g.mean_gain_logAE>0).mean(),
            CNN_AE_wins_all_repeats=int(g.CNN_AE_win_fraction.eq(1).sum()),RF_AE_wins_all_repeats=int(g.CNN_AE_win_fraction.eq(0).sum()),
            CNN_logAE_wins_all_repeats=int(g.CNN_logAE_win_fraction.eq(1).sum()),RF_logAE_wins_all_repeats=int(g.CNN_logAE_win_fraction.eq(0).sum()),
            raw_SE_total_gain=total,raw_SE_high_gain=g.loc[high,'mean_gain_SE'].sum(),
            high_share_of_net_SE_gain=g.loc[high,'mean_gain_SE'].sum()/total if abs(total)>1e-9 else np.nan,
            mean_low_SE_gain=g.loc[~high,'mean_gain_SE'].mean())
        for model in ['RF','CNN']:
            s[model+'_RMSE_over_point_repeat']=np.sqrt(g[f'mean_{model}_SE'].mean())
            s[model+'_MAE_over_point_repeat']=g[f'mean_{model}_AE'].mean()
        summaries.append(s)
        for grouping in ['SOC_group','landcover_observed_2018']:
            for category,h in g.groupby(grouping,observed=True):
                group_summaries.append(dict(**meta,grouping=grouping,category=str(category),n=len(h),
                    RF_RMSE=np.sqrt(h.mean_RF_SE.mean()),CNN_RMSE=np.sqrt(h.mean_CNN_SE.mean()),
                    RF_MAE=h.mean_RF_AE.mean(),CNN_MAE=h.mean_CNN_AE.mean(),
                    gain_AE=h.mean_gain_AE.mean(),gain_logSE=h.mean_gain_logSE.mean(),gain_SE_sum=h.mean_gain_SE.sum()))
        for feature in interesting:
            for loss in ['AE','logSE']:
                rho=float(spearmanr(g[feature],g['mean_gain_'+loss]).statistic)
                associations.append(dict(**meta,feature=feature,loss=loss,spearman_rho=rho,n=len(g)))
            # Tied values may yield fewer than four quantile groups, retained explicitly.
            bins=pd.qcut(g[feature],4,duplicates='drop')
            for cat,h in g.groupby(bins,observed=True):
                quantiles.append(dict(**meta,feature=feature,bin=str(cat),n=len(h),
                    feature_min=h[feature].min(),feature_max=h[feature].max(),
                    gain_AE=h.mean_gain_AE.mean(),gain_logSE=h.mean_gain_logSE.mean()))
    for name,rows in [('gain_summary',summaries),('group_gains',group_summaries),('feature_associations',associations),('feature_quantiles',quantiles)]:
        pd.DataFrame(rows).to_csv(HERE/f'results/{name}.csv',index=False)
    stability=[]
    for (protocol,target),g in point.groupby(['protocol','target']):
        a=g[g.year.eq(2018)].set_index('point_id');b=g[g.year.eq(2022)].set_index('point_id').loc[a.index]
        for loss in ['AE','logSE']:
            stability.append(dict(protocol=protocol,target=target,loss=loss,
                year_gain_spearman=float(spearmanr(a['mean_gain_'+loss],b['mean_gain_'+loss]).statistic),
                winner_agreement_fraction=float(((a['mean_gain_'+loss]>0)==(b['mean_gain_'+loss]>0)).mean())))
    pd.DataFrame(stability).to_csv(HERE/'results/year_stability.csv',index=False)
    (HERE/'source_manifest.json').write_text(json.dumps(dict(stage3_source_library_id='libfile_ad27616539488191b0d5d12e7ba36ef0',
        sha256=manifest,paired_rows=len(paired),point_summary_rows=len(point),
        notes='Same 663 points and 2018 targets; paired losses averaged over repeats, never errors of averaged predictions. Association screening is exploratory.'),indent=2))
    print('Validated paired predictions:',len(paired),'point summaries:',len(point),flush=True)
    print(pd.DataFrame(summaries)[['protocol','year','target','CNN_mean_AE_better_fraction','mean_gain_AE','high_share_of_net_SE_gain']].to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage3-root',type=Path,default=HERE.parent/'target_transform');a=ap.parse_args();main(a.stage3_root.resolve())
