"""Validate paired OOF records and summarize frozen-model spatial uncertainty."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT=Path(__file__).resolve().parent
MODEL_LABELS={'centerRF':'RF: center','contextRF':'RF: center + context','centerMLP':'MLP: center','patchCNN':'CNN: patch + center'}

def main():
    completion=json.loads((ROOT/'results/completion.json').read_text());assert completion['all_completed'] and completion['completed_fits']==1280 and completion['pooled_groups']==256
    design=json.loads((ROOT/'results/design.json').read_text())
    for name,digest in design['input_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    p=pd.read_csv(ROOT/'results/oof_predictions.csv',dtype={'point_id':str});keys=['year','protocol','seed','fold','target','model','point_id']
    assert not p.duplicated(['center']+keys).any() and len(p)==169472
    nom=p[p.center.eq('nominal')].drop(columns='center');gps=p[p.center.eq('gps')].drop(columns='center')
    pair=nom.merge(gps,on=keys,suffixes=('_nominal','_gps'),validate='one_to_one');assert len(pair)==84736
    assert np.array_equal(pair.actual_SOC_nominal,pair.actual_SOC_gps)
    ref=pd.read_csv(ROOT/'inputs/nominal_replay_reference.csv',dtype={'point_id':str})
    replay=nom.merge(ref,on=keys,suffixes=('_new','_reference'),validate='one_to_one')
    assert len(replay)==len(ref)
    max_delta=float(abs(replay.pred_SOC_new-replay.pred_SOC_reference).max())
    # Removing a test row changes float32 matrix-batch shapes. Check exact
    # trained states and allow float32 forward-pass rounding in predictions.
    assert np.allclose(replay.pred_SOC_new,replay.pred_SOC_reference,atol=1e-4,rtol=1e-6),f'Replay mismatch: {max_delta}'
    states=json.loads((ROOT/'inputs/nominal_replay_state_hashes.json').read_text())
    for s in states:
        name=f"nominal_{s['year']}_{s['protocol']}_s{s['seed']}_f{s['fold']}_{s['target']}_{s['model']}"
        ck=torch.load(ROOT/f'results/runs/{name}.pt',weights_only=False,map_location='cpu');h=hashlib.sha256()
        for k,v in sorted(ck['state_dict'].items()):
            a=v.detach().cpu().numpy();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        assert h.hexdigest()==s['state_sha256'] and ck['epochs']==s['epochs'],name
        assert hashlib.sha256(json.dumps(ck['normalization'],sort_keys=True).encode()).hexdigest()==s['normalization_sha256'],name
    qc=pd.read_csv(ROOT/'results/extraction_qc_2018.csv',dtype={'point_id':str}).set_index('point_id')
    pair['actual_SOC']=pair.actual_SOC_nominal
    for center in ['nominal','gps']:
        pair[f'error_{center}']=pair[f'pred_SOC_{center}']-pair.actual_SOC
        pair[f'squared_error_{center}']=pair[f'error_{center}']**2
        pair[f'absolute_error_{center}']=pair[f'error_{center}'].abs()
    point=pair.groupby(['year','protocol','target','model','point_id']).agg(
        actual_SOC=('actual_SOC','first'),nominal_SE=('squared_error_nominal','mean'),gps_SE=('squared_error_gps','mean'),
        nominal_AE=('absolute_error_nominal','mean'),gps_AE=('absolute_error_gps','mean'),
        nominal_error=('error_nominal','mean'),gps_error=('error_gps','mean')).reset_index()
    point=point.merge(qc[['asset_to_gps_m','center_changed']],left_on='point_id',right_index=True,validate='many_to_one')
    point['delta_SE']=point.gps_SE-point.nominal_SE;point['delta_AE']=point.gps_AE-point.nominal_AE
    point.to_csv(ROOT/'results/paired_point_errors.csv',index=False)
    coords=pd.read_csv(ROOT/'inputs/gps_crosswalk_662.csv',dtype={'point_id':str}).set_index('point_id')
    blocks=np.floor(coords[['npz_x_3035','npz_y_3035']].to_numpy()/100000).astype(int)
    coords['block']=[f'{x}_{y}' for x,y in blocks]
    ordered=coords.index.to_numpy();unique=np.unique(coords.block);bindex={b:i for i,b in enumerate(unique)}
    membership=np.array([bindex[b] for b in coords.block]);matrix=np.eye(len(unique))[membership]
    # Resample nominal 100km blocks, preserving paired centers and all repeat
    # predictions. Average point losses over repeats; never treat repeats as n.
    rng=np.random.default_rng(20261010);counts=rng.multinomial(len(unique),np.repeat(1/len(unique),len(unique)),size=2000)
    weights=counts@matrix.T;den=weights.sum(axis=1);y=coords.loc[ordered].index
    rows=[];cached={};sub=[]
    for key,g in point.groupby(['year','protocol','target','model']):
        g=g.set_index('point_id').loc[ordered];actual=g.actual_SOC.to_numpy();a=g.nominal_SE.to_numpy();b=g.gps_SE.to_numpy()
        rmse_a=np.sqrt(a.mean());rmse_b=np.sqrt(b.mean());delta=np.sqrt((weights@b)/den)-np.sqrt((weights@a)/den)
        cached[key]=(rmse_a,rmse_b,delta)
        row=dict(zip(['year','protocol','target','model'],key),nominal_rmse=float(rmse_a),gps_rmse=float(rmse_b),
            delta_rmse=float(rmse_b-rmse_a),delta_rmse_ci_low=float(np.quantile(delta,.025)),delta_rmse_ci_high=float(np.quantile(delta,.975)),
            nominal_r2=float(1-a.mean()/actual.var()),gps_r2=float(1-b.mean()/actual.var()),
            nominal_mae=float(g.nominal_AE.mean()),gps_mae=float(g.gps_AE.mean()),n=662,n_blocks=len(unique))
        rows.append(row)
        for label,keep in [('same_center_cell',~g.center_changed),('changed_center_cell',g.center_changed),('offset_gt20m',g.asset_to_gps_m>20),('offset_le20m',g.asset_to_gps_m<=20)]:
            h=g[keep];sub.append(dict(zip(['year','protocol','target','model'],key),subgroup=label,n=len(h),nominal_rmse=float(np.sqrt(h.nominal_SE.mean())),gps_rmse=float(np.sqrt(h.gps_SE.mean())),delta_mae=float((h.gps_AE-h.nominal_AE).mean())))
    aggregate=pd.DataFrame(rows);aggregate.to_csv(ROOT/'results/paired_summary.csv',index=False);pd.DataFrame(sub).to_csv(ROOT/'results/subgroup_summary.csv',index=False)
    gains=[]
    for year in [2018,2022]:
      for protocol in ['block_100','block_100_buffer20']:
       for target in ['raw','log']:
        for label,center_model,context_model in [('RF_context_vs_center','centerRF','contextRF'),('CNN_vs_MLP','centerMLP','patchCNN')]:
            a0,a1,ad=cached[(year,protocol,target,center_model)];b0,b1,bd=cached[(year,protocol,target,context_model)]
            gain_change=ad-bd
            gains.append(dict(year=year,protocol=protocol,target=target,comparison=label,nominal_rmse_gain=a0-b0,gps_rmse_gain=a1-b1,
                delta_gain=(a1-b1)-(a0-b0),delta_gain_ci_low=float(np.quantile(gain_change,.025)),delta_gain_ci_high=float(np.quantile(gain_change,.975))))
    pd.DataFrame(gains).to_csv(ROOT/'results/context_gain_summary.csv',index=False)
    (ROOT/'results/analysis_validation.json').write_text(json.dumps(dict(all_completed=True,n_predictions=len(p),n_pairs=len(pair),
        n_replay_predictions=len(replay),replay_max_abs_delta=max_delta,n_exact_neural_state_replays=len(states),
        replay_rounding='Unchanged fitted states/normalization/epochs checked bitwise. Removing a test row can change float32 forward-pass rounding.',
        bootstrap_replicates=2000,n_blocks=len(unique),
        bootstrap='Fixed predictions; nominal 100km block resampling, paired centers; mean point losses across repeats, no refitting.',
        subgroup_interpretation='Descriptive subsets; 28 offsets above20m are a small subgroup. CNN-vs-MLP also differs in architecture/capacity.'),indent=2))
    plot(aggregate)
    print(aggregate[['year','protocol','target','model','nominal_rmse','gps_rmse','delta_rmse','delta_rmse_ci_low','delta_rmse_ci_high']].round(3).to_string(index=False))

def plot(d):
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    models=list(MODEL_LABELS);colors={2018:'#24628c',2022:'#ae5b32'}
    for i,protocol in enumerate(['block_100','block_100_buffer20']):
      for j,target in enumerate(['raw','log']):
        ax=axes[i,j];s=d.query('protocol==@protocol and target==@target')
        for year in [2018,2022]:
            a=s.query('year==@year').set_index('model').loc[models];yy=np.arange(4)+( -.12 if year==2018 else .12)
            ax.hlines(yy,a.delta_rmse_ci_low,a.delta_rmse_ci_high,color=colors[year],lw=1.8)
            ax.scatter(a.delta_rmse,yy,s=30,color=colors[year],label=str(year),zorder=3)
        ax.axvline(0,color='#999999',lw=1);ax.set_yticks(np.arange(4),[MODEL_LABELS[m] for m in models]);ax.grid(axis='x',alpha=.15)
        ax.set_title(('100 km blocks' if i==0 else '100 km blocks + nominal 20 km buffer')+' | '+('raw target' if j==0 else 'log1p target'),fontsize=10)
        ax.spines[['top','right']].set_visible(False)
    axes[0,0].invert_yaxis()
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,title='Predictor year',loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False,fontsize=9,title_fontsize=9)
    fig.supxlabel('GPS minus nominal RMSE (g/kg) — negative values favor GPS',fontsize=11,y=.065)
    fig.suptitle('Sensitivity to extraction center on the same 662 points',fontsize=13,y=.985)
    fig.text(.5,.025,'Intervals: 95% paired block bootstrap of fixed OOF losses; 2,000 resamples. Folds unchanged.',ha='center',fontsize=8,color='#555555')
    fig.tight_layout(rect=[0,.12,1,.87]);fig.savefig(ROOT/'results/SOC_GPS_center_comparison.png',dpi=180);fig.savefig(ROOT/'results/SOC_GPS_center_comparison.pdf');plt.close(fig)

if __name__=='__main__':main()
