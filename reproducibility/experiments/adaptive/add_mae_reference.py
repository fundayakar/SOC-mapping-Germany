"""Add a calibration-only, physically MAE-optimal constant blend.
Weighted median gives the exact 1-D convex absolute-loss solution; no test tuning.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from honest_pilot import metrics

HERE=Path(__file__).resolve().parent
NAME='calibrated_constant_physical_MAE'

def optimal_weight(y,r,c):
    delta=c-r;usable=abs(delta)>1e-12
    if not usable.any():return .5
    ratios=(y[usable]-r[usable])/delta[usable];weights=abs(delta[usable]);order=np.argsort(ratios)
    ratios=ratios[order];weights=weights[order]
    return float(np.clip(ratios[np.searchsorted(weights.cumsum(),weights.sum()/2)],0,1))

def main():
    rows=[];diagnostics=[]
    for out in sorted((HERE/'pilot').iterdir()):
        if not out.is_dir():continue
        cal=pd.read_csv(out/'calibration_predictions.csv',dtype={'point_id':str});p=pd.read_csv(out/'predictions.csv',dtype={'point_id':str})
        assert not set(cal.point_id)&set(p.point_id)
        y=cal.actual_SOC.to_numpy();r=cal.pred_RF.to_numpy();c=cal.pred_CNN.to_numpy();w=optimal_weight(y,r,c)
        cal_mae=float(np.mean(abs(r+w*(c-r)-y)))
        grid=np.linspace(0,1,1001);gridloss=np.abs(r[:,None]+(c-r)[:,None]*grid-y[:,None]).mean(axis=0)
        assert cal_mae<=gridloss.min()+1e-9
        p[NAME]=p.contextRF+w*(p.patchCNN-p.contextRF);rows.append(p)
        s=json.loads((out/'completion.json').read_text());diagnostics.append(dict(year=s['year'],target=s['target'],protocol=s['protocol'],fold=s['fold'],
            calibration_CNN_weight=w,calibration_MAE=cal_mae,n_calibration=len(cal),n_test=len(p)))
    p=pd.concat(rows,ignore_index=True);assert len(p)==5304
    p.to_csv(HERE/'results/honest_pilot_predictions.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(HERE/'results/MAE_reference_calibration.csv',index=False)
    strategies=[c for c in p if c not in ['point_id','actual_SOC','year','target','protocol','fold']];assert len(strategies)==14
    scores=[];foldscores=[]
    for k,g in p.groupby(['year','target','protocol']):
        assert len(g)==663
        for strategy in strategies:scores.append(dict(zip(['year','target','protocol'],k),strategy=strategy,**metrics(g.actual_SOC.to_numpy(),g[strategy].to_numpy())))
        for fold,h in g.groupby('fold'):
            for strategy in strategies:foldscores.append(dict(zip(['year','target','protocol'],k),fold=int(fold),strategy=strategy,**metrics(h.actual_SOC.to_numpy(),h[strategy].to_numpy())))
    pd.DataFrame(scores).to_csv(HERE/'results/honest_pilot_scores.csv',index=False)
    pd.DataFrame(foldscores).to_csv(HERE/'results/honest_pilot_fold_scores.csv',index=False)
    (HERE/'MAE_reference_validation.json').write_text(json.dumps(dict(all_passed=True,strategies=14,calibration_jobs=40,
        test_labels_unused_in_weight_selection=True,analytic_weighted_median_checked_against_1001_calibration_grid=True,
        note='Additional fair-loss reference; same frozen experts and same original 13 strategies.'),indent=2))
    s=pd.DataFrame(scores);print(s[s.strategy.isin([NAME,'soft_gate_all_with_predictions','soft_gate_predictions_only'])].query("target=='raw'")[['year','protocol','strategy','RMSE','MAE','log_R2']].round(4).to_string(index=False))

if __name__=='__main__':main()
