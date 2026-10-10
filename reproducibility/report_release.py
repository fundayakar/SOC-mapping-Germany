"""Regenerate canonical summaries from published aggregated score tables."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
def main(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/'experiments/cnn_controls/results/scores_by_repeat.csv')
    d=d[d.variant.isin(['centerRF','contextRF','centerMLP','original_fresh'])].copy()
    d['model']=d.variant.replace({'original_fresh':'patchCNN'});d=d.drop(columns='variant')
    keys=['year','protocol','target','model'];metrics=['raw_rmse','raw_mae','raw_r2','log_r2']
    count=d.groupby(keys).size();assert count.eq(np.where(count.index.get_level_values('protocol')=='block_100',5,3)).all()
    summary=d.groupby(keys)[metrics].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reset_index();d.to_csv(out/'main_scores_by_repeat.csv',index=False);summary.to_csv(out/'main_summary.csv',index=False)
    effects=pd.read_csv(ROOT/'experiments/cnn_controls/results/paired_ablation_effects.csv');effects.to_csv(out/'control_effects.csv',index=False)
    gps=pd.read_csv(ROOT/'experiments/gps/results/paired_summary.csv');assert len(gps)==32 and gps.n.eq(662).all();gps.to_csv(out/'gps_summary.csv',index=False)
    # Keep the GPS square-root-of-mean-loss convention distinct from mean RMSE.
    text=['# Release result index','', 'Main scores are means of pooled OOF metrics computed separately for each repeat; SD is descriptive, not a confidence interval. The CNN baseline is original_fresh.','',
          'GPS RMSE is sqrt(mean squared error across points and repeats) on 662 paired points. Intervals condition on fixed OOF predictions and resample 47 nominal blocks; they do not refit the models.','',
          '## Main benchmark','',summary.round(6).to_csv(index=False),'## GPS sensitivity','',gps.round(6).to_csv(index=False)]
    (out/'RESULTS.md').write_text('\n'.join(text))
    print(json.dumps({'main_repeat_rows':len(d),'main_conditions':len(summary),'GPS_conditions':len(gps),'CNN_reference':'original_fresh'}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
