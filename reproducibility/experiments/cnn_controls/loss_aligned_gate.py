"""Prediction-only convex blending with a directly optimized loss.
Uses frozen, disjoint A/B/C expert predictions from the previous honest pilot.
All hyperparameter selection inside B, spatially grouped (buffered where needed).
Exploratory standard-method prototype; no algorithm novelty claim.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize,check_grad
from scipy.special import expit,logit
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from complete_experiment import a

HERE=Path(__file__).resolve().parent
OUT=HERE/'loss_gate'
LAMBDAS=[.01,.1,1.]
CS=[.1,1.,10.]
SMOOTH_GKG=.01

def features(r,c):return np.column_stack([r,c,c-r])

def const_weight(y,r,c,loss):
    delta=c-r
    if loss=='MSE':return float(np.clip(np.sum(delta*(y-r))/max(np.sum(delta**2),1e-12),0,1))
    keep=abs(delta)>1e-12
    if not keep.any():return .5
    z=(y[keep]-r[keep])/delta[keep];w=abs(delta[keep]);i=np.argsort(z);z=z[i];w=w[i]
    return float(np.clip(z[np.searchsorted(w.cumsum(),w.sum()/2)],0,1))

def objective(theta,z,y,r,c,loss,reg,scale):
    prob=expit(z@theta);delta=c-r;e=r+prob*delta-y
    if loss=='MAE':
        value=np.sqrt(e**2+SMOOTH_GKG**2);der=e/value;factor=scale
    else:value=e**2;der=2*e;factor=scale**2
    val=float(value.mean()/factor+reg*np.sum(theta[1:]**2))
    grad=z.T@(der*delta*prob*(1-prob))/(len(y)*factor)
    grad[1:]+=2*reg*theta[1:]
    return val,grad

def fit_direct(x,y,r,c,loss,reg):
    scaler=StandardScaler().fit(x);xs=scaler.transform(x);z=np.column_stack([np.ones(len(y)),xs])
    scale=max(float(np.median(abs(y-np.median(y)))),1.)
    initial=const_weight(y,r,c,loss);candidates=[]
    for w in [initial,.25,.75]:
        theta=np.zeros(z.shape[1]);theta[0]=logit(np.clip(w,1e-4,1-1e-4))
        theta[0]=np.clip(theta[0],-8,8)
        f=lambda v:objective(v,z,y,r,c,loss,reg,scale)
        result=minimize(f,theta,method='L-BFGS-B',jac=True,bounds=[(-8,8)]+[(-5,5)]*x.shape[1],options={'maxiter':500,'ftol':1e-11,'gtol':1e-7})
        assert np.isfinite(result.fun) and np.isfinite(result.x).all()
        # Always retain its initialization: optimization cannot worsen the fitted
        # regularized training objective relative to its constant start.
        candidates.extend([(f(theta)[0],theta,True,0),(float(result.fun),result.x,bool(result.success),int(result.nit))])
    _,theta,success,nit=min(candidates,key=lambda v:v[0])
    return dict(scaler=scaler,theta=theta,loss=loss,lambda_=reg,scale=scale,optimizer_success=success,iterations=nit)

def weight(model,x):return expit(np.column_stack([np.ones(len(x)),model['scaler'].transform(x)])@model['theta'])

def score(y,p,loss):return float(np.mean(abs(p-y))) if loss=='MAE' else float(np.mean((p-y)**2))

def fit_winner(x,y,r,c,C):
    scaler=StandardScaler().fit(x);labels=(abs(c-y)<abs(r-y)).astype(int)
    if len(np.unique(labels))<2:return dict(scaler=scaler,model=None,constant=float(labels[0]),C=C)
    clf=LogisticRegression(C=C,max_iter=2000,random_state=42).fit(scaler.transform(x),labels)
    return dict(scaler=scaler,model=clf,C=C)

def winner_weight(model,x):
    if model['model'] is None:return np.full(len(x),model['constant'])
    return model['model'].predict_proba(model['scaler'].transform(x))[:,1]

def main(stage3,prior):
    OUT.mkdir(exist_ok=True);(OUT/'inputs').mkdir(exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
    design=dict(source_library_id='libfile_4b4a9b955edc8191b384ce15ee1a563e',source='Frozen raw-target expert predictions from prior honest three-way pilot, outer seed0 only.',
        feature_names=['RF_prediction','CNN_prediction','CNN_minus_RF'],objectives=['physical_MAE','physical_MSE'],
        sigmoid='w=sigmoid(intercept + linear standardized prediction features); prediction=RF+w*(CNN-RF).',
        loss='Smooth physical absolute error (delta .01 g/kg) or physical squared error, normalized by training MAD or MAD squared; slope L2, intercept unpenalized.',
        lambda_grid=LAMBDAS,winner_probability_C_grid=CS,optimization='L-BFGS-B, three fixed starts, max500, bounds intercept±8/slopes±5; keep constant initializations too.',
        hyper_selection='3-fold grouped100km inside calibration B; 20km subtrain-val buffer in buffered protocol. Pooled inner holdout loss selects hyperparameter; final fit B only.',
        guard='Experts remain frozen on A; no test C labels in scaling, fitting, choosing starts or selecting lambda/C. Compare tuned winner-probability gates with identical CV and physical scoring objectives.',
        caveat='Different regularization grids and training losses; this is a practical candidate comparison, not pure isolation of objective effects. Standard conditional blending is not inherently novel.',
        source_sha256={str(p.relative_to(prior)):a.sha(p) for p in [prior/'results/honest_pilot_predictions.csv',prior/'validation.json',prior/'MAE_reference_validation.json']},
        code_sha256=a.sha(__file__))
    (OUT/'design.json').write_text(json.dumps(design,indent=2))
    prior_predictions=pd.read_csv(prior/'results/honest_pilot_predictions.csv',dtype={'point_id':str}).query("target=='raw'")
    assert len(prior_predictions)==2652
    allresults=[];choices=[];hyper=[];gradchecks=[];roles=[];minimum_buffer=1e20
    for (year,protocol,fold),old in prior_predictions.groupby(['year','protocol','fold']):
        source=prior/'pilot'/f'{protocol}_{year}_raw_f{fold}'
        cal=pd.read_csv(source/'calibration_predictions.csv',dtype={'point_id':str})
        assert not set(cal.point_id)&set(old.point_id)
        name=f'{protocol}_{year}_f{fold}'
        cal.to_csv(OUT/f'inputs/{name}_calibration.csv',index=False)
        old.to_csv(OUT/f'inputs/{name}_test_baselines.csv',index=False)
        d=dict(np.load(stage3/f'cnn/data/patches_{year}.npz',allow_pickle=False));ids={p:i for i,p in enumerate(d['point_id'])}
        idx=np.array([ids[p] for p in cal.point_id]);xy=d['xy'][idx]
        blocks_xy=np.floor(xy/100000).astype(int);blocks=np.array([f'{x}_{y}' for x,y in blocks_xy])
        splits=[]
        for inner,(tr,va) in enumerate(GroupKFold(3).split(idx,groups=blocks)):
            if protocol.endswith('buffer20'):tr=tr[cKDTree(xy[va]).query(xy[tr])[0]>20000]
            assert min(len(tr),len(va))>=20 and not set(blocks[tr])&set(blocks[va])
            distance=float(cKDTree(xy[tr]).query(xy[va])[0].min())
            if protocol.endswith('buffer20'):assert distance>20000;minimum_buffer=min(minimum_buffer,distance)
            splits.append((tr,va))
            for role,ii in [('gate_inner_train',tr),('gate_inner_validation',va)]:
                roles.append(pd.DataFrame(dict(year=year,protocol=protocol,fold=fold,inner=inner,role=role,point_id=cal.point_id.iloc[ii].to_numpy(),min_train_val_distance_m=distance)))
        y=cal.actual_SOC.to_numpy();r=cal.pred_RF.to_numpy();c=cal.pred_CNN.to_numpy();x=features(r,c)
        xt=features(old.contextRF.to_numpy(),old.patchCNN.to_numpy());result=old.copy()
        for loss in ['MAE','MSE']:
            reg_scores=[];C_scores=[]
            for reg in LAMBDAS:
                cv=np.full(len(y),np.nan)
                for tr,va in splits:
                    m=fit_direct(x[tr],y[tr],r[tr],c[tr],loss,reg)
                    cv[va]=r[va]+weight(m,x[va])*(c[va]-r[va])
                assert np.isfinite(cv).all();value=score(y,cv,loss);reg_scores.append((value,reg))
                hyper.append(dict(year=year,protocol=protocol,fold=fold,family='direct',objective=loss,hyperparameter=reg,inner_pooled_loss=value))
            for C in CS:
                cv=np.full(len(y),np.nan)
                for tr,va in splits:
                    m=fit_winner(x[tr],y[tr],r[tr],c[tr],C)
                    cv[va]=r[va]+winner_weight(m,x[va])*(c[va]-r[va])
                assert np.isfinite(cv).all();value=score(y,cv,loss);C_scores.append((value,C))
                hyper.append(dict(year=year,protocol=protocol,fold=fold,family='winner_probability',objective=loss,hyperparameter=C,inner_pooled_loss=value))
            _,reg=min(reg_scores);_,C=min(C_scores)
            direct=fit_direct(x,y,r,c,loss,reg);prob=fit_winner(x,y,r,c,C)
            wd=weight(direct,xt);wp=winner_weight(prob,xt)
            result['direct_'+loss]=old.contextRF+wd*(old.patchCNN-old.contextRF)
            result['winner_tuned_'+loss]=old.contextRF+wp*(old.patchCNN-old.contextRF)
            np.testing.assert_allclose(direct['scaler'].mean_,x.mean(axis=0),atol=1e-10,rtol=1e-10)
            assert ((wd>=0)&(wd<=1)).all() and np.isfinite(result['direct_'+loss]).all()
            joblib.dump(direct,OUT/f'models/{name}_direct_{loss}.joblib')
            joblib.dump(prob,OUT/f'models/{name}_winner_{loss}.joblib')
            choices.append(dict(year=year,protocol=protocol,fold=fold,objective=loss,selected_lambda=reg,selected_C=C,
                direct_training_scale=direct['scale'],direct_optimizer_success=direct['optimizer_success'],direct_iterations=direct['iterations'],n_cal=len(cal),n_test=len(old),
                CNN_weight_mean=wd.mean(),CNN_weight_min=wd.min(),CNN_weight_max=wd.max()))
            if fold==0 and protocol=='block_100' and year==2018:
                z=np.column_stack([np.ones(len(y)),direct['scaler'].transform(x)]);theta=np.array([.13,.2,-.15,.07])
                f=lambda v:objective(v,z,y,r,c,loss,reg,direct['scale'])[0]
                jac=lambda v:objective(v,z,y,r,c,loss,reg,direct['scale'])[1]
                error=float(check_grad(f,jac,theta));relative=error/max(np.linalg.norm(jac(theta)),1.)
                assert relative<1e-5;gradchecks.append(dict(objective=loss,absolute_gradient_error=error,relative_gradient_error=relative))
        allresults.append(result);print('Completed loss-aligned gates',name,flush=True)
    p=pd.concat(allresults,ignore_index=True);assert len(p)==2652
    p.to_csv(OUT/'predictions.csv',index=False)
    pd.DataFrame(choices).to_csv(OUT/'selected_hyperparameters.csv',index=False);pd.DataFrame(hyper).to_csv(OUT/'inner_cv.csv',index=False)
    pd.concat(roles,ignore_index=True).to_csv(OUT/'inner_roles.csv',index=False)
    t=a.imports(stage3);scores=[];fs=[]
    variants=[col for col in p if col not in ['point_id','actual_SOC','year','target','protocol','fold']];assert len(variants)==18
    for (year,protocol),g in p.groupby(['year','protocol']):
        assert len(g)==663
        for variant in variants:scores.append(dict(year=year,protocol=protocol,variant=variant,**t.raw_metrics(g.actual_SOC.to_numpy(),g[variant].to_numpy())))
        for fold,h in g.groupby('fold'):
            for variant in variants:fs.append(dict(year=year,protocol=protocol,fold=fold,variant=variant,**t.raw_metrics(h.actual_SOC.to_numpy(),h[variant].to_numpy())))
    pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False);pd.DataFrame(fs).to_csv(OUT/'scores_by_fold.csv',index=False)
    (OUT/'validation.json').write_text(json.dumps(dict(all_passed=True,raw_expert_pairs=20,new_final_gate_models=80,
        objectives=['MAE','MSE'],strategies=18,inner_hyperparam_fits=720,final_test_point_predictions=2652,
        gradient_checks=gradchecks,calibration_only_scaling=True,hyperparameter_selection_calibration_only=True,
        minimum_buffer_inner_distance_m=minimum_buffer,old_frozen_expert_source='prior honest pilot; not fresh ablation experts',exploratory=True),indent=2))
    print(pd.DataFrame(scores).query("variant in ['calibrated_constant_blend','calibrated_constant_physical_MAE','soft_gate_predictions_only','direct_MAE','direct_MSE','winner_tuned_MAE','winner_tuned_MSE']")[['year','protocol','variant','raw_rmse','raw_mae','log_r2']].round(4).to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage3-root',type=Path,default=HERE.parent/'target_transform');ap.add_argument('--prior-root',type=Path,default=HERE.parent/'soc_gain_analysis');x=ap.parse_args();main(x.stage3_root.resolve(),x.prior_root.resolve())
