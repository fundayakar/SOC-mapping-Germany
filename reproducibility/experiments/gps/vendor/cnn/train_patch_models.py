"""Real-patch RF / MLP / CNN controls with training-only epoch selection.
Prepared runner: no scientific CNN result is implied by this file's existence.
"""
from pathlib import Path
import argparse,hashlib,json,random,sys,time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from target_compare import raw_metrics
from prepare_patches import BANDS,CLIMATE

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader,TensorDataset
except ModuleNotFoundError:
    raise SystemExit('PyTorch is required for this runner. The delivered preparation/results do not include a trained CNN.')

class CenterMLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(17,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,image,point):return self.net(point).squeeze(1)

class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # image: N x 15 x 17 x 17 = 14 physical bands + validity mask.
        self.context=nn.Sequential(nn.Conv2d(15,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        # Center skip connection preserves the actual labelled location.
        self.head=nn.Sequential(nn.Linear(32+17,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,image,point):return self.head(torch.cat([self.context(image),point],dim=1)).squeeze(1)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

def normalization(d,idx,target,kind):
    if kind=='centerMLP':
        v=d['X'][idx,:,8,8];mean=v.mean(0);sd=v.std(0)
    else:
        v=d['X'][idx];mask=d['valid'][idx];count=mask.sum()
        mean=(v*mask).sum(axis=(0,2,3))/count
        sd=np.sqrt(((v-mean[None,:,None,None])**2*mask).sum(axis=(0,2,3))/count)
    c=d['climate'][idx];y=np.log1p(d['y'][idx]) if target=='log' else d['y'][idx]
    return dict(mean=mean,sd=np.maximum(sd,1e-6),cmean=c.mean(0),csd=np.maximum(c.std(0),1e-6),ymean=float(y.mean()),ysd=max(float(y.std()),1e-6))

def tensors(d,idx,norm,target):
    image=(d['X'][idx]-norm['mean'][None,:,None,None])/norm['sd'][None,:,None,None]
    image=np.concatenate([image*d['valid'][idx],d['valid'][idx]],axis=1).astype('float32')
    center=(d['X'][idx,:,8,8]-norm['mean'])/norm['sd']
    climate=(d['climate'][idx]-norm['cmean'])/norm['csd']
    point=np.concatenate([center,climate],axis=1).astype('float32')
    y=np.log1p(d['y'][idx]) if target=='log' else d['y'][idx]
    y=((y-norm['ymean'])/norm['ysd']).astype('float32')
    return tuple(torch.from_numpy(v) for v in [image,point,y])

def network_fit(d,train,valid,target,kind,seed,device,epochs,patience,select):
    seed_all(seed);norm=normalization(d,train,target,kind)
    model=(CenterMLP() if kind=='centerMLP' else PatchCNN()).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(*tensors(d,train,norm,target)),batch_size=32,shuffle=True,generator=generator)
    validation=tensors(d,valid,norm,target) if select else None
    best=float('inf');best_epoch=1;stale=0;curve=[]
    for epoch in range(1,epochs+1):
        model.train()
        for image,point,y in loader:
            image,point,y=image.to(device),point.to(device),y.to(device)
            opt.zero_grad();loss=nn.functional.mse_loss(model(image,point),y);loss.backward();opt.step()
        if select:
            model.eval()
            with torch.no_grad():
                image,point,y=(v.to(device) for v in validation)
                error=float(nn.functional.mse_loss(model(image,point),y).item())
            curve.append(dict(epoch=epoch,inner_validation_MSE_standardized=error))
            if error<best-1e-5:best=error;best_epoch=epoch;stale=0
            else:stale+=1
            if stale>=patience:break
    return model,norm,best_epoch if select else epochs,curve

def neural_predict(model,norm,d,test,target,device):
    image,point,_=tensors(d,test,norm,target);model.eval()
    with torch.no_grad():z=model(image.to(device),point.to(device)).cpu().numpy()*norm['ysd']+norm['ymean']
    if target=='log':
        # Overflow is an experiment failure; do not silently cap predictions by test labels.
        z=z.astype(np.float64);pred=np.expm1(z)
    else:pred=z.astype(np.float64)
    pred=np.maximum(0,pred)
    assert np.isfinite(pred).all()
    return pred

def rf_features(d,kind):
    center=d['X'][:,:,8,8]
    if kind=='centerRF':return np.concatenate([center,d['climate']],axis=1)
    pixels=np.where(d['valid'].astype(bool),d['X'],np.nan)
    stats=[f(pixels,axis=(2,3)) for f in [np.nanmean,np.nanstd,np.nanmin,np.nanmax]]
    return np.concatenate([center]+stats+[d['climate'],d['valid'].mean(axis=(1,2,3))[:,None]],axis=1)

def main(args):
    data_path=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    d=dict(np.load(data_path,allow_pickle=False));year=int(d['year'])
    assert d['X'].shape==(663,14,17,17) and np.array_equal(d['bands'],BANDS) and np.array_equal(d['climate_names'],CLIMATE)
    assert (d['valid'][:,0,8,8]==1).all() and np.isfinite(d['X']).all()
    ids={v:i for i,v in enumerate(d['point_id'])}
    folds=pd.read_csv(ROOT/'stage2/results/fold_assignments.csv',dtype={'point_id':str})
    folds=folds.query("cohort=='global_0_20'");all_folds=folds[folds.protocol.eq(args.protocol)&folds.seed.lt(args.repeats)].copy()
    folds=all_folds if args.only_seed is None else all_folds[all_folds.seed.eq(args.only_seed)]
    assert len(folds)>0
    blocks_xy=np.floor(d['xy']/100000).astype(int);blocks=np.array([f'{x}_{y}' for x,y in blocks_xy])
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    torch.set_num_threads(1)
    design=dict(data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        year=year,protocol=args.protocol,requested_repeats=args.repeats,targets=['log','raw'],models=['centerRF','contextRF','centerMLP','patchCNN'],
        n=663,bands=BANDS,climate=CLIMATE,RF=stage2.RF,
        inner='First split of grouped 100km 3-fold CV; buffered protocol also applies a 20km inner buffer. Fixed validation split selects epochs only.',
        epochs=args.epochs,patience=args.patience,device=device,torch_version=torch.__version__,torch_threads=1,
        helper_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'target_compare.py',ROOT/'cnn/prepare_patches.py']},
        loss='MSE on training-standardized target: log1p(SOC) or raw SOC; negative physical predictions clipped at 0, no SOC-dependent clipping.',
        normalization='Fit separately on each inner or outer training set; common validity mask excludes missing pixels; climate scalars are not spatial raster channels.',
        folds_sha256=hashlib.sha256((ROOT/'stage2/results/fold_assignments.csv').read_bytes()).hexdigest())
    fingerprint=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    if (out/'design.json').exists():assert json.loads((out/'design.json').read_text())['fingerprint']==fingerprint,'Use a new output directory for a different design'
    import os
    temporary=out/f'design_{os.getpid()}.tmp';temporary.write_text(json.dumps(dict(design,fingerprint=fingerprint),indent=2));temporary.replace(out/'design.json')
    for (seed,fold),g in folds.groupby(['seed','fold']):
        tr=np.array(sorted(ids[p] for p in g[g.role.eq('train')].point_id));te=np.array(sorted(ids[p] for p in g[g.role.eq('test')].point_id))
        assert not set(blocks[tr])&set(blocks[te])
        outer_distance=cKDTree(d['xy'][tr]).query(d['xy'][te])[0].min()
        patch_distance=cKDTree(d['patch_center_xy'][tr]).query(d['patch_center_xy'][te])[0].min()
        assert patch_distance>np.sqrt(2)*340,'Training and test raster patches may overlap'
        if args.protocol.endswith('buffer20'):assert outer_distance>20000
        a,b=next(GroupKFold(3).split(tr,groups=blocks[tr]));it=tr[a];iv=tr[b]
        if args.protocol.endswith('buffer20'):it=it[cKDTree(d['xy'][iv]).query(d['xy'][it])[0]>20000]
        assert min(len(it),len(iv))>=20 and set(it)<=set(tr) and set(iv)<=set(tr)
        for target in ['log','raw']:
            for kind in ['centerRF','contextRF','centerMLP','patchCNN']:
                name=f'{kind}_{target}_s{seed}_f{fold}';path=out/f'{name}.json'
                if path.exists():continue
                started=time.time();epoch=None;curve=[]
                if kind.endswith('RF'):
                    X=rf_features(d,kind);y=np.log1p(d['y']) if target=='log' else d['y']
                    pred=RandomForestRegressor(**stage2.RF).fit(X[tr],y[tr]).predict(X[te])
                    if target=='log':pred=np.expm1(pred)
                    pred=np.maximum(0,pred)
                else:
                    net_seed=42+100*int(seed)+int(fold)
                    _,_,epoch,curve=network_fit(d,it,iv,target,kind,net_seed,device,args.epochs,args.patience,True)
                    model,norm,_,_=network_fit(d,tr,None,target,kind,net_seed,device,epoch,args.patience,False)
                    pred=neural_predict(model,norm,d,te,target,device)
                    torch.save(dict(state_dict=model.cpu().state_dict(),normalization={k:np.asarray(v).tolist() for k,v in norm.items()},epochs=epoch,fingerprint=fingerprint),out/f'{name}.pt')
                result=dict(model=kind,target=target,year=year,protocol=args.protocol,seed=int(seed),fold=int(fold),fingerprint=fingerprint,
                    point_id=d['point_id'][te].tolist(),actual_SOC=d['y'][te].tolist(),pred_SOC=pred.tolist(),
                    selected_epochs=epoch,inner_curve=curve,metrics=raw_metrics(d['y'][te],pred),elapsed_seconds=time.time()-started)
                tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(path)
                print('Completed',name,'RMSE',round(result['metrics']['raw_rmse'],3),flush=True)
    if args.only_seed is not None:return
    records=[]
    for path in out.glob('*.json'):
        if path.name=='design.json':continue
        r=json.loads(path.read_text())
        if 'point_id' not in r:continue
        assert r['fingerprint']==fingerprint
        meta={k:r[k] for k in ['model','target','year','protocol','seed','fold']}
        records.append(pd.DataFrame(dict(**meta,point_id=r['point_id'],actual_SOC=r['actual_SOC'],pred_SOC=r['pred_SOC'])))
    p=pd.concat(records);p.to_csv(out/'oof_predictions.csv',index=False);scores=[]
    expected_groups=all_folds[['seed','fold']].drop_duplicates().shape[0]*8
    assert p.groupby(['model','target','seed','fold']).ngroups==expected_groups
    for key,g in p.groupby(['model','target','year','protocol','seed']):
        assert len(g)==663 and g.point_id.nunique()==663
        scores.append(dict(zip(['model','target','year','protocol','seed'],key),**raw_metrics(g.actual_SOC.to_numpy(),g.pred_SOC.to_numpy())))
    pd.DataFrame(scores).to_csv(out/'pooled_oof_by_repeat.csv',index=False)
    (out/'completion.json').write_text(json.dumps(dict(all_completed=True,cnn_trained=True,n_point_predictions=len(p),fingerprint=fingerprint),indent=2))

if __name__=='__main__':
    import rf_compare as stage2
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--protocol',choices=['block_100','block_100_buffer20'],default='block_100');p.add_argument('--repeats',type=int,default=5);p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=15);p.add_argument('--device',default='auto');p.add_argument('--only-seed',type=int,default=None,help='Execute one seed without claiming full completion; used by the batch runner');main(p.parse_args())
