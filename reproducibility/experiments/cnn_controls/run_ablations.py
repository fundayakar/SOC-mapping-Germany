"""Same-capacity CNN information controls on unchanged stage-3 spatial folds.
Original training-data normalization shared across all variants; test labels unused.
"""
from pathlib import Path
import argparse,concurrent.futures as cf,hashlib,json,sys,time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold

HERE=Path(__file__).resolve().parent
PRIMARY=['center_tiled','shuffle_314159']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def permutation(ids,seed):
    out=np.broadcast_to(np.arange(289,dtype=np.int16),(len(ids),289)).copy()
    rest=np.delete(np.arange(289),144)
    for i,pid in enumerate(ids):
        key=int.from_bytes(hashlib.sha256(f'{seed}:{pid}'.encode()).digest()[:8],'little')
        out[i,rest]=np.random.default_rng(key).permutation(rest)
    return out

def transformed(d,variant):
    v=dict(d)
    if variant=='center_tiled':v['X']=np.broadcast_to(d['X'][:,:,8:9,8:9],d['X'].shape).copy()
    elif variant.startswith('shuffle_'):
        perm=permutation(d['point_id'],int(variant.split('_')[1]))
        v['X']=np.take_along_axis(d['X'].reshape(len(perm),14,289),perm[:,None,:],axis=2).reshape(d['X'].shape).copy()
    else:assert variant=='original_replay'
    assert np.array_equal(v['X'][:,:,8,8],d['X'][:,:,8,8])
    return v

def imports(root):
    sys.path.insert(0,str(root/'cnn'));sys.path.insert(0,str(root))
    import train_patch_models as t
    t.torch.set_num_threads(1)
    return t

def fit(t,original,d,tr,va,target,seed,epochs,select):
    # Deliberately identical optimization to stage 3, with explicit common norm.
    t.seed_all(seed);norm=t.normalization(original,tr,target,'patchCNN')
    model=t.PatchCNN();opt=t.torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    generator=t.torch.Generator().manual_seed(seed)
    loader=t.DataLoader(t.TensorDataset(*t.tensors(d,tr,norm,target)),batch_size=32,shuffle=True,generator=generator)
    validation=t.tensors(d,va,norm,target) if select else None
    best=float('inf');chosen=1;stale=0;curve=[]
    for epoch in range(1,epochs+1):
        model.train()
        for image,point,y in loader:
            opt.zero_grad();loss=t.nn.functional.mse_loss(model(image,point),y);loss.backward();opt.step()
        if select:
            model.eval()
            with t.torch.no_grad():
                image,point,y=validation;error=float(t.nn.functional.mse_loss(model(image,point),y).item())
            curve.append(dict(epoch=epoch,inner_validation_MSE_standardized=error))
            if error<best-1e-5:best=error;chosen=epoch;stale=0
            else:stale+=1
            if stale>=15:break
    return model,norm,chosen if select else epochs,curve

def job(root,year,target,protocol,seed,fold,variant):
    t=imports(root);started=time.time();design=json.loads((HERE/'design.json').read_text())
    name=f'{protocol}_{year}_{target}_s{seed}_f{fold}_{variant}';out=HERE/'runs';out.mkdir(exist_ok=True)
    path=out/f'{name}.json'
    if path.exists():
        assert json.loads(path.read_text())['fingerprint']==design['fingerprint'] and (out/f'{name}.pt').is_file()
        return name,0
    orig=dict(np.load(root/f'cnn/data/patches_{year}.npz',allow_pickle=False));d=transformed(orig,variant)
    ids={v:i for i,v in enumerate(d['point_id'])}
    folds=pd.read_csv(root/'stage2/results/fold_assignments.csv',dtype={'point_id':str})
    g=folds.query("cohort=='global_0_20' and protocol==@protocol and seed==@seed and fold==@fold")
    tr=np.array(sorted(ids[p] for p in g[g.role.eq('train')].point_id));te=np.array(sorted(ids[p] for p in g[g.role.eq('test')].point_id))
    bxy=np.floor(d['xy']/100000).astype(int);blocks=np.array([f'{a}_{b}' for a,b in bxy])
    assert not set(blocks[tr])&set(blocks[te])
    outer_distance=float(cKDTree(d['xy'][tr]).query(d['xy'][te])[0].min())
    patch_distance=float(cKDTree(d['patch_center_xy'][tr]).query(d['patch_center_xy'][te])[0].min())
    assert patch_distance>np.sqrt(2)*340
    a,b=next(GroupKFold(3).split(tr,groups=blocks[tr]));it=tr[a];iv=tr[b]
    if protocol.endswith('buffer20'):it=it[cKDTree(d['xy'][iv]).query(d['xy'][it])[0]>20000]
    inner_distance=float(cKDTree(d['xy'][it]).query(d['xy'][iv])[0].min())
    if protocol.endswith('buffer20'):assert min(outer_distance,inner_distance)>20000
    netseed=42+100*int(seed)+int(fold)
    _,_,epoch,curve=fit(t,orig,d,it,iv,target,netseed,100,True)
    model,norm,_,_=fit(t,orig,d,tr,None,target,netseed,epoch,False)
    pred=t.neural_predict(model,norm,d,te,target,'cpu')
    replay_delta=None
    if variant=='original_replay':
        folder='results_buffer20' if protocol.endswith('buffer20') else 'results'
        ref=json.loads((root/f'cnn/{folder}/{year}/patchCNN_{target}_s{seed}_f{fold}.json').read_text())
        assert ref['point_id']==d['point_id'][te].tolist() and ref['selected_epochs']==epoch
        np.testing.assert_allclose(pred,ref['pred_SOC'],rtol=1e-8,atol=1e-8)
        replay_delta=float(np.max(abs(pred-np.array(ref['pred_SOC']))))
    checkpoint=dict(state_dict=model.state_dict(),normalization={k:np.asarray(v).tolist() for k,v in norm.items()},
        normalization_dtypes={k:str(np.asarray(v).dtype) for k,v in norm.items()},epochs=epoch,fingerprint=design['fingerprint'],
        variant=variant,net_seed=netseed,parameter_count=sum(p.numel() for p in model.parameters()))
    tmp=out/f'{name}.pt.tmp';t.torch.save(checkpoint,tmp);tmp.replace(out/f'{name}.pt')
    record=dict(fingerprint=design['fingerprint'],variant=variant,year=year,target=target,protocol=protocol,seed=int(seed),fold=int(fold),
        point_id=d['point_id'][te].tolist(),actual_SOC=d['y'][te].tolist(),pred_SOC=pred.tolist(),selected_epochs=epoch,inner_curve=curve,
        expert_train_ids=d['point_id'][tr].tolist(),epoch_train_ids=d['point_id'][it].tolist(),epoch_validation_ids=d['point_id'][iv].tolist(),
        train_test_distance_m=outer_distance,inner_train_valid_distance_m=inner_distance,patch_train_test_distance_m=patch_distance,
        metrics=t.raw_metrics(d['y'][te],pred),original_replay_max_delta=replay_delta,elapsed_seconds=time.time()-started)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record));tmp.replace(path)
    return name,time.time()-started

def main(root,workers):
    t=imports(root)
    design=dict(name='CNN_same_capacity_information_ablation',primary_variants=PRIMARY,secondary_shuffle_seed=271828,
        years=[2018,2022],targets=['log','raw'],protocol_repeats={'block_100':5,'block_100_buffer20':3},
        architecture='Unchanged stage3 PatchCNN, 8449 parameters, same center/climate skip branch and masks.',
        transformations='Center_tiled repeats original 14-band center throughout 17x17. Shuffle preserves center, uses one deterministic point-id hash permutation shared across all 14 bands; both train and test transformed before separate training. Same permutation in both years and repeats.',
        normalization='All variants use original, untransformed training-patch moments. Point/climate tensors therefore exactly identical across variants; statistics fit separately on inner and outer training only.',
        optimization='Adam lr .001 wd .0001 batch32, 100max epochs patience15; first 3-way 100km GroupKFold inner split, with20km inner buffer where relevant; separate epoch selection pervariant.',
        seeds='Neural init and loader exactly stage3: 42+100*outerseed+fold. One secondary permutation seed on outerseed0 only; original replay on seed0/fold0 for8year-target-protocol conditions.',
        interpretation='Separately retrained input controls; not causal pedological inference, external validation, or novelty proof.',
        torch_version=t.torch.__version__,threads=1,code_sha256=sha(__file__),
        source_library_id='libfile_ad27616539488191b0d5d12e7ba36ef0',
        source_sha256={str(p.relative_to(root)):sha(p) for p in [root/'cnn/train_patch_models.py',root/'cnn/prepare_patches.py',root/'target_compare.py',root/'stage2/results/fold_assignments.csv',root/'cnn/data/patches_2018.npz',root/'cnn/data/patches_2022.npz']})
    design['fingerprint']=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    path=HERE/'design.json'
    if path.exists():assert json.loads(path.read_text())['fingerprint']==design['fingerprint'],'Different design needs a new folder.'
    path.write_text(json.dumps(design,indent=2))
    (HERE/'permutations').mkdir(exist_ok=True)
    d=dict(np.load(root/'cnn/data/patches_2018.npz',allow_pickle=False))
    for seed in [314159,271828]:np.savez_compressed(HERE/f'permutations/seed_{seed}.npz',point_id=d['point_id'],mapping=permutation(d['point_id'],seed))
    jobs=[]
    for y in [2018,2022]:
        for target in ['log','raw']:
            for protocol,repeats in [('block_100',5),('block_100_buffer20',3)]:
                for seed in range(repeats):
                    for fold in range(5):
                        for variant in PRIMARY:jobs.append((root,y,target,protocol,seed,fold,variant))
                        if seed==0:jobs.append((root,y,target,protocol,seed,fold,'shuffle_271828'))
                        if seed==0 and fold==0:jobs.append((root,y,target,protocol,seed,fold,'original_replay'))
    assert len(jobs)==368
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(job,*j) for j in jobs]
        for i,f in enumerate(cf.as_completed(futures),1):
            name,seconds=f.result();print(f'Completed {i}/368 {name} ({seconds:.1f}s)',flush=True)
    summarize(root)

def summarize(root):
    (HERE/'results').mkdir(exist_ok=True);records=[];epochs=[]
    for path in sorted((HERE/'runs').glob('*.json')):
        r=json.loads(path.read_text());meta={k:r[k] for k in ['variant','year','target','protocol','seed','fold']}
        records.append(pd.DataFrame(dict(**meta,point_id=r['point_id'],actual_SOC=r['actual_SOC'],pred_SOC=r['pred_SOC'])))
        epochs.append(dict(**meta,epochs=r['selected_epochs'],elapsed_seconds=r['elapsed_seconds']))
    assert len(records)==368
    for folder in ['results','results_buffer20']:
        for year in [2018,2022]:
            p=pd.read_csv(root/f'cnn/{folder}/{year}/oof_predictions.csv',dtype={'point_id':str});p=p[p.model.isin(['patchCNN','centerMLP','centerRF','contextRF'])].rename(columns={'model':'variant'});records.append(p)
    p=pd.concat(records,ignore_index=True);p.to_csv(HERE/'results/all_predictions.csv',index=False)
    t=imports(root);scores=[]
    for k,g in p.groupby(['variant','year','target','protocol','seed']):
        if len(g)!=663:assert k[0]=='original_replay';continue
        assert g.point_id.nunique()==663
        scores.append(dict(zip(['variant','year','target','protocol','seed'],k),**t.raw_metrics(g.actual_SOC.to_numpy(),g.pred_SOC.to_numpy())))
    pd.DataFrame(scores).to_csv(HERE/'results/scores_by_repeat.csv',index=False)
    pd.DataFrame(epochs).to_csv(HERE/'results/epochs.csv',index=False)
    (HERE/'completion.json').write_text(json.dumps(dict(all_completed=True,new_model_fold_fits=368,primary_control_fits=320,
        second_permutation_fits=40,original_replays=8,CNN_parameter_count=8449),indent=2))
    print('All 368 ablation/replay fits completed.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage3-root',type=Path,default=HERE.parent/'target_transform');ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args();main(a.stage3_root.resolve(),a.workers)
