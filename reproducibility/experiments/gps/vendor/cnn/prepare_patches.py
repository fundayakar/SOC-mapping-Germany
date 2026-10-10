"""Validate real GEE patch exports and write PyTorch-friendly NPZ files.
Never replace missing raster patches with tabular predictor vectors.
"""
from pathlib import Path
import argparse,ast,hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BANDS=['B2','B3','B4','B8','B11','B12','NDVI','VV','VH','VV_minus_VH','elev','slope','aspect_sin','aspect_cos']
CLIMATE=['tp_winter_m','t2m_summer_C','sm_annual']

def parse_array(value):
    if not isinstance(value,str):raise ValueError('Raster array missing')
    try:a=json.loads(value)
    except json.JSONDecodeError:a=ast.literal_eval(value)
    a=np.asarray(a,dtype=np.float64)
    if a.shape!=(17,17) or not np.isfinite(a).all():raise ValueError('Expected a finite 17x17 raster array')
    return a

def validate(csv,year,out):
    d=pd.read_csv(csv,dtype={'point_id':str})
    ids=pd.read_csv(ROOT/'stage2/input/point_qc/global_paired_ids.csv',dtype=str).point_id.to_numpy()
    assert len(d)==663 and d.point_id.nunique()==663 and set(d.point_id)==set(ids),'Frozen cohort incomplete/changed'
    d=d.set_index('point_id').loc[ids]
    ref=pd.read_csv(ROOT/f'stage2/input/exports/SOC_unified_{year}.csv',dtype={'point_id':str}).set_index('point_id').loc[ids]
    assert d.Depth.eq('0-20 cm').all() and d.predictor_year.eq(year).all()
    assert np.allclose(pd.to_numeric(d.OC),pd.to_numeric(ref.OC),rtol=0,atol=1e-8)
    assert np.allclose(d[['survey_lon','survey_lat']],ref[['survey_lon','survey_lat']],atol=1e-9,rtol=0)
    X=np.stack([np.stack([parse_array(v) for v in d[b]]) for b in BANDS],axis=1)
    valid=np.stack([parse_array(v) for v in d.valid])[:,None]
    assert np.isin(valid,[0,1]).all() and (valid[:,0,8,8]==1).all()
    climate=d[CLIMATE].to_numpy(np.float64);assert np.isfinite(climate).all()
    assert np.allclose(climate,ref[CLIMATE],rtol=1e-7,atol=1e-8)
    # Grid and center-pixel parity are strict checks, not silent cohort filters.
    grid=pd.read_csv(ROOT/'stage2/input/projected_points_3035.csv',dtype={'point_id':str}).set_index('point_id').loc[ids]
    xy=grid[['x_3035','y_3035']].to_numpy(np.float64)
    centers=d[['patch_center_x3035','patch_center_y3035']].to_numpy(np.float64)
    offset=np.linalg.norm(centers-xy,axis=1)
    assert offset.max()<=np.sqrt(2)*10+.2,'Patch not centered on the containing 20m pixel'
    center=X[:,:,8,8];reference=ref[BANDS].to_numpy(np.float64)
    delta=np.abs(center-reference)
    parity=pd.DataFrame(dict(band=BANDS,max_abs_delta=delta.max(axis=0),mean_abs_delta=delta.mean(axis=0)))
    parity.to_csv(out/f'center_parity_{year}.csv',index=False)
    assert np.allclose(center,reference,rtol=1e-5,atol=1e-6),'Patch center differs from point extraction; inspect center_parity CSV before training'
    coverage=pd.DataFrame(dict(point_id=ids,valid_fraction=valid.mean(axis=(1,2,3)),center_offset_m=offset))
    coverage.to_csv(out/f'patch_coverage_{year}.csv',index=False)
    np.savez_compressed(out/f'patches_{year}.npz',X=X.astype('float32'),valid=valid.astype('float32'),climate=climate.astype('float32'),
        y=pd.to_numeric(d.OC).to_numpy(np.float64),point_id=ids.astype(str),xy=xy,patch_center_xy=centers,year=np.array(year),bands=np.array(BANDS),climate_names=np.array(CLIMATE))
    return dict(year=year,n=663,shape=list(X.shape),mask_shape=list(valid.shape),center_parity_passed=True,
        max_center_offset_m=float(offset.max()),minimum_valid_fraction=float(valid.mean(axis=(1,2,3)).min()),
        input_csv_sha256=hashlib.sha256(Path(csv).read_bytes()).hexdigest(),cnn_trained=False)

def main(paths,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);status=[]
    for year,path in paths.items():status.append(validate(path,year,out))
    (out/'patch_validation.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv2018',required=True);p.add_argument('--csv2022',required=True);p.add_argument('--output',default=str(ROOT/'cnn/data'))
    a=p.parse_args();main({2018:a.csv2018,2022:a.csv2022},a.output)
