"""Validate fresh GPS arrays against nominal overlap and freeze paired data."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/cnn'))
from prepare_patches import parse_array,BANDS,CLIMATE

def prepare(csv,year):
    cross=pd.read_csv(ROOT/'inputs/gps_crosswalk_662.csv',dtype={'point_id':str}).set_index('point_id')
    original=dict(np.load(ROOT/f'data/nominal_{year}.npz',allow_pickle=False));ids=original['point_id']
    d=pd.read_csv(csv,dtype={'point_id':str})
    assert len(d)==662 and d.point_id.nunique()==662 and set(d.point_id)==set(ids)
    d=d.set_index('point_id').loc[ids];cross=cross.loc[ids]
    assert d.center_type.eq('gps').all() and d.predictor_year.eq(year).all() and d.Depth.eq('0-20 cm').all()
    assert np.allclose(pd.to_numeric(d.OC),original['y'],atol=1e-8,rtol=0)
    assert np.allclose(d[['gps_lon','gps_lat']],cross[['GPS_LONG','GPS_LAT']],atol=1e-9,rtol=0)
    assert np.allclose(d[['survey_lon','survey_lat']],cross[['survey_lon','survey_lat']],atol=1e-9,rtol=0)
    X=np.stack([np.stack([parse_array(v) for v in d[b]]) for b in BANDS],axis=1).astype('float32')
    valid=np.stack([parse_array(v) for v in d.valid])[:,None].astype('float32')
    assert np.isin(valid,[0,1]).all() and (valid[:,0,8,8]==1).all(), 'Missing GPS center: stop; do not silently filter cohort.'
    climate=d[CLIMATE].to_numpy('float32');assert np.isfinite(climate).all()
    centers=d[['patch_center_x3035','patch_center_y3035']].to_numpy(float)
    gps_xy=cross[['gps_x_3035','gps_y_3035']].to_numpy(float)
    offset=np.linalg.norm(centers-gps_xy,axis=1);assert offset.max()<=np.sqrt(2)*10+.1
    col=np.stack([parse_array(v) for v in d.grid_col]);row=np.stack([parse_array(v) for v in d.grid_row])
    assert (np.diff(col,axis=2)==1).all() and (np.diff(col,axis=1)==0).all()
    assert (np.diff(row,axis=1)==1).all() and (np.diff(row,axis=2)==0).all()
    row_sign=int(np.sign(np.corrcoef(row[:,8,8],centers[:,1])[0,1]))
    col_sign=int(np.sign(np.corrcoef(col[:,8,8],centers[:,0])[0,1]));assert col_sign==1
    shift=np.rint((centers-original['patch_center_xy'])/20).astype(int)
    dr=shift[:,1]*row_sign;dc=shift[:,0]
    all_delta=[];mask_delta=[];qc=[]
    for i,pid in enumerate(ids):
        nr=np.arange(17)+dr[i];nc=np.arange(17)+dc[i]
        keep_r=(nr>=0)&(nr<17);keep_c=(nc>=0)&(nc<17)
        new_r=np.flatnonzero(keep_r);new_c=np.flatnonzero(keep_c)
        old=original['X'][i][:,nr[keep_r]][:,:,nc[keep_c]]
        new=X[i][:,new_r][:,:,new_c]
        delta=np.abs(new-old);all_delta.append(delta.max(axis=(1,2)))
        oldm=original['valid'][i][:,nr[keep_r]][:,:,nc[keep_c]]
        newm=valid[i][:,new_r][:,:,new_c];mask_delta.append(float(np.max(abs(oldm-newm))))
        assert np.allclose(old,new,rtol=1e-5,atol=1e-5),f'Physical overlap differs: {pid}'
        assert np.array_equal(oldm,newm),f'Mask overlap differs: {pid}'
        assert 0<=8+dr[i]<17 and 0<=8+dc[i]<17
        assert np.array_equal(X[i,:,8,8],original['X'][i,:,8+dr[i],8+dc[i]])
        qc.append(dict(point_id=pid,row_shift=int(dr[i]),col_shift=int(dc[i]),center_changed=bool(dr[i] or dc[i]),
            asset_to_gps_m=float(cross.loc[pid,'asset_to_gps_m']),gps_center_snap_distance_m=float(offset[i]),
            valid_fraction=float(valid[i].mean()),climate_changed=bool(np.any(climate[i]!=original['climate'][i])),
            overlap_max_abs_delta=float(delta.max())))
    np.savez_compressed(ROOT/f'data/gps_{year}.npz',X=X,valid=valid,climate=climate,y=original['y'],
        point_id=ids,xy=original['xy'],patch_center_xy=centers,gps_xy=gps_xy,
        year=original['year'],bands=original['bands'],climate_names=original['climate_names'])
    pd.DataFrame(qc).to_csv(ROOT/f'results/extraction_qc_{year}.csv',index=False)
    maxband=np.stack(all_delta).max(axis=0)
    pd.DataFrame({'band':BANDS,'maximum_overlap_abs_delta':maxband}).to_csv(ROOT/f'results/overlap_parity_{year}.csv',index=False)
    return dict(year=year,n=662,gps_center_all_valid=True,actual_arrays_reextracted=True,common_overlap_parity_passed=True,
        max_overlap_abs_delta=float(maxband.max()),mask_overlap_max_delta=max(mask_delta),
        n_centers_changed=int(sum(x['center_changed'] for x in qc)),n_climate_changed=int(sum(x['climate_changed'] for x in qc)),
        row_world_direction=row_sign,input_sha256=hashlib.sha256(Path(csv).read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv2018',required=True);p.add_argument('--csv2022',required=True);a=p.parse_args()
    results=[prepare(a.csv2018,2018),prepare(a.csv2022,2022)]
    (ROOT/'results/extraction_validation.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
