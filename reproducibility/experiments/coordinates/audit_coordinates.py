"""Reproduce the coordinate audit without refitting any prediction model.

Run from this directory: python audit_coordinates.py
Dependencies: numpy, pandas, scipy, pyproj==3.7.2.
Distances between longitude/latitude pairs use WGS84 ellipsoidal geodesics.
Projected spatial design and patch support checks use EPSG:3035, always_xy.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent
IN = ROOT / 'inputs'
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
sources = {
    'Eurostat_DE_2018_20200213.CSV': 'https://ec.europa.eu/eurostat/cache/lucas/DE_2018_20200213.CSV',
    'ESDAC_readme_LUCAS-SOIL-2018.docx': 'https://esdac.jrc.ec.europa.eu/public_path/shared_folder/dataset/75-LUCAS-SOIL-2018/readme_LUCAS-SOIL-2018-29_07_2022.docx',
    'Eurostat_LUCAS2018_RecordDescriptor.pdf': 'https://ec.europa.eu/eurostat/documents/205002/8072634/LUCAS2018-RecordDescriptor-190611.pdf',
    'LUCAS2018_soil_sampling_instructions.pdf': 'https://esdac.jrc.ec.europa.eu/public_path/shared_folder/doc_pub/JRC105923_LUCAS2018_JRCTechnicalReport.pdf',
}
manifest = [{'file': n, 'url': u, 'retrieved_UTC_date': '2026-10-10',
             'bytes': (ROOT/'source'/n).stat().st_size,
             'sha256': hashlib.sha256((ROOT/'source'/n).read_bytes()).hexdigest()}
            for n, u in sources.items()]
(OUT/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
asset = pd.read_csv(IN/'asset_coordinates_778.csv', dtype={'point_id': str})
patch = pd.read_csv(IN/'patch_support_663.csv', dtype={'point_id': str})
micro = pd.read_csv(ROOT/'source/Eurostat_DE_2018_20200213.CSV',
                    dtype={'POINT_ID': str}, low_memory=False)
fields = ['POINT_ID','TH_LAT','TH_LONG','GPS_LAT','GPS_LONG','GPS_PROJ','GPS_EW',
          'GPS_PREC','OBS_DIST','OBS_TYPE','OBS_DIRECT','SURVEY_DATE',
          'SOIL_STD_TAKEN','SOIL_BLK_0_10_TAKEN','SOIL_BLK_10_20_TAKEN']
z = asset.merge(micro[fields], left_on='point_id', right_on='POINT_ID',
                how='left', validate='one_to_one', indicator=True)
assert z['_merge'].eq('both').all() and len(z)==778
z['in_cohort'] = z.point_id.isin(patch.point_id)
g = Geod(ellps='WGS84')
z['asset_to_theoretical_m'] = g.inv(z.survey_lon, z.survey_lat,
                                     z.TH_LONG, z.TH_LAT)[2]
# Never transform sentinel 88.888888 values as valid coordinates.
# Code 1 is a normal GPS projection, code 2 a GPS problem, code 8 irrelevant.
# Geographic bounds are generous for Germany; east longitude must be code 1.
valid = (z.GPS_PROJ.eq(1) & z.GPS_EW.eq(1) & z.OBS_TYPE.eq(1) &
         z.GPS_LAT.between(47,56) & z.GPS_LONG.between(5,16))
z['gps_usable'] = valid
z.loc[valid, 'asset_to_gps_m'] = g.inv(z.loc[valid,'survey_lon'],
    z.loc[valid,'survey_lat'], z.loc[valid,'GPS_LONG'], z.loc[valid,'GPS_LAT'])[2]
tr = Transformer.from_crs('EPSG:4326','EPSG:3035',always_xy=True)
gx,gy=tr.transform(z.loc[valid,'GPS_LONG'].to_numpy(),z.loc[valid,'GPS_LAT'].to_numpy())
z.loc[valid,'gps_x_3035']=gx;z.loc[valid,'gps_y_3035']=gy
tx,ty=tr.transform(z.TH_LONG.to_numpy(),z.TH_LAT.to_numpy())
z['official_th_x_3035']=tx;z['official_th_y_3035']=ty
z.loc[valid,'gps_changes_100km_cell']=(
    (np.floor(gx/100000)!=np.floor(z.loc[valid,'x_3035']/100000)) |
    (np.floor(gy/100000)!=np.floor(z.loc[valid,'y_3035']/100000)))
z['grid_rounding_changes_100km_cell']=(
    (np.floor(z.x_3035/100000)!=np.floor(np.round(z.x_3035/2000)*2000/100000)) |
    (np.floor(z.y_3035/100000)!=np.floor(np.round(z.y_3035/2000)*2000/100000)))
a = z[z.in_cohort].merge(patch, on='point_id', validate='one_to_one')
assert len(a)==663
v=a.gps_usable
dx=a.loc[v,'gps_x_3035']-a.loc[v,'patch_center_x_3035']
dy=a.loc[v,'gps_y_3035']-a.loc[v,'patch_center_y_3035']
a.loc[v,'gps_within_patch_340m']=(abs(dx)<=170)&(abs(dy)<=170)
a.loc[v,'gps_within_centre_20m_cell']=(abs(dx)<=10)&(abs(dy)<=10)
z.to_csv(OUT/'coordinate_crosswalk_778.csv',index=False)
a.to_csv(OUT/'coordinate_crosswalk_cohort_663.csv',index=False)
summary = {}
for name,s in [('source_778',z),('cohort_663',a)]:
    d=s.loc[s.gps_usable,'asset_to_gps_m']
    summary[name]={'n':len(s),'matched_to_official':int(s['_merge'].eq('both').sum()),
        'asset_to_theoretical_max_m':float(s.asset_to_theoretical_m.max()),
        'gps_usable_n':int(s.gps_usable.sum()),
        'gps_unusable_ids':s.loc[~s.gps_usable,'point_id'].tolist(),
        'gps_displacement_m':{str(q):float(d.quantile(q)) for q in [0,.5,.9,.95,.99,1]},
        'gps_displacement_above_20m_n':int((d>20).sum()),
        'gps_displacement_above_100m_n':int((d>100).sum()),
        'gps_precision_median_m':float(s.loc[s.GPS_PREC<8888,'GPS_PREC'].median()),
        'gps_changes_100km_cell_n':int(s.gps_changes_100km_cell.eq(True).sum()),
        'rounding_changes_100km_cell_n':int(s.grid_rounding_changes_100km_cell.sum())}
summary['cohort_663'].update({
    'gps_within_patch_n':int(a.gps_within_patch_340m.eq(True).sum()),
    'gps_within_centre_cell_n':int(a.gps_within_centre_20m_cell.eq(True).sum()),
    'npz_vs_asset_max_displacement_m':float(np.hypot(a.npz_x_3035-a.x_3035,
                                                        a.npz_y_3035-a.y_3035).max())})
# Existing fixed folds, not recomputed folds. Each repeated role is retained.
roles=pd.read_csv(IN/'buffered_outer_roles.csv',dtype={'point_id':str})
zi=z.set_index('point_id');folds=[];near_pairs=[]
for (seed,fold),s in roles.groupby(['seed','fold']):
    train=zi.loc[s.loc[s.role.eq('train'),'point_id']]
    test=zi.loc[s.loc[s.role.eq('test'),'point_id']]
    nd=cdist(train[['x_3035','y_3035']],test[['x_3035','y_3035']])
    train_gps=train[train.gps_usable];test_gps=test[test.gps_usable]
    gd=cdist(train_gps[['gps_x_3035','gps_y_3035']],test_gps[['gps_x_3035','gps_y_3035']])
    folds.append({'seed':int(seed),'fold':int(fold),'nominal_min_distance_m':float(nd.min()),
        'usable_gps_min_distance_m':float(gd.min()),
        'usable_gps_pair_instances_below_20km':int((gd<20000).sum())})
    for i,j in zip(*np.where(gd<20000)):
        near_pairs.append({'seed':int(seed),'fold':int(fold),
            'train_id':train_gps.index[i],'test_id':test_gps.index[j],
            'gps_distance_m':float(gd[i,j])})
pd.DataFrame(folds).to_csv(OUT/'buffered_outer_fold_distances.csv',index=False)
pd.DataFrame(near_pairs).to_csv(OUT/'buffered_outer_pairs_below20km.csv',index=False)
summary['buffered_outer']={'n_folds':len(folds),
    'nominal_min_distance_m':min(f['nominal_min_distance_m'] for f in folds),
    'usable_gps_min_distance_m':min(f['usable_gps_min_distance_m'] for f in folds),
    'pair_instances_below20km':len(near_pairs),
    'unique_unordered_pairs_below20km':len({tuple(sorted([p['train_id'],p['test_id']])) for p in near_pairs})}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
