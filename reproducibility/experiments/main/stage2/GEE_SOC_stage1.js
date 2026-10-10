/* Germany SOC revision: harmonized, paired 2018/2022 extraction.
 * Paste into GEE Code Editor. Requires access to the configured LUCAS asset.
 * Stage-1 version: CHECK mode samples five points and creates no export tasks.
 * POINTS mode creates the two CSV export tasks. PATCHES is a later stage.
 * Live asset checked: 778 points; POINTID, SURVEY_DATE, Depth and LC labels exist.
 * Both years passed live five-point raster checks on 7 October 2026.
 * Keep table year, coordinates, source IDs and all coverage diagnostics.
 * Raster values at 20 m do not acquire 20 m native information: SRTM ~30 m,
 * ERA5-Land ~9 km and S2 SWIR 20 m. ERA5 is sampled on its native grid.
 */
var LUCAS_ASSET = 'projects/seismic-relic-481709-r8/assets/LUCAS';
var POINT_ID_FIELD = 'POINTID'; // Verified on the actual LUCAS asset, 7 October 2026.
var YEARS = [2018, 2022];
var RUN_MODE = 'CHECK'; // CHECK first; then POINTS; PATCHES only after table QC.
var EXPORT_PATCHES = RUN_MODE === 'PATCHES';
var DRIVE_FOLDER = 'SOC_revision_2018_2022';
var PATCH_RADIUS = 8;       // 17 x 17 cells, 340 m footprint at 20 m.
var CRS = 'EPSG:3035';
var SCALE = 20;
var BANDS = ['B2','B3','B4','B8','B11','B12'];
var PATCH_BANDS = BANDS.concat(['NDVI','VV','VH','elev','slope','aspect_sin','aspect_cos','valid']);
var germany = ee.FeatureCollection('FAO/GAUL/2015/level0')
  .filter(ee.Filter.eq('ADM0_NAME', 'Germany')).geometry();
var source = ee.FeatureCollection(LUCAS_ASSET).filterBounds(germany);
var allPoints = source.map(function(f) {
  var xy = f.geometry().transform('EPSG:4326', 1).coordinates();
  var hasId = ee.Algorithms.If(ee.Algorithms.IsEqual(f.get(POINT_ID_FIELD), null), false, true);
  return f.set({point_id: ee.Algorithms.If(hasId, f.get(POINT_ID_FIELD), f.get('system:index')),
    source_index: f.get('system:index'), survey_lon: xy.get(0), survey_lat: xy.get(1)});
}).select(['point_id','source_index','POINTID','OC','SURVEY_DATE','Depth',
  'LC','LC0_Desc','LC1_Desc','LU','LU1_Desc','survey_lon','survey_lat']);
// Preserve original survey land-cover metadata; do not use other soil
// chemistry measurements or grid IDs as automatically selected predictors.
if (['CHECK','POINTS','PATCHES'].indexOf(RUN_MODE) < 0) { throw new Error('Invalid RUN_MODE'); }
var points = RUN_MODE === 'CHECK' ? allPoints.limit(5) : allPoints;
print('RUN_MODE', RUN_MODE);
print('Source count (OC < LOD retained for explicit Python handling)', allPoints.size());
print('Point ID count (must equal source count)', allPoints.aggregate_count_distinct('point_id'));
print('LC0 distribution (survey metadata)', allPoints.aggregate_histogram('LC0_Desc'));
print('Depth distribution', allPoints.aggregate_histogram('Depth'));
print('Missing OC count (retained explicitly)', allPoints.size().subtract(allPoints.filter(ee.Filter.notNull(['OC'])).size()));
print('Source example: verify original POINTID, OC g/kg and 2018 survey; coordinates are FROM ASSET', points.first());
Map.centerObject(germany, 6);

function date(year, month, day) { return ee.Date.fromYMD(year, month, day); }
function s2Collection(year, startMonth, endMonth) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(germany).filterDate(date(year,startMonth,1), date(year,endMonth,1))
    // Retain the historical scene threshold as a documented design choice.
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
    .map(function(im) {
      var scl=im.select('SCL');
      var clear=scl.neq(0).and(scl.neq(1)).and(scl.neq(2)).and(scl.neq(3))
        .and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
      var refl=im.select(BANDS).multiply(0.0001).updateMask(clear);
      var ndvi=refl.expression('(n-r)/(n+r)',{n:refl.select('B8'),r:refl.select('B4')}).rename('NDVI');
      var bsi=refl.expression('((s+r)-(n+b))/((s+r)+(n+b))',{
        s:refl.select('B11'),r:refl.select('B4'),n:refl.select('B8'),b:refl.select('B2')}).rename('BSI');
      return refl.addBands([ndvi,bsi]).copyProperties(im,['system:time_start']);
    });
}
function medianWithIndices(col) {
  var refl=col.select(BANDS).median();
  var ndvi=refl.expression('(n-r)/(n+r)',{n:refl.select('B8'),r:refl.select('B4')}).rename('NDVI');
  var bsi=refl.expression('((s+r)-(n+b))/((s+r)+(n+b))',{
    s:refl.select('B11'),r:refl.select('B4'),n:refl.select('B8'),b:refl.select('B2')}).rename('BSI');
  return refl.addBands([ndvi,bsi]);
}
function sar(year) {
  var col=ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(germany)
    .filterDate(date(year,1,1),date(year+1,1,1))
    .filter(ee.Filter.eq('instrumentMode','IW'))
    .filter(ee.Filter.eq('orbitProperties_pass','DESCENDING'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'))
    .select(['VV','VH']);
  var med=col.median();
  var db=med.select('VV').subtract(med.select('VH')).rename('VV_minus_VH');
  // The former VV_dB/VH_dB is not a linear polarization ratio.
  var ratio=ee.Image.constant(10).pow(db.divide(10)).rename('VV_VH_linear_ratio');
  return med.addBands([db,ratio,col.select('VV').count().unmask(0).rename('n_s1')]);
}
var elev=ee.Image('USGS/SRTMGL1_003').select('elevation').rename('elev');
var terrain=ee.Terrain.products(elev);
var radians=terrain.select('aspect').multiply(Math.PI/180);
var dem=elev.addBands([terrain.select('slope'),terrain.select('aspect'),
  radians.sin().rename('aspect_sin'),radians.cos().rename('aspect_cos')]);
var era=ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR');
var eraProjection=ee.Image(era.first()).select('temperature_2m').projection();
// Coarse independent land-cover description, never a mineral/organic-soil label.
var corine=ee.Image('COPERNICUS/CORINE/V20/100m/2018').select('landcover').rename('clc2018');
function climate(year) {
  var growing=era.filterDate(date(year,4,1),date(year,11,1));
  return ee.Image.cat([
    growing.select('temperature_2m').mean().subtract(273.15).rename('t2m_growing_C'),
    growing.select('total_precipitation_sum').sum().rename('tp_growing_m'),
    growing.select('volumetric_soil_water_layer_1').mean().rename('sm1_growing'),
    growing.select('volumetric_soil_water_layer_2').mean().rename('sm2_growing'),
    era.filterDate(date(year-1,12,1),date(year,3,1)).select('total_precipitation_sum').sum().rename('tp_winter_m'),
    era.filterDate(date(year,6,1),date(year,9,1)).select('temperature_2m').mean().subtract(273.15).rename('t2m_summer_C'),
    era.filterDate(date(year,1,1),date(year+1,1,1)).select('volumetric_soil_water_layer_1').mean().rename('sm_annual'),
    ee.Image.pixelCoordinates(eraProjection).rename(['era_grid_x','era_grid_y'])
  ]);
}
YEARS.forEach(function(year) {
  var growing=s2Collection(year,4,11);
  var optical=medianWithIndices(growing);
  var nClear=growing.select('B4').count().unmask(0).rename('n_s2_clear');
  // This diagnostic masks individual observations before compositing.
  var bare=growing.map(function(im){ return im.updateMask(im.select('NDVI').lt(0.30)); });
  var nBare=bare.select('B4').count().unmask(0).rename('n_s2_bare030');
  var bareComposite=medianWithIndices(bare).rename(BANDS.concat(['NDVI','BSI']).map(function(b){return b+'_bare030';}));
  var spring=medianWithIndices(s2Collection(year,4,7)).rename(BANDS.concat(['NDVI','BSI']).map(function(b){return b+'_sp';}));
  var autumn=medianWithIndices(s2Collection(year,9,11)).rename(BANDS.concat(['NDVI','BSI']).map(function(b){return b+'_au';}));
  var stack=optical.addBands(sar(year)).addBands(dem).addBands([spring,autumn,bareComposite,nClear,nBare,
    nBare.divide(nClear.max(1)).rename('bare_fraction030')]);
  var table=stack.reduceRegions({collection:points,reducer:ee.Reducer.first(),crs:CRS,scale:SCALE,tileScale:4});
  // Native-grid climate sampling retains grid IDs instead of deducing cells
  // from rounded climate value tuples. reduceRegions retains missing rows.
  table=climate(year).reduceRegions({collection:table,reducer:ee.Reducer.first(),
    crs:eraProjection,tileScale:4}).map(function(f){return f.set('predictor_year',year);});
  table=corine.reduceRegions({collection:table,reducer:ee.Reducer.first().setOutputs(['clc2018']),
    crs:corine.projection(),tileScale:4});
  if (RUN_MODE === 'POINTS') {
    Export.table.toDrive({collection:table,description:'SOC_unified_'+year,
      folder:DRIVE_FOLDER,fileNamePrefix:'SOC_unified_'+year,fileFormat:'CSV'});
  }
  if (RUN_MODE === 'CHECK') {
    print('CHECK '+year+' (5 source points; no export)', table.limit(5));
    print('S2 scene count '+year, growing.size());
  }
  if (EXPORT_PATCHES) {
    // Align to one fixed national grid. Nearest-neighbor resampling is
    // deliberate; no new detail is attributed to the coarse predictors.
    var image=stack.select(PATCH_BANDS.filter(function(b){return b!=='valid';}))
      .reproject({crs:CRS,scale:SCALE});
    var valid=image.mask().reduce(ee.Reducer.min()).rename('valid').unmask(0);
    image=image.unmask(0).addBands(valid);
    var arrays=image.neighborhoodToArray(ee.Kernel.square({radius:PATCH_RADIUS,units:'pixels'}));
    var patches=arrays.sampleRegions({collection:points,
      properties:['point_id','source_index','POINTID','OC','SURVEY_DATE','Depth','LC','LC0_Desc','LC1_Desc','survey_lon','survey_lat'],crs:CRS,scale:SCALE,
      geometries:true,tileScale:4}).map(function(f){return f.set('predictor_year',year);});
    Export.table.toDrive({collection:patches,description:'SOC_patch17_'+year,folder:DRIVE_FOLDER,fileNamePrefix:'SOC_patch17_'+year,fileFormat:'CSV'});
    print('Patch count and first patch: validate against source IDs',patches.size(),patches.first());
  }
});
