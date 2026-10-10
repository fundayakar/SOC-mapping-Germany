/* Stage 3 CNN data: 14 physical raster channels + common validity mask; three climate scalars stay in a tabular branch.
 * CHECK: 5 points, no export. PATCHES: frozen 663-point cohort, two CSV exports.
 * Unchanged harmonized image preparation from stage 1; corrected sampleRegions projection argument.
 */
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
var RUN_MODE = 'PATCHES'; // Both years passed live five-point array shape checks; CHECK reruns those checks without exports.
var EXPORT_PATCHES = RUN_MODE === 'PATCHES';
var DRIVE_FOLDER = 'SOC_CNN_stage3';
var PATCH_RADIUS = 8;       // 17 x 17 cells, 340 m footprint at 20 m.
var CRS = 'EPSG:3035';
var SCALE = 20;
var BANDS = ['B2','B3','B4','B8','B11','B12'];
var PATCH_BANDS = BANDS.concat(['NDVI','VV','VH','VV_minus_VH','elev','slope','aspect_sin','aspect_cos']);
var COHORT_IDS = [40523090, 40703150, 40742982, 40783020, 40783066, 40802992, 40803010, 40822936, 40843122, 40862974, 40882942, 40882988, 40883024, 40902992, 40903100, 40923018, 40963188, 40983046, 41003008, 41003224, 41043190, 41063212, 41082902, 41083172, 41083226, 41083280, 41122926, 41123106, 41142930, 41143228, 41223208, 41243258, 41303178, 41303214, 41303268, 41323026, 41323278, 41343142, 41363184, 41383140, 41383180, 41383234, 41383284, 41403088, 41403316, 41423338, 41423356, 41462942, 41462950, 41463292, 41483196, 41502948, 41523186, 41523334, 41543058, 41543076, 41583286, 41602942, 41603078, 41603104, 41603222, 41603320, 41603342, 41623298, 41642752, 41642978, 41643360, 41643396, 41663232, 41682724, 41682750, 41702722, 41703028, 41703214, 41703268, 41723044, 41742978, 41743032, 41743072, 41743198, 41743302, 41743306, 41763100, 41763142, 41763268, 41763334, 41782772, 41782888, 41782970, 41783032, 41783068, 41783204, 41783234, 41783288, 41783342, 41783356, 41783392, 41802956, 41823036, 41823058, 41823112, 41823184, 41823212, 41823284, 41823346, 41823378, 41843340, 41843358, 41863092, 41882776, 41883064, 41883154, 41883208, 41902738, 41902954, 41903314, 41903350, 41923186, 41962754, 41962952, 41963126, 41963222, 41963248, 41963360, 41982722, 41983298, 42003148, 42003184, 42003238, 42022908, 42023304, 42063100, 42063322, 42082918, 42083076, 42083224, 42083332, 42083350, 42103248, 42123178, 42142970, 42143270, 42143366, 42163298, 42182748, 42183148, 42183342, 42223056, 42223066, 42243136, 42263008, 42263076, 42263094, 42263098, 42283108, 42283198, 42283288, 42283356, 42303034, 42322970, 42323042, 42323068, 42323086, 42323104, 42323114, 42323144, 42343028, 42343226, 42362802, 42362968, 42363036, 42363076, 42363108, 42363202, 42363374, 42383304, 42403002, 42403026, 42403110, 42403260, 42403344, 42403354, 42403530, 42422746, 42423010, 42423082, 42423100, 42443026, 42443112, 42443310, 42443472, 42462780, 42462906, 42463090, 42463122, 42463162, 42463294, 42463356, 42482812, 42483154, 42483226, 42483262, 42483322, 42483466, 42502766, 42502820, 42503014, 42503180, 42503248, 42503342, 42503366, 42522758, 42523208, 42543032, 42543054, 42543086, 42543090, 42543140, 42543212, 42543248, 42543270, 42562962, 42563034, 42563304, 42582994, 42583038, 42583062, 42583182, 42583236, 42602746, 42602758, 42602992, 42603006, 42603070, 42603100, 42603190, 42603298, 42603492, 42622896, 42623080, 42623116, 42623368, 42642924, 42643126, 42643158, 42643198, 42643222, 42663262, 42682790, 42682952, 42683042, 42683078, 42683096, 42683204, 42683456, 42683482, 42703082, 42723058, 42723086, 42723126, 42723238, 42723266, 42723338, 42723504, 42723522, 42743070, 42762738, 42762760, 42763062, 42763074, 42763084, 42763102, 42763174, 42763282, 42763318, 42763506, 42782830, 42783106, 42783118, 42803098, 42803112, 42822960, 42823374, 42843466, 42862808, 42863078, 42863176, 42863320, 42863486, 42883028, 42903112, 42903252, 42903450, 42903482, 42903508, 42923502, 42943084, 42943098, 42943146, 42943458, 42962764, 42962776, 42963078, 42963124, 42982824, 42983224, 42983458, 42983494, 43002766, 43003518, 43022722, 43022740, 43022782, 43023088, 43023106, 43023370, 43023484, 43042730, 43043060, 43043086, 43043230, 43043486, 43083104, 43083126, 43083184, 43083270, 43083338, 43083392, 43103214, 43123092, 43123120, 43123128, 43142746, 43143096, 43162914, 43162972, 43163044, 43163098, 43163458, 43182730, 43182744, 43182820, 43183122, 43202722, 43202746, 43203016, 43222730, 43222748, 43223108, 43223266, 43263094, 43263126, 43263238, 43263454, 43283124, 43303228, 43303296, 43322934, 43323106, 43323456, 43343040, 43343458, 43362748, 43363252, 43363356, 43382746, 43383214, 43383316, 43403168, 43423280, 43442986, 43443018, 43443126, 43443464, 43463466, 43483260, 43483278, 43483300, 43483308, 43502746, 43502764, 43503106, 43503136, 43503258, 43503456, 43523256, 43542748, 43543126, 43562776, 43582726, 43583104, 43583108, 43583194, 43583302, 43622788, 43622798, 43622816, 43622824, 43622838, 43642800, 43643304, 43662756, 43662760, 43663170, 43663254, 43663264, 43663296, 43663318, 43682776, 43703224, 43703382, 43723032, 43723306, 43723446, 43742782, 43743118, 43743244, 43743280, 43743460, 43763176, 43763252, 43763306, 43783136, 43783280, 43802766, 43803090, 43803230, 43803270, 43822800, 43843278, 43843308, 43843470, 43882738, 43882828, 43883134, 43883278, 43883310, 43923244, 43942820, 43943180, 43943284, 43983288, 44023246, 44023336, 44042800, 44042830, 44043276, 44062810, 44063148, 44063188, 44063202, 44083090, 44083162, 44083356, 44103118, 44103280, 44103316, 44123180, 44143298, 44163086, 44163216, 44163248, 44163274, 44163288, 44163400, 44203092, 44203246, 44203254, 44203300, 44223106, 44223258, 44223286, 44243098, 44243278, 44263126, 44282800, 44283316, 44303068, 44342838, 44343058, 44343140, 44363178, 44383062, 44383080, 44402818, 44403106, 44422810, 44422846, 44443230, 44463286, 44482834, 44482856, 44483096, 44483302, 44483338, 44502938, 44503244, 44503370, 44523270, 44523310, 44523360, 44562976, 44563084, 44563192, 44563260, 44602792, 44603022, 44603256, 44603314, 44603404, 44623090, 44623270, 44643268, 44663072, 44663338, 44702744, 44703302, 44742732, 44762764, 44762836, 44763100, 44763262, 44763286, 44763294, 44763430, 44782914, 44783296, 44783310, 44783454, 44802780, 44803176, 44803324, 44822740, 44823280, 44823334, 44842784, 44842892, 44843230, 44843348, 44862848, 44882852, 44883126, 44883184, 44883234, 44883396, 44903340, 44923048, 44923174, 44923272, 44923290, 44923314, 44942974, 44943268, 44962878, 44962896, 44962918, 44963130, 44963202, 44963220, 44963472, 44983158, 44983234, 45002758, 45002866, 45003106, 45022834, 45022946, 45023176, 45023284, 45023302, 45043316, 45062870, 45063274, 45063310, 45102904, 45103272, 45103300, 45103308, 45122758, 45123226, 45123286, 45123304, 45143274, 45162820, 45162852, 45163194, 45163216, 45182836, 45183250, 45202838, 45202880, 45202898, 45203194, 45222812, 45222866, 45243238, 45243306, 45262854, 45263232, 45282846, 45282904, 45283260, 45283278, 45283408, 45323152, 45383240, 45383374, 45422842, 45423284, 45423302, 45423320, 45442854, 45443304, 45463200, 45482808, 45482830, 45502846, 45502864, 45523324, 45562852, 45583298, 45583334, 45603292, 45662866, 45703230, 45703360, 45723100, 45723304, 45723316, 45743122, 45743266, 45782870, 45803232, 45803340, 45823296, 45843316, 45843352, 45863256, 45863292, 45883320, 45903358, 45922856, 45923114, 45923158, 45923252, 45923270, 45963310, 45963320, 45963328, 45963382, 45983106, 45983142, 45983214, 45983232, 46003300, 46043346, 46043364, 46063266, 46103168, 46103252, 46183264, 46283230, 46303244]; // Frozen paired 0-20 cm cohort, n=663.
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
allPoints = allPoints.filter(ee.Filter.inList('POINTID', COHORT_IDS)).filter(ee.Filter.eq('Depth','0-20 cm'));
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
  if (EXPORT_PATCHES || RUN_MODE === 'CHECK') {
    var projection = ee.Projection(CRS).atScale(SCALE);
    var image=stack.select(PATCH_BANDS).reproject(projection);
    var valid=image.mask().reduce(ee.Reducer.min()).rename('valid').unmask(0, false);
    image=image.unmask(0, false).addBands(valid);
    var arrays=image.neighborhoodToArray(ee.Kernel.square({radius:PATCH_RADIUS,units:'pixels'}));
    var props=['point_id','source_index','POINTID','OC','SURVEY_DATE','Depth','LC0_Desc','survey_lon','survey_lat',
      'tp_winter_m','t2m_summer_C','sm_annual','era_grid_x','era_grid_y'];
    var patches=arrays.sampleRegions({collection:table, properties:props,
      projection:projection, scale:SCALE, geometries:true, tileScale:4}).map(function(f){
        var ll=f.geometry().transform('EPSG:4326',0.1).coordinates();
        var xy=f.geometry().transform(CRS,0.1).coordinates();
        return f.set({predictor_year:year, patch_center_lon:ll.get(0),patch_center_lat:ll.get(1),
          patch_center_x3035:xy.get(0),patch_center_y3035:xy.get(1)});
      });
    print('PATCH count '+year, patches.size());
    print('PATCH band names '+year, PATCH_BANDS);
    if (RUN_MODE === 'CHECK') {
      var f=ee.Feature(patches.first());
      var shapes=ee.Dictionary.fromLists(PATCH_BANDS.concat(['valid']),PATCH_BANDS.concat(['valid']).map(function(b){
        var a=ee.Array(f.get(b));return a.length().toList();
      }));
      print('PATCH shapes '+year,shapes);
      print('PATCH center QC '+year,f.select(props.concat(['predictor_year','patch_center_lon','patch_center_lat'])));
    }
    if (EXPORT_PATCHES) {
      Export.table.toDrive({collection:patches,description:'SOC_CNN_patch17_'+year,folder:DRIVE_FOLDER,
        fileNamePrefix:'SOC_CNN_patch17_'+year,fileFormat:'CSV',
        selectors:props.concat(['predictor_year','patch_center_lon','patch_center_lat','patch_center_x3035','patch_center_y3035']).concat(PATCH_BANDS).concat(['valid'])});
    }
  }
});
