var germany=ee.FeatureCollection('FAO/GAUL/2015/level0').filter(ee.Filter.eq('ADM0_NAME','Germany')).geometry();
var src=ee.FeatureCollection('projects/seismic-relic-481709-r8/assets/LUCAS').filterBounds(germany);
var rows=src.toList(src.size()).map(function(obj){var f=ee.Feature(obj);var xy=f.geometry().transform('EPSG:3035',0.1).coordinates();var ll=f.geometry().transform('EPSG:4326',0.1).coordinates();return ee.List([f.get('POINTID'),xy.get(0),xy.get(1),ll.get(0),ll.get(1)]);});
print('RF_GRID_JSON',ee.String.encodeJSON(rows));
print('RF_GRID_COUNT',src.size());