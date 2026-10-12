# GeoJSON Sources

## `ukraine_oblasts.geojson`
27 oblast-level boundaries. Source: geoBoundaries (`shapeISO`/`shapeID`/`shapeGroup` properties).
Loaded via `alertpred.app.map.load_ukraine_geojson`, which rewinds every ring - the
source file's winding order renders inverted (fills the whole canvas instead of each
oblast) in this project's Plotly version.

## `ukraine_raions.geojson`
136 raion-level boundaries reflecting Ukraine's 2020 administrative reform (490 -> 136
raions). Source: [slawomirmatuszak/ukrainian_geodata](https://github.com/slawomirmatuszak/ukrainian_geodata)
(`rayony.geojson`), licensed **CC BY 4.0**. Attribution: Slawomir Matuszak /
ukrainian_geodata, with thanks to Fedir Gontsa per the source repo's README.

Properties: `fid` (integer index, not a stable external id), `rayon` (Ukrainian raion
name, matching `location_title` on alerts.in.ua raion-level alerts).

**Known naming drift vs. the alerts.in.ua API** (~5 of 136 raions, decommunization
renames made after this file was built): e.g. `Новоград-Волинський район` here is
`Звягельський район` in the API; `Володимир-Волинський район` here is `Володимирський
район` in the API. An alias table will be needed when joining on raion name.

An alternative candidate, [zheniaslius/kyiv-air-map](https://github.com/zheniaslius/kyiv-air-map)'s
`public/data/raions.geojson`, had a cleaner 118/118 name match against cached API data
and includes `name_en`/oblast grouping, but the repo has **no stated license**, so it
was not used here.
