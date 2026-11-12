# RainNet Province Attribution (38 Provinsi) — Scientific Method & Limitations

## Objective

Convert the binary RainNet segmentation of one contemporary Himawari-9 B13
r2w satellite image into **indicative cloud-coverage summaries** for all 38
Indonesian provinces, with labels on the same image. This does **not** predict
rain amount in mm/h, a rainfall probability, or future precipitation.
JMA HRP product is used solely as an exploratory second visual reference.

## Geographic registration

- **JMA Users' Guide for Heavy Rainfall Potential Images** (July 2015, ver.4):
  https://www.data.jma.go.jp/mscweb/data/himawari/Users_Guide.pdf
- Coverage: upper-left 90°E, 30°N; lower-right 165°E, 15°S.
- JMA grid spacing: **0.05°** in longitude and latitude.
- r2w large imagery inspected at **1501 × 901 pixels**, inclusive endpoints.
- Adopt the explicit grid mapping for raster sample coordinates:
  `pixel_x = (longitude_deg - 90.0) / 0.05`
  `pixel_y = (30.0 - latitude_deg) / 0.05`
- If the JMA image changes size or alignment, province calculations are
  **disabled** rather than silently stretching polygons to fit.
- The regional boundary raster is approximated by nearest-pixel polygon
  coverage, not ground-measured land area in km².

## Provincial boundary source, license and attribution

- Dataset: *Indonesia GeoJSON & TopoJSON Maps (38 Provinces)*
- Repository: https://github.com/denyherianto/indonesia-geojson-topojson-maps-with-38-provinces
- File: GeoJSON/indonesia-38-provinces.geojson
- Pinned upstream revision: **80aa4a90ac7eef5136eaa2b551b6aa5d198be27d**
- Data license: **CC BY 4.0** (see upstream DATA_LICENSE.md)
- Saved unchanged in `app/assets/indonesia_38_provinces.geojson`
- Source URL, license, SHA-256, provenance: `app/assets/province_source.json`
- Source provenance/geometry requires independent verification against official
  BIG/Mendagri administrative boundaries before operational use.
- Re-fetch procedure: `py -3.8 scripts/fetch_province_boundaries.py`.
  This requires internet and validates the number of unique provinces.

## Per-province indicators

For every province polygon, rasterize its land pixels on the registered JMA
grid. The model outputs a probability per pixel; use binary mask at the
dashboard's threshold (default 0.5).

`RainNet_coverage_pct = 100 × (# RainNet-positive province pixels) /
                             (# pixels covered by province polygon)`

`JMA_coverage_pct = 100 × (# visual-magenta HRP province pixels) /
                          (# province pixels)`

Classification labels are **descriptive spatial-coverage bins**, not
meteorological hazard, rainfall intensity, or probability thresholds:

- Below 0.5% OR fewer than 3 positive pixels: **Belum terindikasi**
- 0.5% up to below 3%: **Sebaran lokal**
- 3% up to below 10%: **Sebaran luas**
- 10% and above: **Sebaran sangat luas**

The report includes all 38 provinces (also those with no positive pixels).
The map labels the 12 highest-ranked provinces with positive coverage when
there is sufficient visual space. Full province results are available in the
on-screen table and as a CSV export. The labelled image and JSON report are
written locally to `outputs/runs/YYYY-MM-DD/`.

## Scientific limitations

1. Segmentation is at **observation time**. No multi-frame sequence model
   forecasting +30/+60/+120 minutes is available. Do not call this a weather
   forecast or imply deterministic rainfall in any named province.
2. B13 cloud-top infrared radiance has a different scope from observed
   surface precipitation; cloud-top segmentation is a screening indicator.
3. Reference HRP color extraction from JPEG is approximate and may include
   compression artifacts or map overlays. It is not independent rain-gauge truth.
4. Provincial polygons are approximate in image pixel space: cloud locations
   over boundaries, small offshore islands, and small provinces may have high
   classification uncertainty; labels are illustrative.
5. This workflow is for research interpretation and must not be used alone
   for public warnings or evacuation decisions. Consult official BMKG notices.
