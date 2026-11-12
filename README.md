# RainNet V1 — Identifying Potential Heavy Rainfall

**RainNet: Convolutional Neural Network for Convective Cloud Segmentation and Identifying Potential Heavy Rainfall**

Research project associated with the 2024 doctoral dissertation of Agus Perdana Windarto, *Pengembangan Model Convolutional Neural Network untuk Segmentasi Awan Konvektif dalam Mengidentifikasi Potensi Hujan Deras.*

## What this repository provides

A **VS Code / Streamlit multi-agent desktop-local research dashboard** that retrieves synchronized Himawari **B13 Infrared** and **Heavy Rainfall Potential (HRP)** images from Japan Meteorological Agency (JMA) and applies the original trained **RainNet** model. The system compares its segmentation visually with JMA's HRP product, produces an annotated yellow RainNet overlay, and reports indicative cloud coverage for **38 Indonesian provinces** with geographic labels, exportable CSV, and a saved JSON audit report.

**Not a rainfall gauge or an operational warning system.** HRP indicates possible or probable rainfall associated with deep convective clouds, not measured rainfall in millimeters. Derived-mask similarity is exploratory agreement with JMA imagery, not independent ground-truth validation or a forecast of future rainfall.

### Multi-agent architecture

1. **Acquisition Agent** — downloads matching B13 and HRP product images (same region and time asset key).
2. **Quality & Alignment Agent** — validates image dimensions, timestamps, freshness, and the supported trained domain.
3. **RainNet Inference Agent** — loads authentic dissertation checkpoint, segments original RGB B13 images in 400×400 pixel tiles.
4. **Reference Comparison Agent** — estimates magenta HRP image mask; calculates screen-space Dice and IoU with scientific caveats.
5. **Province Attribution Agent** — rasterizes licensed 38-province GeoJSON at the documented JMA 0.05-degree grid, ranks indicative cloud-covered pixels, and labels province names on the satellite image.
6. **Reporting Agent** — produces a timestamped JSON report, labeled image and exportable 38-province CSV via the GUI.

### Local workspace

All data, code, checkpoints, reports, logs, and historical experiments are in:

`D:\DISERTASI_S3\PROGRAM_RAINNET`

The user's original dissertation source directories are preserved separately. Their contents were copied and checked before integration.

### Run with VS Code on Windows

Open `D:\DISERTASI_S3\PROGRAM_RAINNET` in VS Code. Select Python **3.8.10** with TensorFlow **2.13.0**, then run:

```powershell
python -m streamlit run app/dashboard.py --server.port 8501
```

Or double-click `run_rainnet.cmd`. The local dashboard opens at **http://localhost:8501**. Use VS Code Run and Debug with the provided `.vscode/launch.json`.

The repository does **not** bundle the trained model or the original 68 GB dissertation materials. Place the following original files in `models/selected/` on your own computer:
- `rainnet_binary_best_05.h5` (RainNet with dropout 0.5; default)
- `rainnet_binary_baseline_00.h5` (optional)
- `unet_binary.h5` (optional)
- `rainnet_multiclass.h5` (research archive, not used by binary UI)
- `unet_multiclass.h5` (research archive, not used by binary UI)

The original 2024 RainNet architecture source modules are provided under `app/architectures/` for code reproducibility. If the full local historical archive exists, it is preferred.

**Model-selection audit:** five archived dropout checkpoints were compared using the complete 118-image Dataset 4 validation set with normalized 0–1 B13 inputs. The highest observed mean validation DSC was **0.816416** for **dropout 0.5** (mean IoU **0.703124**); the difference from dropout 0.1 is small and not statistically established. See [checkpoint audit](docs/MODEL_VALIDATION.md).

Model loading is version-dependent: the tested environment was TensorFlow 2.13.0 on Python 3.8.10.

`scripts/select_and_hash_models.py` can copy and verify the selected weights **from the original dissertation folder**, if you have that folder. `scripts/consolidate_legacy_experiments.py` performs a copy-and-verify archive; it does not delete originals.

### 38 Indonesian provinces and yellow RainNet overlay

RainNet's segmented pixels are displayed in **yellow**, JMA HRP annotations
remain **magenta**, and overlapping areas are **white**. See the labeled
full-width province image, sortable table of all 38 provinces, and CSV export
in the dashboard. Values are the **proportion of rasterized province pixels
classified as potential convective clouds at the image observation time**,
not probability of rain, rain rate, or a forecast for upcoming hours.

The geographic registration uses JMA r2w native 1501×901 pixels, northwest
(90°E, 30°N), southeast (165°E, 15°S) and 0.05°/pixel. Province polygons
are attributed to [38-province GeoJSON](https://github.com/denyherianto/indonesia-geojson-topojson-maps-with-38-provinces)
published under CC BY 4.0 (attribution in [the methods document](docs/PROVINCE_ATTRIBUTION.md)).
This research overlay is approximate and not an official meteorological warning.

The yellow-image and province-label changes also remove obsolete Streamlit
`use_column_width` rendering parameters and instead use
`use_container_width=True`.

### Live imagery and freshness

JMA dynamic time selectors can sometimes reference yesterday's JPEG file even though a later nominal time appears in the website. The Acquisition Agent tries recent choices, checks **Last-Modified** and verifies that B13+HRP are not stale or inconsistent. It refuses to present a stale pair as real-time data. The app uses Southeast Asia **Large (`r2w`)** matching the historical RainNet image composition and cross-links to the public `r2s` page requested by the researcher.

Auto-refresh is performed **every ten minutes while the Streamlit tab is active**. This is near-real-time satellite monitoring, subject to JMA publication latency and network availability; it is not a guaranteed low-latency sensor feed. The application explicitly reports timestamps.

### Validation and tested environment

- Windows / Python 3.8.10 / TensorFlow 2.13.0 (CPU inference).
- Historical dropout-0.5 RainNet and U-Net binary checkpoints were actually loaded successfully.
- 8 isolated agent integration unit tests passed.
- End-to-end live JMA acquisition + saved RainNet inference + comparison produced a JSON report and image in the local output directory on **12 November 2026**.
- The historical training setup used 1,000 epochs, filter sequence `[4, 8, 16, 32, 64]`, learning rate `1e-4`, and dropout `0.5` for the selected RainNet design.
- GPU/CUDA support on the particular deployment is not claimed. The tested Python environment uses TensorFlow CPU.

### Research cautions

- Cloud-top B13 imagery is not direct precipitation measurement.
- JMA HRP uses multiple infrared channels, whereas original RainNet used B13 imagery to learn a segmentation mask.
- Pixel thresholds for visual-magenta extraction can be affected by JPEG artifacts.
- All quantitative similarity values are *agreement with a derived HRP image*, not independent meteorological validation.
- Meteorological field validation requires gauge, radar, AWS/ARG, and geospatial/temporal alignment.
- Do not use this dashboard alone for evacuation decisions or public flood alerts.

### Data sources and acknowledgements

[JMA Himawari HRP product](https://www.data.jma.go.jp/mscweb/data/himawari/sat_hrp.php?area=r2s) · [JMA Users' Guide](https://www.data.jma.go.jp/mscweb/data/himawari/Users_Guide.pdf) · [RSMC Tokyo HRP documentation](https://www.jma.go.jp/jma/jma-eng/jma-center/nowcasting/).

Please cite the original dissertation when using this research implementation. JMA retains rights and terms over its imagery; imagery is downloaded live from JMA, not republished in this repository.

### Disclaimer on licensing

No license is granted for redistributed source data or original trained checkpoints; ask the rights holder for permission before republishing the original dissertation material. Code provided in this repository is for scholarly reproducibility and educational research pending an explicit repository license.
