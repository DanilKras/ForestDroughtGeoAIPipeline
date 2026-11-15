# Forest Vulnerability to Hotter Droughts: Multi-Scale Cloud-Native GeoAI Pipeline

[![CI Pipeline](https://github.com/DanilKras/ForestDroughtGeoAIPipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/DanilKras/ForestDroughtGeoAIPipeline/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![DVC](https://img.shields.io/badge/data-versioned%20by%20DVC-orange.svg)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, configuration-driven Geospatial Machine Learning pipeline designed to quantify, explain, and project
**forest ecosystem vulnerability to hotter droughts and compound climate extremes** (focusing on Norway spruce, *Picea
abies*).

The architecture bridges the scale gap by uniting **macro-scale regional bioclimatic vulnerability modeling (30 m)**
with **micro-scale individual crown stress diagnostics (1 m)** across the Sopron–Fertő ecosystem (Hungary) and European
spruce biomes.

---

## Interactive Dashboard & Model Attribution

|                Real-Time Spatial Risk & SHAP Attribution                 |                 Global Factor Attribution (SHAP Beeswarm)                  |
|:------------------------------------------------------------------------:|:--------------------------------------------------------------------------:|
| ![Interactive UI Preview](data/03_results/figures/dashboard_preview.png) | ![SHAP Beeswarm Summary](data/03_results/figures/spatial_shap_summary.png) |

> 🔗 **Video
Walkthrough:** [Project Demonstration Video](https://drive.google.com/file/d/1ZfSfTBwKpOTrdrGbKoAzGnVFcybMXxF1/view?usp=sharing)

---

## Experimental Benchmarks & Validation

All macro-scale experiments are evaluated via **5-fold Spatial Block `GroupKFold`** (150 km blocks derived from
empirical terrain/climate semivariograms) with a **strict pre-season temporal cutoff** (`< T-01-01`) to eliminate both
spatial autocorrelation and temporal target leakage:

| Scale / Experiment                               | Predictor Set & Temporal Window                                | Evaluation Protocol                       |                Metric                |
|:-------------------------------------------------|:---------------------------------------------------------------|:------------------------------------------|:------------------------------------:|
| **Macro Full Model (2018 Mega-Drought)**         | 11 features (climate strictly `1995–2017`, `< 2018-01-01`)     | 5-Fold Spatial Block CV (150 km)          |    **ROC-AUC: `0.9802 ± 0.0101`**    |
| **Macro Static Ablation (2018 Target)**          | 7 static terrain/edaphic features (**zero climate variables**) | 5-Fold Spatial Block CV (150 km)          |    **ROC-AUC: `0.8304 ± 0.0627`**    |
| **Macro Temporal Generalization (2020 Drought)** | 11 features (climate strictly `1995–2019`, `< 2020-01-01`)     | 5-Fold Spatial Block CV (150 km)          |    **ROC-AUC: `0.9759 ± 0.0243`**    |
| **Micro Multi-Sensor Crown Fusion**              | UAV LiDAR CHM (1 m) + RGB/NIR (20 cm) + Sentinel-1/2           | Cross-Attention Transformer (640k crowns) | **ROC-AUC: `0.88` / PR-AUC: `0.34`** |

### Key Methodological Highlights

- **Leakage-Free Pre-Season Formulation:** Ground-truth vulnerability labels are stratified via summer Climatic Water
  Deficit ($\text{CWD} = \text{PET} - \text{PPT}$), while bioclimatic predictors (`BIO1`, `BIO4`, `BIO12`, `BIO15`) are
  strictly truncated on December 31 of the preceding year (`< T-01-01`), preserving full 12-month annual cycles for
  operational pre-season forest management.
- **Physical Attribution (TreeSHAP):** Post-cutoff attribution confirms that thermal background (`BIO1`), cumulative
  precipitation (`BIO12`), elevation gradients, and soil water-retention properties (`SoilGrids` clay content & pH)
  govern predicted drought vulnerability.
- **Weak Supervision for Crown Dieback:** Overcomes manual annotation scarcity using allometric LiDAR crown-to-height
  residuals ($Z\text{-score} < -1.65$, 4.8% prevalence), outperforming the random baseline by >7× on PR-AUC.
- **Dual-Engine Spatial Inference:** Combines sub-5ms local 11-band GeoTIFF inference for focal landscapes (Sopron /
  Lake Fertő) with on-demand cloud-native STAC/Zarr streaming and CMIP6 (SSP2-4.5 / SSP5-8.5) 2050 climate projections.

---

## Multi-Scale Predictor Stack

### Macro-Scale Bioclimatic & Edaphic Drivers (30 m)

| Domain              | Predictor Variable                  | Source / Resolution                 | Physiological / Ecological Mechanism                                   |
|:--------------------|:------------------------------------|:------------------------------------|:-----------------------------------------------------------------------|
| **Topography**      | Elevation, Slope, Northness, TWI    | Copernicus GLO-30 (30 m)            | Lapse rates, runoff velocity, solar insolation & moisture accumulation |
| **Land Cover**      | Forest Canopy Mask                  | ESA WorldCover (10 m)               | Restricts inference to forest stands (Class 10)                        |
| **Soil Properties** | Soil pH ($H_2O$) & Clay Content (%) | ISRIC SoilGrids 2.0 (250 m)         | Rhizosphere biochemistry & capillary water-holding capacity            |
| **Macroclimate**    | `BIO1`, `BIO4`, `BIO12`, `BIO15`    | TerraClimate Zarr (`1995` to `T-1`) | Pre-season thermal regime, moisture budget & intra-annual seasonality  |

### Micro-Scale Multi-Sensor Canopy Signals (1 m / Single Crown)

| Sensor Modality    | Resolution           | Extracted Feature Space                                             | Physiological Function                                    |
|:-------------------|:---------------------|:--------------------------------------------------------------------|:----------------------------------------------------------|
| **UAV LiDAR**      | 1 m (Normalized CHM) | Crown radius, 3D height, allometric ratios                          | Physical crown thinning, top dieback & structural loss    |
| **UAV Orthophoto** | 20 cm (RGB / NIR)    | Local spectral texture & canopy openness                            | Visible needle chlorosis, defoliation & branch flagging   |
| **Sentinel-1 SAR** | 10 m / 6–12 days     | Backscatter ($\sigma^0_\text{VV}, \sigma^0_\text{VH}$, VH/VV ratio) | Dielectric permittivity & pre-visual canopy moisture loss |
| **Sentinel-2 MSI** | 10–20 m / 5 days     | NDWI, NDRE, NDVI seasonal profiles                                  | Water absorption bands & red-edge chlorophyll breakdown   |

---

## Quick Start

### Environment & Data Setup

```bash
git clone [https://github.com/DanilKras/ForestDroughtGeoAIPipeline.git](https://github.com/DanilKras/ForestDroughtGeoAIPipeline.git)
cd ForestDroughtGeoAIPipeline

uv venv .venv && source .venv/bin/activate
uv sync
dvc pull
```

### Launch Full Stack via Docker Compose

```bash
docker-compose up --build -d
```

| Service                   | Endpoint                     | Description                                                 |
|:--------------------------|:-----------------------------|:------------------------------------------------------------|
| **Streamlit Dashboard**   | `http://localhost:8501`      | Interactive Leaflet vulnerability map & live SHAP breakdown |
| **FastAPI Inference API** | `http://localhost:8000/docs` | Dual-engine spatial inference OpenAPI documentation         |
| **Airflow Webserver**     | `http://localhost:8081`      | Automated ETL, training & raster generation DAG             |

### Standalone Macro Pipeline Execution

```bash
python -m src.macro.data_extraction           # EU-Forest & GBIF occurrence ingestion
python -m src.macro.data_processing           # 1 km spatial rarefaction (EPSG:3035)
python -m src.macro.generate_drought_labels   # Summer CWD stratification
python -m src.macro.feature_extraction        # Batched STAC DEM/WorldCover, SoilGrids VRT & TerraClimate Zarr
python -m src.macro.model_training            # 150 km Spatial Block CV, Ablation & XGBoost training
python -m src.macro.inference                 # Regional 11-band GeoTIFF & risk raster export
python -m utils.generate_future_risk_maps     # CMIP6 2050 climate horizon projections
```

<details>
<summary><b>Repository Structure</b></summary>

```text
├── api/                # FastAPI dual-engine spatial inference service
├── config/params.yaml  # Single Source of Truth (predictors, bbox, hyperparameters)
├── dags/               # Apache Airflow orchestration DAG (pipeline_dag.py)
├── data/               # DVC-tracked raw, processed, micro_soe & result rasters/figures
├── frontend/           # Streamlit + Leaflet interactive XAI dashboard
├── models/             # Trained XGBoost (macro) & Cross-Attention Transformer (micro) weights
├── notebooks/          # Semivariogram & exploratory spatial analysis
├── src/
│   ├── macro/          # Regional ETL, CWD labeling, Spatial Block CV & GeoTIFF inference
│   └── micro/          # UAV LiDAR CHM & multi-sensor crown fusion modules
├── tests/              # Physical monotonicity audits & config validation
└── utils/              # Config loader, logger & CMIP6 2050 scenario generator
```

</details>

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.