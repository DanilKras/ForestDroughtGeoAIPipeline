import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import rasterio
import requests
import streamlit as st
from folium.raster_layers import ImageOverlay
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.prepared import prep
from streamlit_folium import st_folium

from utils.config_loader import load_config

config = load_config()
REGION_NAME = config["project"].get("region_name", "Study Ecosystem")
BBOX = config["spatial"]["bbox"]
BBOX_CONFIG = {
    "min_lon": BBOX[0],
    "min_lat": BBOX[1],
    "max_lon": BBOX[2],
    "max_lat": BBOX[3],
}

DEFAULT_CENTER = [
    (BBOX_CONFIG["min_lat"] + BBOX_CONFIG["max_lat"]) / 2,
    (BBOX_CONFIG["min_lon"] + BBOX_CONFIG["max_lon"]) / 2,
]

st.set_page_config(
    page_title="Forest Vulnerability GeoAI",
    page_icon="🌲",
    layout="wide",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"], .stMarkdown, .stText {
            font-family: 'Inter', sans-serif !important;
        }
        .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }
        header { visibility: hidden !important; }

        div[data-testid="stMetric"] {
            background-color: #1e293b;
            border: 1px solid #334155;
            padding: 10px 14px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        div[data-testid="stMetricValue"] { 
            font-size: 1.3rem !important; 
            font-weight: 700 !important; 
            color: #f8fafc !important; 
        }
        div[data-testid="stMetricLabel"] { 
            font-size: 0.75rem !important; 
            font-weight: 500 !important; 
            color: #94a3b8 !important; 
            text-transform: uppercase; 
            letter-spacing: 0.5px; 
        }

        .risk-banner {
            padding: 14px 16px; 
            border-radius: 6px; 
            font-weight: 600;
            font-size: 1.1rem;
            text-align: center; 
            margin: 8px 0 16px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            letter-spacing: 0.3px;
        }
        .risk-high { background: linear-gradient(135deg, rgba(231,76,60,0.15), rgba(231,76,60,0.05)); color: #ff6b6b; border-left: 5px solid #e74c3c; }
        .risk-medium { background: linear-gradient(135deg, rgba(241,196,15,0.15), rgba(241,196,15,0.05)); color: #fcd34d; border-left: 5px solid #f1c40f; }
        .risk-low { background: linear-gradient(135deg, rgba(46,204,113,0.15), rgba(46,204,113,0.05)); color: #4ade80; border-left: 5px solid #2ecc71; }
        .risk-nodata { background: linear-gradient(135deg, rgba(148,163,184,0.15), rgba(148,163,184,0.05)); color: #94a3b8; border-left: 5px solid #64748b; }

        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

BACKEND_HOST = os.getenv("BACKEND_HOST", "http://localhost:8000")
PREDICT_URL_LOCAL = f"{BACKEND_HOST}/predict/local"
PREDICT_URL_CLOUD = f"{BACKEND_HOST}/predict/cloud"

BOUNDARY_PATH = "data/00_external/focal_forest_boundary.geojson"

if "map_center" not in st.session_state:
    st.session_state.map_center = DEFAULT_CENTER
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 12
if "target_lat" not in st.session_state:
    st.session_state.target_lat = DEFAULT_CENTER[0]
if "target_lon" not in st.session_state:
    st.session_state.target_lon = DEFAULT_CENTER[1]
if "inference_results" not in st.session_state:
    st.session_state.inference_results = None
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "map_bounds_fitted" not in st.session_state:
    st.session_state.map_bounds_fitted = False


@st.cache_resource(show_spinner=False)
def load_spatial_boundary(geojson_path: str):
    if not os.path.exists(geojson_path):
        return None, None, None
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_geom = unary_union([shape(f["geometry"]) for f in data["features"]])
    prepared = prep(raw_geom)
    min_lon, min_lat, max_lon, max_lat = raw_geom.bounds
    leaflet_bounds = [[min_lat, min_lon], [max_lat, max_lon]]
    return data, prepared, leaflet_bounds


@st.cache_data(show_spinner=False)
def get_risk_raster_overlay(tif_path: str, is_delta: bool = False):
    if not os.path.exists(tif_path):
        return None, None
    try:
        with rasterio.open(tif_path) as src:
            bounds = [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]
            data = src.read(
                1,
                out_shape=(max(1, src.height // 4), max(1, src.width // 4)),
                resampling=Resampling.bilinear,
            )

            rgba = np.zeros((*data.shape, 4), dtype=np.uint8)

            if is_delta:
                mask = (data >= -1.0) & (data <= 1.0) & (data != -9999.0)
                norm = plt.Normalize(vmin=-0.4, vmax=0.4)
                cmap = plt.colormaps["bwr"]
            else:
                mask = (data >= 0.0) & (data <= 1.0) & (data != -9999.0)
                norm = plt.Normalize(vmin=0.0, vmax=1.0)
                cmap = plt.colormaps["YlOrRd"]

            colored = (cmap(norm(data)) * 255).astype(np.uint8)
            rgba[mask] = colored[mask]
            rgba[~mask] = [0, 0, 0, 0]

            return rgba, bounds
    except Exception:
        return None, None


def export_history_to_gpkg(history_records: list) -> bytes:
    df = pd.DataFrame(history_records)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "forest_inspections.gpkg"
        gdf.to_file(tmp_file, layer="canopy_vulnerability_inspections", driver="GPKG")
        with open(tmp_file, "rb") as f:
            return f.read()


def plot_risk_gauge(probability: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 32, "family": "Inter, sans-serif"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                "bar": {"color": "#3b82f6", "thickness": 0.25},
                "steps": [
                    {"range": [0, 33], "color": "rgba(46, 204, 113, 0.45)"},
                    {"range": [33, 66], "color": "rgba(241, 196, 15, 0.45)"},
                    {"range": [66, 100], "color": "rgba(231, 76, 60, 0.45)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.8,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=190,
        margin=dict(l=10, r=10, t=10, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
    )
    return fig


def plot_shap_bar(shap_dict: dict):
    if not shap_dict:
        return None
    friendly_names = {
        "bio1_mean_temp": "Mean Temp (BIO1)",
        "bio4_temp_seasonality": "Temp Seasonality (BIO4)",
        "bio12_annual_precip": "Annual Precip (BIO12)",
        "bio15_precip_seasonality": "Precip Seasonality (BIO15)",
        "elevation": "Elevation (DEM)",
        "slope_deg": "Slope (°)",
        "northness_aspect": "Aspect (Northness)",
        "twi_moisture_index": "Moisture (TWI)",
        "landcover_class": "Land Cover",
        "soil_ph": "Soil pH",
        "soil_clay_percent": "Soil Clay (%)",
    }
    df = pd.DataFrame(
        [{"Feature": friendly_names.get(k, k), "SHAP Value": v} for k, v in shap_dict.items()]
    )
    df["Abs_SHAP"] = df["SHAP Value"].abs()
    df = df.sort_values(by="Abs_SHAP", ascending=True)
    df["Impact"] = df["SHAP Value"].apply(
        lambda x: "Increases Vulnerability" if x > 0 else "Decreases Vulnerability"
    )

    fig = px.bar(
        df,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        color="Impact",
        color_discrete_map={
            "Increases Vulnerability": "#ef4444",
            "Decreases Vulnerability": "#3b82f6",
        },
    )
    fig.update_layout(
        height=260,
        margin=dict(l=5, r=5, t=5, b=5),
        showlegend=False,
        xaxis_title="Attribution (Log-odds)",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.15)", zerolinecolor="#64748b")
    fig.update_yaxes(showgrid=False)
    return fig


st.title("Forest GeoAI: Drought Vulnerability")
st.caption(
    "Cloud-Native Multi-Scale Geospatial Inference (Copernicus DEM, WorldCover, TerraClimate Zarr, SoilGrids)."
)

raw_geojson, prepared_boundary, boundary_bounds = load_spatial_boundary(BOUNDARY_PATH)

top_col1, top_col2 = st.columns([2, 1], gap="medium")

with top_col2:
    st.subheader("Controls & Inspection")
    st.write(
        f"**Lat:** `{st.session_state.target_lat:.5f}` | **Lon:** `{st.session_state.target_lon:.5f}`"
    )

    st.markdown("##### Climate Projection Horizon")
    selected_scenario = st.selectbox(
        "Vulnerability Envelope:",
        options=[
            "Baseline (Current Climate)",
            "SSP2-4.5 (2050 Horizon)",
            "SSP5-8.5 (2050 Extreme)",
            "Delta Shift: SSP5-8.5 vs Baseline",
        ],
        index=0,
    )

    st.markdown("##### Inference Engine")
    engine_mode = st.radio(
        "Backend Routing:",
        options=[f"Local Precomputed ({REGION_NAME})", "Cloud Streaming (Global STAC/Zarr)"],
        captions=[
            "Latency: <5ms. Uses local 11-band GeoTIFF stack.",
            "Latency: ~15-30s. On-demand global planetary assets.",
        ],
        index=0,
    )

    is_local_mode = "Local" in engine_mode
    in_bounds = (
        BBOX_CONFIG["min_lat"] <= st.session_state.target_lat <= BBOX_CONFIG["max_lat"]
        and BBOX_CONFIG["min_lon"] <= st.session_state.target_lon <= BBOX_CONFIG["max_lon"]
    )

    if is_local_mode and not in_bounds:
        st.warning(
            f"Point is outside focal BBOX [{BBOX_CONFIG['min_lon']}, {BBOX_CONFIG['min_lat']}, {BBOX_CONFIG['max_lon']}, {BBOX_CONFIG['max_lat']}]. Switch to 'Cloud Streaming' for global inference."
        )
        st.button("Run Spatial Inference", disabled=True, use_container_width=True)
    else:
        if st.button("Run Spatial Inference", type="primary", use_container_width=True):
            target_endpoint = PREDICT_URL_LOCAL if is_local_mode else PREDICT_URL_CLOUD
            call_timeout = 10 if is_local_mode else 120
            loading_msg = (
                "Sampling local 11-band raster stack..."
                if is_local_mode
                else "Streaming STAC DEM, WorldCover & Zarr climate..."
            )

            with st.spinner(loading_msg):
                try:
                    resp = requests.post(
                        target_endpoint,
                        json={
                            "lat": st.session_state.target_lat,
                            "lon": st.session_state.target_lon,
                        },
                        timeout=call_timeout,
                    )
                    if resp.status_code == 422:
                        st.error(f"Validation Error: {resp.json().get('detail')}")
                    else:
                        resp.raise_for_status()
                        res = resp.json()
                        st.session_state.inference_results = res

                        if res.get("is_forest", False):
                            feats = res.get("extracted_features") or {}
                            new_entry = {
                                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                                "lat": st.session_state.target_lat,
                                "lon": st.session_state.target_lon,
                                "risk_probability": res["risk_probability"],
                                "risk_level": res["risk_level"],
                                "elevation_m": feats.get("elevation_m", 0.0),
                                "bio1_temp_c": feats.get("bio1_temp_c", 0.0),
                            }
                            if not any(
                                abs(p["lat"] - new_entry["lat"]) < 1e-5
                                and abs(p["lon"] - new_entry["lon"]) < 1e-5
                                for p in st.session_state.prediction_history
                            ):
                                st.session_state.prediction_history.append(new_entry)
                except requests.exceptions.RequestException as e:
                    st.error(f"Backend API error: {e}")

with top_col1:
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        min_zoom=3,
        max_zoom=17,
        tiles=None,
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Satellite",
        name="Satellite (Esri)",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Topography (OSM)",
        overlay=False,
        control=True,
    ).add_to(m)

    scenario_map_files = {
        "Baseline (Current Climate)": (
            "data/03_results/maps/regional_vulnerability_map.tif",
            False,
        ),
        "SSP2-4.5 (2050 Horizon)": (
            "data/03_results/maps/regional_vulnerability_ssp245_2050.tif",
            False,
        ),
        "SSP5-8.5 (2050 Extreme)": (
            "data/03_results/maps/regional_vulnerability_ssp585_2050.tif",
            False,
        ),
        "Delta Shift: SSP5-8.5 vs Baseline": (
            "data/03_results/maps/regional_vulnerability_delta_ssp585_2050.tif",
            True,
        ),
    }

    tif_target, is_delta_mode = scenario_map_files.get(
        selected_scenario, ("data/03_results/maps/regional_vulnerability_map.tif", False)
    )
    risk_img, r_bounds = get_risk_raster_overlay(tif_target, is_delta=is_delta_mode)

    if risk_img is not None:
        ImageOverlay(
            image=risk_img,
            bounds=r_bounds,
            opacity=0.65,
            name=f"{selected_scenario} Layer",
            overlay=True,
            interactive=False,
        ).add_to(m)

    if raw_geojson:
        folium.GeoJson(
            data=raw_geojson,
            name="Ecosystem Boundary",
            style_function=lambda f: {
                "fillColor": "#3b82f6",
                "fillOpacity": 0.06,
                "color": "#60a5fa",
                "weight": 2.2,
                "dashArray": "6, 6",
            },
        ).add_to(m)

        if not st.session_state.map_bounds_fitted and boundary_bounds:
            m.fit_bounds(boundary_bounds, padding=(15, 15))
            st.session_state.map_bounds_fitted = True

    for pt in st.session_state.prediction_history:
        p_val = pt["risk_probability"]
        color = "#ef4444" if p_val >= 0.66 else "#f59e0b" if p_val >= 0.33 else "#10b981"
        folium.CircleMarker(
            location=[pt["lat"], pt["lon"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=f"Vulnerability: {p_val * 100:.1f}%",
        ).add_to(m)

    folium.Marker(
        [st.session_state.target_lat, st.session_state.target_lon],
        popup=f"Target: {st.session_state.target_lat:.4f}, {st.session_state.target_lon:.4f}",
        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
    ).add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    map_data = st_folium(
        m,
        height=430,
        use_container_width=True,
        returned_objects=["last_clicked", "center", "zoom"],
        key="global_forest_map",
    )

    if map_data and map_data.get("last_clicked"):
        c_lat = map_data["last_clicked"]["lat"]
        c_lon = map_data["last_clicked"]["lng"]
        if (
            abs(c_lat - st.session_state.target_lat) > 1e-5
            or abs(c_lon - st.session_state.target_lon) > 1e-5
        ):
            st.session_state.target_lat = c_lat
            st.session_state.target_lon = c_lon
            if map_data.get("center"):
                st.session_state.map_center = [
                    map_data["center"]["lat"],
                    map_data["center"]["lng"],
                ]
            if map_data.get("zoom"):
                st.session_state.map_zoom = map_data["zoom"]
            st.rerun()

if st.session_state.inference_results:
    st.markdown("---")
    res = st.session_state.inference_results
    prob = float(res.get("risk_probability", 0.0))
    is_forest = res.get("is_forest", True)

    b_col1, b_col2, b_col3 = st.columns([1.0, 1.8, 1.6], gap="medium")

    with b_col1:
        st.markdown("### Canopy Vulnerability")
        if not is_forest:
            st.markdown(
                '<div class="risk-banner risk-nodata">Non-Forest / Excluded Area</div>',
                unsafe_allow_html=True,
            )
            st.caption("Point falls outside forest canopy mask.")
        else:
            st.plotly_chart(
                plot_risk_gauge(prob),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            if prob >= 0.66:
                st.markdown(
                    '<div class="risk-banner risk-high">High Vulnerability Stand</div>',
                    unsafe_allow_html=True,
                )
            elif prob >= 0.33:
                st.markdown(
                    '<div class="risk-banner risk-medium">Moderate Stress Zone</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="risk-banner risk-low">Resilient Canopy Stand</div>',
                    unsafe_allow_html=True,
                )

    with b_col2:
        st.markdown("### Attribution Drivers (SHAP)")
        shap_chart = plot_shap_bar(res.get("shap_values", {}))
        if shap_chart:
            st.plotly_chart(
                shap_chart,
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("Attribution unavailable for non-forest locations.")

    with b_col3:
        st.markdown("### Predictors (11 Variables)")
        feats = res.get("extracted_features")
        if is_forest and feats:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption("Topography")
                st.metric("Elevation", f"{feats.get('elevation_m', 0):.0f} m")
                st.metric("Slope", f"{feats.get('slope_deg', 0):.1f}°")
                st.metric("Northness", f"{feats.get('northness_aspect', 0):.2f}")
                st.metric("TWI", f"{feats.get('twi_moisture_index', 0):.1f}")
            with c2:
                st.caption("Climate (30y)")
                st.metric("Mean Temp", f"{feats.get('bio1_temp_c', 0):.1f} °C")
                st.metric("Temp Seas.", f"{feats.get('bio4_seasonality', 0):.0f}")
                st.metric("Ann. Precip", f"{feats.get('bio12_precip_mm', 0):.0f} mm")
                st.metric("Precip Seas.", f"{feats.get('bio15_precip_seas', 0):.1f}")
            with c3:
                st.caption("Soil & Cover")
                st.metric("Land Cover", "Forest" if feats.get("landcover_class") == 10 else "Other")
                st.metric("Soil pH", f"{feats.get('soil_ph', 0):.2f}")
                st.metric("Clay Content", f"{feats.get('soil_clay_percent', 0):.1f} %")
        else:
            st.caption("No biological predictors extracted.")

if st.session_state.prediction_history:
    st.markdown("---")
    st.subheader("Field Inspection History & Export")
    hist_df = pd.DataFrame(st.session_state.prediction_history)

    st.dataframe(
        hist_df[
            [
                "timestamp",
                "lat",
                "lon",
                "risk_probability",
                "risk_level",
                "elevation_m",
                "bio1_temp_c",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    d_col1, d_col2 = st.columns([1, 4])
    with d_col1:
        gpkg_data = export_history_to_gpkg(st.session_state.prediction_history)
        st.download_button(
            label="Download GPKG Inspection Layer",
            data=gpkg_data,
            file_name=f"forest_inspections_{datetime.now().strftime('%Y%m%d_%H%M')}.gpkg",
            mime="application/geopackage+sqlite3",
            type="primary",
            use_container_width=True,
        )
    with d_col2:
        if st.button("Clear Points"):
            st.session_state.prediction_history = []
            st.rerun()
