import time
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from pygbif import occurrences as occ
from pygbif import species
from tqdm import tqdm

from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_downloaded_eu_forest(
    file_path: Path,
    target_species: str = "Picea abies",
    max_records: Optional[int] = 80000,
) -> pd.DataFrame:
    """
    Parses the JRC EU-Forest dataset with EPSG:3035 coordinates
    and reprojects occurrences to EPSG:4326 (WGS84).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"EU-Forest file not found: {file_path}")

    logger.info(f"Reading EU-Forest dataset: {file_path}")
    df_raw = pd.read_csv(file_path, low_memory=False)
    df_raw.columns = [c.strip().upper() for c in df_raw.columns]

    species_col = next(
        (c for c in ["SPECIES NAME", "SPECIES", "TAXA"] if c in df_raw.columns), None
    )
    if not species_col:
        raise KeyError(f"Species column not found. Available columns: {df_raw.columns.tolist()}")

    species_mask = (
        df_raw[species_col].astype(str).str.contains(target_species, case=False, na=False)
    )
    df_species = df_raw[species_mask].copy()
    logger.info(f"Found {len(df_species)} records for species '{target_species}'.")

    if df_species.empty:
        raise ValueError(
            f"Species '{target_species}' not found in dataset. "
            f"Available sample taxa: {df_raw[species_col].dropna().unique()[:5]}"
        )

    x_col = "X" if "X" in df_species.columns else "COORD_X"
    y_col = "Y" if "Y" in df_species.columns else "COORD_Y"

    gdf_metric = gpd.GeoDataFrame(
        df_species,
        geometry=gpd.points_from_xy(df_species[x_col], df_species[y_col]),
        crs="EPSG:3035",
    )

    logger.info("Reprojecting coordinates from EPSG:3035 to EPSG:4326 (WGS84)...")
    gdf_wgs84 = gdf_metric.to_crs("EPSG:4326")

    country_col = "COUNTRY" if "COUNTRY" in gdf_wgs84.columns else None

    df_out = pd.DataFrame(
        {
            "obs_id": [f"eu_forest_{i}" for i in range(len(gdf_wgs84))],
            "species": target_species,
            "country": gdf_wgs84[country_col] if country_col else "Europe",
            "lat": gdf_wgs84.geometry.y,
            "lon": gdf_wgs84.geometry.x,
            "year": 2018,
            "source": "EU_FOREST_JRC",
            "coordinate_uncertainty_m": 25.0,
            "target": 1,
        }
    )

    df_out = df_out.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["lat", "lon"])

    if max_records and len(df_out) > max_records:
        df_out = df_out.sample(n=max_records, random_state=42).reset_index(drop=True)

    logger.info(f"Extracted {len(df_out)} unique WGS84 coordinates for '{target_species}'.")
    return df_out


def fetch_gbif_occurrences(
    target_species: str = "Picea abies",
    target_records: int = 10000,
) -> pd.DataFrame:
    """
    Retrieves verified research-grade occurrences from the GBIF API.
    """
    config = load_config()
    start_year = config["temporal"]["start_date"][:4]
    end_year = config["temporal"]["end_date"][:4]

    logger.info(f"Resolving taxonKey for '{target_species}'...")
    match = species.name_backbone(species=target_species)
    taxon_key = match.get("usageKey") or match.get("usage", {}).get("key")

    if not taxon_key:
        logger.error(f"TaxonKey for '{target_species}' was not found in GBIF.")
        return pd.DataFrame()

    limit = 300
    offset = 0
    raw_records = []
    safe_target = min(target_records, 10000)

    search_params = {
        "taxonKey": taxon_key,
        "hasCoordinate": True,
        "hasGeospatialIssue": False,
        "continent": "EUROPE",
        "coordinateUncertaintyInMeters": "0,250",
        "year": f"{start_year},{end_year}",
        "basisOfRecord": [
            "HUMAN_OBSERVATION",
            "OBSERVATION",
            "PRESERVED_SPECIMEN",
            "MACHINE_OBSERVATION",
        ],
        "limit": limit,
    }

    logger.info(f"Requesting {safe_target} records from GBIF...")
    pbar = tqdm(total=safe_target, desc="GBIF Extraction")

    while len(raw_records) < safe_target and offset < 10000:
        response = occ.search(**search_params, offset=offset)
        records = response.get("results", [])

        if not records:
            break

        raw_records.extend(records)
        offset += limit
        pbar.update(len(records))
        time.sleep(0.12)

        if response.get("endOfRecords") or len(raw_records) >= response.get("count", 0):
            break

    pbar.close()
    raw_records = raw_records[:safe_target]

    parsed = [
        {
            "obs_id": f"gbif_{r.get('key')}",
            "species": r.get("species", target_species),
            "country": r.get("country", "Europe"),
            "lat": r.get("decimalLatitude"),
            "lon": r.get("decimalLongitude"),
            "year": r.get("year"),
            "source": "GBIF",
            "coordinate_uncertainty_m": r.get("coordinateUncertaintyInMeters", 50.0),
            "target": 1,
        }
        for r in raw_records
    ]

    df = pd.DataFrame(parsed).dropna(subset=["lat", "lon"]).drop_duplicates(subset=["lat", "lon"])
    logger.info(f"Retrieved {len(df)} unique occurrences from GBIF.")
    return df


def extract_from_local(file_path: Path, target_species: str) -> pd.DataFrame:
    """
    Loads local forest inventory data (Shapefile, GeoPackage, CSV)
    and standardizes coordinates to WGS84.
    """
    logger.info(f"Reading local inventory source: {file_path}")
    if file_path.suffix.lower() in [".gpkg", ".shp", ".geojson"]:
        gdf = gpd.read_file(file_path)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        df = pd.DataFrame(
            {
                "obs_id": [f"local_{i}" for i in range(len(gdf))],
                "species": target_species,
                "country": "Local_Inventory",
                "lat": gdf.geometry.y,
                "lon": gdf.geometry.x,
                "year": gdf.get("year", 2022),
                "source": "LOCAL_INVENTORY",
                "coordinate_uncertainty_m": 10.0,
                "target": 1,
            }
        )
    else:
        df = pd.read_csv(file_path, low_memory=False)
        cols_upper = [c.strip().upper() for c in df.columns]

        if "X" in cols_upper and "Y" in cols_upper:
            return parse_downloaded_eu_forest(file_path, target_species=target_species)

        rename_dict = {
            "latitude": "lat",
            "decimal_latitude": "lat",
            "longitude": "lon",
            "decimal_longitude": "lon",
            "lat": "lat",
            "lon": "lon",
        }
        df = df.rename(
            columns={c: rename_dict[c.lower()] for c in df.columns if c.lower() in rename_dict}
        )

        if "lat" not in df.columns or "lon" not in df.columns:
            raise KeyError(
                f"Coordinates (lat/lon or X/Y) not found in {file_path.name}. "
                f"Detected columns: {df.columns.tolist()}"
            )

        if "obs_id" not in df.columns:
            df["obs_id"] = [f"local_{i}" for i in range(len(df))]
        df["species"] = target_species
        df["source"] = "LOCAL_INVENTORY"
        df["target"] = 1

    return df.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["lat", "lon"])


def extract_all_raw_data() -> pd.DataFrame:
    """
    Main extraction pipeline routing input sources based on configuration.
    """
    config = load_config()
    source_type = config.get("data_source", {}).get("type", "eu_forest").lower()
    target_species = config["project"].get("target_species", "Picea abies")
    output_path = Path(config["paths"]["raw_data"])
    max_records = config.get("data_source", {}).get("target_records", 25000)

    logger.info(
        f"Starting raw occurrence extraction: source='{source_type}', species='{target_species}'"
    )

    raw_dir = output_path.parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    local_path_str = config.get("data_source", {}).get("local_path")
    local_csv_path = Path(local_path_str) if local_path_str else (raw_dir / "EUForestspecies.csv")

    if not local_csv_path.exists():
        candidates = (
            list(raw_dir.glob("*EUForestspecies*.csv"))
            + list(raw_dir.glob("*EU*Forest*.csv"))
            + list(raw_dir.glob("*species*.csv"))
            + list(raw_dir.glob("*occurrence*.csv"))
        )
        if candidates:
            local_csv_path = candidates[0]

    if source_type in ["eu_forest", "local"]:
        if local_csv_path.exists():
            df = parse_downloaded_eu_forest(
                file_path=local_csv_path,
                target_species=target_species,
                max_records=max_records,
            )
        else:
            logger.warning(
                f"Inventory file not found at {local_csv_path}. Falling back to GBIF API..."
            )
            df = fetch_gbif_occurrences(target_species=target_species, target_records=max_records)

    elif source_type == "gbif":
        df = fetch_gbif_occurrences(target_species=target_species, target_records=max_records)

    else:
        logger.warning(f"Unknown data source '{source_type}'. Falling back to GBIF...")
        df = fetch_gbif_occurrences(target_species=target_species, target_records=max_records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, engine="pyarrow", compression="snappy")
    else:
        df.to_csv(output_path, index=False)

    logger.info(f"Extraction completed. Saved {len(df)} records to: {output_path}")
    return df


if __name__ == "__main__":
    extract_all_raw_data()
