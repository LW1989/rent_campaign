"""
Rent Campaign Analysis Functions

This module contains all the core functions for processing census and geographic data
to identify areas suitable for rental campaigns. Functions include data loading,
spatial analysis, threshold-based filtering, and export utilities.

The main workflow processes:
1. Census data on heating types and energy sources
2. Renter demographic data  
3. Geographic boundaries and districts
4. Address data via Overpass API

Key functions:
- load_geojson_folder: Load multiple GeoJSON files
- get_rent_campaign_df: Main analysis pipeline
- filter_squares_invoting_distirct: Spatial filtering by districts
- get_all_addresses: Address extraction via Overpass API
- export functions: Save results to various formats
"""



import os
import re
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from loguru import logger, logger as _default_logger
from shapely.geometry import Point, Polygon, MultiPolygon
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from typing import Dict, Iterable, Tuple, Literal, List, Optional, Union
from pathlib import Path
import xarray as xr
from scipy.ndimage import generic_filter


def extract_district_name(name_string: str) -> str:
    """
    Extract district name from various input formats.
    
    This function handles multiple naming conventions:
    - Oberhausen format: "BTW 2017 | Stimmbezirk 0203 | Linke 14.35 %" → "Stimmbezirk 0203"
    - Heide/Neumünster format: "39.0" → "39.0"
    
    Parameters
    ----------
    name_string : str
        Name string in any supported format
        
    Returns
    -------
    str
        Extracted district name, stripped of whitespace
        
    Examples
    --------
    >>> extract_district_name("BTW 2017 | Stimmbezirk 0203 | Linke 14.35 %")
    'Stimmbezirk 0203'
    >>> extract_district_name("39.0")
    '39.0'
    """
    if "|" in name_string:
        # Oberhausen format: extract middle part between pipes
        parts = name_string.split("|")
        if len(parts) >= 2:
            return parts[1].strip()
        else:
            # Fallback if split doesn't produce expected parts
            return name_string.strip()
    else:
        # Heide/Neumünster format: use full name as-is
        return name_string.strip()


def load_geojson_folder(folder_path: Union[str, Path]) -> Dict[str, gpd.GeoDataFrame]:
    """
    Load all GeoJSON files from a folder into a dictionary of GeoDataFrames.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing GeoJSON files.

    Returns
    -------
    dict
        Dictionary where keys are filenames (without extension)
        and values are corresponding GeoDataFrames.
    """
    geojson_dict = {}

    logger.info(f"Scanning folder: {folder_path}")

    if not os.path.isdir(folder_path):
        logger.error(f"Provided path is not a folder: {folder_path}")
        return geojson_dict
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".geojson"):
            file_path = os.path.join(folder_path, filename)
            try:
                logger.debug(f"Loading file: {filename}")
                gdf = gpd.read_file(file_path)
                key = os.path.splitext(filename)[0]
                geojson_dict[key] = gdf
                logger.success(f"Loaded '{filename}' with {len(gdf)} records.")
            except Exception as e:
                logger.exception(f"Error loading {filename}: {e}")
    
    logger.info(f"Finished loading {len(geojson_dict)} GeoJSON files.")

    return geojson_dict



def plot_histogram(df: pd.DataFrame, column: str, bins: int = 30, title: Optional[str] = None):
    """
    Plot a histogram for a numeric column in a (Geo)DataFrame.

    - Coerces to numeric and drops NaNs.
    - Returns the Matplotlib Axes for further customization.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Column name to plot.
    bins : int
        Number of bins.
    title : Optional[str]
        Optional plot title.
    """
    s = pd.to_numeric(df[column], errors="coerce").dropna()

    fig, ax = plt.subplots()
    ax.hist(s, bins=bins)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Histogram of {column} (n={len(s)})")
    return ax

def import_gdf(path: Union[str, Path], crs: str) -> gpd.GeoDataFrame:
    """
    Load a geospatial file and reproject it to the specified CRS.
    
    Parameters
    ----------
    path : str or Path
        Path to the geospatial file (shapefile, GeoJSON, etc.)
    crs : str
        Target coordinate reference system (e.g., "EPSG:3035")
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame reprojected to the target CRS
    """
    # Load the shapefile
    logger.debug(f"Start loading file at {path}")

    gdf = gpd.read_file(path)

    # Verify the current CRS 
    logger.info(f" Loaded file with CRS: {gdf.crs}")

    # Re-project to desired CRS (example: EPSG:3035)
    krs_gdf = gdf.to_crs(crs)

    # Quick confirmation
    logger.info(f"Re-projected file to CRS: {krs_gdf.crs}")

    return krs_gdf

def filter_overlapping(gdf_1, gdf_2):
    """
    Return rows of gdf_2 that spatially intersect the (single) geometry in gdf_1.

    Parameters
    ----------
    gdf_1 : geopandas.GeoDataFrame
        Must contain exactly one row (the reference geometry).
    gdf_2 : geopandas.GeoDataFrame
        Features to test against gdf_1's geometry.

    Returns
    -------
    geopandas.GeoDataFrame
        Subset of gdf_2 that intersects gdf_1's geometry.
        Returned in gdf_2's original CRS.
    """
    if len(gdf_1) != 1:
        raise ValueError("gdf_1 must contain exactly one row.")
    if gdf_1.crs is None or gdf_2.crs is None:
        raise ValueError("Both GeoDataFrames must have a valid CRS set.")

    # Align CRS
    g2 = gdf_2.to_crs(gdf_1.crs) if gdf_2.crs != gdf_1.crs else gdf_2

    # Get the reference geometry (fix if invalid)
    geom1 = gdf_1.geometry.iloc[0]
    if geom1 is None or geom1.is_empty:
        return g2.iloc[0:0].copy()
    if not geom1.is_valid:
        geom1 = geom1.buffer(0)

    # Keep features that intersect
    mask = g2.intersects(geom1)
    out = g2.loc[mask].copy()

    # Return in original CRS of gdf_2
    if out.crs != gdf_2.crs:
        out = out.to_crs(gdf_2.crs)

    return out

def calc_total(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Calculate row-wise totals for specified columns and add as 'total' column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    cols : List[str]
        List of column names to sum across rows
        
    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with additional 'total' column
    """
    df_copy = df.copy()
    df_copy["total"] = df_copy[cols].sum(axis=1)
    return df_copy

import geopandas as gpd

def squares_in_municipality(
    mun: gpd.GeoDataFrame,
    heating_df: gpd.GeoDataFrame,
    *,
    mun_index=None,
    mun_filter: Optional[dict] = None,
    mode: str = "intersects",       # "intersects" | "within" | "covered_by"
    min_overlap_ratio: float = 0.0  # fraction of square area that must overlap (>0 keeps all intersects)
) -> gpd.GeoDataFrame:
    """
    Select squares from `heating_df` that satisfy a spatial relation to one municipality polygon.

    mode="intersects" counts partial overlaps (and boundary touches).
    min_overlap_ratio > 0 filters out tiny slivers by requiring
    (area(square ∩ muni) / area(square)) >= min_overlap_ratio.
    """
    if mun.empty:
        raise ValueError("`mun` is empty.")
    if heating_df.empty:
        return heating_df.iloc[0:0].copy()

    # pick municipality geometry
    if mun_filter is not None:
        col, val = next(iter(mun_filter.items()))
        muni_geom = mun.loc[mun[col] == val].geometry.iloc[0]
    else:
        if mun_index is None:
            raise ValueError("Provide mun_index or mun_filter.")
        try:
            muni_geom = mun.loc[mun_index].geometry
        except KeyError:
            muni_geom = mun.iloc[int(mun_index)].geometry

    # CRS align
    if mun.crs is None or heating_df.crs is None:
        raise ValueError("Both GeoDataFrames must have a CRS.")
    if heating_df.crs != mun.crs:
        heating_df = heating_df.to_crs(mun.crs)

    # fast candidate filter via sindex
    try:
        idx = heating_df.sindex.query(muni_geom, predicate="intersects")
        candidates = heating_df.iloc[idx].copy()
    except Exception:
        candidates = heating_df.copy()

    # primary predicate
    if mode not in {"intersects", "within", "covered_by"}:
        raise ValueError("mode must be one of: 'intersects', 'within', 'covered_by'.")

    mask = getattr(candidates, mode)(muni_geom)
    result = candidates.loc[mask].copy()

    # optional: require minimum overlapping fraction of each square
    if min_overlap_ratio > 0:
        # compute intersection areas (2D; Z is ignored)
        inter = result.geometry.intersection(muni_geom)
        overlap_frac = inter.area / result.geometry.area
        result = result.loc[overlap_frac >= min_overlap_ratio].copy()

    return result.reset_index(drop=True)



def _safe_union_all(gdf: gpd.GeoDataFrame, logger=_default_logger):
    """GeoPandas ≥0.14: union_all(); older: unary_union."""
    try:
        logger.debug("Using GeoDataFrame.union_all() to dissolve geometries.")
        return gdf.union_all()
    except AttributeError:
        logger.debug("GeoPandas.union_all not available, falling back to unary_union (deprecated).")
        return gdf.unary_union

def _poly_string_from_polygon(poly: Polygon) -> str:
    """Overpass poly format is 'lat lon lat lon ...' (note the order!)."""
    coords = list(poly.exterior.coords)
    return " ".join(f"{lat} {lon}" for lon, lat in coords)

def _polygonize(geom, logger=_default_logger):
    """Return list[Polygon] from Polygon/MultiPolygon/GeometryCollection."""
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    logger.debug("Cleaning non-(Multi)Polygon geometry with buffer(0).")
    cleaned = geom.buffer(0)
    if isinstance(cleaned, Polygon):
        return [cleaned]
    if isinstance(cleaned, MultiPolygon):
        return list(cleaned.geoms)
    raise ValueError("Could not form polygon(s) from the input geometry.")

def _request_overpass(query: str, *, timeout=180, retries=4, logger=_default_logger):
    
    OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
    ]
    
    last_err = None
    for attempt in range(retries):
        for base in OVERPASS_URLS:
            try:
                logger.debug(f"POST Overpass attempt={attempt+1}/{retries} endpoint={base}")
                resp = requests.post(base, data={"data": query}, timeout=timeout)
                if resp.status_code == 429:
                    logger.warning(f"Overpass 429 (rate limit) at {base}.")
                    raise requests.HTTPError("429 Too Many Requests", response=resp)
                resp.raise_for_status()
                logger.success(f"Overpass request OK via {base} (status {resp.status_code}).")
                return resp.json()
            except Exception as e:
                last_err = e
                logger.warning(f"Overpass request failed on {base}: {e}")
        sleep_s = 2 ** attempt
        logger.info(f"Backing off {sleep_s}s before retrying all endpoints…")
        time.sleep(sleep_s)
    logger.error("All Overpass endpoints failed after retries.")
    raise last_err

def _elements_to_gdf(elements: list, logger=_default_logger) -> gpd.GeoDataFrame:
    rows = []
    for el in elements:
        tags = el.get("tags", {}) or {}
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
            if lat is None or lon is None:
                continue
            geom = Point(lon, lat)
        else:
            c = el.get("center")
            if not c:
                continue
            geom = Point(c["lon"], c["lat"])
        rows.append({
            "housenumber": tags.get("addr:housenumber"),
            "street":      tags.get("addr:street"),
            "postcode":    tags.get("addr:postcode"),
            "city":        tags.get("addr:city"),
            "name":        tags.get("name"),
            "geometry":    geom
        })
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    logger.debug(f"Built GeoDataFrame from Overpass elements: {len(gdf)} rows.")
    return gdf

def addresses_in_squares_overpass_unified(
    squares_gdf: gpd.GeoDataFrame,
    *,
    logger=_default_logger
) -> gpd.GeoDataFrame:
    """
    One-shot Overpass query for all squares:
    - Dissolves squares to a (Multi)Polygon in EPSG:4326
    - Requests nodes/ways/relations with addr:housenumber
    - Returns points joined back to the original squares
    """
    t0 = time.perf_counter()
    logger.info("Starting addresses_in_squares_overpass_unified")

    if squares_gdf.empty:
        logger.error("squares_gdf is empty.")
        return squares_gdf.iloc[0:0].copy()
    if squares_gdf.crs is None:
        raise ValueError("squares_gdf must have a CRS.")

    logger.debug(f"Input CRS: {squares_gdf.crs}. Reprojecting to EPSG:4326 for Overpass.")
    sq_wgs = squares_gdf.to_crs(4326)

    logger.debug("Dissolving squares into a single geometry…")
    geom = _safe_union_all(sq_wgs, logger=logger)
    polygons = _polygonize(geom, logger=logger)
    logger.info(f"Dissolve produced {len(polygons)} polygon(s) for querying.")

    # Build a single Overpass query with multiple (poly:"...") clauses
    logger.debug("Composing Overpass query with poly filters…")
    poly_clauses = "\n".join(
        f'  node["addr:housenumber"](poly:"{_poly_string_from_polygon(p)}");\n'
        f'  way["addr:housenumber"](poly:"{_poly_string_from_polygon(p)}");\n'
        f'  relation["addr:housenumber"](poly:"{_poly_string_from_polygon(p)}");'
        for p in polygons
    )
    query = f"""
    [out:json][timeout:180];
    (
    {poly_clauses}
    );
    out tags center;
    """.strip()
    logger.debug(f"Overpass query length: {len(query)} chars")

    # Request
    data = _request_overpass(query, timeout=180, retries=4, logger=logger)
    elements = data.get("elements", [])
    logger.info(f"Overpass returned {len(elements)} elements.")

    if not elements:
        cols = ["housenumber", "street", "postcode", "city", "name", "geometry"]
        empty = gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=4326)
        logger.warning("No elements found in Overpass response. Returning empty GeoDataFrame.")
        return empty

    addr_points = _elements_to_gdf(elements, logger=logger)

    # Join addresses back to the input squares (original CRS)
    logger.debug("Reprojecting address points back to the input CRS and spatial-joining to squares…")
    addr_points_proj = addr_points.to_crs(squares_gdf.crs)
    before_join = len(addr_points_proj)
    result = gpd.sjoin(addr_points_proj, squares_gdf, predicate="within", how="inner")
    after_join = len(result)
    logger.info(f"Spatial join kept {after_join}/{before_join} points within squares.")

    result = result.drop(columns=[c for c in ["index_right"] if c in result.columns]).reset_index(drop=True)

    dt = time.perf_counter() - t0
    logger.success(f"addresses_in_squares_overpass_unified done in {dt:.2f}s. Rows: {len(result)}")

    if "name" in result.columns:
        result.drop(columns=["name"], inplace=True)
    return result

def add_umap_tooltip(gdf):
    def yesno(v): return "✅" if bool(v) else "❌"
    gdf = gdf.copy()
    gdf["tooltip"] = (
        "<b>" + gdf.get("district_name", "").astype(str) + "</b><br>"
        "<ul>"
        + "<li>Central heating: " + gdf["central_heating_flag"].map(yesno) + "</li>"
        + "<li>Fossil heating: "   + gdf["fossil_heating_flag"].map(yesno)   + "</li>"
        + "<li>Fernwärme: "        + gdf["fernwaerme_flag"].map(yesno)       + "</li>"
        + "<li>Renter: "           + gdf["renter_flag"].map(yesno)           + "</li>"
        + "</ul>"
    )
    return gdf


def get_renter_share(renter_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert ownership percentage to renter share percentage.
    
    Parameters
    ----------
    renter_df : gpd.GeoDataFrame
        DataFrame with 'Eigentuemerquote' column (ownership percentage)
        
    Returns
    -------
    gpd.GeoDataFrame
        DataFrame with 'renter_share' column and geometry
    """
    renter_df["Eigentuemerquote"] = (100 - renter_df["Eigentuemerquote"]) / 100
    renter_df.rename(columns={"Eigentuemerquote": "renter_share"}, inplace=True)
    renter_df["renter_share"] = renter_df["renter_share"].round(2)

    # Include GITTER_ID_100m if it exists, otherwise just return geometry and calculated column
    if "GITTER_ID_100m" in renter_df.columns:
        return renter_df[["geometry", "GITTER_ID_100m", "renter_share"]]
    else:
        return renter_df[["geometry", "renter_share"]]

def get_heating_type(heating_type: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate heating type shares from heating type data.
    
    Parameters
    ----------
    heating_type : gpd.GeoDataFrame
        DataFrame with heating type columns
        
    Returns
    -------
    gpd.GeoDataFrame
        DataFrame with individual heating type shares, central_heating_share, and geometry
    """
    heating_cols = ["Fernheizung", "Etagenheizung", "Blockheizung", "Zentralheizung", "Einzel_Mehrraumoefen", "keine_Heizung"]
    heating_type = calc_total(heating_type, heating_cols)
    
    # Calculate individual shares
    for col in heating_cols:
        heating_type[col + "_share"] = heating_type[col] / heating_type["total"]
    
    # Calculate central heating share (Zentralheizung)
    heating_type["central_heating_share"] = heating_type["Zentralheizung_share"]

    # Include GITTER_ID_100m if it exists, otherwise just return geometry and calculated columns
    base_cols = ["geometry"] + [col + "_share" for col in heating_cols] + ["central_heating_share"]
    if "GITTER_ID_100m" in heating_type.columns:
        base_cols.insert(1, "GITTER_ID_100m")
    
    return heating_type[base_cols]

def get_energy_type(energy_type: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate energy type shares from energy type data.
    
    Parameters
    ---------- 
    energy_type : gpd.GeoDataFrame
        DataFrame with energy type columns
        
    Returns
    -------
    gpd.GeoDataFrame
        DataFrame with aggregated energy type shares and geometry
    """
    energy_cols = ["Gas", "Heizoel", "Holz_Holzpellets", "Biomasse_Biogas", "Solar_Geothermie_Waermepumpen", "Strom", "Kohle", "Fernwaerme", "kein_Energietraeger"]
    energy_type = calc_total(energy_type, energy_cols)
    
    # Calculate aggregated shares (matching the prototype)
    energy_type["fossil_heating_share"] = (energy_type["Gas"] + energy_type["Heizoel"] + energy_type["Kohle"]) / energy_type["total"]
    energy_type["renewable_share"] = (energy_type["Holz_Holzpellets"] + energy_type["Biomasse_Biogas"] + energy_type["Solar_Geothermie_Waermepumpen"] + energy_type["Strom"]) / energy_type["total"]
    energy_type["no_energy_type"] = energy_type["kein_Energietraeger"] / energy_type["total"]
    energy_type["fernwaerme_share"] = energy_type["Fernwaerme"] / energy_type["total"]

    # Include GITTER_ID_100m if it exists, otherwise just return geometry and calculated columns
    base_cols = ["geometry", "fossil_heating_share", "renewable_share", "no_energy_type", "fernwaerme_share"]
    if "GITTER_ID_100m" in energy_type.columns:
        base_cols.insert(1, "GITTER_ID_100m")
    
    return energy_type[base_cols]


def merge_dfs(list_of_dfs: List[pd.DataFrame], on_col: str, how: str) -> pd.DataFrame:
    """
    Merge multiple DataFrames sequentially on a specified column.
    
    Parameters
    ----------
    list_of_dfs : List[pd.DataFrame]
        List of DataFrames to merge
    on_col : str
        Column name to merge on
    how : str
        Type of merge (e.g., 'inner', 'outer', 'left', 'right')
        
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with reset index
    """
    if not list_of_dfs:
        return pd.DataFrame()
    
    merged_df = list_of_dfs[0].copy()
    for i, df in enumerate(list_of_dfs[1:], 1):
        # Use suffixes to avoid column name conflicts, but prefer the first occurrence
        merged_df = pd.merge(merged_df, df, on=on_col, how=how, suffixes=('', f'_merge_{i}'))
    
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df

def make_pie(row: pd.Series, cols: list, labels: dict) -> list:
    """
    Create pie chart data structure for visualization.
    
    Parameters
    ----------
    row : pd.Series
        Row containing share columns
    cols : list
        List of column names to include in pie chart
    labels : dict
        Mapping from column names to display labels
        
    Returns
    -------
    list
        List of dictionaries with label and value for pie chart
    """
    return [
        {"label": labels[col], "value": row[col]}
        for col in cols
    ]

def calc_rent_campaign_flags(
        rent_campaign_df, 
        threshold_dict={
            "central_heating_thres":0.6,
            "fossil_heating_thres":0.6,
            "fernwaerme_thres":0.2,
            "renter_share":0.6,
            "etagenheizung_thres":0.6
            }
            ):

    rent_campaign_df["central_heating_flag"] = rent_campaign_df["central_heating_share"] > threshold_dict["central_heating_thres"]
    rent_campaign_df["fossil_heating_flag"] = rent_campaign_df["fossil_heating_share"] > threshold_dict["fossil_heating_thres"]
    rent_campaign_df["fernwaerme_flag"] = rent_campaign_df["fernwaerme_share"] > threshold_dict["fernwaerme_thres"]
    rent_campaign_df["renter_flag"] = rent_campaign_df["renter_share"] > threshold_dict["renter_share"]
    rent_campaign_df["etagenheizung_flag"] = rent_campaign_df["Etagenheizung_share"] > threshold_dict["etagenheizung_thres"]

    rent_campaign_df=rent_campaign_df[rent_campaign_df["renter_flag"]==True]

    return rent_campaign_df

def get_rent_campaign_df(
        heating_type, 
        energy_type, 
        renter_df, 
        heating_typeshare_list,
        energy_type_share_list,
        heating_labels,
        energy_labels,
        threshold_dict=None):
    
    if threshold_dict is None:
        threshold_dict={
            "central_heating_thres":0.6,
            "fossil_heating_thres":0.6,
            "fernwaerme_thres":0.2,
            "renter_share":0.6,
            "etagenheizung_thres":0.6
            }

    logger.debug(f"Running get_heating_type with heating_type shape: {heating_type.shape}")
    heating_type_df=get_heating_type(heating_type)
    logger.debug(f"Running get_energy_type with energy_type shape: {energy_type.shape}")
    energy_type_df=get_energy_type(energy_type)
    logger.debug(f"Running get_renter_share with renter_df shape: {renter_df.shape}")
    renter_df=get_renter_share(renter_df)
   
    logger.debug(f"Merging DataFrames with shapes: heating_type_df={heating_type_df.shape}, energy_type_df={energy_type_df.shape}, renter_df={renter_df.shape}")
    
    # Check which DataFrames have GITTER_ID_100m column for merging
    merge_col = "GITTER_ID_100m" if "GITTER_ID_100m" in heating_type_df.columns else "geometry"
    logger.debug(f"Using '{merge_col}' column for merging")
    
    rent_campaign_df=merge_dfs(
        list_of_dfs=[heating_type_df, energy_type_df, renter_df], 
        on_col=merge_col, 
        how="inner")
    logger.debug(f"Resulting rent_campaign_df shape: {rent_campaign_df.shape}")

    # Create pie columns first (before filtering)
    logger.debug("Calculating pie chart values for heating and energy types")
    rent_campaign_df["heating_pie"] = rent_campaign_df.apply(
        lambda r: make_pie(r, heating_typeshare_list, heating_labels), axis=1
    )

    rent_campaign_df["energy_pie"] = rent_campaign_df.apply(
        lambda r: make_pie(r, energy_type_share_list, energy_labels), axis=1
    )

    # Calculate flags based on thresholds
    logger.debug("Calculating rent campaign flags")
    rent_campaign_df=calc_rent_campaign_flags(rent_campaign_df, threshold_dict)

    # Drop original share columns
    rent_campaign_df = rent_campaign_df.drop(columns=heating_typeshare_list + energy_type_share_list)

    # Include GITTER_ID_100m if it exists, otherwise just return geometry and other columns
    base_cols = ["geometry", "central_heating_flag", "fossil_heating_flag", "fernwaerme_flag", "renter_flag", "etagenheizung_flag", "renter_share", "heating_pie", "energy_pie"]
    if "GITTER_ID_100m" in rent_campaign_df.columns:
        base_cols.insert(1, "GITTER_ID_100m")
    
    return rent_campaign_df[base_cols]


def filter_squares_invoting_distirct(bezirke_dict, rent_campaign_df):
    results_dict = {}
    color_dict = {}  # NEW: Store district colors
    work_crs = "EPSG:3035"

    for key, gdf in bezirke_dict.items():
        try:
            overlapping_list = []
            logger.info(f"Processing {key} with {len(gdf)} geometries")
            
            # Extract district color from _umap_options
            district_color = None
            if "_umap_options" in gdf.columns and len(gdf) > 0:
                try:
                    umap_opts = gdf["_umap_options"].iloc[0]
                    # Handle both string (JSON) and dict formats
                    if isinstance(umap_opts, str):
                        import json
                        umap_opts = json.loads(umap_opts)
                    district_color = umap_opts.get("fillColor", None)
                    if district_color:
                        logger.debug(f"Extracted color {district_color} for district {key}")
                except Exception as e:
                    logger.warning(f"Could not extract color from {key}: {e}")
                    district_color = None
            
            # Store color for this district
            color_dict[key] = district_color
            
            gdf.to_crs(work_crs)
            for index, row in gdf.iterrows():
                try:
                    overlap = squares_in_municipality(gdf, rent_campaign_df, mun_index=index, min_overlap_ratio=0.10)
                    name = extract_district_name(gdf["name"].iloc[0])
                    overlap["district_name"] = name
                    overlapping_list.append(overlap)
                except Exception as e:
                    logger.error(f"Error processing geometry {index} for district {key}: {e}")
                    continue
            
            try:
                overlapping_df = pd.concat(overlapping_list, ignore_index=True)
            except Exception as e:
                logger.error(f"Error concatenating overlapping list for {key}: {e}")
                # Create empty result for this district
                empty_df = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=work_crs)
                results_dict[key] = empty_df
                continue
                
            results_dict[key] = overlapping_df
            logger.debug(f"Successfully processed district {key}: {len(overlapping_df)} squares")
            
        except Exception as e:
            logger.error(f"Failed to process district {key}: {e}")
            logger.warning(f"Skipping district {key} and continuing with remaining districts")
            # Create empty result for this district
            empty_df = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=work_crs)
            results_dict[key] = empty_df
            continue

    return results_dict, color_dict  # Return both results and colors

def get_all_addresses(results_dict):
    addresses_results_dict={}
    for key, gdf in results_dict.items():
        try:
            logger.info(f"Results for {key}: {len(gdf)} overlapping geometries")
            addresses=addresses_in_squares_overpass_unified(gdf)
            addresses=add_umap_tooltip(addresses)
            addresses_results_dict[key]=addresses
            logger.debug(f"Successfully processed addresses for {key}: {len(addresses)} addresses")
        except Exception as e:
            logger.error(f"Failed to process addresses for district {key}: {e}")
            logger.warning(f"Skipping district {key} and continuing with remaining districts")
            # Create empty GeoDataFrame with same structure for failed district
            empty_columns = ['housenumber', 'street', 'postcode', 'city', 'geometry']
            # Add flag columns if they exist in the original squares
            if hasattr(gdf, 'columns'):
                flag_columns = ['central_heating_flag', 'fossil_heating_flag', 'fernwaerme_flag', 'renter_flag', 'wucher_miete_flag']
                for flag_col in flag_columns:
                    if flag_col in gdf.columns:
                        empty_columns.append(flag_col)
            
            empty_addresses = gpd.GeoDataFrame(columns=empty_columns, 
                                             geometry='geometry', crs=gdf.crs if hasattr(gdf, 'crs') else None)
            addresses_results_dict[key] = empty_addresses
            continue

    return addresses_results_dict


def export_geodf_dict_to_geojson(gdf_dict, output_dir, prefix="layer"):
    """
    Export a dict of GeoDataFrames to GeoJSON files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for key, gdf in gdf_dict.items():
        out_path = os.path.join(output_dir, f"{prefix}_{key}.geojson")
        logger.info(f"Exporting {key} to {out_path}")
        gdf.to_file(out_path, driver="GeoJSON")

def gdf_dict_to_crs(d, crs):
    out = {}
    for k, v in d.items():
        try:
            if isinstance(v, gpd.GeoDataFrame):
                out[k] = v.to_crs(crs)
                logger.debug(f"Successfully transformed {k} to CRS {crs}")
            else:
                out[k] = v
        except Exception as e:
            logger.error(f"Failed to transform {k} to CRS {crs}: {e}")
            logger.warning(f"Skipping {k} due to transformation error")
            # Keep original data even if transformation fails
            out[k] = v
            continue
    return out


FlagCols = ("central_heating_flag", "fossil_heating_flag", "fernwaerme_flag", "renter_flag", "wucher_miete_flag")


def flags_to_key(row: pd.Series) -> str:
    """
    Generate a binary flag key from boolean flag columns.
    
    Creates a 4-character binary string representing the state of:
    - central_heating_flag (position 0)
    - fossil_heating_flag (position 1) 
    - fernwaerme_flag (position 2)
    - wucher_miete_flag (position 3)
    
    Parameters
    ----------
    row : pd.Series
        Row containing boolean flag columns
        
    Returns
    -------
    str
        4-character binary string like "1010" 
    """
    # Handle missing flag columns gracefully - default to False
    central_heating = row.get('central_heating_flag', False)
    fossil_heating = row.get('fossil_heating_flag', False)
    fernwaerme = row.get('fernwaerme_flag', False)
    wucher_miete = row.get('wucher_miete_flag', False)
    
    return (
        f"{int(central_heating)}"
        f"{int(fossil_heating)}"
        f"{int(fernwaerme)}"
        f"{int(wucher_miete)}"
    )


def add_conversation_starters(gdf: gpd.GeoDataFrame, conversation_starters: dict) -> gpd.GeoDataFrame:
    """
    Add conversation starter columns to GeoDataFrame based on flag combinations.
    
    Adds two new columns:
    - flag_key: 4-character binary string representing flag states
    - conversation_start: Mapped conversation starter text
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with boolean flag columns
    conversation_starters : dict
        Dictionary mapping flag keys to conversation starter texts
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with additional conversation starter columns
    """
    gdf = gdf.copy()
    
    # Handle empty GeoDataFrames
    if gdf.empty:
        logger.warning("Empty GeoDataFrame passed to add_conversation_starters - returning empty result")
        gdf["flag_key"] = pd.Series([], dtype=str)
        gdf["conversation_start"] = pd.Series([], dtype=str)
        return gdf
    
    # Generate flag keys
    gdf["flag_key"] = gdf.apply(flags_to_key, axis=1)
    
    # Map conversation starters, with fallback for missing keys
    default_starter = "Hallo, ich bin von der Linken. Wie geht es Ihnen mit den Wohn- und Nebenkosten?"
    gdf["conversation_start"] = gdf["flag_key"].map(conversation_starters).fillna(default_starter)
    
    return gdf


def _ensure_flags(gdf: gpd.GeoDataFrame, flag_cols: Iterable[str] = FlagCols) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    for c in flag_cols:
        if c not in gdf.columns:
            gdf[c] = False
    return gdf


def add_umap_tooltip(gdf: gpd.GeoDataFrame, title_col: str = "district_name") -> gpd.GeoDataFrame:
    """Create a rich HTML tooltip based on standard flag columns. Works for addresses AND squares."""
    def yesno(v): return "✅" if bool(v) else "❌"
    gdf = _ensure_flags(gdf)

    # robust title series even if column is missing
    if title_col in gdf.columns:
        title = gdf[title_col].astype(str)
    else:
        title = pd.Series([""] * len(gdf), index=gdf.index)

    gdf = gdf.copy()
    gdf["tooltip"] = (
        "<b>" + title + "</b><br>"
        "<ul>"
        + "<li>Central heating: " + gdf["central_heating_flag"].map(yesno) + "</li>"
        + "<li>Fossil heating: "   + gdf["fossil_heating_flag"].map(yesno)   + "</li>"
        + "<li>Fernwärme: "        + gdf["fernwaerme_flag"].map(yesno)       + "</li>"
        + "<li>Renter: "           + gdf["renter_flag"].map(yesno)           + "</li>"
        + "<li>Wucher Miete: "     + gdf["wucher_miete_flag"].map(yesno)     + "</li>"
        + "</ul>"
    )
    return gdf


def _make_desc(row: pd.Series) -> str:
    def yesno(v): 
        return "✅ yes" if bool(v) else "❌ no"
    return (
        "<b>Flags</b><ul>"
        f"<li>Central heating: {yesno(row['central_heating_flag'])}</li>"
        f"<li>Fossil heating: {yesno(row['fossil_heating_flag'])}</li>"
        f"<li>Fernwärme: {yesno(row['fernwaerme_flag'])}</li>"
        f"<li>Renter: {yesno(row['renter_flag'])}</li>"
        f"<li>Wucher Miete: {yesno(row['wucher_miete_flag'])}</li>"
        "</ul>"
    )


def _geom_kind(gdf: gpd.GeoDataFrame) -> Literal["point", "polygon", "mixed"]:
    kinds = set(gdf.geometry.geom_type.unique())
    if kinds <= {"Point"}:
        return "point"
    if kinds <= {"Polygon", "MultiPolygon"}:
        return "polygon"
    return "mixed"


def export_gdf_to_umap_geojson(
    gdf: gpd.GeoDataFrame,
    out_path: str,
    *,
    name_cols: Tuple[str, str] = ("street", "housenumber"),
    feature_type: Literal["auto", "addresses", "squares"] = "auto",
    title_col: str = "district_name",
    exclude_cols: Optional[List[str]] = None,
    selection_type: Literal["old_selection", "new_selection"] = "old_selection",
    override_color: Optional[str] = None,
    population_stats: Optional[dict] = None,
    square_city_mapping: Optional[dict] = None,
    opacity_config: Optional[dict] = None,
) -> str:
    """
    Generic exporter for uMap-ready GeoJSON.
    Handles both point 'addresses' and polygon 'squares'.

    Adds:
      - 'name'            : human-readable title (address for points; district/centroid for polygons)
      - 'description'     : HTML with flag summary
      - '_umap_options'   : styling dict (marker options for points; stroke/fill for polygons)
      - 'tooltip'         : same HTML as description but with bold title (nice in uMap)

    Parameters
    ----------
    gdf : GeoDataFrame of POINT or (Multi)POLYGON geometries.
    out_path : output .geojson path
    name_cols : columns to build 'name' for addresses (points)
    feature_type : force behavior ("addresses" or "squares") or "auto" to infer from geometry
    title_col : column used as title (e.g., "district_name")
    selection_type : data source type for color coding
            - "old_selection" → red #e74c3c (voting districts) or inherited color
            - "new_selection" → grey #9e9e9e (city-level data)
    override_color : hex color to override default (only applied for old_selection)
            - If provided and selection_type is "old_selection", uses this color
            - Enables color inheritance from input district files
            - Example: "#2d0a41" (purple), "#ff0000" (red), "#2E8B57" (green)
    population_stats : dict, optional
        City-level population statistics from calculate_city_population_stats()
        Format: {city_name: {"min": float, "max": float, "count": int}}
    square_city_mapping : dict, optional
        Mapping of square indices to city names
        Format: {index: city_name}
    opacity_config : dict, optional
        Configuration for opacity scaling with keys:
        - enabled (bool): Whether to use dynamic opacity
        - min_opacity (float): Minimum opacity value
        - max_opacity (float): Maximum opacity value
        - fallback_opacity (float): Default when scaling unavailable
        - population_column (str): Column name for population data
        - use_robust_scaling (bool): Enable percentile-based clipping
        - lower_percentile (float): Lower clip boundary (%)
        - upper_percentile (float): Upper clip boundary (%)
    """
    if gdf.empty:
        raise ValueError("Input GeoDataFrame is empty.")
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a CRS set.")

    gdf = gdf.copy().to_crs(4326)
    gdf = _ensure_flags(gdf)

    inferred = _geom_kind(gdf)
    if feature_type == "auto":
        feature_type = "addresses" if inferred == "point" else "squares"
    
    # Get opacity configuration
    if opacity_config is None:
        from params import OPACITY_SCALING_CONFIG
        opacity_config = OPACITY_SCALING_CONFIG
    
    # Determine color: override takes precedence for old_selection, fallback to defaults
    if override_color and selection_type == "old_selection":
        # Use district-specific color from input file
        color = override_color
        logger.debug(f"Using inherited color: {color}")
    else:
        # Use default color based on selection type
        color = "#e74c3c" if selection_type == "old_selection" else "#9e9e9e"

    # Build 'name'
    if feature_type == "addresses":
        def make_name(row):
            vals = []
            for c in name_cols:
                if c in row and pd.notna(row[c]) and str(row[c]).strip():
                    vals.append(str(row[c]).strip())
            if vals:
                return " ".join(vals)
            # graceful fallbacks
            for alt in ("name", "addr:street", "addr:housenumber", title_col):
                if alt in row and pd.notna(row[alt]) and str(row[alt]).strip():
                    return str(row[alt]).strip()
            # last resort: point coords
            x, y = row.geometry.x, row.geometry.y
            return f"{x:.5f}, {y:.5f}"
    else:  # squares (polygons)
        def make_name(row):
            # Prefer district_name (or custom title_col); fallback to centroid coords
            if title_col in row and pd.notna(row[title_col]) and str(row[title_col]).strip():
                return str(row[title_col]).strip()
            cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
            return f"Area @ {cx:.5f}, {cy:.5f}"

    gdf["name"] = gdf.apply(make_name, axis=1)
    gdf["description"] = gdf.apply(_make_desc, axis=1)

    # Tooltip with bold title + flags
    gdf = add_umap_tooltip(gdf, title_col=title_col)

    # Styling for uMap
    if feature_type == "addresses":
        gdf["_umap_options"] = gdf.apply(
            lambda _: {
                "iconClass": "Circle",     # e.g. "Circle", "Drop", "Ball"
                "color": color,            # marker/outline color
                # more options possible: "fillColor", "opacity", "weight", etc.
            },
            axis=1,
        )
    else:  # squares
        # Determine if we should use dynamic opacity
        use_dynamic_opacity = (
            opacity_config.get("enabled", False) and
            population_stats is not None and
            square_city_mapping is not None and
            feature_type == "squares"
        )
        
        if use_dynamic_opacity:
            logger.debug("Using population-based dynamic opacity scaling")
            
            def make_umap_options(row):
                # Get square's city
                city_name = square_city_mapping.get(row.name, None)
                
                # Get city population stats
                city_stats = population_stats.get(city_name, {}) if city_name else {}
                city_min = city_stats.get("min", None)
                city_max = city_stats.get("max", None)
                
                # Get square's population
                pop_col = opacity_config.get("population_column", "Einwohner")
                population = row.get(pop_col, None)
                
                # Calculate opacity
                if (city_min is not None and city_max is not None and 
                    population is not None and not pd.isna(population)):
                    
                    # Use the helper function
                    opacity = calculate_population_opacity(
                        population,
                        city_min,
                        city_max,
                        opacity_config.get("min_opacity", 0.1),
                        opacity_config.get("max_opacity", 0.6)
                    )
                else:
                    # Fallback for missing data
                    opacity = opacity_config.get("fallback_opacity", 0.15)
                
                return {
                    "color": color,
                    "weight": 2,
                    "opacity": 0.9,
                    "fillColor": color,
                    "fillOpacity": opacity,  # ✅ Dynamic opacity
                }
            
            gdf["_umap_options"] = gdf.apply(make_umap_options, axis=1)
        
        else:
            # Use static opacity (backward compatibility)
            logger.debug("Using static fillOpacity (dynamic scaling disabled or data unavailable)")
            fallback_opacity = opacity_config.get("fallback_opacity", 0.15)
            
            gdf["_umap_options"] = gdf.apply(
                lambda _: {
                    "color": color,
                    "weight": 2,
                    "opacity": 0.9,
                    "fillColor": color,
                    "fillOpacity": fallback_opacity,
                },
                axis=1,
            )

    # Remove excluded columns if specified
    if exclude_cols:
        columns_to_drop = [col for col in exclude_cols if col in gdf.columns]
        if columns_to_drop:
            gdf = gdf.drop(columns=columns_to_drop)
            logger.debug(f"Excluded columns from output: {columns_to_drop}")
    
    # Write GeoJSON
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path


def save_all_to_geojson(
    results_dict: Dict[str, gpd.GeoDataFrame],
    base_path: str = "/Users/lutz/Documents/red_data/Zenus_2022/output/",
    *,
    kind: Literal["addresses", "squares", "auto"] = "auto",
    file_prefix: Optional[str] = None,
    name_cols: Tuple[str, str] = ("street", "housenumber"),
    title_col: str = "district_name",
    exclude_cols: Optional[List[str]] = None,
    selection_type: Literal["old_selection", "new_selection"] = "old_selection",
    district_colors: Optional[Dict[str, str]] = None,
    population_stats: Optional[dict] = None,
    square_city_mapping: Optional[Dict[str, dict]] = None,
) -> None:
    """
    Generic batch saver for a dict of homogeneous GeoDataFrames
    (either all 'addresses' or all 'squares'). If `kind='auto'`, it infers from geometry.

    Parameters
    ----------
    population_stats : dict, optional
        City-level population statistics for opacity scaling
    square_city_mapping : dict, optional
        Nested dict mapping districts to their square-city mappings
        Format: {district_name: {square_index: city_name}}

    Example:
        save_all_to_geojson(addresses_results_dict, kind="addresses")
        save_all_to_geojson(squares_results_dict,   kind="squares")
    """
    if file_prefix is None:
        file_prefix = f"umap_{'auto' if kind=='auto' else kind}_"

    for key, gdf in results_dict.items():
        # Ensure base_path ends with a directory separator
        base_path_normalized = base_path.rstrip('/') + '/'
        out_path = f"{base_path_normalized}{file_prefix}{key}.geojson"
        logger.info(f"Exporting {kind} for {key} to {out_path}")
        try:
            # per-GDF auto if requested
            per_kind = kind
            if per_kind == "auto":
                per_kind = "addresses" if _geom_kind(gdf) == "point" else "squares"

            # Get district-specific color if available
            override_color = None
            if district_colors and key in district_colors:
                override_color = district_colors[key]
            
            # Get district-specific square-city mapping if available
            district_mapping = None
            if square_city_mapping and key in square_city_mapping:
                district_mapping = square_city_mapping[key]

            export_gdf_to_umap_geojson(
                gdf,
                out_path,
                name_cols=name_cols,
                feature_type=per_kind,     # "addresses" or "squares"
                title_col=title_col,
                exclude_cols=exclude_cols,
                selection_type=selection_type,
                override_color=override_color,
                population_stats=population_stats,
                square_city_mapping=district_mapping,
            )
        except Exception as e:
            logger.error(f"Error exporting {key} to {out_path}: {e}")
            continue


def import_dfs(path, sep):
    df_dict={}
    for file in os.listdir(path):

        if file.endswith(".csv"):
            df=pd.read_csv(os.path.join(path, file), sep=sep)
            df_dict[file.split(".")[0]]=df
    return df_dict

def drop_cols(df, cols):
    df_copy=df.copy()
    existing_cols=[col for col in cols if col in df_copy.columns]
    df_copy=df_copy.drop(columns=existing_cols, axis=1)
    return df_copy


def convert_to_float(df):
    df_copy=df.copy()
    string_columns = df_copy.select_dtypes(include=['object']).columns.tolist()
    for cols in string_columns:
        if cols != "GITTER_ID_100m":
            df_copy[cols]=df_copy[cols].str.replace(",", ".")
            df_copy[cols] = pd.to_numeric(df_copy[cols], errors='coerce')
    #df_copy=df_copy.str.replace(",", ".").astype(float)
    df_copy=df_copy.fillna(0)
    return df_copy

def create_geodataframe(df, gitter_id_column='GITTER_ID_100m'):
    """
    Convert pandas DataFrame to GeoDataFrame using GITTER_ID_100m column
    """
    # Create geometry column
    geometries = df[gitter_id_column].apply(gitter_id_to_polygon)
    
    # Remove rows where geometry creation failed
    valid_mask = geometries.notna()
    df_valid = df[valid_mask].copy()
    geometries_valid = geometries[valid_mask]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df_valid, geometry=geometries_valid, crs="EPSG:3035")
    
    return gdf


def process_df(path, sep, cols_to_drop, on_col, drop_how, how, gitter_id_column):
    logger.info("Starting process_df")
    logger.debug(f"Parameters received - path: {path}, sep: {sep}, cols_to_drop: {cols_to_drop}, "
                 f"on_col: {on_col}, drop_how: {drop_how}, how: {how}, gitter_id_column: {gitter_id_column}")

    df_dict = import_dfs(path, sep)
    logger.info(f"{len(df_dict)} dataframes imported from {path}")

    total = len(df_dict)

    for idx, (key, df) in enumerate(df_dict.items(), start=1):
        logger.debug(f"Processing dataframe {idx}/{total} – '{key}'")

        # clean it
        
        df = drop_cols(df, cols_to_drop)
        df = convert_to_float(df)
        df = create_geodataframe(df, gitter_id_column)
        # Keep GITTER_ID_100m column for merging instead of dropping it

        # write the cleaned frame back into the dict
        df_dict[key] = df

    # merged_df = merge_dfs(df_list, on_col, how)
    # logger.info("DataFrames merged successfully")

    # merged_df = drop_na(merged_df, drop_how)
    # logger.info("Missing values dropped")

    # merged_df = create_geodataframe(merged_df, gitter_id_column)
    # logger.info("GeoDataFrame created")

    logger.success("process_df completed successfully")
    return df_dict


def process_demographics(path, sep, cols_to_drop, gitter_id_column, demographics_datasets):
    """
    Process demographics CSV files and create a merged demographics GeoDataFrame.
    
    Parameters
    ----------
    path : str
        Path to directory containing CSV files
    sep : str
        CSV separator character
    cols_to_drop : list
        List of columns to drop from datasets
    gitter_id_column : str
        Name of the GITTER_ID column
    demographics_datasets : dict
        Dictionary mapping demographic categories to CSV filenames (without extension)
        
    Returns
    -------
    gpd.GeoDataFrame
        Merged demographics GeoDataFrame with all demographic data
    """
    logger.info("Starting process_demographics")
    logger.debug(f"Parameters - path: {path}, sep: {sep}, demographics_datasets: {demographics_datasets}")
    
    # Import all CSV files
    df_dict = import_dfs(path, sep)
    logger.info(f"{len(df_dict)} dataframes imported from {path}")
    
    # Filter to only demographics datasets
    demographics_dfs = {}
    missing_files = []
    
    for category, filename in demographics_datasets.items():
        if filename in df_dict:
            demographics_dfs[category] = df_dict[filename]
            logger.debug(f"Found demographics dataset: {category} -> {filename}")
        else:
            missing_files.append(filename)
            logger.warning(f"Missing demographics file: {filename}")
    
    if missing_files:
        logger.error(f"Missing required demographics files: {missing_files}")
        raise FileNotFoundError(f"Missing demographics files: {missing_files}")
    
    # Process each demographics dataset
    processed_dfs = []
    for category, df in demographics_dfs.items():
        logger.debug(f"Processing demographics dataset: {category}")
        
        # Clean the dataset
        df = drop_cols(df, cols_to_drop)
        df = convert_to_float(df)
        df = create_geodataframe(df, gitter_id_column)
        # Keep GITTER_ID_100m column for merging
        
        # Store processed dataframe
        processed_dfs.append(df)
    
    # Merge all demographics datasets
    logger.info("Merging demographics datasets")
    
    # Use GITTER_ID_100m for merging if available, otherwise use geometry
    merge_col = "GITTER_ID_100m" if "GITTER_ID_100m" in processed_dfs[0].columns else "geometry"
    logger.debug(f"Using '{merge_col}' column for merging demographics data")
    
    demographic_df = merge_dfs(processed_dfs, merge_col, "inner")
    
    # Clean up multiple geometry columns - keep only the first one and drop others
    geometry_cols = [col for col in demographic_df.columns if col.startswith('geometry')]
    if len(geometry_cols) > 1:
        logger.debug(f"Found multiple geometry columns: {geometry_cols}. Keeping only the first one.")
        # Keep the first geometry column and drop the rest
        cols_to_drop = geometry_cols[1:]
        demographic_df = demographic_df.drop(columns=cols_to_drop)
        # Ensure the remaining geometry column is set as the active geometry
        demographic_df = demographic_df.set_geometry('geometry')
    
    # Fill missing values with 0
    demographic_df.fillna(0, inplace=True)
    
    logger.info(f"Demographics processing completed. Final dataset shape: {demographic_df.shape}")
    logger.debug(f"Demographics columns: {list(demographic_df.columns)}")
    
    return demographic_df

import geopandas as gpd
import os

def save_geodataframe(gdf, output_path, file_format='geojson'):
    """
    Save a single GeoDataFrame to disk.

    Parameters:
    - gdf (gpd.GeoDataFrame): GeoDataFrame to save.
    - output_path (str): Full path where the file will be saved.
    - file_format (str): File format to save the GeoDataFrame. Default is 'geojson'.

    Supported formats include: 'gpkg', 'shp', 'geojson', etc.
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError(f"Input is not a GeoDataFrame: {type(gdf)}")
    
    try:
        gdf.to_file(output_path, driver=_get_driver(file_format))
        logger.info(f"Saved demographics GeoDataFrame to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving GeoDataFrame to {output_path}: {e}")
        raise


def save_geodataframes(gdf_dict, output_dir='output', file_format='gpkg'):
    """
    Saves all GeoDataFrames in a dictionary to disk using their keys as filenames.

    Parameters:
    - gdf_dict (dict): Dictionary where each value is a GeoDataFrame.
    - output_dir (str): Directory where the files will be saved.
    - file_format (str): File format to save the GeoDataFrames. Default is 'gpkg'.

    Supported formats include: 'gpkg', 'shp', 'geojson', etc.
    """
    os.makedirs(output_dir, exist_ok=True)

    for key, gdf in gdf_dict.items():
        if not isinstance(gdf, gpd.GeoDataFrame):
            print(f"Skipping key '{key}': Not a GeoDataFrame.")
            continue

        filename = f"{key}.{file_format}"
        filepath = os.path.join(output_dir, filename)
        
        try:
            gdf.to_file(filepath, driver=_get_driver(file_format))
            print(f"Saved: {filepath}")
        except Exception as e:
            print(f"Error saving '{key}': {e}")

def _get_driver(file_format):
    """Maps file formats to appropriate GeoPandas drivers."""
    format_driver_map = {
        'gpkg': 'GPKG',
        'shp': 'ESRI Shapefile',
        'geojson': 'GeoJSON'
    }
    return format_driver_map.get(file_format.lower(), 'GPKG')

def gitter_id_to_polygon(gitter_id):
    """
    Convert GITTER_ID_100m to a polygon geometry.
    Format: CRS3035RES100mN2691700E4341100
    """
    # Extract coordinates using regex
    match = re.match(r'CRS3035RES100mN(\d+)E(\d+)', gitter_id)
    if match:
        north = int(match.group(1))
        east = int(match.group(2))
        
        # Create 100m x 100m square polygon
        # The coordinates are the southwest corner of the grid cell
        polygon = Polygon([
            (east, north),           # SW
            (east + 100, north),     # SE
            (east + 100, north + 100), # NE
            (east, north + 100),     # NW
            (east, north)            # Close the polygon
        ])
        return polygon
    else:
        return None


# =============================================================================
# WUCHER MIETE (RENT GOUGING) DETECTION FUNCTIONS
# =============================================================================

def detect_neighbor_outliers(da: xr.DataArray, method: str = 'median', threshold: float = 3.0, size: int = 3) -> xr.DataArray:
    """
    Identify strong outliers relative to neighboring pixels using a customizable neighborhood.
    
    This function detects rent gouging by comparing each grid cell's rent to its spatial neighbors.
    A cell is flagged as an outlier if its rent is significantly higher than the local neighborhood.
    
    Parameters
    ----------
    da : xr.DataArray
        2D DataArray with rent values (floats) and NaNs for missing data.
    method : str, default 'median'
        Statistical method to compute neighbor central tendency: 'mean' or 'median'.
        'median' is more robust to outliers in the neighborhood.
    threshold : float, default 3.0
        Number of standard deviations above neighbor mean/median to flag as outlier.
        Lower values are more sensitive (detect more outliers).
    size : int, default 3
        Size of the square neighborhood (must be odd, e.g., 3, 5, 7).
        Larger sizes consider more distant neighbors.
        
    Returns
    -------
    xr.DataArray
        Boolean DataArray: True = potential rent gouging, False = normal rent.
        Same coordinates and dimensions as input.
        
    Raises
    ------
    ValueError
        If da is not 2D, size is not odd, or method is not 'mean'/'median'.
        
    Examples
    --------
    >>> rent_array = xr.DataArray([[5, 6, 15], [7, 8, 6], [5, 7, 6]])
    >>> outliers = detect_neighbor_outliers(rent_array, threshold=2.0, size=3)
    >>> outliers.values
    array([[False, False,  True],
           [False, False, False], 
           [False, False, False]])
    """
    if da.ndim != 2:
        raise ValueError("detect_neighbor_outliers only works on 2D DataArrays")
    if size % 2 == 0:
        raise ValueError("size must be an odd integer")
    if method not in ['mean', 'median']:
        raise ValueError("method must be 'mean' or 'median'")
    
    center_index = (size * size) // 2
    
    # Create a closure that captures the method and threshold parameters
    def make_outlier_checker(method_param, threshold_param, center_idx):
        def check_outlier(window):
            """Check if center pixel is an outlier relative to neighbors."""
            center = window[center_idx]
            neighbors = np.delete(window, center_idx)
            neighbors = neighbors[~np.isnan(neighbors)]
            
            # Skip if center is NaN or no valid neighbors
            if np.isnan(center) or len(neighbors) == 0:
                return False
                
            # Compute neighbor statistic and variability
            if method_param == 'mean':
                stat = np.mean(neighbors)
                std = np.std(neighbors)
            elif method_param == 'median':
                stat = np.median(neighbors)
                std = np.std(neighbors)
            else:
                return False  # This shouldn't happen due to validation above
            
            # Flag as outlier if significantly above neighbors
            # Only flag high outliers (potential rent gouging), not low outliers
            return (center - stat) > threshold_param * std
        return check_outlier
    
    # Create the outlier checker with captured parameters
    outlier_checker = make_outlier_checker(method, threshold, center_index)
    
    # Apply the outlier check to every pixel using a sliding window
    outlier_mask = generic_filter(
        da.values,
        function=outlier_checker,
        size=size,
        mode='constant',
        cval=np.nan
    )
    
    # Ensure boolean dtype
    outlier_mask = outlier_mask.astype(bool)
    
    return xr.DataArray(outlier_mask, coords=da.coords, dims=da.dims, name='wucher_miete_outlier')


def gdf_to_xarray(gdf: gpd.GeoDataFrame, value_col: str, grid_size: int = 100) -> xr.DataArray:
    """
    Convert GeoDataFrame with regular 100m grid to xarray DataArray.
    
    This function assumes the input GeoDataFrame represents a regular grid where each
    geometry is a 100m x 100m square. It extracts the grid coordinates and creates
    a 2D array suitable for spatial analysis.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with regular grid geometries and a value column.
        Geometries should be 100m x 100m squares in EPSG:3035.
    value_col : str
        Name of the column containing values to convert to array.
    grid_size : int, default 100
        Size of each grid cell in meters (should be 100 for Zensus data).
        
    Returns
    -------
    xr.DataArray
        2D DataArray with spatial coordinates and values.
        NaN values indicate missing data.
        
    Raises
    ------
    ValueError
        If gdf is empty, value_col doesn't exist, or CRS is not EPSG:3035.
        
    Examples
    --------
    >>> # Create sample grid GeoDataFrame
    >>> gdf = create_sample_grid()
    >>> da = gdf_to_xarray(gdf, 'rent_per_sqm')
    >>> da.shape
    (50, 100)  # Example dimensions
    """
    if gdf.empty:
        raise ValueError("GeoDataFrame is empty")
    if value_col not in gdf.columns:
        raise ValueError(f"Column '{value_col}' not found in GeoDataFrame")
    if gdf.crs != "EPSG:3035":
        logger.warning(f"Expected EPSG:3035, got {gdf.crs}. Reprojecting...")
        gdf = gdf.to_crs("EPSG:3035")
    
    # Extract grid coordinates from geometry centroids
    centroids = gdf.geometry.centroid
    x_coords = centroids.x.values
    y_coords = centroids.y.values
    
    # Determine grid bounds and create regular grid
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Create coordinate arrays (aligned to grid centers)
    x_range = np.arange(x_min, x_max + grid_size, grid_size)
    y_range = np.arange(y_min, y_max + grid_size, grid_size)
    
    # Initialize array with NaN
    data_array = np.full((len(y_range), len(x_range)), np.nan)
    
    # Fill array with values at correct positions
    for idx, (x, y, val) in enumerate(zip(x_coords, y_coords, gdf[value_col])):
        # Find nearest grid indices
        x_idx = np.argmin(np.abs(x_range - x))
        y_idx = np.argmin(np.abs(y_range - y))
        
        # Only assign if position is close enough (within half grid size)
        if (abs(x_range[x_idx] - x) <= grid_size/2 and 
            abs(y_range[y_idx] - y) <= grid_size/2):
            data_array[y_idx, x_idx] = val
    
    # Create xarray DataArray with proper coordinates
    da = xr.DataArray(
        data_array,
        coords={'y': y_range, 'x': x_range},
        dims=['y', 'x'],
        name=value_col,
        attrs={
            'grid_size': grid_size,
            'crs': 'EPSG:3035',
            'units': 'EUR/sqm' if 'rent' in value_col.lower() else 'unknown'
        }
    )
    
    return da


def xarray_to_gdf(da: xr.DataArray, template_gdf: gpd.GeoDataFrame, result_col: str = 'outlier') -> gpd.GeoDataFrame:
    """
    Convert xarray DataArray results back to GeoDataFrame format.
    
    This function maps the 2D array results back to the original GeoDataFrame
    geometries, preserving spatial relationships and allowing integration with
    other geospatial data.
    
    Parameters
    ----------
    da : xr.DataArray
        2D DataArray with analysis results (e.g., outlier detection).
    template_gdf : gpd.GeoDataFrame
        Original GeoDataFrame used as template for geometries and structure.
        Must have the same spatial extent as the DataArray.
    result_col : str, default 'outlier'
        Name for the column containing the DataArray values.
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with original geometries and new result column.
        Rows with NaN results are excluded.
        
    Raises
    ------
    ValueError
        If template_gdf is empty or dimensions don't match expectations.
        
    Examples
    --------
    >>> outliers_da = detect_neighbor_outliers(rent_array)
    >>> result_gdf = xarray_to_gdf(outliers_da, original_gdf, 'wucher_miete')
    >>> result_gdf['wucher_miete'].sum()  # Count outliers
    42
    """
    if template_gdf.empty:
        raise ValueError("Template GeoDataFrame is empty")
    
    # Extract coordinates and values from DataArray
    x_coords = da.coords['x'].values
    y_coords = da.coords['y'].values
    grid_size = da.attrs.get('grid_size', 100)
    
    # Get template centroids for mapping
    template_centroids = template_gdf.geometry.centroid
    template_x = template_centroids.x.values
    template_y = template_centroids.y.values
    
    # Create result column with default values
    result_values = []
    valid_indices = []
    
    for idx, (x, y) in enumerate(zip(template_x, template_y)):
        # Find nearest grid position in DataArray
        x_idx = np.argmin(np.abs(x_coords - x))
        y_idx = np.argmin(np.abs(y_coords - y))
        
        # Check if position is within reasonable distance
        if (abs(x_coords[x_idx] - x) <= grid_size/2 and 
            abs(y_coords[y_idx] - y) <= grid_size/2):
            
            value = da.values[y_idx, x_idx]
            if not np.isnan(value):
                result_values.append(value)
                valid_indices.append(idx)
    
    # Create result GeoDataFrame with only valid results
    if valid_indices:
        result_gdf = template_gdf.iloc[valid_indices].copy()
        result_gdf[result_col] = result_values
        result_gdf = result_gdf.reset_index(drop=True)
    else:
        # Return empty GeoDataFrame with correct structure
        result_gdf = template_gdf.iloc[0:0].copy()
        result_gdf[result_col] = pd.Series([], dtype=da.dtype)
    
    return result_gdf


# =============================================================================
# METRIC CARD FUNCTIONALITY
# =============================================================================

def load_city_boundaries(path: str = "/Users/lutz/Documents/red_data/Zenus_2022/vg250_ebenen_0101/VG250_KRS.shp", 
                        crs: str = "EPSG:3035") -> gpd.GeoDataFrame:
    """
    Load city and Landkreis boundaries for spatial analysis.
    
    Parameters
    ----------
    path : str
        Path to the VG250_KRS.shp file
    crs : str
        Target coordinate reference system
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with city/Landkreis boundaries
    """
    logger.info(f"Loading city boundaries from: {path}")
    krs_gdf = import_gdf(path, crs)
    
    # Select only required columns
    krs_gdf = krs_gdf[["GEN", "BEZ", "geometry"]]
    
    logger.info(f"Loaded {len(krs_gdf)} city/Landkreis boundaries")
    return krs_gdf


def map_districts_to_cities(results_dict: dict, krs_gdf: gpd.GeoDataFrame) -> dict:
    """
    Map each district's squares to their corresponding city or Landkreis.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with district names as keys and GeoDataFrames as values
    krs_gdf : gpd.GeoDataFrame
        City/Landkreis boundaries
        
    Returns
    -------
    dict
        Dictionary mapping district names to city/Landkreis information
    """
    logger.info("Mapping districts to cities/Landkreise")
    
    district_city_mapping = {}
    
    for district_name, district_gdf in results_dict.items():
        if district_gdf.empty:
            logger.warning(f"Empty district {district_name}, skipping city mapping")
            district_city_mapping[district_name] = None
            continue
            
        try:
            # Ensure same CRS
            district_gdf_proj = district_gdf.to_crs(krs_gdf.crs)
            
            # Find overlapping cities/Landkreise
            overlapping_cities = []
            
            for idx, city_row in krs_gdf.iterrows():
                city_geom = city_row.geometry
                city_name = city_row['GEN']
                city_type = city_row['BEZ']
                
                # Check if any squares in this district overlap with this city
                overlapping_squares = district_gdf_proj[district_gdf_proj.geometry.intersects(city_geom)]
                
                if not overlapping_squares.empty:
                    overlapping_cities.append({
                        'city_name': city_name,
                        'city_type': city_type,
                        'overlap_count': len(overlapping_squares),
                        'geometry': city_geom
                    })
            
            if overlapping_cities:
                # Find the city with the most overlapping squares
                best_match = max(overlapping_cities, key=lambda x: x['overlap_count'])
                district_city_mapping[district_name] = best_match
                logger.debug(f"District {district_name} mapped to {best_match['city_name']} ({best_match['city_type']})")
            else:
                logger.warning(f"No city/Landkreis found for district {district_name}")
                district_city_mapping[district_name] = None
                
        except Exception as e:
            logger.error(f"Error mapping district {district_name} to city: {e}")
            district_city_mapping[district_name] = None
    
    logger.info(f"Successfully mapped {len([v for v in district_city_mapping.values() if v is not None])} districts to cities")
    return district_city_mapping


def calculate_city_means(rent_campaign_df: gpd.GeoDataFrame, krs_gdf: gpd.GeoDataFrame, 
                        metric_columns: list) -> dict:
    """
    Calculate mean values for each metric column per city/Landkreis.
    
    Parameters
    ----------
    rent_campaign_df : gpd.GeoDataFrame
        Full rent campaign data with all squares
    krs_gdf : gpd.GeoDataFrame
        City/Landkreis boundaries
    metric_columns : list
        List of column names to calculate means for
        
    Returns
    -------
    dict
        Dictionary mapping city names to mean values for each metric
    """
    logger.info(f"Calculating city means for {len(metric_columns)} metrics")
    
    # Ensure same CRS
    rent_campaign_proj = rent_campaign_df.to_crs(krs_gdf.crs)
    
    city_means = {}
    
    for idx, city_row in krs_gdf.iterrows():
        city_name = city_row['GEN']
        city_geom = city_row.geometry
        
        # Find squares within this city
        city_squares = rent_campaign_proj[rent_campaign_proj.geometry.intersects(city_geom)]
        
        if not city_squares.empty:
            city_means[city_name] = {}
            
            for metric_col in metric_columns:
                if metric_col in city_squares.columns:
                    # Calculate mean, excluding NaN values
                    valid_values = city_squares[metric_col].dropna()
                    if not valid_values.empty:
                        city_means[city_name][metric_col] = valid_values.mean()
                    else:
                        city_means[city_name][metric_col] = None
                else:
                    logger.warning(f"Column {metric_col} not found in city squares for {city_name}")
                    city_means[city_name][metric_col] = None
        else:
            logger.warning(f"No squares found for city {city_name}")
            city_means[city_name] = {col: None for col in metric_columns}
    
    logger.info(f"Calculated means for {len(city_means)} cities")
    return city_means


def calculate_population_opacity(
    population: float,
    city_min: float,
    city_max: float,
    min_opacity: float = 0.1,
    max_opacity: float = 0.6
) -> float:
    """
    Calculate fillOpacity based on population density within city with automatic clipping.
    
    Parameters
    ----------
    population : float
        Population count for this square
    city_min : float
        Minimum population in the city/Landkreis (or lower percentile if robust scaling)
    city_max : float
        Maximum population in the city/Landkreis (or upper percentile if robust scaling)
    min_opacity : float
        Minimum opacity for least populated squares (default: 0.1 = very faint)
    max_opacity : float
        Maximum opacity for most populated squares (default: 0.6 = bold)
    
    Returns
    -------
    float
        Opacity value between min_opacity and max_opacity
    
    Examples
    --------
    >>> # Square with min population in city
    >>> calculate_population_opacity(10, 10, 100, 0.1, 0.6)
    0.1
    
    >>> # Square with max population in city
    >>> calculate_population_opacity(100, 10, 100, 0.1, 0.6)
    0.6
    
    >>> # Square with median population
    >>> calculate_population_opacity(55, 10, 100, 0.1, 0.6)
    0.35  # (55-10)/(100-10) = 0.5, so 0.1 + (0.5 * 0.5) = 0.35
    """
    # Handle edge cases - use max opacity for missing data
    if pd.isna(population) or pd.isna(city_min) or pd.isna(city_max):
        return max_opacity  # Fallback: full opacity ensures visibility
    
    if city_max == city_min:
        # All squares have same population (or only one square)
        return (min_opacity + max_opacity) / 2  # Use middle opacity
    
    # Clip population to [city_min, city_max] range
    # This handles both outliers and values outside percentile bounds
    clipped_pop = np.clip(population, city_min, city_max)
    
    # Min-max normalization on clipped value
    normalized = (clipped_pop - city_min) / (city_max - city_min)
    
    # Safety clamp (should already be in [0,1] due to clipping)
    normalized = max(0.0, min(1.0, normalized))
    
    # Map to opacity range
    opacity = min_opacity + (normalized * (max_opacity - min_opacity))
    
    return opacity


def calculate_city_population_stats(
    rent_campaign_df: gpd.GeoDataFrame,
    krs_gdf: gpd.GeoDataFrame,
    population_column: str = "Einwohner",
    use_robust_scaling: bool = True,
    lower_percentile: float = 5,
    upper_percentile: float = 95
) -> dict:
    """
    Calculate population statistics for each city with optional outlier handling.
    
    This function identifies population ranges within each administrative
    district to enable relative opacity scaling. Supports robust percentile-based
    boundaries to handle outliers in both directions.
    
    Parameters
    ----------
    rent_campaign_df : gpd.GeoDataFrame
        Full rent campaign data with all squares
    krs_gdf : gpd.GeoDataFrame
        City/Landkreis boundaries (e.g., VG250_KRS.shp)
    population_column : str
        Name of population column (default: "Einwohner")
    use_robust_scaling : bool
        If True, use percentile-based boundaries instead of min/max (default: True)
    lower_percentile : float
        Lower percentile for clipping (default: 5)
    upper_percentile : float
        Upper percentile for clipping (default: 95)
    
    Returns
    -------
    dict
        Dictionary mapping city names to population statistics:
        {
            "City Name": {
                "min": float,        # Minimum (or lower percentile)
                "max": float,        # Maximum (or upper percentile)
                "p5": float,         # 5th percentile (if robust)
                "p95": float,        # 95th percentile (if robust)
                "actual_min": float, # Actual minimum (if robust)
                "actual_max": float, # Actual maximum (if robust)
                "count": int,        # Number of squares
                "outliers_low": int, # Count of low outliers clipped (if robust)
                "outliers_high": int # Count of high outliers clipped (if robust)
            }
        }
    
    Examples
    --------
    >>> stats = calculate_city_population_stats(rent_df, cities_df)
    >>> stats["Essen"]
    {'min': 10.0, 'max': 450.0, 'count': 1234, ...}
    """
    logger.info(f"Calculating population statistics for cities using column '{population_column}'")
    
    # Ensure same CRS
    rent_campaign_proj = rent_campaign_df.to_crs(krs_gdf.crs)
    
    # Check if population column exists
    if population_column not in rent_campaign_proj.columns:
        logger.warning(f"Population column '{population_column}' not found. Available columns: {list(rent_campaign_proj.columns)}")
        return {}
    
    city_stats = {}
    
    for idx, city_row in krs_gdf.iterrows():
        city_name = city_row['GEN']
        city_geom = city_row.geometry
        
        # Find squares within this city
        city_squares = rent_campaign_proj[rent_campaign_proj.geometry.intersects(city_geom)]
        
        if not city_squares.empty:
            # Get valid population values (exclude NaN)
            pop_values = city_squares[population_column].dropna()
            
            if not pop_values.empty and len(pop_values) > 1:
                if use_robust_scaling:
                    # Use percentiles for robust boundaries
                    p_lower = np.percentile(pop_values, lower_percentile)
                    p_upper = np.percentile(pop_values, upper_percentile)
                    
                    city_stats[city_name] = {
                        "min": float(p_lower),   # Robust lower boundary
                        "max": float(p_upper),   # Robust upper boundary
                        "p5": float(np.percentile(pop_values, 5)),
                        "p95": float(np.percentile(pop_values, 95)),
                        "actual_min": float(pop_values.min()),
                        "actual_max": float(pop_values.max()),
                        "count": len(pop_values),
                        "outliers_low": int((pop_values < p_lower).sum()),
                        "outliers_high": int((pop_values > p_upper).sum())
                    }
                    
                    logger.debug(
                        f"City {city_name}: robust range [{p_lower:.1f}, {p_upper:.1f}], "
                        f"actual range [{pop_values.min():.1f}, {pop_values.max():.1f}], "
                        f"outliers: {city_stats[city_name]['outliers_low']} low, "
                        f"{city_stats[city_name]['outliers_high']} high"
                    )
                else:
                    # Standard min-max
                    city_stats[city_name] = {
                        "min": float(pop_values.min()),
                        "max": float(pop_values.max()),
                        "count": len(pop_values)
                    }
                    logger.debug(f"City {city_name}: standard range [{pop_values.min():.1f}, {pop_values.max():.1f}], {len(pop_values)} squares")
            elif len(pop_values) == 1:
                # Only one square with population
                single_val = float(pop_values.iloc[0])
                city_stats[city_name] = {
                    "min": single_val,
                    "max": single_val,
                    "count": 1
                }
                logger.debug(f"City {city_name}: single square with population {single_val}")
            else:
                logger.warning(f"No valid population data for city {city_name}")
                city_stats[city_name] = {"min": None, "max": None, "count": 0}
        else:
            logger.warning(f"No squares found for city {city_name}")
            city_stats[city_name] = {"min": None, "max": None, "count": 0}
    
    logger.info(f"Calculated population stats for {len(city_stats)} cities")
    return city_stats


def map_squares_to_cities(
    results_dict: dict,
    krs_gdf: gpd.GeoDataFrame
) -> dict:
    """
    Create a mapping of each square to its containing city/Landkreis.
    
    Uses spatial join to determine which city each square belongs to.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of district GeoDataFrames with squares
    krs_gdf : gpd.GeoDataFrame
        City/Landkreis boundaries
    
    Returns
    -------
    dict
        Nested dictionary: {district_name: {square_index: city_name}}
    
    Examples
    --------
    >>> mapping = map_squares_to_cities(results_dict, krs_gdf)
    >>> mapping["Oberhausen_117_htwk_hochburg"][0]
    'Oberhausen'
    """
    logger.info("Mapping squares to cities for opacity scaling")
    
    square_city_mapping = {}
    
    for district_name, district_gdf in results_dict.items():
        if district_gdf.empty:
            square_city_mapping[district_name] = {}
            continue
        
        # Ensure same CRS
        district_proj = district_gdf.to_crs(krs_gdf.crs)
        
        # Spatial join to find which city each square is in
        joined = gpd.sjoin(
            district_proj,
            krs_gdf[['GEN', 'geometry']],
            how='left',
            predicate='intersects'
        )
        
        # Create mapping: index -> city name
        district_mapping = {}
        for idx, row in joined.iterrows():
            city_name = row.get('GEN', None)
            district_mapping[idx] = city_name
        
        square_city_mapping[district_name] = district_mapping
        
        # Log statistics
        cities_found = set([c for c in district_mapping.values() if c is not None])
        logger.debug(f"District {district_name}: {len(district_gdf)} squares mapped to {len(cities_found)} cities")
    
    return square_city_mapping


def create_metric_card(value: float, group_mean: float, metric_id: str, metric_label: str) -> dict:
    """
    Create a metric card dictionary for a single metric.
    
    Parameters
    ----------
    value : float
        Individual value for this square
    group_mean : float
        Mean value for the city/Landkreis
    metric_id : str
        Unique identifier for the metric
    metric_label : str
        Human-readable label for the metric
        
    Returns
    -------
    dict
        Metric card dictionary with all required fields
    """
    if pd.isna(value) or pd.isna(group_mean) or group_mean == 0:
        return {
            "id": metric_id,
            "label": metric_label,
            "value": value,
            "group_mean": group_mean,
            "abs_diff": None,
            "pct_diff": None,
            "direction": "equal"
        }
    
    abs_diff = value - group_mean
    pct_diff = (value / group_mean) - 1
    
    if abs_diff > 0:
        direction = "above"
    elif abs_diff < 0:
        direction = "below"
    else:
        direction = "equal"
    
    return {
        "id": metric_id,
        "label": metric_label,
        "value": value,
        "group_mean": group_mean,
        "abs_diff": abs_diff,
        "pct_diff": pct_diff,
        "direction": direction
    }


def add_metric_cards_to_districts(results_dict: dict, district_city_mapping: dict, 
                                 city_means: dict, metric_config: dict) -> dict:
    """
    Add metric cards to each district's squares.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with district GeoDataFrames
    district_city_mapping : dict
        Mapping of districts to cities
    city_means : dict
        Mean values per city for each metric
    metric_config : dict
        Configuration for metric columns and labels
        
    Returns
    -------
    dict
        Updated results_dict with metric cards added
    """
    logger.info("Adding metric cards to district squares")
    
    enhanced_results = {}
    
    for district_name, district_gdf in results_dict.items():
        if district_gdf.empty:
            enhanced_results[district_name] = district_gdf
            continue
            
        try:
            # Get city information for this district
            city_info = district_city_mapping.get(district_name)
            
            if city_info is None:
                logger.warning(f"No city mapping for district {district_name}, skipping metric cards")
                enhanced_results[district_name] = district_gdf
                continue
            
            city_name = city_info['city_name']
            city_means_for_city = city_means.get(city_name, {})
            
            # Create a copy to avoid modifying original
            enhanced_gdf = district_gdf.copy()
            
            # Add metric cards for each row
            metric_cards = []
            
            for idx, row in enhanced_gdf.iterrows():
                row_metric_cards = {}
                
                for metric_col, metric_info in metric_config.items():
                    if metric_col in row:
                        value = row[metric_col]
                        group_mean = city_means_for_city.get(metric_col)
                        
                        metric_card = create_metric_card(
                            value=value,
                            group_mean=group_mean,
                            metric_id=metric_info['id'],
                            metric_label=metric_info['label']
                        )
                        
                        row_metric_cards[metric_info['id']] = metric_card
                
                metric_cards.append(row_metric_cards)
            
            # Add metric_cards column
            enhanced_gdf['metric_cards'] = metric_cards
            
            enhanced_results[district_name] = enhanced_gdf
            logger.debug(f"Added metric cards to {len(enhanced_gdf)} squares in district {district_name}")
            
        except Exception as e:
            logger.error(f"Error adding metric cards to district {district_name}: {e}")
            enhanced_results[district_name] = district_gdf
    
    logger.info(f"Successfully added metric cards to {len(enhanced_results)} districts")
    return enhanced_results


def detect_wucher_miete(
    rent_gdf: gpd.GeoDataFrame, 
    rent_column: str = 'durchschnMieteQM',
    method: str = 'median',
    threshold: float = 2.5,
    neighborhood_size: int = 5,
    min_rent_threshold: float = 3.0,
    min_neighbors: int = 3
) -> gpd.GeoDataFrame:
    """
    Detect potential rent gouging (Wucher Miete) in spatial rent data.
    
    This function identifies grid cells where rent is significantly higher than
    the local neighborhood, which may indicate rent gouging. It uses spatial
    outlier detection on a regular grid to find anomalous rent prices.
    
    Parameters
    ----------
    rent_gdf : gpd.GeoDataFrame
        GeoDataFrame with rent data on regular 100m grid.
        Must have geometries and rent column.
    rent_column : str, default 'durchschnMieteQM'
        Name of column containing rent per square meter values.
    method : str, default 'median'
        Statistical method for neighbor comparison: 'mean' or 'median'.
    threshold : float, default 2.5
        Number of standard deviations above neighbors to flag as outlier.
    neighborhood_size : int, default 5
        Size of neighborhood for comparison (must be odd: 3, 5, 7).
    min_rent_threshold : float, default 3.0
        Minimum rent per sqm to consider (filters out very low/invalid rents).
    min_neighbors : int, default 3
        Minimum number of neighbors required for valid comparison.
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with original data plus 'wucher_miete_flag' column.
        True values indicate potential rent gouging.
        Only includes cells with valid outlier analysis.
        
    Raises
    ------
    ValueError
        If rent_gdf is empty, rent_col doesn't exist, or parameters are invalid.
        
    Examples
    --------
    >>> rent_data = load_rent_data()
    >>> wucher_results = detect_wucher_miete(
    ...     rent_data, 
    ...     threshold=2.0,
    ...     neighborhood_size=7
    ... )
    >>> print(f"Found {wucher_results['wucher_miete_flag'].sum()} potential cases")
    Found 1247 potential cases
    
    Notes
    -----
    This function performs the following steps:
    1. Filters rent data to remove invalid/low values
    2. Converts GeoDataFrame to regular grid array
    3. Applies spatial outlier detection
    4. Converts results back to GeoDataFrame format
    5. Returns only cells flagged as outliers
    """
    if rent_gdf.empty:
        raise ValueError("Rent GeoDataFrame is empty")
    if rent_column not in rent_gdf.columns:
        raise ValueError(f"Rent column '{rent_column}' not found in GeoDataFrame")
    if neighborhood_size % 2 == 0:
        raise ValueError("neighborhood_size must be odd")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    
    logger.info(f"Starting Wucher Miete detection on {len(rent_gdf):,} grid cells")
    
    # Filter valid rent data
    valid_rent_mask = (
        rent_gdf[rent_column].notna() & 
        (rent_gdf[rent_column] >= min_rent_threshold)
    )
    filtered_gdf = rent_gdf[valid_rent_mask].copy()
    
    if filtered_gdf.empty:
        logger.warning("No valid rent data after filtering")
        empty_result = rent_gdf.iloc[0:0].copy()
        empty_result['wucher_miete_flag'] = pd.Series([], dtype=bool)
        return empty_result
    
    logger.info(f"Analyzing {len(filtered_gdf):,} cells with valid rent data (≥{min_rent_threshold} EUR/sqm)")
    
    # Convert to xarray for grid analysis
    logger.debug("Converting GeoDataFrame to xarray grid")
    rent_array = gdf_to_xarray(filtered_gdf, rent_column)
    logger.debug(f"Created {rent_array.shape} grid array")
    
    # Detect outliers
    logger.debug(f"Running outlier detection (method={method}, threshold={threshold}, size={neighborhood_size})")
    outliers = detect_neighbor_outliers(
        rent_array, 
        method=method,
        threshold=threshold,
        size=neighborhood_size
    )
    
    # Convert back to GeoDataFrame
    logger.debug("Converting outlier results back to GeoDataFrame")
    result_gdf = xarray_to_gdf(outliers, filtered_gdf, 'wucher_miete_flag')
    
    # Filter to only return flagged outliers
    wucher_cases = result_gdf[result_gdf['wucher_miete_flag'] == True].copy()
    
    logger.info(f"Detected {len(wucher_cases):,} potential Wucher Miete cases")
    
    if len(wucher_cases) > 0:
        rent_stats = wucher_cases[rent_column].describe()
        logger.info(f"Outlier rent stats: mean={rent_stats['mean']:.2f}, "
                   f"median={rent_stats['50%']:.2f}, max={rent_stats['max']:.2f} EUR/sqm")
    
    return wucher_cases
