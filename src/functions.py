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

    return renter_df[["geometry", "renter_share"]]

def get_heating_type(heating_type: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate central heating share from heating type data.
    
    Parameters
    ----------
    heating_type : gpd.GeoDataFrame
        DataFrame with heating type columns
        
    Returns
    -------
    gpd.GeoDataFrame
        DataFrame with 'central_heating_share' column and geometry
    """
    heating_type = calc_total(heating_type, ["Fernheizung", "Etagenheizung", "Blockheizung", "Zentralheizung", "Einzel_Mehrraumoefen", "keine_Heizung"])
    heating_type["central_heating_share"] = heating_type["Zentralheizung"] / heating_type["total"]

    return heating_type[["geometry", "central_heating_share"]]

def get_energy_type(energy_type: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate fossil heating and district heating shares from energy type data.
    
    Parameters
    ---------- 
    energy_type : gpd.GeoDataFrame
        DataFrame with energy type columns
        
    Returns
    -------
    gpd.GeoDataFrame
        DataFrame with 'fossil_heating_share', 'fernwaerme_share' columns and geometry
    """
    energy_type = calc_total(energy_type, ["Gas", "Heizoel", "Holz_Holzpellets", "Biomasse_Biogas", "Solar_Geothermie_Waermepumpen", "Strom", "Kohle", "Fernwaerme", "kein_Energietraeger"])
    energy_type["fossil_heating_share"] = (energy_type["Gas"] + energy_type["Heizoel"] + energy_type["Kohle"]) / energy_type["total"]
    energy_type["fernwaerme_share"] = energy_type["Fernwaerme"] / energy_type["total"]

    return energy_type[["geometry", "fossil_heating_share", "fernwaerme_share"]]


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
    merged_df = list_of_dfs[0]
    for df in list_of_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=on_col, how=how)
    merged_df.reset_index(drop=False, inplace=True)
    return merged_df

def calc_rent_campaign_flags(
        rent_campaign_df, 
        threshold_dict={
            "central_heating_thres":0.6,
            "fossil_heating_thres":0.6,
            "fernwaerme_thres":0.2,
            "renter_share":0.6
            }
            ):

    rent_campaign_df["central_heating_flag"] = rent_campaign_df["central_heating_share"] > threshold_dict["central_heating_thres"]
    rent_campaign_df["fossil_heating_flag"] = rent_campaign_df["fossil_heating_share"] > threshold_dict["fossil_heating_thres"]
    rent_campaign_df["fernwaerme_flag"] = rent_campaign_df["fernwaerme_share"] > threshold_dict["fernwaerme_thres"]
    rent_campaign_df["renter_flag"] = rent_campaign_df["renter_share"] > threshold_dict["renter_share"]

    rent_campaign_df=rent_campaign_df[rent_campaign_df["renter_flag"]==True]


    return rent_campaign_df[["geometry", "central_heating_flag", "fossil_heating_flag", "fernwaerme_flag", "renter_flag"]]

def get_rent_campaign_df(
        heating_type, 
        energy_type, 
        renter_df, 
        threshold_dict=None):
    
    if threshold_dict is None:
        threshold_dict={
            "central_heating_thres":0.6,
            "fossil_heating_thres":0.6,
            "fernwaerme_thres":0.2,
            "renter_share":0.6
            }

    logger.debug(f"Running get_heating_type with heating_type shape: {heating_type.shape}")
    heating_type_df=get_heating_type(heating_type)
    logger.debug(f"Running get_energy_type with energy_type shape: {energy_type.shape}")
    energy_type_df=get_energy_type(energy_type)
    logger.debug(f"Running get_renter_share with renter_df shape: {renter_df.shape}")
    renter_df=get_renter_share(renter_df)
   
    logger.debug(f"Merging DataFrames with shapes: heating_type_df={heating_type_df.shape}, energy_type_df={energy_type_df.shape}, renter_df={renter_df.shape}")
    rent_campaign_df=merge_dfs(
        list_of_dfs=[heating_type_df, energy_type_df, renter_df], 
        on_col="geometry", 
        how="inner")
    logger.debug(f"Resulting rent_campaign_df shape: {rent_campaign_df.shape}")

    # Calculate flags based on thresholds
    logger.debug("Calculating rent campaign flags")
    rent_campaign_df=calc_rent_campaign_flags(rent_campaign_df, threshold_dict)

    return rent_campaign_df


def filter_squares_invoting_distirct(bezirke_dict, rent_campaign_df):
    results_dict={}
    work_crs = "EPSG:3035"

    for key, gdf in bezirke_dict.items():
        overlapping_list=[]
        logger.info(f"Processing {key} with {len(gdf)} geometries")
        
        gdf.to_crs(work_crs)
        for index, row in gdf.iterrows():
            overlap = squares_in_municipality(gdf, rent_campaign_df, mun_index=index, min_overlap_ratio=0.10)
            name=gdf["name"].iloc[0].split("|")[1].strip()
            overlap["district_name"]=name
            overlapping_list.append(overlap)
        try:
            overlapping_df = pd.concat(overlapping_list, ignore_index=True)
        except Exception as e:
            logger.error(f"Error concatenating overlapping list for {key}: {e}")
            continue
        results_dict[key]=overlapping_df

    return results_dict

def get_all_addresses(results_dict):
    addresses_results_dict={}
    for key, gdf in results_dict.items():
        logger.info(f"Results for {key}: {len(gdf)} overlapping geometries")
        addresses=addresses_in_squares_overpass_unified(gdf)
        addresses=add_umap_tooltip(addresses)
        addresses_results_dict[key]=addresses

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
        out[k] = v.to_crs(crs) if isinstance(v, gpd.GeoDataFrame) else v
    return out


FlagCols = ("central_heating_flag", "fossil_heating_flag", "fernwaerme_flag", "renter_flag")


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
                "color": "#e74c3c",        # marker/outline color
                # more options possible: "fillColor", "opacity", "weight", etc.
            },
            axis=1,
        )
    else:  # squares
        # polygon style: stroke + fill
        gdf["_umap_options"] = gdf.apply(
            lambda _: {
                "color": "#e74c3c",        # stroke color
                "weight": 2,               # stroke width
                "opacity": 0.9,            # stroke opacity
                "fillColor": "#e74c3c",    # fill color
                "fillOpacity": 0.15,       # fill opacity for visibility
            },
            axis=1,
        )

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
) -> None:
    """
    Generic batch saver for a dict of homogeneous GeoDataFrames
    (either all 'addresses' or all 'squares'). If `kind='auto'`, it infers from geometry.

    Example:
        save_all_to_geojson(addresses_results_dict, kind="addresses")
        save_all_to_geojson(squares_results_dict,   kind="squares")
    """
    if file_prefix is None:
        file_prefix = f"umap_{'auto' if kind=='auto' else kind}_"

    for key, gdf in results_dict.items():
        out_path = f"{base_path}{file_prefix}{key}.geojson"
        logger.info(f"Exporting {kind} for {key} to {out_path}")
        try:
            # per-GDF auto if requested
            per_kind = kind
            if per_kind == "auto":
                per_kind = "addresses" if _geom_kind(gdf) == "point" else "squares"

            export_gdf_to_umap_geojson(
                gdf,
                out_path,
                name_cols=name_cols,
                feature_type=per_kind,     # "addresses" or "squares"
                title_col=title_col,
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
        df = drop_cols(df, ["GITTER_ID_100m"])

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

import geopandas as gpd
import os

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