"""
USGS ComCat Data Acquisition Script
Fetches global earthquake events filtered for ShakeMap and PAGER impact metrics:
- Red, Orange, Yellow alert events (2000-2024)
- Green alert events & ShakeMap events (2018-2024)
- Correlates with sensor error metrics (horizontalError, depthError, magError)
"""

import os
import sys
import io
import time
import json
import urllib.request
import pandas as pd
import numpy as np

BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def query_url(url, retries=3, delay=2):
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as e:
            print(f"Warning: Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(delay)
    return None

def fetch_high_impact_events():
    """Fetch all available historical red, orange, yellow alert earthquakes (2000-2024).

    Sensor columns come from the CSV endpoint, exactly like fetch_period_data() does for
    ordinary events. (An earlier version read them from GeoJSON, which has no
    horizontalError / depthError / magError. Those columns were then missing for almost
    every non-green event and never for green ones, so missingness leaked the label.)
    GeoJSON is still queried, only for mmi.
    """
    chunks = []
    for alert in ["red", "orange", "yellow"]:
        print(f"Querying PAGER alert level '{alert}' (2000-2024)...")
        query = f"starttime=2000-01-01&endtime=2024-01-01&alertlevel={alert}"

        mmi_by_id = {}
        raw_geo = query_url(f"{BASE_URL}?format=geojson&{query}")
        if raw_geo:
            data = json.loads(raw_geo.decode())
            mmi_by_id = {f["id"]: f["properties"].get("mmi") for f in data["features"]}

        raw_csv = query_url(f"{BASE_URL}?format=csv&{query}")
        if raw_csv:
            df_csv = pd.read_csv(io.StringIO(raw_csv.decode()))
            df_csv["mmi"] = df_csv["id"].map(mmi_by_id)
            df_csv["alert"] = alert
            chunks.append(df_csv)
            print(f"  Found {len(df_csv)} events with alert '{alert}'")
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

def fetch_period_data(start_year, end_year, min_mag=4.8):
    """Fetch global seismic events with ShakeMap & sensor errors in yearly chunks."""
    all_chunks = []
    for yr in range(start_year, end_year):
        start_date = f"{yr}-01-01"
        end_date = f"{yr+1}-01-01"
        print(f"Querying USGS ComCat for {yr} (min magnitude {min_mag})...")

        # 1. Fetch GeoJSON for mmi and alert
        url_geo = f"{BASE_URL}?format=geojson&starttime={start_date}&endtime={end_date}&minmagnitude={min_mag}"
        raw_geo = query_url(url_geo)
        geo_dict = {}
        if raw_geo:
            data = json.loads(raw_geo.decode())
            for f in data["features"]:
                p = f["properties"]
                geo_dict[f["id"]] = {
                    "mmi": p.get("mmi"),
                    "alert": p.get("alert"),
                    "place": p.get("place")
                }

        # 2. Fetch CSV for sensor errors (horizontalError, depthError, magError, etc.)
        url_csv = f"{BASE_URL}?format=csv&starttime={start_date}&endtime={end_date}&minmagnitude={min_mag}"
        raw_csv = query_url(url_csv)
        if raw_csv:
            df_csv = pd.read_csv(io.StringIO(raw_csv.decode()))
            # Filter for events that have ShakeMap mmi or PAGER alert
            df_csv["mmi"] = df_csv["id"].map(lambda x: geo_dict.get(x, {}).get("mmi"))
            df_csv["alert"] = df_csv["id"].map(lambda x: geo_dict.get(x, {}).get("alert"))
            all_chunks.append(df_csv)
            print(f"  Year {yr}: retrieved {len(df_csv)} events ({df_csv['alert'].notna().sum()} alerts, {df_csv['mmi'].notna().sum()} ShakeMaps)")

    if all_chunks:
        return pd.concat(all_chunks, ignore_index=True)
    return pd.DataFrame()

def main():
    os.makedirs("data/raw", exist_ok=True)
    out_parquet = "data/raw/usgs_comcat_raw.parquet"
    out_csv = "data/raw/usgs_comcat_raw.csv"

    print("=== Starting USGS ComCat Data Acquisition ===")
    
    # Fetch high impact events across 2000-2024
    df_high = fetch_high_impact_events()
    
    # Fetch modern period (2018-2024)
    df_modern = fetch_period_data(2018, 2024, min_mag=4.8)
    
    # Combine and deduplicate
    cols_to_keep = [
        "id", "time", "latitude", "longitude", "depth", "mag", "magType",
        "nst", "gap", "dmin", "rms", "horizontalError", "depthError", "magError",
        "mmi", "alert", "place"
    ]

    for col in cols_to_keep:
        if col not in df_high.columns:
            df_high[col] = np.nan
        if col not in df_modern.columns:
            df_modern[col] = np.nan

    df_combined = pd.concat([df_high[cols_to_keep], df_modern[cols_to_keep]], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["id"]).reset_index(drop=True)
    
    # Filter for events that have either ShakeMap mmi or PAGER alert
    df_filtered = df_combined[df_combined["alert"].notna() | df_combined["mmi"].notna()].copy()

    # For events with mmi where alert was unassigned, assign alert based on standard USGS PAGER MMI thresholds
    # MMI < 4: green, 4 <= MMI < 6: yellow, 6 <= MMI < 7.5: orange, MMI >= 7.5: red
    def impute_alert(row):
        if pd.notna(row["alert"]):
            return str(row["alert"]).lower()
        mmi = row["mmi"]
        if pd.isna(mmi):
            return "green"
        if mmi < 4.0:
            return "green"
        elif mmi < 6.0:
            return "yellow"
        elif mmi < 7.5:
            return "orange"
        else:
            return "red"

    df_filtered["alert"] = df_filtered.apply(impute_alert, axis=1)
    
    # Standardize time to UTC ISO string / datetime
    df_filtered["time"] = pd.to_datetime(df_filtered["time"], format="mixed", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Ensure event_id column naming convention
    df_filtered = df_filtered.rename(columns={"id": "event_id"})

    print(f"\nFinal dataset shape: {df_filtered.shape}")
    print("Alert class distribution:")
    print(df_filtered["alert"].value_counts())
    print("\nNon-null counts:")
    print(df_filtered.notna().sum())

    df_filtered.to_parquet(out_parquet, index=False)
    df_filtered.to_csv(out_csv, index=False)
    print(f"\nSaved raw dataset to:")
    print(f"  - {out_parquet}")
    print(f"  - {out_csv}")

if __name__ == "__main__":
    main()

