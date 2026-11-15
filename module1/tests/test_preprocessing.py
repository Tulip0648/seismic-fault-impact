"""
Unit & Integration Tests for Module 1 Preprocessing & Geospatial Transformations
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0

def test_spatial_ecef_transformation():
    # Test point 1: Equator (lat=0, lon=0, depth=0)
    # Expected: x = 6371.0, y = 0.0, z = 0.0
    r = EARTH_RADIUS_KM - 0.0
    lat_rad = np.radians(0.0)
    lon_rad = np.radians(0.0)
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    assert np.isclose(x, 6371.0, atol=1e-5)
    assert np.isclose(y, 0.0, atol=1e-5)
    assert np.isclose(z, 0.0, atol=1e-5)

    # Test point 2: North Pole (lat=90, lon=0, depth=10)
    # Expected: x = 0.0, y = 0.0, z = 6361.0
    r2 = EARTH_RADIUS_KM - 10.0
    lat_rad2 = np.radians(90.0)
    lon_rad2 = np.radians(0.0)
    x2 = r2 * np.cos(lat_rad2) * np.cos(lon_rad2)
    y2 = r2 * np.cos(lat_rad2) * np.sin(lon_rad2)
    z2 = r2 * np.sin(lat_rad2)
    assert np.isclose(x2, 0.0, atol=1e-5)
    assert np.isclose(y2, 0.0, atol=1e-5)
    assert np.isclose(z2, 6361.0, atol=1e-5)

def test_dataset_partitions_exist():
    assert os.path.exists("data/processed/train_base.parquet")
    assert os.path.exists("data/processed/val_base.parquet")
    assert os.path.exists("data/processed/test_base.parquet")

def test_no_data_leakage_and_schemas():
    df_train = pd.read_parquet("data/processed/train_base.parquet")
    df_val = pd.read_parquet("data/processed/val_base.parquet")
    df_test = pd.read_parquet("data/processed/test_base.parquet")

    # 1. Check disjoint event IDs
    train_ids = set(df_train["event_id"])
    val_ids = set(df_val["event_id"])
    test_ids = set(df_test["event_id"])

    assert len(train_ids.intersection(val_ids)) == 0, "Data leakage: Train and Val IDs overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Data leakage: Train and Test IDs overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Data leakage: Val and Test IDs overlap!"

    # 2. Check column schemas match
    assert list(df_train.columns) == list(df_val.columns)
    assert list(df_train.columns) == list(df_test.columns)

    # 3. Check for unexpected NaNs in critical engineered columns
    critical_cols = [
        "x_ecef", "y_ecef", "z_ecef", "lat_rad", "lon_rad",
        "pca_sensor_1", "pca_sensor_2", "pca_sensor_3",
        "alert", "alert_label", "mmi"
    ]
    for col in critical_cols:
        assert df_train[col].isnull().sum() == 0, f"Unexpected NaN in train: {col}"
        assert df_val[col].isnull().sum() == 0, f"Unexpected NaN in val: {col}"
        assert df_test[col].isnull().sum() == 0, f"Unexpected NaN in test: {col}"

def test_stratification_proportions():
    df_train = pd.read_parquet("data/processed/train_base.parquet")
    df_test = pd.read_parquet("data/processed/test_base.parquet")

    train_pcts = df_train["alert"].value_counts(normalize=True)
    test_pcts = df_test["alert"].value_counts(normalize=True)

    for alert_cat in ["green", "yellow", "orange", "red"]:
        diff = abs(train_pcts.get(alert_cat, 0) - test_pcts.get(alert_cat, 0))
        assert diff < 0.015, f"Stratification mismatch for {alert_cat}: diff={diff:.4f}"

def test_artifacts_loadable():
    preprocessor = joblib.load("module1/artifacts/preprocessor.joblib")
    pca = joblib.load("module1/artifacts/pca_model.joblib")
    
    assert "imputer" in preprocessor
    assert "scaler" in preprocessor
    assert "ohe" in preprocessor
    assert pca.n_components_ == 3
    assert pca.explained_variance_ratio_.sum() > 0.85
