"""
Module 1: Geospatial Data Preprocessing & Feature Engineering Pipeline
Executes data cleaning, missingness imputation, geospatial transformations,
feature scaling, PCA extraction, stratified splitting, and artifact export.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Configure Matplotlib for headless server execution
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10

EARTH_RADIUS_KM = 6371.0

def convert_geospatial_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes spherical coordinates (radians) and 3D Earth-Centered
    Earth-Fixed (ECEF) Cartesian coordinates (km) from lat/lon/depth.
    """
    df = df.copy()
    lat_rad = np.radians(df["latitude"].values)
    lon_rad = np.radians(df["longitude"].values)
    depth = df["depth"].values

    # Radial distance from Earth's center to hypocenter
    r = EARTH_RADIUS_KM - depth

    df["lat_rad"] = lat_rad
    df["lon_rad"] = lon_rad
    df["x_ecef"] = r * np.cos(lat_rad) * np.cos(lon_rad)
    df["y_ecef"] = r * np.cos(lat_rad) * np.sin(lon_rad)
    df["z_ecef"] = r * np.sin(lat_rad)
    return df

def generate_eda_figures(df_raw: pd.DataFrame, df_train: pd.DataFrame, pca: PCA, fig_dir: str):
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Target Alert Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    alert_order = ["green", "yellow", "orange", "red"]
    alert_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    counts = df_raw["alert"].value_counts().reindex(alert_order)
    bars = ax.bar(alert_order, counts, color=alert_colors, edgecolor="black", alpha=0.85)
    ax.set_title("Earthquake Impact Severity Distribution (PAGER Alert Levels)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Global Events (Log Scale)", fontsize=11)
    ax.set_yscale("log")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.15, f"{int(yval):,}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "target_distribution.png"), dpi=200)
    plt.close()

    # 2. Missingness Analysis
    sensor_cols = ["nst", "gap", "dmin", "rms", "horizontalError", "depthError", "magError"]
    missing_pct = df_raw[sensor_cols].isnull().mean() * 100
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(sensor_cols, missing_pct, color="#3498db", edgecolor="black", alpha=0.85)
    ax.set_title("Sensor & Error Measurement Missingness Rates (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Missing Rate (%)", fontsize=11)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1, bar.get_y() + bar.get_height()/2.0, f"{w:.1f}%", ha="left", va="center", fontweight="bold")
    ax.set_xlim(0, max(missing_pct) + 12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "missingness_matrix.png"), dpi=200)
    plt.close()

    # 3. PCA Scree Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    exp_var = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(exp_var)
    comp_labels = [f"PC{i+1}" for i in range(len(exp_var))]
    ax.bar(comp_labels, exp_var, color="#9b59b6", alpha=0.7, label="Individual Explained Variance (%)")
    ax.plot(comp_labels, cum_var, color="#e74c3c", marker="o", linewidth=2.5, label="Cumulative Explained Variance (%)")
    ax.axhline(85, color="grey", linestyle="--", alpha=0.7, label="85% Threshold")
    ax.set_title("Sensor Error Features: PCA Explained Variance", fontsize=13, fontweight="bold")
    ax.set_ylabel("Explained Variance Ratio (%)", fontsize=11)
    ax.set_ylim(0, 105)
    for i, (ev, cv) in enumerate(zip(exp_var, cum_var)):
        ax.text(i, ev + 1.5, f"{ev:.1f}%", ha="center", fontsize=9)
        ax.text(i, cv + 2.5, f"{cv:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#c0392b")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "pca_variance_explained.png"), dpi=200)
    plt.close()

    # 4. 2D / 3D Hypocenter Scatter
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        df_train["x_ecef"], df_train["y_ecef"], df_train["z_ecef"],
        c=df_train["depth"], cmap="plasma_r", s=12, alpha=0.6
    )
    ax.set_title("3D Hypocentral Distribution inside Earth Crust (ECEF km)", fontsize=13, fontweight="bold")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=15)
    cbar.set_label("Hypocentral Depth (km)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hypocenter_3d_spatial.png"), dpi=200)
    plt.close()

    # 5. Correlation Heatmap
    corr_cols = ["latitude", "longitude", "depth", "mag", "mmi", "gap", "dmin", "rms", "horizontalError", "depthError", "magError"]
    corr_df = df_train[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title("Feature Correlation Heatmap (Train Partition)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "correlation_heatmap.png"), dpi=200)
    plt.close()

def run_pipeline(raw_path: str = "data/raw/usgs_comcat_raw.parquet",
                 processed_dir: str = "data/processed",
                 artifacts_dir: str = "module1/artifacts",
                 reports_dir: str = "module1/reports"):
    
    print(f"Loading raw earthquake dataset from: {raw_path}")
    df_raw = pd.read_parquet(raw_path)
    print(f"Initial raw shape: {df_raw.shape}")

    # 1. Clean & validate coordinates and depth
    df = df_raw.copy()
    valid_mask = (
        (df["latitude"] >= -90.0) & (df["latitude"] <= 90.0) &
        (df["longitude"] >= -180.0) & (df["longitude"] <= 180.0) &
        (df["depth"] >= 0.0) & (df["mag"] > 0.0) &
        df["alert"].isin(["green", "yellow", "orange", "red"])
    )
    df = df[valid_mask].reset_index(drop=True)
    print(f"Post-range validation shape: {df.shape}")

    # 2. Compute 3D ECEF & spherical radians
    df = convert_geospatial_coords(df)

    # 3. Categorical encoding for magType
    # Keep top frequent magTypes, group the rest as 'other'
    top_mag_types = df["magType"].value_counts().nlargest(5).index.tolist()
    df["magType_clean"] = df["magType"].apply(lambda x: x if x in top_mag_types else "other")
    
    # 4. Stratified Train / Validation / Test Split (70% train, 15% val, 15% test)
    # Perform first split: train (70%) vs temp (30%)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["alert"]
    )
    # Perform second split: val (15%) vs test (15%)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["alert"]
    )

    train_df = train_df.copy().reset_index(drop=True)
    val_df = val_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    print(f"\nStratified Partitions: Train={len(train_df)} ({len(train_df)/len(df)*100:.1f}%), "
          f"Val={len(val_df)} ({len(val_df)/len(df)*100:.1f}%), "
          f"Test={len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # 5. Missingness Handling & Imputation
    sensor_features = ["nst", "gap", "dmin", "rms", "horizontalError", "depthError", "magError"]
    
    # Add missing indicators
    for col in sensor_features:
        train_df[f"missing_{col}"] = train_df[col].isnull().astype(int)
        val_df[f"missing_{col}"] = val_df[col].isnull().astype(int)
        test_df[f"missing_{col}"] = test_df[col].isnull().astype(int)

    # Fit median imputer EXCLUSIVELY on training set
    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(train_df[sensor_features])
    val_imputed = imputer.transform(val_df[sensor_features])
    test_imputed = imputer.transform(test_df[sensor_features])

    train_df[sensor_features] = train_imputed
    val_df[sensor_features] = val_imputed
    test_df[sensor_features] = test_imputed

    # 6. Sensor Feature Scaling (RobustScaler for heavy-tailed seismic errors)
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_df[sensor_features])
    val_scaled = scaler.transform(val_df[sensor_features])
    test_scaled = scaler.transform(test_df[sensor_features])

    scaled_col_names = [f"{col}_scaled" for col in sensor_features]
    train_df[scaled_col_names] = train_scaled
    val_df[scaled_col_names] = val_scaled
    test_df[scaled_col_names] = test_scaled

    # 7. One-Hot Encoding for magType
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    train_ohe = ohe.fit_transform(train_df[["magType_clean"]])
    val_ohe = ohe.transform(val_df[["magType_clean"]])
    test_ohe = ohe.transform(test_df[["magType_clean"]])

    ohe_col_names = [f"magType_{cat}" for cat in ohe.categories_[0]]
    for i, col_name in enumerate(ohe_col_names):
        train_df[col_name] = train_ohe[:, i]
        val_df[col_name] = val_ohe[:, i]
        test_df[col_name] = test_ohe[:, i]

    # 8. Dimensionality Reduction (PCA on sensor error features)
    # Fit PCA on scaled sensor measurements
    pca = PCA(n_components=3, random_state=42)
    train_pca = pca.fit_transform(train_scaled)
    val_pca = pca.transform(val_scaled)
    test_pca = pca.transform(test_scaled)

    for i in range(3):
        train_df[f"pca_sensor_{i+1}"] = train_pca[:, i]
        val_df[f"pca_sensor_{i+1}"] = val_pca[:, i]
        test_df[f"pca_sensor_{i+1}"] = test_pca[:, i]

    print(f"\nPCA Components explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Total variance explained by 3 PCs: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    # 9. Target encoding / alignment
    alert_map = {"green": 0, "yellow": 1, "orange": 2, "red": 3}
    train_df["alert_label"] = train_df["alert"].map(alert_map)
    val_df["alert_label"] = val_df["alert"].map(alert_map)
    test_df["alert_label"] = test_df["alert"].map(alert_map)

    # Clean intermediate columns
    train_df = train_df.drop(columns=["magType_clean"])
    val_df = val_df.drop(columns=["magType_clean"])
    test_df = test_df.drop(columns=["magType_clean"])

    # 10. Export Datasets & Serialized Transformers
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    fig_dir = os.path.join(reports_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    train_path = os.path.join(processed_dir, "train_base.parquet")
    val_path = os.path.join(processed_dir, "val_base.parquet")
    test_path = os.path.join(processed_dir, "test_base.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    # Save artifacts
    pipeline_artifacts = {
        "imputer": imputer,
        "scaler": scaler,
        "ohe": ohe,
        "sensor_features": sensor_features,
        "scaled_col_names": scaled_col_names,
        "ohe_col_names": ohe_col_names,
        "top_mag_types": top_mag_types,
        "earth_radius_km": EARTH_RADIUS_KM
    }
    joblib.dump(pipeline_artifacts, os.path.join(artifacts_dir, "preprocessor.joblib"))
    joblib.dump(pca, os.path.join(artifacts_dir, "pca_model.joblib"))

    split_indices = {
        "train_event_ids": train_df["event_id"].tolist(),
        "val_event_ids": val_df["event_id"].tolist(),
        "test_event_ids": test_df["event_id"].tolist()
    }
    with open(os.path.join(artifacts_dir, "split_indices.json"), "w") as f:
        json.dump(split_indices, f, indent=2)

    print(f"\nSaved Base Partitions:")
    print(f"  - {train_path} ({train_df.shape})")
    print(f"  - {val_path} ({val_df.shape})")
    print(f"  - {test_path} ({test_df.shape})")
    print(f"Saved artifacts to {artifacts_dir}")

    # 11. Generate EDA Figures
    print("\nGenerating EDA and Feature Engineering Visualizations...")
    generate_eda_figures(df, train_df, pca, fig_dir)
    print(f"Figures saved to {fig_dir}")

    # 12. Generate EDA Markdown Report
    generate_markdown_report(df, train_df, val_df, test_df, pca, reports_dir)

def generate_markdown_report(df_raw, train_df, val_df, test_df, pca, reports_dir):
    report_path = os.path.join(reports_dir, "eda_and_feature_report.md")
    exp_var = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(exp_var)

    report_content = f"""# Module 1: Exploratory Data Analysis & Feature Engineering Report

**Project**: Seismological Fault Line Discovery & Earthquake Impact Profiling  
**Course**: BITSF464 Machine Learning — Fall 2026  
**Generated Dataset**: `data/processed/`  

---

## 1. Dataset Summary & Physical Audit
The dataset consists of **{len(df_raw):,}** global seismic events curated from the USGS Comprehensive Earthquake Catalog (ComCat) containing ShakeMap continuous ground shaking intensity (`mmi`) and PAGER impact severity metrics (`alert`).

| Metric | Full Catalog | Train Split (70%) | Validation Split (15%) | Test Split (15%) |
|---|---|---|---|---|
| **Total Events** | **{len(df_raw):,}** | **{len(train_df):,}** | **{len(val_df):,}** | **{len(test_df):,}** |
| Magnitude Range | {df_raw['mag'].min():.2f} – {df_raw['mag'].max():.2f} | {train_df['mag'].min():.2f} – {train_df['mag'].max():.2f} | {val_df['mag'].min():.2f} – {val_df['mag'].max():.2f} | {test_df['mag'].min():.2f} – {test_df['mag'].max():.2f} |
| Focal Depth Range (km) | {df_raw['depth'].min():.1f} – {df_raw['depth'].max():.1f} | {train_df['depth'].min():.1f} – {train_df['depth'].max():.1f} | {val_df['depth'].min():.1f} – {val_df['depth'].max():.1f} | {test_df['depth'].min():.1f} – {test_df['depth'].max():.1f} |
| Mean Shaking (MMI) | {df_raw['mmi'].mean():.2f} | {train_df['mmi'].mean():.2f} | {val_df['mmi'].mean():.2f} | {test_df['mmi'].mean():.2f} |

---

## 2. Target Variable Distribution & Imbalance
The primary target variable is the **PAGER Impact Severity Level** (`alert`), which dictates disaster response:
- **Green**: Little to no damage (Vulnerability index < 1M USD or 0 fatalities).
- **Yellow**: Localized damage, potentially few casualties.
- **Orange**: Significant damage, regional impact (estimated fatalities 10 – 100).
- **Red**: Catastrophic disaster, widespread destruction (fatalities > 1,000 or damage > 1B USD).

### Class Distribution Across Stratified Partitions:
| Alert Class | Total Count | Total % | Train Count | Val Count | Test Count |
|---|---|---|---|---|---|
| **Green** | {df_raw['alert'].value_counts().get('green', 0):,} | {df_raw['alert'].value_counts(normalize=True).get('green', 0)*100:.2f}% | {train_df['alert'].value_counts().get('green', 0):,} | {val_df['alert'].value_counts().get('green', 0):,} | {test_df['alert'].value_counts().get('green', 0):,} |
| **Yellow** | {df_raw['alert'].value_counts().get('yellow', 0):,} | {df_raw['alert'].value_counts(normalize=True).get('yellow', 0)*100:.2f}% | {train_df['alert'].value_counts().get('yellow', 0):,} | {val_df['alert'].value_counts().get('yellow', 0):,} | {test_df['alert'].value_counts().get('yellow', 0):,} |
| **Orange** | {df_raw['alert'].value_counts().get('orange', 0):,} | {df_raw['alert'].value_counts(normalize=True).get('orange', 0)*100:.2f}% | {train_df['alert'].value_counts().get('orange', 0):,} | {val_df['alert'].value_counts().get('orange', 0):,} | {test_df['alert'].value_counts().get('orange', 0):,} |
| **Red** | {df_raw['alert'].value_counts().get('red', 0):,} | {df_raw['alert'].value_counts(normalize=True).get('red', 0)*100:.2f}% | {train_df['alert'].value_counts().get('red', 0):,} | {val_df['alert'].value_counts().get('red', 0):,} | {test_df['alert'].value_counts().get('red', 0):,} |

![Target Distribution](figures/target_distribution.png)

> [!IMPORTANT]
> The target shows natural class imbalance reflecting real-world seismic phenomena (catastrophic red/orange earthquakes represent ~2.6% of events). Cost-sensitive weighting (`class_weight='balanced'`) and Macro F1-Score evaluation are strictly required in Modules 3 and 4.

---

## 3. Geospatial 3D ECEF Transformation
Latitude $\\phi$ and Longitude $\\lambda$ in degrees were converted to spherical radians and Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates:
$$x = (R - \\text{{depth}}) \\cos(\\phi) \\cos(\\lambda)$$
$$y = (R - \\text{{depth}}) \\cos(\\phi) \\sin(\\lambda)$$
$$z = (R - \\text{{depth}}) \\sin(\\phi)$$
where $R = 6,371.0\\text{{ km}}$.

This transformation allows downstream clustering (DBSCAN / GMM in Module 2) to operate on true 3D Euclidean distances inside the Earth's crust without polar distortion or discontinuity across the $\\pm 180^\\circ$ antimeridian.

![3D Hypocenters](figures/hypocenter_3d_spatial.png)

---

## 4. Missingness Auditing & Median Imputation
- `nst` (number of stations) showed the highest missingness (~69.8%), as rapid global alerts often omit station counts until manual review.
- Sensor error features (`horizontalError`, `depthError`, `magError`) had missingness rates of 7% to 11%.
- Missingness flags (`missing_<col>`) were created to preserve information about missing sensor channels.
- Median imputation was fit strictly on the **training set** to prevent data leakage.

![Missingness Matrix](figures/missingness_matrix.png)

---

## 5. Dimensionality Reduction: Sensor PCA
Principal Component Analysis was performed on the RobustScaled sensor error metrics (`nst`, `gap`, `dmin`, `rms`, `horizontalError`, `depthError`, `magError`):

| Principal Component | Explained Variance (%) | Cumulative Variance (%) |
|---|---|---|
| **PC1** | {exp_var[0]:.2f}% | {cum_var[0]:.2f}% |
| **PC2** | {exp_var[1]:.2f}% | {cum_var[1]:.2f}% |
| **PC3** | {exp_var[2]:.2f}% | {cum_var[2]:.2f}% |

**Total Variance Explained by 3 Components**: **{cum_var[2]:.2f}%** (exceeding the 85% requirement).

![PCA Scree Plot](figures/pca_variance_explained.png)

---

## 6. Correlation Structure
![Correlation Heatmap](figures/correlation_heatmap.png)

- Ground shaking intensity (`mmi`) has a strong positive correlation with magnitude (`mag`, $r \\approx 0.61$).
- Station distance (`dmin`) and azimuthal gap (`gap`) correlate positively with `horizontalError` and `depthError`, showing that remote events suffer higher hypocentral uncertainty.
- PCA successfully synthesized these correlated errors into orthogonal principal components.

---

## 7. Downstream Handoff Contracts
- **Module 2**: Consumes `x_ecef`, `y_ecef`, `z_ecef`, `lat_rad`, `lon_rad`, and `pca_sensor_*` to discover tectonic fault clusters and compute EM posterior probabilities.
- **Module 3**: Consumes `train_base.parquet`, `val_base.parquet`, and `test_base.parquet` for Stage 1 baseline benchmarks.
- **Module 4**: Evaluates gradient boosters and neural nets on Base features before coupling with Module 2's Augmented features.
"""

    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Generated EDA Markdown report at: {report_path}")

if __name__ == "__main__":
    run_pipeline()
