"""
Module 4: Unified end-to-end inference pipeline.

    Raw event dict --(M1 preprocessor)--> Base features --(M2 clusterers)--> Augmented features --(M4 model)--> alert

Every transformation re-uses the *fitted* artifacts saved by Modules 1 and 2
(`module1/artifacts/preprocessor.joblib`, `pca_model.joblib`,
`module2/artifacts/dbscan_model.joblib`, `gmm_model.joblib`), so a raw event goes
through exactly the same maths as the training data did.

The features are computed column-wise with numpy and assembled into a DataFrame once
at the end: adding ~60 columns one at a time to a pandas frame costs far more than
the maths itself for the single-event case (`predict_single`).

This class is in a .py file (not the notebook) so that the exported
`end_to_end_pipeline.joblib` can be un-pickled from anywhere, e.g. from the repo root:

    import joblib
    pipeline = joblib.load("module4/artifacts/end_to_end_pipeline.joblib")
    pipeline.predict_single({...})
"""

import time

import numpy as np
import pandas as pd

ALERT_CLASSES = ["green", "yellow", "orange", "red"]

RAW_FIELDS = ["latitude", "longitude", "depth", "mag", "magType",
              "nst", "gap", "dmin", "rms", "horizontalError", "depthError", "magError"]
NUMERIC_FIELDS = [c for c in RAW_FIELDS if c != "magType"]


def _frame(values: np.ndarray, columns, reference):
    """Give sklearn the same input type it was fitted on (named DataFrame or bare array)."""
    names = getattr(reference, "feature_names_in_", None)
    return pd.DataFrame(values, columns=list(columns)) if names is not None else values


class EarthquakeImpactPipeline:
    def __init__(self, preprocessor, pca, dbscan_model, gmm_model, model, feature_cols,
                 model_name="model", class_names=ALERT_CLASSES):
        self.preprocessor = preprocessor          # dict from module1/artifacts/preprocessor.joblib
        self.pca = pca                            # module1/artifacts/pca_model.joblib
        self.dbscan_model = dbscan_model          # dict from module2/artifacts/dbscan_model.joblib
        self.gmm_model = gmm_model                # dict from module2/artifacts/gmm_model.joblib
        self.model = model                        # fitted Module 4 classifier (predict_proba)
        self.feature_cols = list(feature_cols)    # exact column order the model was trained on
        self.model_name = model_name
        self.class_names = list(class_names)

    # ------------------------------------------------------------------ input handling
    @staticmethod
    def _to_columns(raw) -> dict:
        """dict / list of dicts / DataFrame -> {field: 1-D array}, missing fields as NaN."""
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, pd.DataFrame):
            raw = pd.DataFrame(list(raw))
        n = len(raw)
        cols = {}
        for c in NUMERIC_FIELDS:
            cols[c] = (pd.to_numeric(raw[c], errors="coerce").to_numpy(dtype=float)
                       if c in raw.columns else np.full(n, np.nan))
        mt = raw["magType"] if "magType" in raw.columns else pd.Series(["other"] * n)
        cols["magType"] = np.array(["other" if pd.isna(v) else str(v).strip().lower() for v in mt], dtype=object)
        return cols

    # ------------------------------------------------------------------ Module 1 replica
    def _module1(self, c: dict) -> dict:
        p = self.preprocessor
        lat, lon, depth, mag = c["latitude"], c["longitude"], c["depth"], c["mag"]
        bad = ~((lat >= -90) & (lat <= 90) & (lon >= -180) & (lon <= 180) & (depth >= 0) & (mag > 0))
        if bad.any():
            raise ValueError(f"Invalid coordinates/depth/magnitude in rows {list(np.flatnonzero(bad))}")

        f = {"latitude": lat, "longitude": lon, "depth": depth, "mag": mag}
        # 3D ECEF + spherical radians (same formula and radius as Module 1)
        R = p["earth_radius_km"]
        lat_r, lon_r = np.radians(lat), np.radians(lon)
        r = R - depth
        f["lat_rad"], f["lon_rad"] = lat_r, lon_r
        f["x_ecef"] = r * np.cos(lat_r) * np.cos(lon_r)
        f["y_ecef"] = r * np.cos(lat_r) * np.sin(lon_r)
        f["z_ecef"] = r * np.sin(lat_r)

        # missingness flags -> median imputation -> robust scaling
        sensors = p["sensor_features"]
        S = np.column_stack([c[s] for s in sensors])
        for j, s in enumerate(sensors):
            f[f"missing_{s}"] = np.isnan(S[:, j]).astype(int)
        S_imp = p["imputer"].transform(_frame(S, sensors, p["imputer"]))
        S_scaled = p["scaler"].transform(_frame(S_imp, sensors, p["scaler"]))
        for j, s in enumerate(sensors):
            f[s] = S_imp[:, j]
        for j, s in enumerate(p["scaled_col_names"]):
            f[s] = S_scaled[:, j]

        # magType one-hot (rare types -> 'other', exactly like Module 1)
        mt = np.where(np.isin(c["magType"], p["top_mag_types"]), c["magType"], "other").reshape(-1, 1)
        ohe = p["ohe"].transform(pd.DataFrame(mt, columns=["magType_clean"]))
        for j, name in enumerate(p["ohe_col_names"]):
            f[name] = ohe[:, j]

        # sensor PCA (fit on the scaled sensors in Module 1)
        pcs = self.pca.transform(_frame(S_scaled, p["scaled_col_names"], self.pca))
        for i in range(pcs.shape[1]):
            f[f"pca_sensor_{i + 1}"] = pcs[:, i]
        return f

    # ------------------------------------------------------------------ Module 2 replica
    def _module2(self, f: dict) -> dict:
        db = self.dbscan_model
        X = np.column_stack([f[c] for c in db["feature_cols"]])
        dist, idx = db["core_nn"].kneighbors(X)
        dist_km = dist[:, 0] * db["km_per_unit"]
        labels = np.where(dist_km <= db["eps_km"] + 1e-9, db["core_labels"][idx[:, 0]], -1).astype("int32")
        f["dbscan_cluster"] = labels
        f["is_background_noise"] = (labels == -1).astype("int32")
        f["dist_to_nearest_fault_km"] = dist_km

        g = self.gmm_model
        Zin = np.column_stack([f[c] for c in g["feature_cols"]])
        Z = g["scaler"].transform(_frame(Zin, g["feature_cols"], g["scaler"]))
        proba = g["gmm"].predict_proba(Z)
        f["gmm_cluster"] = proba.argmax(axis=1).astype("int32")
        f["gmm_log_likelihood"] = g["gmm"].score_samples(Z)
        for k in range(proba.shape[1]):
            f[f"gmm_prob_{k}"] = proba[:, k]
        return f

    # ------------------------------------------------------------------ public API
    def transform(self, raw) -> pd.DataFrame:
        """Raw events (dict, list of dicts or DataFrame) -> full Augmented feature frame."""
        return pd.DataFrame(self._module2(self._module1(self._to_columns(raw))))

    def _model_input(self, feats: pd.DataFrame) -> pd.DataFrame:
        return feats[self.feature_cols]

    def predict_proba(self, raw) -> np.ndarray:
        return self.model.predict_proba(self._model_input(self.transform(raw)))

    def predict(self, raw) -> np.ndarray:
        return np.asarray(self.class_names)[self.predict_proba(raw).argmax(axis=1)]

    def predict_single(self, event: dict) -> dict:
        t0 = time.perf_counter()
        feats = self.transform(event)
        proba = self.model.predict_proba(self._model_input(feats))[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        row = feats.iloc[0]
        return {
            "predicted_alert": self.class_names[int(np.argmax(proba))],
            "probabilities": {c: float(p) for c, p in zip(self.class_names, proba)},
            "nearest_fault_distance_km": float(row["dist_to_nearest_fault_km"]),
            "dbscan_cluster": int(row["dbscan_cluster"]),
            "is_background_noise": bool(row["is_background_noise"]),
            "gmm_cluster": int(row["gmm_cluster"]),
            "model": self.model_name,
            "latency_ms": latency_ms,
        }
