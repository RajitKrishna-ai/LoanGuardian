"""
Model utilities used by the Flask app.

- load_latest_model: loads the most recent model .pkl from deployment/model/
- predict_df: prepares input DataFrame to model features and returns predictions + explain
- retrain_model_from_processed_csvs: simple retraining function using processed CSVs in ../data/processed/
"""

import os
import glob
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Tuple, Any, List, Dict

# Optional: shap for explanation (try/except so  my app still runs if shap is not installed)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(ROOT, "deployment", "model")
DATA_PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODEL_NAME_PREFIX = "loanguardian_model"

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------
# Loading model(s)
# --------------------------
def load_latest_model() -> Tuple[Any, Dict]:
    """
    Load the latest model file in MODEL_DIR based on timestamp in filename.
    Returns (model_obj, metadata_dict) or (None, None) if not found.
    """
    model_files = sorted(glob.glob(os.path.join(MODEL_DIR, f"{MODEL_NAME_PREFIX}_*.pkl")), reverse=True)
    if not model_files:
        return None, None
    latest = model_files[0]
    try:
        model = joblib.load(latest)
        meta = {"path": latest, "loaded_at": datetime.utcnow().isoformat() + "Z"}
        return model, meta
    except Exception as e:
        print("Failed to load model:", e)
        return None, None


# --------------------------
# Prediction wrapper
# --------------------------
def _align_columns(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure dataframe has the same columns as training. If missing columns, create them with 0s.
    If extra columns, drop them.
    NOTE: In production, use a persisted preprocessing pipeline with column ordering.
    """
    # Try to infer required columns from model (if scikit-learn pipeline with named_steps)
    required = None
    # If the model was saved along with feature list, attempt to load it
    # Model might be a dict with {'model':..., 'features':[...]}
    if isinstance(model, dict) and 'model' in model and 'features' in model:
        required = model['features']
        model_obj = model['model']
    else:
        model_obj = model

    if required is None:
        # Best-effort: keep input columns
        return X, model_obj

    # Create missing cols
    for c in required:
        if c not in X.columns:
            X[c] = 0

    # Drop extras
    X = X[required]
    return X, model_obj


def predict_df(model, df_in: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Predict on a small pandas DataFrame.
    Returns (preds, probas, explain_list) where explain_list is None or a list of small dicts.
    """
    # Basic validation
    if df_in.empty:
        raise ValueError("Empty input")

    # Align columns
    X_aligned, model_obj = _align_columns(model, df_in.copy())

    # If the saved model was a mapping with 'model' we used model_obj above
    if isinstance(model_obj, dict) and 'model' in model_obj:
        model_obj = model_obj['model']

    # Predictions
    if hasattr(model_obj, "predict_proba"):
        probas = model_obj.predict_proba(X_aligned)[:, 1]
    else:
        # fallback: decision function or binary predict
        try:
            probas = model_obj.decision_function(X_aligned)
            probas = 1 / (1 + np.exp(-probas))  # sigmoid
        except Exception:
            probas = model_obj.predict(X_aligned).astype(float)

    preds = (probas >= 0.5).astype(int)

    # Explainability (SHAP) - keep it small and readable: top 3 features per record
    explain_list = None
    if SHAP_AVAILABLE:
        try:
            explainer = shap.Explainer(model_obj, X_aligned)
            shap_values = explainer(X_aligned)
            explain_list = []
            for sv, row in zip(shap_values.values, X_aligned.to_dict(orient="records")):
                # get top 3 absolute contributors
                idx = np.argsort(np.abs(sv))[-3:][::-1]
                top_feats = []
                for i in idx:
                    top_feats.append({"feature": X_aligned.columns[i], "shap_value": float(sv[i]), "feature_value": row[X_aligned.columns[i]]})
                explain_list.append(top_feats)
        except Exception as e:
            explain_list = None

    return preds, probas, explain_list


# --------------------------
# Retrain helper (simple)
# --------------------------
def retrain_model_from_processed_csvs(model_type="random_forest") -> Tuple[Any, Dict]:
    """
    Very simple retrain function that:
    - reads X_train_res.csv and y_train_res.csv from data/processed
    - trains a model (RandomForest or XGBoost)
    - saves a timestamped artifact in deployment/model/
    - returns (loaded_model, meta)
    """

    # Locate processed training files
    X_path = os.path.join(DATA_PROCESSED_DIR, "X_train_res.csv")
    y_path = os.path.join(DATA_PROCESSED_DIR, "y_train_res.csv")

    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        raise FileNotFoundError("Processed training files not found. Expected at: " + X_path)

    X_train = pd.read_csv(X_path)
    y_train = pd.read_csv(y_path).squeeze()

    # Simple model choices
    if model_type.lower() in ("xgboost", "xgb"):
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            model.fit(X_train, y_train)
        except Exception as e:
            raise RuntimeError("XGBoost not available or training failed: " + str(e))
    else:
        # default: random forest
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)

    # Save model with metadat
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_filename = f"{MODEL_NAME_PREFIX}_{timestamp}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)

    # Save a dict to preserve feature order
    to_save = {"model": model, "features": list(X_train.columns)}
    joblib.dump(to_save, model_path)

    meta = {"model_path": model_path, "trained_at": timestamp, "model_type": model_type}
    return load_latest_model()  # returns (model, meta) for the file we just saved
