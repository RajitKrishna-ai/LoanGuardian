"""
LoanGuardian Flask API (deployment/flask_app/app.py)

Endpoints:
- GET =/health
- POST /predict   (body: JSON single record or list of records)
- POST /retrain   (requir API key in header 'rajit-api-key')

Notes:
- Model artifacts are expected under ../deployment/model/*.pkl
- Training triggers use processed CSVs located at ../data/processed/
- Explainability (SHAP) is optional and only enabled if installed and this model supports it.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, request, jsonify, send_file, abort
import pandas as pd
import numpy as np
import joblib

# Local helper module for model training / preprocessing
from model_utils import (
    load_latest_model,
    predict_df,
    retrain_model_from_processed_csvs,
    MODEL_DIR,
    MODEL_NAME_PREFIX,
)

# --- Configuration (change as needed) -
API_KEY = os.getenv("LOANGUARDIAN_API_KEY", "change_me_locally")  # required for /retrain
MODEL_DIR = MODEL_DIR  # from model_utils
STATIC_LOGO_PATH = "/mnt/data/477df273-4a81-4efa-9923-6077e595c71a.png"  # your uploaded file

# --- Logging --
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("LoanGuardianAPI")

app = Flask(__name__)

# Load model at startup (best-effort)
MODEL, MODEL_META = load_latest_model()
if MODEL is not None:
    logger.info(f"Loaded model: {MODEL_META}")
else:
    logger.warning("No model found at startup. /predict will return error until model is trained or loaded.")


# --------------------------
# Health endpoint
# --------------------------
@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint. Returns service status + model metadata (if loaded).
    """
    status = {
        "service": "LoanGuardian API",
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_loaded": MODEL is not None,
        "model_meta": MODEL_META if MODEL_META is not None else {}
    }
    return jsonify(status), 200


# --------------------------
# Predict endpoint
# --------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict endpoint.
    Accepts JSON with either:
    - a single record: {"LoanAmount": 100000, "Income": 12000, ...}
    - or a list of records: {"data": [ {...}, {...} ]}

    Returns for each input:
    - predicted_class: 0/1
    - predicted_proba: probability for class 1
    - (optional) explain: top shap features when available
    """
    global MODEL
    if MODEL is None:
        return jsonify({"error": "No model loaded. Please /retrain or place a model in deployment/model/"}), 503

    try:
        payload = request.get_json(force=True)
    except Exception as e:
        logger.exception("Failed to parse JSON payload")
        return jsonify({"error": "Invalid JSON payload"}), 400

    # Accept both {"data": [...]} or a direct dict (single record)
    if isinstance(payload, dict) and "data" in payload:
        records = payload["data"]
    elif isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        # single record
        records = [payload]
    else:
        return jsonify({"error": "Payload must be a JSON object or list"}), 400

    # Convert to DataFrame (this is where you must ensure your input features align with training)
    try:
        df_in = pd.DataFrame.from_records(records)
    except Exception as e:
        logger.exception("Failed to convert payload to DataFrame")
        return jsonify({"error": "Input records could not be parsed into a table"}), 400

    # Core predict helper returns (preds, proba, explain_data)
    try:
        preds, probas, explain = predict_df(MODEL, df_in)
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Build response
    results = []
    for i in range(len(df_in)):
        item = {
            "input_index": i,
            "predicted_class": int(preds[i]),
            "predicted_proba": float(probas[i]),
        }
        # include explainability when available (small summary)
        if explain is not None:
            item["explain"] = explain[i]
        results.append(item)

    return jsonify({"results": results, "model_meta": MODEL_META}), 200


# --------------------------
# Retrain endpoint
# --------------------------
@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Retrains a new model from processed CSVs under ../data/processed/.
    Protected by simple API key provided as header 'x-api-key'.
    Note: This is synchronous for demo purposes. In production use a job queue or Airflow.
    """
    global MODEL, MODEL_META

    # API key check
    req_key = request.headers.get("x-api-key", "")
    if req_key != API_KEY:
        return jsonify({"error": "Unauthorized. Provide correct x-api-key header."}), 401

    # Optional JSON params (e.g., model_type)
    params = request.get_json(silent=True) or {}
    model_type = params.get("model_type", "random_forest")  # default

    try:
        # retrain_model_from_processed_csvs returns (model, meta)
        model, meta = retrain_model_from_processed_csvs(model_type=model_type)
        if model is None:
            return jsonify({"error": "Retrain failed, check server logs"}), 500

        # update in-memory model
        MODEL = model
        MODEL_META = meta
        return jsonify({"status": "retrained", "model_meta": meta}), 200

    except Exception as e:
        logger.exception("Retrain failed")
        return jsonify({"error": f"Retrain failed: {str(e)}"}), 500


# --------------------------
# Static file endpoint (serves the uploaded screenshot)
# --------------------------
@app.route("/static/logo", methods=["GET"])
def logo():
    """
    Serve the provided screenshot for README or dashboard preview.
    Uses the development environment absolute path (provided earlier).
    """
    if not os.path.exists(STATIC_LOGO_PATH):
        return jsonify({"error": "Logo not found on server"}), 404
    return send_file(STATIC_LOGO_PATH, mimetype='image/png')


# --------------------------
# Run (for only local development)
# --------------------------
if __name__ == "__main__":
    # Useful for local debugging: set FLASK_ENV=development
    app.run(host="0.0.0.0", port=8000, debug=False)
