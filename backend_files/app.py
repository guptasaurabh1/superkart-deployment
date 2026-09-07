"""
SuperKart Sales Prediction - Flask Backend API
------------------------------------------------
Exposes two endpoints:
  POST /v1/predict       -> online (single-record) inference
  POST /v1/predictbatch  -> batch inference from an uploaded CSV file
  GET  /                 -> simple health check

The model is a scikit-learn Pipeline (ColumnTransformer + tuned
RandomForestRegressor) that was trained and serialized in this
notebook as `superkart_model.joblib`.
"""

import io
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# ---------------------------------------------------------------
# App & model initialization
# ---------------------------------------------------------------
superkart_api = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "superkart_model.joblib")
model = joblib.load(MODEL_PATH)

# The exact feature order the pipeline was fitted on.
FEATURE_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


def _prepare_dataframe(records: pd.DataFrame) -> pd.DataFrame:
    """Ensure incoming data has every expected column, in the right order."""
    missing = [c for c in FEATURE_COLUMNS if c not in records.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    return records[FEATURE_COLUMNS]


@superkart_api.get("/")
def health_check():
    """Simple health check so we can confirm the container is alive."""
    return jsonify({"status": "ok", "message": "SuperKart sales prediction API is running."})


@superkart_api.post("/v1/predict")
def predict():
    """
    Online inference: accepts a single JSON record and returns one prediction.
    """
    try:
        payload = request.get_json(force=True)
        input_df = pd.DataFrame([payload])
        input_df = _prepare_dataframe(input_df)
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": round(float(prediction), 2)})
    except Exception as exc:  # noqa: BLE001 - return a clean error to the client
        return jsonify({"error": str(exc)}), 400


@superkart_api.post("/v1/predictbatch")
def predict_batch():
    """
    Batch inference: accepts a CSV file (multipart/form-data, field name
    'file') containing one or more records, and returns predictions for
    every row as a JSON object keyed by row index.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part named 'file' in the request."}), 400

        file = request.files["file"]
        csv_bytes = file.read()
        batch_df = pd.read_csv(io.BytesIO(csv_bytes))
        batch_df = _prepare_dataframe(batch_df)

        predictions = model.predict(batch_df)
        result = pd.Series(np.round(predictions, 2), name="Predicted_Product_Store_Sales_Total")
        return result.to_json(orient="index")
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    # Port 7860 is forwarded publicly from the GitHub Codespace so the
    # notebook can reach this API directly for online/batch inference.
    superkart_api.run(host="0.0.0.0", port=7860)
