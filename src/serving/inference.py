"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load HF-hosted ML model and feature metadata
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully
"""

import os
import pandas as pd
import mlflow
from huggingface_hub import hf_hub_download

# === MODEL LOADING CONFIGURATION ===
HF_MODEL_REPO = "vaishnavibitla/telco-churn-model"  # Hugging Face model repo
PIPELINE_FILE = "pipeline.pkl"  # file name in the HF repo

try:
    # Download pipeline.pkl from HF Hub
    local_model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=PIPELINE_FILE)
    # Load MLflow pyfunc model from downloaded path
    model = mlflow.pyfunc.load_model(local_model_path)
    print(f"✅ Model loaded successfully from Hugging Face Hub: {local_model_path}")
except Exception as e:
    raise Exception(f"❌ Failed to download or load model from HF Hub: {e}")

# === FEATURE SCHEMA LOADING ===
# Load feature columns file from same HF repo
try:
    feature_file_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="feature_columns.txt")
    with open(feature_file_path, "r") as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from HF Hub")
except Exception as e:
    raise Exception(f"❌ Failed to load feature columns from HF Hub: {e}")

# === FEATURE TRANSFORMATION CONSTANTS ===
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Numeric type coercion
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Binary feature encoding
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().map(mapping).fillna(0).astype(int)

    # One-hot encoding for remaining categorical features
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Boolean to integer conversion
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Feature alignment
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df

def predict(input_dict: dict) -> str:
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        preds = model.predict(df_enc)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        result = preds[0] if isinstance(preds, (list, tuple)) else preds
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    return "Likely to churn" if result == 1 else "Not likely to churn"