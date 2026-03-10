import numpy as np
import joblib
import os


def load_artifacts(model_name="gaussian", artifacts_dir="data/processed"):
    scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
    pca = joblib.load(os.path.join(artifacts_dir, "pca.pkl"))
    model = joblib.load(os.path.join(artifacts_dir, f"{model_name}_model.pkl"))
    return scaler, pca, model


def predict(input_data, model_name="gaussian", artifacts_dir="data/processed"):
    scaler, pca, model = load_artifacts(model_name, artifacts_dir)
    scaled = scaler.transform(input_data)
    reduced = pca.transform(scaled)
    prediction = model.predict(reduced)
    return prediction


def predict_proba(input_data, model_name="gaussian", artifacts_dir="data/processed"):
    scaler, pca, model = load_artifacts(model_name, artifacts_dir)
    scaled = scaler.transform(input_data)
    reduced = pca.transform(scaled)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(reduced)
    else:
        proba = None
    return proba
