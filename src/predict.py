#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# make 'src' importable when running: python src/predict.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from utils.config import load_config  # noqa: E402

FIELDS = [
    "pay_amount",
    "pay_verified",
    "response_time_hours",
    "complaints_count",
    "complaints_payment_related",
    "info_complete",
    "profile_complete",
    "past_jobs",
    "past_paid_jobs",
    "account_age_days",
]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # same feature engineering as preprocess
    df["paid_ratio"] = df["past_paid_jobs"] / df["past_jobs"].replace(0, 1)
    for c in ["pay_amount", "response_time_hours", "account_age_days"]:
        df[f"log_{c}"] = np.log1p(df[c])
    return df

def main():
    p = argparse.ArgumentParser(description="Predict risk label for a job posting JSON.")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--json", required=True, help="Path to input JSON file")
    p.add_argument("--model", choices=["logreg","xgb"], default="logreg")
    args = p.parse_args()

    cfg = load_config(args.config)
    adir = Path(cfg["paths"]["preprocess_dir"])
    mdir = Path(cfg["paths"]["models_dir"])

    # load artifacts
    scaler = joblib.load(adir / "scaler.pkl")
    le = joblib.load(adir / "label_encoder.pkl")
    feature_names_path = adir / "feature_names.json"
    feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))

    # model
    model_path = mdir / f"{args.model}.pkl"
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}. Train first or choose --model logreg/xgb correctly.")
    model = joblib.load(model_path)

    # read input json
    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    # support single dict or list of dicts
    rows = payload if isinstance(payload, list) else [payload]

    # validate fields
    for i, r in enumerate(rows):
        missing = [f for f in FIELDS if f not in r]
        if missing:
            raise SystemExit(f"Row {i} missing fields: {missing}")

    df = pd.DataFrame(rows, columns=FIELDS)

    # add engineered features to match training
    df_feat = add_features(df)
    # ensure order matches training feature_names
    X = df_feat[feature_names].values
    Xs = scaler.transform(X)

    y_pred = model.predict(Xs)
    labels = le.inverse_transform(y_pred)

    # optional: also return probabilities when supported
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xs)
        classes = le.classes_.tolist()

    # print results
    for i, lbl in enumerate(labels):
        out = {"pred_label": lbl}
        if proba is not None:
            # add top-3 classes with probs
            probs = sorted(zip(classes, proba[i].tolist()), key=lambda x: x[1], reverse=True)[:3]
            out["top3"] = [{"class": c, "prob": round(p, 4)} for c, p in probs]
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
