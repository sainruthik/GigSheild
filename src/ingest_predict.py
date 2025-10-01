#!/usr/bin/env python3
"""
Continuous ingestion & prediction (no Kafka required).

- Polls a folder (data/incoming) for new JSON/CSV files
- Scores them with the saved scaler/encoder/model
- Writes predictions CSVs to data/predictions/
- Moves processed files to data/processed/ (errors -> data/error/)
- (Optional) logs a tiny MLflow run per file with counts & label distribution

Run:
  python src/ingest_predict.py
"""

import argparse
import json
import sys
import time
import shutil
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# make 'src' importable when running as: python src/ingest_predict.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.config import load_config  # noqa: E402

# Optional MLflow logging (respects your config mlflow.enabled + ingestion.mlflow_per_file)
try:
    import mlflow as mlf
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False

# Required raw input fields (same as training)
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
    """Feature engineering identical to preprocessing script."""
    df = df.copy()
    df["paid_ratio"] = df["past_paid_jobs"] / df["past_jobs"].replace(0, 1)
    for c in ["pay_amount", "response_time_hours", "account_age_days"]:
        df[f"log_{c}"] = np.log1p(df[c])
    return df

def _ensure_dirs(*dirs: Path):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def load_artifacts(cfg, model_name: str):
    """Load scaler, label encoder, feature names, and chosen model."""
    adir = Path(cfg["paths"]["preprocess_dir"])
    mdir = Path(cfg["paths"]["models_dir"])

    scaler: StandardScaler = joblib.load(adir / "scaler.pkl")
    le = joblib.load(adir / "label_encoder.pkl")
    feature_names = json.loads((adir / "feature_names.json").read_text(encoding="utf-8"))

    model_path = mdir / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. "
                                f"Train first or set ingestion.model_name to a saved model.")
    model = joblib.load(model_path)
    return scaler, le, feature_names, model

def read_payload(path: Path) -> pd.DataFrame:
    """Read JSON or CSV into a DataFrame with required FIELDS."""
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else [payload]
        df = pd.DataFrame(rows)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    # Validate columns
    missing = [c for c in FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields {missing} in file {path.name}")
    # Keep only expected columns & correct order
    df = df[FIELDS].copy()
    return df

def predict_df(df_raw: pd.DataFrame, scaler, feature_names: List[str], model, le) -> pd.DataFrame:
    """Return predictions (label + probs) aligned with training features."""
    df_feat = add_features(df_raw)
    X = df_feat[feature_names].values
    Xs = scaler.transform(X)

    y_pred = model.predict(Xs)
    labels = le.inverse_transform(y_pred)

    out = pd.DataFrame({"pred_label": labels})

    # Top-3 probabilities (if supported)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xs)
        classes = le.classes_.tolist()
        # also expose probability of 'high_risk' if present
        if "high_risk" in classes:
            idx_hr = classes.index("high_risk")
            out["prob_high_risk"] = np.round(proba[:, idx_hr], 6)
        # pack top3 as JSON string per row
        top3_list = []
        for i in range(proba.shape[0]):
            ranked = sorted(zip(classes, proba[i].tolist()), key=lambda x: x[1], reverse=True)[:3]
            top3_list.append(json.dumps([{"class": c, "prob": round(p, 6)} for c, p in ranked]))
        out["top3"] = top3_list
    return out

def log_to_mlflow(cfg, file_path: Path, preds: pd.DataFrame):
    """Log a small run per processed file (counts + label distribution)."""
    mlcfg = cfg.get("mlflow", {})
    ingest_cfg = cfg.get("ingestion", {})
    if not (mlcfg.get("enabled", True) and ingest_cfg.get("mlflow_per_file", True) and _HAS_MLFLOW):
        return

    # ensure tracking/experiment set
    mlf.set_tracking_uri(mlcfg.get("tracking_uri", "file:artifacts/mlruns"))
    mlf.set_experiment(mlcfg.get("experiment", "gigshield-risk"))
    tags = mlcfg.get("run_tags", {"project": "GigShieldRisk", "mode": "ingest_predict"})

    with mlf.start_run(run_name=f"ingest_{file_path.name}"):
        mlf.set_tags(tags)
        mlf.log_param("file_name", file_path.name)
        mlf.log_param("num_rows", len(preds))
        # label distribution
        counts = preds["pred_label"].value_counts().to_dict()
        for k, v in counts.items():
            mlf.log_metric(f"count_{k}", int(v))

def process_one_file(cfg, scaler, le, feature_names, model, src_path: Path, preds_dir: Path, processed_dir: Path, error_dir: Path, append_log_path: Path):
    """Score one file and move it to processed/error."""
    try:
        df_raw = read_payload(src_path)
        preds = predict_df(df_raw, scaler, feature_names, model, le)

        # merge with some original fields if you want context (e.g., pay_amount/account_age_days)
        summary_cols = ["pay_amount", "account_age_days", "response_time_hours", "pay_verified"]
        keep_cols = [c for c in summary_cols if c in df_raw.columns]
        result = pd.concat([df_raw[keep_cols].reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

        # write per-file predictions
        out_name = src_path.stem + "_preds.csv"
        out_path = preds_dir / out_name
        result.to_csv(out_path, index=False, encoding="utf-8-sig")

        # also append to rolling log
        if append_log_path.exists():
            result.assign(source_file=src_path.name).to_csv(append_log_path, mode="a", index=False, header=False, encoding="utf-8-sig")
        else:
            result.assign(source_file=src_path.name).to_csv(append_log_path, mode="w", index=False, header=True, encoding="utf-8-sig")

        # MLflow logging (optional)
        log_to_mlflow(cfg, src_path, preds)

        # move original to processed
        shutil.move(str(src_path), str(processed_dir / src_path.name))
        print(f"[OK] {src_path.name} -> {out_path.name} ({len(result)} rows)")
    except Exception as e:
        # move to error + write error text
        print(f"[ERR] {src_path.name}: {e}")
        try:
            shutil.move(str(src_path), str(error_dir / src_path.name))
        except Exception:
            pass
        err_txt = error_dir / (src_path.stem + ".error.txt")
        err_txt.write_text(str(e), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Continuous folder polling for ingestion + prediction.")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ing = cfg.get("ingestion", {})

    incoming_dir = Path(ing.get("incoming_dir", "data/incoming"))
    processed_dir = Path(ing.get("processed_dir", "data/processed"))
    error_dir = Path(ing.get("error_dir", "data/error"))
    preds_dir = Path(ing.get("predictions_dir", "data/predictions"))
    poll_s = float(ing.get("poll_interval_seconds", 2))
    model_name = ing.get("model_name", "logreg")
    exts = [x.lower().lstrip(".") for x in ing.get("exts", ["json", "csv"])]

    _ensure_dirs(incoming_dir, processed_dir, error_dir, preds_dir)

    # load scaler/encoder/feature order + model
    scaler, le, feature_names, model = load_artifacts(cfg, model_name=model_name)

    append_log_path = preds_dir / "predictions_log.csv"

    print("Watching:", incoming_dir.resolve())
    print("Model:", model_name, "| Poll every", poll_s, "sec")
    print("Accepting types:", exts)
    while True:
        try:
            # collect all matching files
            files = []
            for ext in exts:
                files.extend(incoming_dir.glob(f"*.{ext}"))
            # process in name order (oldest first would be os.stat; simple is fine)
            for f in sorted(files):
                process_one_file(cfg, scaler, le, feature_names, model, f, preds_dir, processed_dir, error_dir, append_log_path)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as loop_err:
            # never die; just log
            print("[LoopError]", loop_err)
        time.sleep(poll_s)

if __name__ == "__main__":
    main()
