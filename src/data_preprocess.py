#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Make 'src' importable when running as: python src/data_preprocess.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from utils.config import load_config  # noqa: E402

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["paid_ratio"] = df["past_paid_jobs"] / df["past_jobs"].replace(0, 1)
    for c in ["pay_amount", "response_time_hours", "account_age_days"]:
        df[f"log_{c}"] = np.log1p(df[c])
    return df

def main():
    p = argparse.ArgumentParser(description="Preprocess dataset and save arrays/artifacts.")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--csv", default=None, help="override input csv path")
    p.add_argument("--outdir", default=None, help="override artifacts dir")
    p.add_argument("--test-size", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    raw_csv = Path(args.csv or cfg["paths"]["raw_csv"])
    outdir = Path(args.outdir or cfg["paths"]["preprocess_dir"])
    test_size = args.test_size if args.test_size is not None else cfg["preprocess"]["test_size"]
    seed = cfg.get("seed", 42) if args.seed is None else args.seed

    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_csv).drop(columns=["job_id", "client_id"])
    df = add_features(df)

    y = df["risk_label"]
    X = df.drop(columns=["risk_label"])
    feature_names = list(X.columns)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y_enc, test_size=test_size, stratify=y_enc, random_state=seed
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Save arrays
    np.save(outdir / "X_train.npy", X_train_s)
    np.save(outdir / "X_test.npy", X_test_s)
    np.save(outdir / "y_train.npy", y_train)
    np.save(outdir / "y_test.npy", y_test)

    # Save artifacts
    joblib.dump(scaler, outdir / "scaler.pkl")
    joblib.dump(le, outdir / "label_encoder.pkl")
    (outdir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    print(f"Saved arrays & artifacts → {outdir.resolve()}")
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

if __name__ == "__main__":
    main()
