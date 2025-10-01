#!/usr/bin/env python3
"""
Hyperparameter search with MLflow logging.

Outputs:
- artifacts/models/best_params.yaml
- artifacts/models/search_report_logreg.json
- artifacts/models/search_report_xgb.json (if enabled)
- MLflow runs for each model's grid search (best CV score + test metrics)
"""
import argparse, sys, json
from pathlib import Path
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Make 'src' importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from utils.config import load_config  # noqa: E402

import mlflow
import mlflow.sklearn
import yaml

def setup_mlflow(cfg):
    ml = cfg.get("mlflow", {})
    mlflow.set_tracking_uri(ml.get("tracking_uri", "file:artifacts/mlruns"))
    mlflow.set_experiment(ml.get("experiment", "gigshield-risk"))
    return ml

def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def save_yaml(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

def main():
    p = argparse.ArgumentParser(description="Grid search models and log to MLflow.")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--artifacts", default=None)
    p.add_argument("--models-dir", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    adir = Path(args.artifacts or cfg["paths"]["preprocess_dir"])
    mdir = Path(args.models_dir or cfg["paths"]["models_dir"]); mdir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(adir / "X_train.npy")
    X_test  = np.load(adir / "X_test.npy")
    y_train = np.load(adir / "y_train.npy")
    y_test  = np.load(adir / "y_test.npy")
    le = joblib.load(adir / "label_encoder.pkl")
    classes = le.classes_.tolist()

    ml_cfg = setup_mlflow(cfg)
    tags = ml_cfg.get("run_tags", {"project": "GigShieldRisk"})
    scoring = cfg["search"].get("scoring", "f1_macro")
    cv = cfg["search"].get("cv", 3)
    n_jobs = cfg["search"].get("n_jobs", -1)

    best_params = {}

    # ---- Logistic Regression ----
    with mlflow.start_run(run_name="gridsearch_logreg"):
        mlflow.set_tags(tags)
        param_grid_lr = cfg["search"].get("logistic", {})
        mlflow.log_params({"search_model": "logreg", "cv": cv, "scoring": scoring, "n_jobs": n_jobs})
        mlflow.log_param("grid_size", sum(len(v) for v in param_grid_lr.values()))

        lr = LogisticRegression(max_iter=cfg["model"]["logreg"]["max_iter"], multi_class="multinomial")
        gs = GridSearchCV(lr, param_grid_lr, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=1, refit=True)
        gs.fit(X_train, y_train)

        best_params["logistic"] = gs.best_params_
        mlflow.log_params({f"best_{k}": v for k, v in gs.best_params_.items()})
        mlflow.log_metric("best_cv_score", gs.best_score_)

        # evaluate on test set
        y_pred = gs.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        cr = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        mlflow.log_metrics({"test_accuracy": acc, "test_f1_macro": f1m})
        save_json(mdir / "search_report_logreg.json", {
            "best_params": gs.best_params_, "best_cv_score": gs.best_score_,
            "classification_report": cr, "confusion_matrix": cm, "classes": classes
        })
        mlflow.log_artifact(str(mdir / "search_report_logreg.json"), artifact_path="reports")

        # also drop full cv_results_
        import pandas as pd
        pd.DataFrame(gs.cv_results_).to_csv(mdir / "logreg_cv_results.csv", index=False)
        mlflow.log_artifact(str(mdir / "logreg_cv_results.csv"), artifact_path="reports")

    # ---- XGBoost (optional) ----
    if cfg["model"].get("use_xgboost", True):
        try:
            from xgboost import XGBClassifier
            with mlflow.start_run(run_name="gridsearch_xgboost"):
                mlflow.set_tags(tags)
                param_grid_xgb = cfg["search"].get("xgb", {})
                mlflow.log_params({"search_model": "xgb", "cv": cv, "scoring": scoring, "n_jobs": n_jobs})
                mlflow.log_param("grid_size", sum(len(v) for v in param_grid_xgb.values()))

                xgb = XGBClassifier(
                    random_state=cfg.get("seed", 42),
                    eval_metric=cfg["model"]["xgb"].get("eval_metric", "mlogloss"),
                    n_jobs=n_jobs,
                )
                gsx = GridSearchCV(xgb, param_grid_xgb, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=1, refit=True)
                gsx.fit(X_train, y_train)

                best_params["xgb"] = gsx.best_params_
                mlflow.log_params({f"best_{k}": v for k, v in gsx.best_params_.items()})
                mlflow.log_metric("best_cv_score", gsx.best_score_)

                ypx = gsx.predict(X_test)
                accx = accuracy_score(y_test, ypx)
                f1x = f1_score(y_test, ypx, average="macro")
                crx = classification_report(y_test, ypx, target_names=classes, output_dict=True)
                cmx = confusion_matrix(y_test, ypx).tolist()

                mlflow.log_metrics({"test_accuracy": accx, "test_f1_macro": f1x})
                save_json(mdir / "search_report_xgb.json", {
                    "best_params": gsx.best_params_, "best_cv_score": gsx.best_score_,
                    "classification_report": crx, "confusion_matrix": cmx, "classes": classes
                })
                mlflow.log_artifact(str(mdir / "search_report_xgb.json"), artifact_path="reports")

                import pandas as pd
                pd.DataFrame(gsx.cv_results_).to_csv(mdir / "xgb_cv_results.csv", index=False)
                mlflow.log_artifact(str(mdir / "xgb_cv_results.csv"), artifact_path="reports")
        except ImportError:
            print("\n[Info] XGBoost not installed; skipping XGB search (pip install xgboost).")

    # save consolidated best params
    save_yaml(mdir / "best_params.yaml", best_params)
    print(f"\nSaved best params → {mdir / 'best_params.yaml'}")

if __name__ == "__main__":
    main()
