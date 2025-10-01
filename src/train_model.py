#!/usr/bin/env python3
import argparse, sys, json
from pathlib import Path
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Make 'src' importable when running as: python src/train_model.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from utils.config import load_config  # noqa: E402

# --- MLflow & plotting ---
import mlflow as mlf
import mlflow.sklearn
import matplotlib.pyplot as plt

def save_confusion_png(cm: np.ndarray, labels: list[str], out_path: Path):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def setup_mlflow(cfg):
    ml_cfg = cfg.get("mlflow", {})
    if ml_cfg.get("enabled", True):
        mlf.set_tracking_uri(ml_cfg.get("tracking_uri", "file:artifacts/mlruns"))
        mlf.set_experiment(ml_cfg.get("experiment", "gigshield-risk"))
    return ml_cfg

def main():
    p = argparse.ArgumentParser(description="Train models on preprocessed arrays (with MLflow logging).")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--artifacts", default=None, help="override preprocessing dir")
    p.add_argument("--out", default=None, help="override models dir")
    p.add_argument("--use-xgboost", default=None, help="true/false (override config)")
    args = p.parse_args()

    cfg = load_config(args.config)
    adir = Path(args.artifacts or cfg["paths"]["preprocess_dir"])
    mdir = Path(args.out or cfg["paths"]["models_dir"])
    mdir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(adir / "X_train.npy")
    X_test  = np.load(adir / "X_test.npy")
    y_train = np.load(adir / "y_train.npy")
    y_test  = np.load(adir / "y_test.npy")
    le = joblib.load(adir / "label_encoder.pkl")
    class_names = le.classes_.tolist()

    # --- MLflow setup ---
    ml_cfg = setup_mlflow(cfg)
    tags = ml_cfg.get("run_tags", {"project": "GigShieldRisk"})
    ml_enabled = ml_cfg.get("enabled", True)

    # =============== Logistic Regression ===============
    max_iter = cfg["model"]["logreg"]["max_iter"]
    clf = LogisticRegression(max_iter=max_iter, multi_class="multinomial")

    if ml_enabled:
        run_ctx = mlf.start_run(run_name="logreg")
        mlf.set_tags(tags)
        mlf.log_params({"model": "logreg", "max_iter": max_iter})
    else:
        run_ctx = None

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    cr = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Logistic Regression ===")
    print(json.dumps(cr, indent=2))
    print("Confusion Matrix:\n", cm)

    if ml_enabled:
        mlf.log_metrics({"accuracy": acc, "f1_macro": f1_macro})
        for cls in class_names:
            mlf.log_metric(f"f1_{cls}", cr[cls]["f1-score"])
        cm_png = mdir / "logreg_confusion.png"
        save_confusion_png(cm, class_names, cm_png)
        mlf.log_artifact(str(cm_png), artifact_path="plots")
        (mdir / "logreg_classification_report.json").write_text(json.dumps(cr, indent=2), encoding="utf-8")
        mlf.log_artifact(str(mdir / "logreg_classification_report.json"), artifact_path="reports")
        mlf.sklearn.log_model(clf, artifact_path="model", registered_model_name=None)

    joblib.dump(clf, mdir / "logreg.pkl")
    print(f"Saved model → {mdir / 'logreg.pkl'}")

    if run_ctx is not None:
        mlf.end_run()

    # =============== XGBoost (optional) ===============
    use_xgb = cfg["model"]["use_xgboost"] if args.use_xgboost is None else (str(args.use_xgboost).lower() == "true")
    if use_xgb:
        try:
            from xgboost import XGBClassifier
            xgb_cfg = cfg["model"]["xgb"]
            xgb = XGBClassifier(
                n_estimators=xgb_cfg["n_estimators"],
                learning_rate=xgb_cfg["learning_rate"],
                max_depth=xgb_cfg["max_depth"],
                subsample=xgb_cfg["subsample"],
                colsample_bytree=xgb_cfg["colsample_bytree"],
                random_state=cfg.get("seed", 42),
                eval_metric=xgb_cfg.get("eval_metric", "mlogloss"),
            )

            if ml_enabled:
                run_ctx = mlf.start_run(run_name="xgboost")
                mlf.set_tags(tags)
                mlf.log_params({"model": "xgboost", **xgb_cfg})
            else:
                run_ctx = None

            xgb.fit(X_train, y_train)
            y_pred_xgb = xgb.predict(X_test)

            acc_xgb = accuracy_score(y_test, y_pred_xgb)
            f1_xgb = f1_score(y_test, y_pred_xgb, average="macro")
            cr_xgb = classification_report(y_test, y_pred_xgb, target_names=class_names, output_dict=True)
            cm_xgb = confusion_matrix(y_test, y_pred_xgb)

            print("\n=== XGBoost ===")
            print(json.dumps(cr_xgb, indent=2))
            print("Confusion Matrix:\n", cm_xgb)

            if ml_enabled:
                mlf.log_metrics({"accuracy": acc_xgb, "f1_macro": f1_xgb})
                for cls in class_names:
                    mlf.log_metric(f"f1_{cls}", cr_xgb[cls]["f1-score"])
                cm_png_x = mdir / "xgb_confusion.png"
                save_confusion_png(cm_xgb, class_names, cm_png_x)
                mlf.log_artifact(str(cm_png_x), artifact_path="plots")
                (mdir / "xgb_classification_report.json").write_text(json.dumps(cr_xgb, indent=2), encoding="utf-8")
                mlf.log_artifact(str(mdir / "xgb_classification_report.json"), artifact_path="reports")
                import mlflow.xgboost
                mlflow_xgb = mlflow.xgboost if hasattr(mlflow, "xgboost") else None  # safety
                if mlflow_xgb:
                    mlflow_xgb.log_model(xgb, artifact_path="model", registered_model_name=None)

            joblib.dump(xgb, mdir / "xgb.pkl")
            print(f"Saved model → {mdir / 'xgb.pkl'}")

            if run_ctx is not None:
                mlf.end_run()
        except ImportError:
            print("\n[Info] XGBoost not installed; skipping (pip install xgboost).")

if __name__ == "__main__":
    main()
