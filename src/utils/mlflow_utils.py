from pathlib import Path
import json
import mlflow
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def setup_mlflow(cfg):
    """Configure MLflow tracking from YAML config; returns mlflow or None if disabled."""
    mcfg = cfg.get("mlflow", {})
    if not mcfg.get("enabled", True):
        return None
    mlflow.set_tracking_uri(mcfg.get("tracking_uri", "file:artifacts/mlruns"))
    mlflow.set_experiment(mcfg.get("experiment_name", "default"))
    return mlflow

def log_dict_as_artifact(d: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(out_path))

def log_classification_metrics(y_true, y_pred, class_names, prefix=""):
    """Log per-class and aggregate metrics to MLflow."""
    cr = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    # Aggregate metrics
    for agg_key in ["macro avg", "weighted avg"]:
        for metric in ["precision", "recall", "f1-score"]:
            mlflow.log_metric(f"{prefix}{agg_key.replace(' ', '_')}_{metric}", cr[agg_key][metric])
    # Per-class F1 (optional: precision/recall too)
    for cname in class_names:
        mlflow.log_metric(f"{prefix}f1_{cname}", cr[cname]["f1-score"])
    return cr

def log_confusion_matrix(y_true, y_pred, class_names, out_png: Path, prefix=""):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(out_png))
    # Also log raw counts
    mlflow.log_dict({"classes": list(class_names), "matrix": cm.tolist()}, f"{prefix}confusion_matrix.json")
