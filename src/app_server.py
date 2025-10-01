#!/usr/bin/env python3
# FastAPI app that serves the dashboard UI and JSON APIs for predictions

import sys
import json
from pathlib import Path
from typing import Optional, List

import pandas as pd
from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---- Make the repo root importable, then import config loader ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from src.utils.config import load_config  # expects config/config.yaml
except Exception as e:
    raise RuntimeError(
        "Could not import load_config from src.utils.config. "
        "Ensure you have 'src/utils/config.py' and you're running uvicorn from the project root."
    ) from e

app = FastAPI(title="GigShield Risk Monitor", version="1.0.0")

# ---- CORS (relaxed for local dev) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- Serve static frontend (single-file React) ----
WEB_DIR = PROJECT_ROOT / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ------------------------- Helpers -------------------------
def _predictions_log_path(cfg) -> Path:
    """Return path to the rolling predictions CSV."""
    ing = cfg.get("ingestion", {})
    pdir = Path(ing.get("predictions_dir", "data/predictions"))
    return pdir / "predictions_log.csv"


def _read_log_df(log_path: Path) -> pd.DataFrame:
    """Read predictions log; ensure required columns exist; resilient to older formats."""
    base_cols = [
        "pay_amount", "account_age_days", "response_time_hours", "pay_verified",
        "pred_label", "prob_high_risk", "top3", "source_file"
    ]
    if not log_path.exists():
        return pd.DataFrame(columns=base_cols)

    df = pd.read_csv(log_path)

    # Ensure expected columns exist
    for col in base_cols:
        if col not in df.columns:
            if col == "prob_high_risk":
                df[col] = 0.0
            else:
                df[col] = None

    # Derive prob_high_risk from top3 if missing/zero but top3 present
    def _extract_prob(s):
        if pd.isna(s):
            return 0.0
        try:
            arr = json.loads(s) if isinstance(s, str) else []
            for item in arr:
                if item.get("class") == "high_risk":
                    return float(item.get("prob", 0.0))
        except Exception:
            return 0.0
        return 0.0

    if "prob_high_risk" in df.columns and "top3" in df.columns:
        mask_zero = pd.to_numeric(df["prob_high_risk"], errors="coerce").fillna(0.0) == 0.0
        if mask_zero.any():
            df.loc[mask_zero, "prob_high_risk"] = df.loc[mask_zero, "top3"].apply(_extract_prob)

    # Normalize a couple of types
    if "pred_label" in df.columns:
        df["pred_label"] = df["pred_label"].astype(str)
    if "pay_verified" in df.columns:
        df["pay_verified"] = pd.to_numeric(df["pay_verified"], errors="coerce").fillna(0).astype(int)

    return df


# ------------------------- Routes -------------------------
@app.get("/")
def root():
    """Serve the dashboard page."""
    index = WEB_DIR / "index.html"
    if not index.exists():
        # Friendly message if index.html not created yet
        return JSONResponse(
            status_code=200,
            content={
                "message": "Dashboard not found. Create 'web/index.html' or use the provided template.",
                "hint": "Place your index.html under the 'web/' folder, then refresh.",
            },
        )
    return FileResponse(str(index))


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Serve favicon if present; otherwise return empty to avoid 404 noise."""
    ico = WEB_DIR / "favicon.ico"
    png = WEB_DIR / "favicon.png"
    if ico.exists():
        return FileResponse(str(ico))
    if png.exists():
        return FileResponse(str(png))
    return Response(status_code=204)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/summary")
def summary(limit: int = Query(500, ge=1, le=100000)):
    """Summary KPIs over the most recent rows."""
    cfg = load_config()
    log_path = _predictions_log_path(cfg)
    df = _read_log_df(log_path)

    if df.empty:
        return {
            "total": 0, "by_label": {}, "avg_pay": None, "avg_age": None, "high_risk_prob_70_plus": 0
        }

    df = df.tail(limit)
    by_label = df["pred_label"].value_counts().to_dict()

    avg_pay = round(float(pd.to_numeric(df["pay_amount"], errors="coerce").mean()), 2) if "pay_amount" in df.columns else None
    avg_age = round(float(pd.to_numeric(df["account_age_days"], errors="coerce").mean()), 2) if "account_age_days" in df.columns else None

    hr_over_70 = 0
    if "prob_high_risk" in df.columns:
        try:
            hr_over_70 = int((pd.to_numeric(df["prob_high_risk"], errors="coerce").fillna(0.0) >= 0.70).sum())
        except Exception:
            hr_over_70 = 0

    return {
        "total": int(len(df)),
        "by_label": by_label,
        "avg_pay": avg_pay,
        "avg_age": avg_age,
        "high_risk_prob_70_plus": hr_over_70,
    }


@app.get("/api/predictions")
def predictions(
    limit: int = Query(200, ge=1, le=20000),
    label: Optional[str] = Query(None, pattern="^(safe|caution|high_risk)$"),
    min_prob_high_risk: float = Query(0.0, ge=0.0, le=1.0),
    pay_min: Optional[float] = Query(None, ge=0.0),
    pay_max: Optional[float] = Query(None, ge=0.0),
):
    """Return recent prediction rows with optional filters."""
    cfg = load_config()
    log_path = _predictions_log_path(cfg)
    df = _read_log_df(log_path)
    if df.empty:
        return {"rows": []}

    # Filters
    if label:
        df = df[df["pred_label"] == label]

    if "prob_high_risk" in df.columns and min_prob_high_risk:
        df = df[pd.to_numeric(df["prob_high_risk"], errors="coerce").fillna(0.0) >= float(min_prob_high_risk)]

    if pay_min is not None:
        df = df[pd.to_numeric(df["pay_amount"], errors="coerce").fillna(0.0) >= float(pay_min)]
    if pay_max is not None:
        df = df[pd.to_numeric(df["pay_amount"], errors="coerce").fillna(0.0) <= float(pay_max)]

    # Keep the newest rows
    df = df.tail(limit)

    def _fmt_row(r: pd.Series) -> dict:
        def _float(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return default

        def _int(x, default=0):
            try:
                return int(float(x))
            except Exception:
                return default

        return {
            "pay_amount": _float(r.get("pay_amount")),
            "account_age_days": _int(r.get("account_age_days")),
            "response_time_hours": _float(r.get("response_time_hours")),
            "pay_verified": _int(r.get("pay_verified")),
            "pred_label": str(r.get("pred_label", "")),
            "prob_high_risk": _float(r.get("prob_high_risk")),
            "top3": r.get("top3", None),
            "source_file": r.get("source_file", None),
        }

    rows: List[dict] = [_fmt_row(r) for _, r in df.iterrows()]
    return {"rows": rows}


# Optional: explicit /app route if you prefer /app over /
@app.get("/app")
def app_page():
    index = WEB_DIR / "index.html"
    if not index.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "index.html not found in /web. Create it and refresh."},
        )
    return FileResponse(str(index))
