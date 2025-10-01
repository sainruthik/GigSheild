#!/usr/bin/env python3
"""
Continuous data feeder for GigShield Risk.

Generates synthetic job-posting rows (matching your training schema) and
drops them into the incoming folder on a schedule.

Run (defaults: JSON, 1 row per file, every 3s):
    python src/feeder.py

Customize:
    python src/feeder.py --mode csv --batch-size 50 --interval 5 --jitter 2 --prefix batch --stop-after 20
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Make 'src' importable when running as: python src/feeder.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
try:
    from utils.config import load_config  # uses config/config.yaml if present
except Exception:
    load_config = None  # allow running without YAML

# Fields expected by your ingest/predict pipeline
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

def gen_one(rng: np.random.Generator) -> dict:
    """Generate one synthetic row consistent with your training distributions."""
    # Account age
    is_new = rng.random() < 0.35
    age = int(rng.integers(0, 30)) if is_new else int(rng.integers(30, 2000))

    # Profile completion
    p_prof = float(np.clip((age / 2000) * 0.7 + 0.2, 0.05, 0.98))
    profile_complete = int(rng.random() < p_prof)

    # Payment verification
    p_payv = float(np.clip(0.1 + 0.6 * profile_complete + 0.0002 * age, 0.05, 0.98))
    pay_verified = int(rng.random() < p_payv)

    # History
    lam_jobs = float(np.clip(age / 180, 0, 15))
    past_jobs = int(min(100, rng.poisson(lam_jobs)))
    paid_frac_base = 0.2 + 0.55 * pay_verified + 0.25 * (age > 365)
    paid_frac = float(np.clip(paid_frac_base + rng.normal(0, 0.1), 0, 1))
    past_paid_jobs = int(min(past_jobs, round(paid_frac * past_jobs)))

    # Complaints
    zero_inflate = rng.random() < (0.70 - 0.25 * pay_verified + 0.1 * (age < 30))
    complaints_count = 0 if zero_inflate else int(rng.poisson(1 + 2 * (1 - pay_verified)))
    complaints_payment_related = int(complaints_count > 0 and rng.random() < (0.25 + 0.45 * (1 - pay_verified)))

    # Response time (hours)
    rt_mu = 2.5 + 0.6 * (1 - pay_verified) + 0.3 * (age < 30) + 0.2 * (1 - profile_complete)
    response_time_hours = float(np.clip(np.exp(rng.normal(rt_mu, 0.6)), 0.25, 240))
    response_time_hours = round(response_time_hours, 2)

    # Info completeness
    p_info = float(np.clip(0.3 + 0.4 * profile_complete + 0.2 * (age > 60) - 0.2 * (response_time_hours > 72), 0.05, 0.98))
    info_complete = int(rng.random() < p_info)

    # Pay
    pay = float(rng.normal(500, 200))
    if rng.random() < 0.08:
        pay *= float(rng.uniform(2, 6))
    pay_amount = float(np.clip(pay, 25, 10000))
    pay_amount = round(pay_amount, 2)

    return {
        "pay_amount": pay_amount,
        "pay_verified": pay_verified,
        "response_time_hours": response_time_hours,
        "complaints_count": complaints_count,
        "complaints_payment_related": complaints_payment_related,
        "info_complete": info_complete,
        "profile_complete": profile_complete,
        "past_jobs": past_jobs,
        "past_paid_jobs": past_paid_jobs,
        "account_age_days": age,
    }

def gen_batch(n: int, rng: np.random.Generator) -> List[dict]:
    return [gen_one(rng) for _ in range(n)]

def atomic_write_bytes(path: Path, data: bytes):
    """Write to a temporary file then rename for atomicity (prevents partial reads)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
    tmp.replace(path)  # atomic on most platforms

def main():
    ap = argparse.ArgumentParser(description="Continuously feed synthetic job rows into data/incoming/")
    ap.add_argument("--config", default="config/config.yaml", help="Path to YAML (optional)")
    ap.add_argument("--outdir", default=None, help="Target incoming dir; defaults to ingestion.incoming_dir")
    ap.add_argument("--mode", choices=["json", "csv"], default="json", help="Output file type")
    ap.add_argument("--batch-size", type=int, default=1, help="Rows per file")
    ap.add_argument("--interval", type=float, default=3.0, help="Seconds between files")
    ap.add_argument("--jitter", type=float, default=1.0, help="Random extra seconds [0..jitter]")
    ap.add_argument("--prefix", default="stream", help="Filename prefix")
    ap.add_argument("--stop-after", type=int, default=None, help="Number of files to produce (None=infinite)")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    args = ap.parse_args()

    # Load config if available
    incoming_default = "data/incoming"
    if load_config:
        try:
            cfg = load_config(args.config)
            incoming_default = cfg.get("ingestion", {}).get("incoming_dir", incoming_default)
        except Exception:
            cfg = {}
    else:
        cfg = {}

    outdir = Path(args.outdir or incoming_default)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    count = 0

    print(f"Feeding to: {outdir.resolve()}")
    print(f"Mode: {args.mode} | batch-size: {args.batch_size} | interval: {args.interval}s (+ <={args.jitter}s jitter)")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            count += 1
            rows = gen_batch(args.batch_size, rng)
            ts = int(time.time())
            if args.mode == "json":
                filename = f"{args.prefix}_{ts}_{count}.json"
                payload = rows[0] if args.batch_size == 1 else rows
                atomic_write_bytes(outdir / filename, json.dumps(payload, indent=2).encode("utf-8"))
            else:
                filename = f"{args.prefix}_{ts}_{count}.csv"
                df = pd.DataFrame(rows, columns=FIELDS)
                # ensure column order
                df = df[FIELDS]
                atomic_write_bytes(outdir / filename, df.to_csv(index=False).encode("utf-8"))

            print(f"[+] wrote {filename}  ({args.batch_size} row{'s' if args.batch_size!=1 else ''})")

            if args.stop_after and count >= args.stop_after:
                print("\nDone — produced", count, "files.")
                break

            time.sleep(args.interval + rng.uniform(0, max(args.jitter, 0.0)))
    except KeyboardInterrupt:
        print("\nStopped by user after", count, "files.")

if __name__ == "__main__":
    main()
