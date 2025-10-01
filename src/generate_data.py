#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys

# Make 'src' importable when running as: python src/generate_data.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from utils.config import load_config  # noqa: E402

def main():
    p = argparse.ArgumentParser(description="Generate synthetic freelance job risk dataset.")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--rows", type=int, default=None, help="override rows")
    p.add_argument("--out", type=str, default=None, help="override output csv path")
    p.add_argument("--seed", type=int, default=None, help="override seed")
    args = p.parse_args()

    cfg = load_config(args.config)
    rows = args.rows if args.rows is not None else cfg["generate"]["rows"]
    out_csv = Path(args.out or cfg["paths"]["raw_csv"])
    seed = cfg.get("seed", 42) if args.seed is None else args.seed

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    N = rows

    # --- generate fields ---
    is_new = rng.random(N) < 0.35
    account_age_days = np.where(is_new, rng.integers(0, 30, size=N), rng.integers(30, 2000, size=N))

    profile_complete_prob = np.clip((account_age_days/2000)*0.7 + 0.2, 0.05, 0.98)
    profile_complete = (rng.random(N) < profile_complete_prob).astype(int)

    pay_verified_prob = np.clip(0.1 + 0.6*profile_complete + 0.0002*account_age_days, 0.05, 0.98)
    pay_verified = (rng.random(N) < pay_verified_prob).astype(int)

    past_jobs_lambda = np.clip(account_age_days/180, 0, 15)
    past_jobs = np.clip(rng.poisson(past_jobs_lambda), 0, 100)

    paid_frac_base = 0.2 + 0.55*pay_verified + 0.25*(account_age_days > 365)
    paid_frac_noise = rng.normal(0, 0.1, size=N)
    paid_frac = np.clip(paid_frac_base + paid_frac_noise, 0, 1)
    past_paid_jobs = np.minimum(past_jobs, (paid_frac * past_jobs).round().astype(int))

    zero_inflate = rng.random(N) < (0.70 - 0.25*pay_verified + 0.1*(account_age_days < 30))
    complaints_poisson = rng.poisson(1 + 2*(1 - pay_verified))
    complaints_count = np.where(zero_inflate, 0, complaints_poisson).astype(int)

    complaints_payment_related = ((complaints_count > 0) &
                                  (rng.random(N) < (0.25 + 0.45*(1 - pay_verified)))).astype(int)

    rt_mu = 2.5 + 0.6*(1 - pay_verified) + 0.3*(account_age_days < 30) + 0.2*(1 - profile_complete)
    rt_sigma = 0.6
    response_time_hours = np.clip(np.exp(rng.normal(rt_mu, rt_sigma, N)), 0.25, 240).round(2)

    info_complete_prob = np.clip(0.3 + 0.4*profile_complete + 0.2*(account_age_days > 60)
                                 - 0.2*(response_time_hours > 72), 0.05, 0.98)
    info_complete = (rng.random(N) < info_complete_prob).astype(int)

    base_pay = rng.normal(500, 200, size=N)
    outliers = rng.random(N) < 0.08
    base_pay[outliers] *= rng.uniform(2, 6, size=outliers.sum())
    pay_amount = np.clip(base_pay, 25, 10000).round(2)

    job_id = np.array([f"J{1000+i}" for i in range(N)])
    client_id = np.array([f"C{1 + rng.integers(1, 1200)}" for _ in range(N)])

    def assign_label(i: int) -> str:
        pct_paid = past_paid_jobs[i] / max(past_jobs[i], 1)
        high_risk = (
            (pay_verified[i] == 0 and pct_paid < 0.5) or
            (complaints_payment_related[i] == 1 and complaints_count[i] >= 2) or
            (info_complete[i] == 0 and response_time_hours[i] > 72) or
            (account_age_days[i] < 14 and profile_complete[i] == 0 and pay_verified[i] == 0)
        )
        safe = (
            (pay_verified[i] == 1) and
            (complaints_count[i] == 0) and
            (profile_complete[i] == 1) and
            (info_complete[i] == 1) and
            (response_time_hours[i] <= 48) and
            ((past_paid_jobs[i] >= 1) or (account_age_days[i] >= 120)) and
            (past_paid_jobs[i] >= 0.6 * past_jobs[i] if past_jobs[i] > 0 else True)
        )
        return "high_risk" if high_risk else ("safe" if safe else "caution")

    risk_label = np.array([assign_label(i) for i in range(N)])

    df = pd.DataFrame({
        "job_id": job_id, "client_id": client_id, "pay_amount": pay_amount,
        "pay_verified": pay_verified, "response_time_hours": response_time_hours,
        "complaints_count": complaints_count, "complaints_payment_related": complaints_payment_related,
        "info_complete": info_complete, "profile_complete": profile_complete,
        "past_jobs": past_jobs, "past_paid_jobs": past_paid_jobs, "account_age_days": account_age_days,
        "risk_label": risk_label
    })

    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} | shape={df.shape}")

if __name__ == "__main__":
    main()
