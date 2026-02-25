from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Weights:
    cost: float = 0.40
    service: float = 0.20
    quality: float = 0.10
    risk: float = 0.20
    esg: float = 0.05
    diversity: float = 0.05

def minmax(series: pd.Series, higher_is_better: bool) -> pd.Series:
    x = series.astype(float).replace([np.inf, -np.inf], np.nan)
    if x.isna().all():
        return pd.Series(0.5, index=series.index)
    x = x.fillna(x.median())
    lo, hi = float(x.min()), float(x.max())
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    scaled = (x - lo) / (hi - lo)
    return scaled if higher_is_better else (1 - scaled)

def score_candidates(df: pd.DataFrame, w: Weights) -> pd.DataFrame:
    df = df.copy()
    df = df[df["feasible_flag"] == 1].reset_index(drop=True)
    if df.empty:
        return df

    # Component scores (0..1)
    df["score_cost"] = minmax(df["est_total_cost_usd"], higher_is_better=False)

    # Service: blend OTIF (higher better) and total time (lower better)
    total_time = df["lead_time_days"] + df["transit_days"]
    df["score_service"] = 0.6 * minmax(df["otif_pct"], True) + 0.4 * minmax(total_time, False)

    df["score_quality"] = minmax(df["defect_ppm"], higher_is_better=False)

    # Risk: average subrisks (assume 0..100, higher = worse)
    risk_raw = df[["cyber_risk", "financial_risk", "geo_risk"]].astype(float).fillna(50).mean(axis=1)
    df["score_risk"] = minmax(risk_raw, higher_is_better=False)

    df["score_esg"] = minmax(df["esg_score"].fillna(50), higher_is_better=True)
    df["score_diversity"] = df["diversity_flag"].fillna(False).astype(bool).astype(int)

    df["score_total"] = (
        w.cost * df["score_cost"] +
        w.service * df["score_service"] +
        w.quality * df["score_quality"] +
        w.risk * df["score_risk"] +
        w.esg * df["score_esg"] +
        w.diversity * df["score_diversity"]
    )

    # Explainability: keep component breakdown
    df["rationale_json"] = df.apply(lambda r: {
        "cost_usd": float(r["est_total_cost_usd"]),
        "components": {
            "cost": float(r["score_cost"]),
            "service": float(r["score_service"]),
            "quality": float(r["score_quality"]),
            "risk": float(r["score_risk"]),
            "esg": float(r["score_esg"]),
            "diversity": float(r["score_diversity"])
        }
    }, axis=1)

    return df.sort_values(["request_id", "line_id", "score_total"], ascending=[True, True, False])
