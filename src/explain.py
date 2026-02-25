from __future__ import annotations
from typing import Dict, Any, List, Tuple
import pandas as pd

def _rank_reason_codes(row: pd.Series) -> List[Dict[str, Any]]:
    """
    Generate human-readable reason codes based on component scores (0..1).
    """
    reasons = []
    # Thresholds are configurable; keep simple for MVP.
    if row.get("score_cost", 0) >= 0.75:
        reasons.append({"code": "COST_ADVANTAGE", "detail": "Landed cost is among the lowest options."})
    if row.get("score_service", 0) >= 0.75:
        reasons.append({"code": "SERVICE_STRONG", "detail": "High OTIF and/or short total lead+transit time."})
    if row.get("score_quality", 0) >= 0.75:
        reasons.append({"code": "QUALITY_STRONG", "detail": "Low defects relative to other options."})
    if row.get("score_risk", 0) >= 0.75:
        reasons.append({"code": "RISK_LOW", "detail": "Lower combined cyber/financial/geo risk vs peers."})
    if row.get("score_esg", 0) >= 0.75:
        reasons.append({"code": "ESG_STRONG", "detail": "Higher ESG score vs peers."})
    if row.get("score_diversity", 0) >= 1.0:
        reasons.append({"code": "DIVERSE_SUPPLIER", "detail": "Supplier meets diversity criteria."})

    # If nothing triggers, provide generic reason
    if not reasons:
        reasons.append({"code": "BALANCED_OPTION", "detail": "Balanced trade-offs across cost, service, risk, and quality."})
    return reasons

def build_constraint_trace(candidates: pd.DataFrame) -> Dict[str, Any]:
    """
    Create an audit trace for feasibility filtering.
    Assumes columns: qty, moq, capacity_monthly, feasible_flag
    """
    df = candidates.copy()
    trace = {
        "total_candidates": int(len(df)),
        "feasible_candidates": int((df["feasible_flag"] == 1).sum()),
        "infeasible_candidates": int((df["feasible_flag"] == 0).sum()),
        "infeasible_breakdown": []
    }

    # Diagnose infeasibility causes (simple MVP)
    infeasible = df[df["feasible_flag"] == 0].copy()
    for _, r in infeasible.iterrows():
        causes = []
        if float(r["qty"]) < float(r["moq"]):
            causes.append("MOQ_NOT_MET")
        if float(r["qty"]) > float(r["capacity_monthly"]):
            causes.append("CAPACITY_EXCEEDED")
        trace["infeasible_breakdown"].append({
            "supplier_id": r["supplier_id"],
            "supplier_name": r.get("name", None),
            "causes": causes or ["OTHER"]
        })
    return trace

def add_contributions(scored: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    Adds columns showing contribution of each component:
      contrib_cost = weight_cost * score_cost, etc.
    """
    df = scored.copy()
    df["contrib_cost"] = float(weights["cost"]) * df["score_cost"]
    df["contrib_service"] = float(weights["service"]) * df["score_service"]
    df["contrib_quality"] = float(weights["quality"]) * df["score_quality"]
    df["contrib_risk"] = float(weights["risk"]) * df["score_risk"]
    df["contrib_esg"] = float(weights["esg"]) * df["score_esg"]
    df["contrib_diversity"] = float(weights["diversity"]) * df["score_diversity"]
    df["contrib_total"] = (
        df["contrib_cost"] + df["contrib_service"] + df["contrib_quality"] +
        df["contrib_risk"] + df["contrib_esg"] + df["contrib_diversity"]
    )
    return df

def build_supplier_explanation(row: pd.Series) -> Dict[str, Any]:
    """
    Builds an explainability payload for one supplier candidate.
    """
    return {
        "supplier_id": row["supplier_id"],
        "supplier_name": row.get("name", None),
        "feasible_flag": int(row.get("feasible_flag", 1)),
        "inputs": {
            "est_total_cost_usd": float(row.get("est_total_cost_usd", 0)),
            "unit_price_usd": float(row.get("unit_price_usd", 0)),
            "lane_cost_usd": float(row.get("lane_cost_usd", 0)),
            "lead_time_days": float(row.get("lead_time_days", 0)),
            "transit_days": float(row.get("transit_days", 0)),
            "otif_pct": float(row.get("otif_pct", 0)) if pd.notna(row.get("otif_pct", None)) else None,
            "defect_ppm": float(row.get("defect_ppm", 0)) if pd.notna(row.get("defect_ppm", None)) else None,
            "cyber_risk": float(row.get("cyber_risk", 0)) if pd.notna(row.get("cyber_risk", None)) else None,
            "financial_risk": float(row.get("financial_risk", 0)) if pd.notna(row.get("financial_risk", None)) else None,
            "geo_risk": float(row.get("geo_risk", 0)) if pd.notna(row.get("geo_risk", None)) else None,
            "esg_score": float(row.get("esg_score", 0)) if pd.notna(row.get("esg_score", None)) else None,
            "diversity_flag": bool(row.get("diversity_flag", False)),
        },
        "normalized_scores": {
            "cost": float(row.get("score_cost", 0)),
            "service": float(row.get("score_service", 0)),
            "quality": float(row.get("score_quality", 0)),
            "risk": float(row.get("score_risk", 0)),
            "esg": float(row.get("score_esg", 0)),
            "diversity": float(row.get("score_diversity", 0)),
        },
        "contributions": {
            "cost": float(row.get("contrib_cost", 0)),
            "service": float(row.get("contrib_service", 0)),
            "quality": float(row.get("contrib_quality", 0)),
            "risk": float(row.get("contrib_risk", 0)),
            "esg": float(row.get("contrib_esg", 0)),
            "diversity": float(row.get("contrib_diversity", 0)),
            "total": float(row.get("contrib_total", row.get("score_total", 0))),
        },
        "score_total": float(row.get("score_total", 0)),
        "reason_codes": _rank_reason_codes(row),
    }

def counterfactual_to_beat_winner(
    winner: pd.Series,
    challenger: pd.Series,
    weights: Dict[str, float]
) -> Dict[str, Any]:
    """
    Simple counterfactual: how much would challenger need to improve (by component)
    to exceed winner, holding other components constant?
    We estimate needed delta in the SINGLE best lever component.
    """
    win = float(winner["score_total"])
    ch = float(challenger["score_total"])
    gap = win - ch
    if gap <= 0:
        return {"status": "already_beats_winner", "gap": gap}

    # candidate levers = components where challenger can potentially increase score (max 1.0)
    levers = [
        ("cost", float(challenger["score_cost"]), float(winner["score_cost"]), float(weights["cost"])),
        ("service", float(challenger["score_service"]), float(winner["score_service"]), float(weights["service"])),
        ("quality", float(challenger["score_quality"]), float(winner["score_quality"]), float(weights["quality"])),
        ("risk", float(challenger["score_risk"]), float(winner["score_risk"]), float(weights["risk"])),
        ("esg", float(challenger["score_esg"]), float(winner["score_esg"]), float(weights["esg"])),
        ("diversity", float(challenger["score_diversity"]), float(winner["score_diversity"]), float(weights["diversity"])),
    ]

    # Find lever requiring smallest delta score to close gap: delta = gap / weight
    best = None
    for name, ch_score, win_score, w in levers:
        if w <= 0:
            continue
        max_improve = 1.0 - ch_score
        needed = gap / w
        feasible = needed <= max_improve + 1e-9
        candidate = (needed, feasible, name, ch_score, w)
        if best is None or candidate[0] < best[0]:
            best = candidate

    needed, feasible, lever, ch_score, w = best
    return {
        "status": "feasible_single_lever" if feasible else "requires_multiple_levers",
        "gap_to_winner": gap,
        "best_lever": lever,
        "lever_weight": w,
        "challenger_lever_score": ch_score,
        "required_increase_in_lever_score": needed,
        "note": "This is a simplified counterfactual based on normalized component scores (0..1)."
    }
