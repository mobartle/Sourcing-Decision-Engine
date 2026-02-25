from __future__ import annotations
import argparse
import pandas as pd
from io_utils import read_csv, read_config, write_csv, write_json
from scoring import Weights, score_candidates
from optimize import optimize_award_min_cost
from explain import (
    build_constraint_trace,
    add_contributions,
    build_supplier_explanation,
    counterfactual_to_beat_winner
)

def build_candidates(request_id: str) -> pd.DataFrame:
    req = read_csv("request_lines.csv")
    si = read_csv("supplier_item.csv")
    perf = read_csv("performance.csv")
    risk = read_csv("risk.csv")
    logi = read_csv("logistics.csv")
    sup = read_csv("suppliers.csv")

    req = req[req["request_id"] == request_id].copy()
    if req.empty:
        return pd.DataFrame()

    df = (req.merge(si, on="item_id", how="left")
            .merge(perf, on=["supplier_id","item_id"], how="left")
            .merge(risk, on="supplier_id", how="left")
            .merge(logi, on=["supplier_id","ship_to"], how="left")
            .merge(sup, on="supplier_id", how="left"))

    # Feasibility checks (MVP)
    df["feasible_flag"] = 1
    df.loc[df["qty"] < df["moq"], "feasible_flag"] = 0
    df.loc[df["qty"] > df["capacity_monthly"], "feasible_flag"] = 0

    df["lane_cost_usd"] = df["lane_cost_usd"].fillna(0)
    df["transit_days"] = df["transit_days"].fillna(0)

    df["est_total_cost_usd"] = (df["unit_price_usd"] * df["qty"]) + df["lane_cost_usd"]

    df["diversity_flag"] = df["diversity_flag"].astype(str).str.lower().isin(["true","1","yes"])
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request_id", required=True)
    p.add_argument("--optimize", default="false")
    args = p.parse_args()

    optimize_flag = str(args.optimize).lower() in ["true", "1", "yes"]

    config = read_config()
    w_cfg = config.get("weights", {})
    constraints = config.get("constraints", {})
    max_suppliers = int(constraints.get("max_suppliers", 2))

    weights_dict = {
        "cost": float(w_cfg.get("cost", 0.40)),
        "service": float(w_cfg.get("service", 0.20)),
        "quality": float(w_cfg.get("quality", 0.10)),
        "risk": float(w_cfg.get("risk", 0.20)),
        "esg": float(w_cfg.get("esg", 0.05)),
        "diversity": float(w_cfg.get("diversity", 0.05)),
    }

    w = Weights(**weights_dict)

    candidates = build_candidates(args.request_id)
    if candidates.empty:
        write_json({"request_id": args.request_id, "message": "No candidates found"}, "recommendations.json")
        return

    # Decision trace: feasibility filtering audit
    decision_trace = {
        "request_id": args.request_id,
        "constraints": {"max_suppliers": max_suppliers},
        "feasibility_trace": build_constraint_trace(candidates)
    }
    write_json(decision_trace, "decision_trace.json")

    scored = score_candidates(candidates, w)
    if scored.empty:
        write_json({"request_id": args.request_id, "message": "No feasible candidates"}, "recommendations.json")
        write_json({"request_id": args.request_id, "message": "No feasible candidates"}, "explainability.json")
        return

    # Add contribution columns for explainability
    scored = add_contributions(scored, weights_dict)

    # Save scored candidates
    export_cols = [
        "request_id","line_id","item_id","qty","ship_to",
        "supplier_id","name",
        "unit_price_usd","lane_cost_usd","est_total_cost_usd",
        "otif_pct","defect_ppm","cyber_risk","financial_risk","geo_risk",
        "esg_score","diversity_flag",
        "score_total","score_cost","score_service","score_quality","score_risk","score_esg","score_diversity",
        "contrib_cost","contrib_service","contrib_quality","contrib_risk","contrib_esg","contrib_diversity","contrib_total"
    ]
    write_csv(scored[export_cols].copy(), "scored_candidates.csv")

    # Top-N recommendations per line
    recs = []
    for (rid, line_id), grp in scored.groupby(["request_id","line_id"]):
        top = grp.head(3)
        recs.append({
            "request_id": rid,
            "line_id": int(line_id),
            "recommendations": [
                {
                    "supplier_id": r["supplier_id"],
                    "supplier_name": r["name"],
                    "score_total": float(r["score_total"]),
                    "est_total_cost_usd": float(r["est_total_cost_usd"]),
                    "reason_codes": build_supplier_explanation(r)["reason_codes"],
                    "contributions": build_supplier_explanation(r)["contributions"]
                }
                for _, r in top.iterrows()
            ]
        })
    write_json({"request_id": args.request_id, "results": recs}, "recommendations.json")

    # Explainability output: full explanation pack per line
    explain_pack = {"request_id": args.request_id, "lines": []}
    for (rid, line_id), grp in scored.groupby(["request_id","line_id"]):
        grp = grp.reset_index(drop=True)
        winner = grp.iloc[0]  # top by score_total
        suppliers = [build_supplier_explanation(r) for _, r in grp.iterrows()]

        # counterfactuals vs winner (for top challengers)
        counterfactuals = []
        for i in range(1, min(4, len(grp))):  # top 3 challengers
            ch = grp.iloc[i]
            counterfactuals.append({
                "challenger_supplier_id": ch["supplier_id"],
                "result": counterfactual_to_beat_winner(winner, ch, weights_dict)
            })

        explain_pack["lines"].append({
            "line_id": int(line_id),
            "winner_by_score": {
                "supplier_id": winner["supplier_id"],
                "supplier_name": winner.get("name", None),
                "score_total": float(winner["score_total"]),
                "reasons": build_supplier_explanation(winner)["reason_codes"]
            },
            "suppliers": suppliers,
            "counterfactuals_vs_winner": counterfactuals,
            "weights_used": weights_dict
        })

    write_json(explain_pack, "explainability.json")

    # Optional optimization: min-cost award per line
    if optimize_flag:
        plans = []
        for (rid, line_id), grp in scored.groupby(["request_id","line_id"]):
            award_df = optimize_award_min_cost(grp, max_suppliers=max_suppliers)
            total_cost = 0.0
            award_rows = []
            for _, a in award_df.iterrows():
                s = a["supplier_id"]
                aq = float(a["award_qty"])
                up = float(grp.loc[grp.supplier_id == s, "unit_price_usd"].iloc[0])
                total_cost += aq * up
                award_rows.append({"supplier_id": s, "award_qty": aq, "unit_price_usd": up})
            plans.append({
                "request_id": rid,
                "line_id": int(line_id),
                "objective": "minimize_unit_cost",
                "max_suppliers": max_suppliers,
                "award_plan": award_rows,
                "purchase_cost_usd": total_cost
            })
        write_json({"request_id": args.request_id, "plans": plans}, "award_plan.json")

if __name__ == "__main__":
    main()
