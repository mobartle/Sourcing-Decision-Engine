from __future__ import annotations
import argparse
import pandas as pd
from io_utils import read_csv, read_config, write_csv, write_json
from scoring import Weights, score_candidates
from optimize import optimize_award_min_cost

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

    # Feasibility checks
    # Note: required_date logic omitted for simplicity (would require today's date comparison)
    df["feasible_flag"] = 1
    df.loc[df["qty"] < df["moq"], "feasible_flag"] = 0
    df.loc[df["qty"] > df["capacity_monthly"], "feasible_flag"] = 0

    # Total cost (estimate)
    df["lane_cost_usd"] = df["lane_cost_usd"].fillna(0)
    df["transit_days"] = df["transit_days"].fillna(0)

    df["est_total_cost_usd"] = (df["unit_price_usd"] * df["qty"]) + df["lane_cost_usd"]

    # Clean types
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

    w = Weights(
        cost=float(w_cfg.get("cost", 0.40)),
        service=float(w_cfg.get("service", 0.20)),
        quality=float(w_cfg.get("quality", 0.10)),
        risk=float(w_cfg.get("risk", 0.20)),
        esg=float(w_cfg.get("esg", 0.05)),
        diversity=float(w_cfg.get("diversity", 0.05)),
    )

    candidates = build_candidates(args.request_id)
    if candidates.empty:
        write_json({"request_id": args.request_id, "message": "No candidates found"}, "recommendations.json")
        return

    scored = score_candidates(candidates, w)
    if scored.empty:
        write_json({"request_id": args.request_id, "message": "No feasible candidates"}, "recommendations.json")
        return

    # Save scored candidates
    export_cols = [
        "request_id","line_id","item_id","qty","ship_to",
        "supplier_id","name",
        "unit_price_usd","lane_cost_usd","est_total_cost_usd",
        "otif_pct","defect_ppm","cyber_risk","financial_risk","geo_risk",
        "esg_score","diversity_flag",
        "score_total","score_cost","score_service","score_quality","score_risk","score_esg","score_diversity"
    ]
    scored_out = scored[export_cols].copy()
    write_csv(scored_out, "scored_candidates.csv")

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
                    "rationale": r["rationale_json"]
                }
                for _, r in top.iterrows()
            ]
        })
    write_json({"request_id": args.request_id, "results": recs}, "recommendations.json")

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
