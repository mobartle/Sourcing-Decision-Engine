from __future__ import annotations
import pandas as pd
import pulp

def optimize_award_min_cost(df_line: pd.DataFrame, max_suppliers: int = 2) -> pd.DataFrame:
    """
    Min-cost award subject to:
      - satisfy full quantity
      - supplier capacity
      - max number of suppliers used
    """
    df_line = df_line.copy()
    if df_line.empty:
        return pd.DataFrame()

    qty = float(df_line["qty"].iloc[0])

    suppliers = df_line["supplier_id"].tolist()
    unit_price = {s: float(df_line.loc[df_line.supplier_id == s, "unit_price_usd"].iloc[0]) for s in suppliers}
    cap = {s: float(df_line.loc[df_line.supplier_id == s, "capacity_monthly"].iloc[0]) for s in suppliers}

    x = pulp.LpVariable.dicts("award_qty", suppliers, lowBound=0, cat="Continuous")
    y = pulp.LpVariable.dicts("used", suppliers, lowBound=0, upBound=1, cat="Binary")

    model = pulp.LpProblem("sourcing_award", pulp.LpMinimize)
    model += pulp.lpSum(unit_price[s] * x[s] for s in suppliers)

    model += pulp.lpSum(x[s] for s in suppliers) == qty

    for s in suppliers:
        model += x[s] <= cap[s]
        model += x[s] <= qty * y[s]

    model += pulp.lpSum(y[s] for s in suppliers) <= max_suppliers

    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return pd.DataFrame()

    out = []
    for s in suppliers:
        aq = float(pulp.value(x[s]))
        if aq > 1e-6:
            out.append({"supplier_id": s, "award_qty": aq})
    return pd.DataFrame(out)
