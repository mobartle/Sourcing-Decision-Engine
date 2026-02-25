# Sourcing Decision Engine 

This repo demonstrates a sourcing decision engine that:
1) builds a feasible candidate set
2) scores suppliers using configurable weights (transparent, explainable)
3) optionally runs award optimization (min cost with constraints)
4) produces audited outputs (CSV/JSON)

## Quickstart


## Explainability Outputs
- outputs/decision_trace.json: feasibility filtering and constraints applied
- outputs/explainability.json: per-supplier breakdown of scores, contributions, and reason codes
- outputs/explainability_detailed.csv - Deeper explainability dataset — enterprise audit ready
- outputs/counterfactual_analysis.csv - Why challengers didn’t win

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/engine.py --request_id RQ1001 --optimize true
