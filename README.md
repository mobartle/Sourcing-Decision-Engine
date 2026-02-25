# Sourcing Decision Engine (Example Repo)

This repo demonstrates a sourcing decision engine that:
1) builds a feasible candidate set
2) scores suppliers using configurable weights (transparent, explainable)
3) optionally runs award optimization (min cost with constraints)
4) produces audited outputs (CSV/JSON)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/engine.py --request_id RQ1001 --optimize true
