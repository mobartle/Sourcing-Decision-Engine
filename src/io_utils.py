from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[1]

def read_csv(name: str) -> pd.DataFrame:
    path = ROOT / "inputs" / name
    return pd.read_csv(path)

def read_config() -> Dict[str, Any]:
    path = ROOT / "inputs" / "config_weights.json"
    return json.loads(path.read_text())

def write_csv(df: pd.DataFrame, name: str) -> None:
    out = ROOT / "outputs" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

def write_json(obj: Any, name: str) -> None:
    out = ROOT / "outputs" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2))
