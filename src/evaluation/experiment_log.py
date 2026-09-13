from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_experiment(
    experiment_name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: str | Path="results/experiments",
)->Path:
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record={
        "experiment_name": experiment_name,
        "timestamp_utc": timestamp,
        "config": config,
        "metrics": metrics,
    }
    out_path=output_dir/f"{experiment_name}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    return out_path

def load_all_experiments(experiments_dir: str|Path="results/experiments")->list[dict]:
    experiments_dir=Path(experiments_dir)
    records=[]
    for path in sorted(experiments_dir.glob("*.json")):
        with open(path, "r") as f:
            records.append(json.load(f))
    return sorted(records, key=lambda r: r["timestamp_utc"])
