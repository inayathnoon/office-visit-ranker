"""Generate the estate, the organisation and the trips, plus the truth set.

Two roots:

* ``data/raw/``   - what a workplace-services team would actually have: an
                    office directory, an HR extract, a space allocation table,
                    trips and the badge taps recorded during them.
* ``data/truth/`` - the choice weights, the exploration flag per trip, and the
                    accuracy ceiling they imply. Read only by the grading code.

The exploration flag in particular must never leak. It is the label for "no
feature could have predicted this", and a model that could see it would score
above a ceiling that is supposed to be unreachable.
"""

from __future__ import annotations

import json
import shutil

import pandas as pd

from ..config import GROUND_TRUTH, RAW_DIR, TRUTH_DIR, Config, load_config
from .estate import generate_offices, generate_org, generate_space_allocation
from .trips import PUBLISHED_TRIP_COLUMNS, generate_trips


def _write(frame: pd.DataFrame, root, name: str) -> int:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "part-000.parquet", index=False)
    return len(frame)


def generate_all(cfg: Config | None = None, clean: bool = True) -> dict:
    cfg = cfg or load_config()
    if clean:
        for root in (RAW_DIR, TRUTH_DIR):
            if root.exists():
                shutil.rmtree(root)

    offices = generate_offices(cfg)
    employees = generate_org(cfg, offices)
    allocation = generate_space_allocation(cfg, offices)
    trips, taps, realised = generate_trips(cfg, employees, offices, allocation)

    counts = {
        "offices": _write(offices, RAW_DIR, "offices"),
        "employees": _write(employees, RAW_DIR, "employees"),
        "space_allocation": _write(allocation, RAW_DIR, "space_allocation"),
        "trips": _write(trips[PUBLISHED_TRIP_COLUMNS], RAW_DIR, "trips"),
        "trip_taps": _write(taps, RAW_DIR, "trip_taps"),
    }
    _write(trips, TRUTH_DIR, "trips_truth")

    ground_truth = {
        "profile": cfg.profile_name,
        "seed": cfg.seed,
        "start_date": cfg.start_date.isoformat(),
        "end_date": cfg.end_date.isoformat(),
        "train_end": cfg.train_end.isoformat(),
        "validation_end": cfg.validation_end.isoformat(),
        "row_counts": counts,
        # The weights the model is trying to recover. Normalised alongside the
        # raw values so they can be compared with a feature importance, which
        # has no natural scale.
        "choice_weights": cfg.choice.weights,
        "choice_weights_normalised": {
            k: round(v / sum(cfg.choice.weights.values()), 4) for k, v in cfg.choice.weights.items()
        },
        "exploration_share_configured": cfg.choice.exploration_share,
        "noise_scale": cfg.choice.noise_scale,
        "realised": realised,
    }
    GROUND_TRUTH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH.write_text(json.dumps(ground_truth, indent=2, sort_keys=True))
    return {"row_counts": counts, "ground_truth": ground_truth}


if __name__ == "__main__":  # pragma: no cover
    result = generate_all()
    for table, count in result["row_counts"].items():
        print(f"{table:20s} {count:>10,}")
    realised = result["ground_truth"]["realised"]
    print()
    for key, value in realised.items():
        print(f"  {key:32s} {value}")
