"""L1 reproduction of one bounded Young et al. result.

The fresh result is the randomized difference in means for the continuous
outcome on the published common-eligibility sample. Cached DR/IPW/OM/PSM files
are recorded as published artifacts, not as fresh estimates.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml

from p4_causal_validation.estimators.regression import linear_adjustment
from p4_causal_validation.estimators.unadjusted import difference_in_means
from p4_causal_validation.data.product_release import (
    COVARIATES,
    OUTCOME_CONTINUOUS,
    load_cohort,
)


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _fresh_dim(frame: pd.DataFrame, cohort: str, regime: str) -> dict:
    result = difference_in_means(
        frame[OUTCOME_CONTINUOUS],
        frame["D"],
        estimand="ATE_continuous_common_eligibility" if regime == "common_cutoff" else "ATE_continuous_as_released",
        outcome=OUTCOME_CONTINUOUS,
        target_population=f"{cohort} sample, {regime}",
        role="fresh computation; randomized benchmark only if cohort is experimental",
    )
    young_root = Path.home() / "Desktop" / "projects" / "_p4_external_sources" / "young-reevaluating"
    sys.path.insert(0, str(young_root))
    from src.data_loading import difference_in_means as upstream

    upstream_estimate, upstream_variance = upstream(pd.DataFrame({"y": frame[OUTCOME_CONTINUOUS], "D": frame["D"]}))
    payload = result.as_dict()
    payload["upstream_function_estimate"] = float(upstream_estimate)
    payload["upstream_function_variance"] = float(upstream_variance)
    payload["matches_upstream_function"] = abs(payload["estimate"] - float(upstream_estimate)) < 1e-10
    payload["computation"] = "fresh"
    payload["cohort"] = cohort
    payload["regime"] = regime
    return payload


def _fresh_regression(frame: pd.DataFrame, cohort: str) -> dict:
    result = linear_adjustment(
        frame[OUTCOME_CONTINUOUS].to_numpy(),
        frame["D"].to_numpy(),
        frame[COVARIATES].astype(float).to_numpy(),
        estimand="ATE_continuous_common_eligibility",
        outcome=OUTCOME_CONTINUOUS,
        target_population=f"{cohort} common eligibility Outcome_Continuous <= 44",
    )
    payload = result.as_dict()
    payload["computation"] = "fresh"
    payload["cohort"] = cohort
    payload["regime"] = "common_cutoff"
    return payload


def _cached_summaries(cache_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(cache_dir.glob("*.pkl")):
        record = {"file": path.name, "bytes": path.stat().st_size, "computation": "published_cached_artifact"}
        if path.stat().st_size > 200_000:
            record["loaded"] = False
            record["reason"] = "Propensity array was hashed in Gate 0 and not treated as a fresh estimate."
            rows.append(record)
            continue
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        record["loaded"] = True
        record["python_type"] = type(obj).__name__
        if isinstance(obj, pd.DataFrame):
            record["columns"] = [str(column) for column in obj.columns]
            record["records"] = json.loads(obj.to_json(orient="records"))
        elif isinstance(obj, dict):
            record["keys"] = [str(key) for key in obj]
            record["preview"] = {str(key): _plain(value) for key, value in list(obj.items())[:30]}
        else:
            record["preview"] = _plain(obj)
        rows.append(record)
    return rows


def _plain(value):
    if isinstance(value, float):
        return value
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in list(value.items())[:20]}
    return repr(value)[:500]


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "--config":
        raise SystemExit(
            "Usage: python -m p4_causal_validation.experiments.reproduce_young --config configs/reproduction.yaml"
        )
    config = _load_config(Path(sys.argv[2]))
    data_dir = Path(config["young_data_dir"])
    output = Path(config["output_json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fresh = []
    for cohort in ("experimental", "observed"):
        for regime, upper in (("as_released", None), ("common_cutoff", 44.0)):
            frame = load_cohort(data_dir, cohort, upper)
            fresh.append(_fresh_dim(frame, cohort, regime))
        common = load_cohort(data_dir, cohort, 44.0)
        try:
            fresh.append(_fresh_regression(common, cohort))
        except Exception as exc:
            fresh.append(
                {
                    "cohort": cohort,
                    "regime": "common_cutoff",
                    "estimator": "linear_regression_hc1",
                    "computation": "fresh_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    payload = {
        "selected_fresh_result": (
            "Experimental difference in means for Outcome_Continuous on Outcome_Continuous <= 44. "
            "This is a randomized ATE estimate with sampling uncertainty, not the paper's cached DR benchmark."
        ),
        "not_run": [
            "FLAML AutoML nuisance search",
            "R Matching propensity-score matching",
            "Fresh doubly robust refit",
        ],
        "fresh": fresh,
        "cached_artifacts": _cached_summaries(Path(config["young_cache_dir"])),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
