"""Gate 0 source and data audit. Descriptive only: no causal model fitting."""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import sys
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = Path.home() / "Desktop" / "projects" / "_p4_external_sources"
YOUNG = SOURCES / "young-reevaluating"
WSC = SOURCES / "wscdata"
CRAN_DIR = SOURCES / "cran"
REPORTS = ROOT / "reports"
ACCESS_DATE = "2026-10-26"

YOUNG_URL = "https://github.com/microsoft/Reevaluating-Causal-Estimation-Methods"
WSC_URL = "https://github.com/jzangela/WSCdata"
CRAN_URL = "https://cran.r-project.org/src/contrib/WSCdata_0.1.2.tar.gz"
PAPER_YOUNG = "https://arxiv.org/abs/2601.11845"
PAPER_WSC = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12959902/"

POST_TREATMENT = {"mathPost", "vocabPost", "MCSpost", "Math_Train_WSC"}
TREATMENT_COLS = {"mathGrp", "mathSel"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {
        "path": str(path),
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def summarize_series(series: pd.Series) -> dict:
    numeric = pd.to_numeric(series, errors="coerce")
    out = {
        "dtype": str(series.dtype),
        "n": int(series.shape[0]),
        "n_missing": int(series.isna().sum()),
        "n_unique": int(series.nunique(dropna=True)),
    }
    if numeric.notna().any():
        out.update(
            {
                "min": float(np.nanmin(numeric.to_numpy())),
                "max": float(np.nanmax(numeric.to_numpy())),
                "mean": float(np.nanmean(numeric.to_numpy())),
                "std_ddof1": float(np.nanstd(numeric.to_numpy(), ddof=1)),
            }
        )
    return out


def column_table(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for name in frame.columns:
        row = {"column": str(name)}
        row.update(summarize_series(frame[name]))
        rows.append(row)
    return rows


class _RestrictedUnpickler(pickle.Unpickler):
    """Allow only numeric containers from the published result pickles."""

    def find_class(self, module, name):
        root = module.split(".")[0]
        if root not in {"builtins", "collections", "numpy", "pandas", "copyreg", "_codecs"}:
            raise RuntimeError(f"Refusing pickle class {module}.{name}")
        return super().find_class(module, name)


def safe_load_pickle(path: Path):
    with path.open("rb") as handle:
        return _RestrictedUnpickler(handle).load()


def pickle_summary(obj, depth: int = 0) -> dict:
    if isinstance(obj, dict):
        return {
            "type": "dict",
            "keys": [str(key) for key in list(obj)[:40]],
            "values": {str(key): pickle_summary(obj[key], depth + 1) for key in list(obj)[:20]},
        }
    if isinstance(obj, pd.DataFrame):
        preview = obj.head(8).to_dict(orient="records")
        return {
            "type": "DataFrame",
            "shape": list(obj.shape),
            "columns": [str(col) for col in obj.columns],
            "preview": json.loads(pd.DataFrame(preview).to_json(orient="records")),
        }
    if isinstance(obj, pd.Series):
        return {"type": "Series", "name": str(obj.name), "n": int(len(obj)), "head": obj.head(5).tolist()}
    if isinstance(obj, np.ndarray):
        flat = obj.reshape(-1)
        return {
            "type": "ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "min": float(np.nanmin(flat)) if flat.size and np.issubdtype(obj.dtype, np.number) else None,
            "max": float(np.nanmax(flat)) if flat.size and np.issubdtype(obj.dtype, np.number) else None,
        }
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return {"type": type(obj).__name__, "value": obj if not isinstance(obj, float) else float(obj)}
    if isinstance(obj, (list, tuple)) and depth < 2:
        return {"type": type(obj).__name__, "n": len(obj), "head": [pickle_summary(item, depth + 1) for item in obj[:5]]}
    return {"type": type(obj).__name__, "repr": repr(obj)[:300]}


def audit_young() -> dict:
    exp_path = YOUNG / "Data" / "FINAL_PUBLIC_experimental.parquet"
    obs_path = YOUNG / "Data" / "FINAL_PUBLIC_observed.parquet"
    exp = pd.read_parquet(exp_path)
    obs = pd.read_parquet(obs_path)
    files = [file_record(exp_path), file_record(obs_path), file_record(YOUNG / "Data" / "README.md")]
    saved = []
    for path in sorted((YOUNG / "saved_outputs").glob("*.pkl")):
        record = file_record(path)
        if path.stat().st_size > 200_000:
            record["loaded"] = False
            record["reason"] = "Large propensity array recorded by hash only; not unpickled in Gate 0."
        else:
            obj = safe_load_pickle(path)
            record["loaded"] = True
            record["content"] = pickle_summary(obj)
        saved.append(record)

    def arm_summary(frame: pd.DataFrame, label: str) -> dict:
        treatment = frame["New_Feature"].astype(bool)
        eligible = frame["Outcome_Continuous"] <= 44
        out = {
            "label": label,
            "n": int(len(frame)),
            "n_columns": int(frame.shape[1]),
            "columns": [str(col) for col in frame.columns],
            "n_missing_cells": int(frame.isna().sum().sum()),
            "treatment_true": int(treatment.sum()),
            "treatment_false": int((~treatment).sum()),
            "treatment_rate": float(treatment.mean()),
            "eligible_le_44": int(eligible.sum()),
            "dropped_gt_44": int((~eligible).sum()),
            "dropped_treated_gt_44": int(((~eligible) & treatment).sum()),
            "dropped_control_gt_44": int(((~eligible) & ~treatment).sum()),
            "outcomes": {
                "Outcome_Continuous": summarize_series(frame["Outcome_Continuous"]),
                "Outcome_Binary": summarize_series(frame["Outcome_Binary"]),
            },
        }
        return out

    license_files = [path.name for path in YOUNG.iterdir() if "LICENSE" in path.name.upper() or "COPYING" in path.name.upper()]
    return {
        "source_url": YOUNG_URL,
        "paper_url": PAPER_YOUNG,
        "commit": "a6633e12506a71ed9b96ae12edbc4ce4a808feb8",
        "commit_date": "2026-10-23T11:54:15-07:00",
        "access_date": ACCESS_DATE,
        "license_files_present": license_files,
        "readme_license_claim": "README states MIT and points to LICENSE, but no LICENSE file is present in this commit.",
        "files": files,
        "experimental": arm_summary(exp, "experimental"),
        "observed": arm_summary(obs, "observed"),
        "schema_equal": list(exp.columns) == list(obs.columns),
        "saved_outputs": saved,
        "column_summaries_experimental": column_table(exp),
        "column_summaries_observed": column_table(obs),
    }


def load_wsc_frame(path: Path) -> pd.DataFrame:
    import rdata

    parsed = rdata.parser.parse_file(path)
    converted = rdata.conversion.convert(parsed)
    if isinstance(converted, pd.DataFrame):
        return converted
    if isinstance(converted, dict):
        if "WSCdata" in converted and isinstance(converted["WSCdata"], pd.DataFrame):
            return converted["WSCdata"]
        frames = {key: value for key, value in converted.items() if isinstance(value, pd.DataFrame)}
        if len(frames) == 1:
            return next(iter(frames.values()))
        raise RuntimeError(f"Unexpected rda objects: { {key: type(value).__name__ for key, value in converted.items()} }")
    raise RuntimeError(f"Unexpected rda type: {type(converted).__name__}")


def download_cran() -> dict:
    CRAN_DIR.mkdir(parents=True, exist_ok=True)
    destination = CRAN_DIR / "WSCdata_0.1.2.tar.gz"
    if not destination.exists():
        urllib.request.urlretrieve(CRAN_URL, destination)
    record = file_record(destination)
    record["url"] = CRAN_URL
    with tarfile.open(destination, "r:gz") as archive:
        members = [member.name for member in archive.getmembers() if member.name.endswith("data/WSCdata.rda") or member.name.endswith("DESCRIPTION")]
        record["matched_members"] = members
        extracted = CRAN_DIR / "WSCdata_0.1.2_extract"
        archive.extractall(extracted, filter="data")
    cran_rda = extracted / "WSCdata" / "data" / "WSCdata.rda"
    cran_desc = extracted / "WSCdata" / "DESCRIPTION"
    record["cran_rda"] = file_record(cran_rda)
    description = cran_desc.read_text(encoding="utf-8", errors="replace")
    version_line = next(line for line in description.splitlines() if line.startswith("Version:"))
    license_line = next(line for line in description.splitlines() if line.startswith("License:"))
    record["cran_version_line"] = version_line
    record["cran_license_line"] = license_line
    return record


def audit_wsc() -> dict:
    github_rda = WSC / "data" / "WSCdata.rda"
    frame = load_wsc_frame(github_rda)
    cran = download_cran()
    cran_frame = load_wsc_frame(Path(cran["cran_rda"]["path"]))
    same_values = frame.equals(cran_frame)
    cell = (
        frame.assign(mathGrp=frame["mathGrp"].astype(int), mathSel=frame["mathSel"].astype(int))
        .groupby(["mathSel", "mathGrp"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    cell_rows = []
    for row in cell.to_dict(orient="records"):
        sel = int(row["mathSel"])
        grp = int(row["mathGrp"])
        if sel == 1 and grp == 1:
            label = "G1 prefer math, assigned math; QED concordant treated"
        elif sel == 1 and grp == 0:
            label = "G2 prefer math, assigned vocabulary; RCT counterfactual for math preference"
        elif sel == 0 and grp == 1:
            label = "G3 prefer vocabulary, assigned math"
        elif sel == 0 and grp == 0:
            label = "G4 prefer vocabulary, assigned vocabulary; QED concordant control"
        else:
            label = "unmapped"
        cell_rows.append({"mathSel": sel, "mathGrp": grp, "n": int(row["n"]), "role": label})

    qed = frame.loc[frame["mathSel"].astype(int) == frame["mathGrp"].astype(int)]
    integrity = {}
    for outcome, treatment in [("mathPost", "mathGrp"), ("vocabPost", "mathGrp")]:
        treated = frame.loc[frame[treatment].astype(int) == 1, outcome].astype(float)
        control = frame.loc[frame[treatment].astype(int) == 0, outcome].astype(float)
        estimate = float(treated.mean() - control.mean())
        variance = float(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
        integrity[outcome] = {
            "label": "Descriptive full-sample difference in means used only to verify treatment coding against the package example. Not an L2 result.",
            "n_treated": int(len(treated)),
            "n_control": int(len(control)),
            "estimate": estimate,
            "se": float(np.sqrt(variance)),
            "package_readme_lm_mathPost_only": 0.7603 if outcome == "mathPost" else None,
        }

    return {
        "source_url": WSC_URL,
        "paper_url": PAPER_WSC,
        "doi": "https://doi.org/10.1353/obs.2025.a963647",
        "github_commit": "6360a5247567c0ff632cd12aa2f6b3cfff92b2b6",
        "github_commit_date": "2025-11-20T13:47:40-06:00",
        "github_description_version": "0.1.1.9000",
        "access_date": ACCESS_DATE,
        "license": "MIT + file LICENSE; COPYRIGHT HOLDER: WSCdata authors; year 2024",
        "github_rda": file_record(github_rda),
        "cran": cran,
        "github_rda_equals_cran_rda_values": bool(same_values),
        "github_rda_sha_equals_cran_rda_sha": file_record(github_rda)["sha256"] == cran["cran_rda"]["sha256"],
        "n": int(len(frame)),
        "columns": [str(col) for col in frame.columns],
        "column_summaries": column_table(frame),
        "cells": cell_rows,
        "n_math_assigned": int((frame["mathGrp"].astype(int) == 1).sum()),
        "n_vocab_assigned": int((frame["mathGrp"].astype(int) == 0).sum()),
        "n_prefer_math": int((frame["mathSel"].astype(int) == 1).sum()),
        "n_prefer_vocab": int((frame["mathSel"].astype(int) == 0).sum()),
        "n_qed_concordant": int(len(qed)),
        "qed_treatment_equals_preference": bool(
            (qed["mathGrp"].astype(int) == qed["mathSel"].astype(int)).all()
        ),
        "post_treatment_columns_present": [col for col in frame.columns if col in POST_TREATMENT],
        "coding_check_difference_in_means": integrity,
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "young": audit_young(),
        "wsc": audit_wsc(),
    }
    destination = REPORTS / "gate0_audit_raw.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {destination}")
    print("WSC_N", payload["wsc"]["n"])
    print("WSC_COLS", payload["wsc"]["columns"])
    print("CELLS", payload["wsc"]["cells"])
    print("SHA_EQUAL", payload["wsc"]["github_rda_sha_equals_cran_rda_sha"])
    print("VALUES_EQUAL", payload["wsc"]["github_rda_equals_cran_rda_values"])
    print("YOUNG_N", payload["young"]["experimental"]["n"], payload["young"]["observed"]["n"])
    print("ELIGIBLE", payload["young"]["experimental"]["eligible_le_44"], payload["young"]["observed"]["eligible_le_44"])


if __name__ == "__main__":
    main()
