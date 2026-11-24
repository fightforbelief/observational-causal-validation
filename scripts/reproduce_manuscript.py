"""Rebuild exploratory manuscript exhibits from a pinned public WSCdata file."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from audit_gate0 import load_wsc_frame

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "exploratory" / "wsc_reanalysis"
VALIDATION = ROOT / "exploratory" / "discordant_validation"
RDA_SHA256 = "b47be7d8f2c0d54af9cc71f3602183e61307a964e701a2cdcf3944bcef85286b"


def verify_exhibits(run):
    actual = pd.read_csv(run / "results/main_contrasts.csv")
    reference = pd.read_csv(SOURCE / "published/main_contrasts.csv")
    pd.testing.assert_frame_equal(actual, reference, rtol=1e-7, atol=1e-7)
    adjusted = json.loads((run / "results/multiplicity.json").read_text())
    expected = json.loads((SOURCE / "published/multiplicity.json").read_text())
    if adjusted.keys() != expected.keys():
        raise ValueError("Multiplicity check has unexpected contrasts")
    for key, value in expected.items():
        if isinstance(value, dict):
            for field, target in value.items():
                np.testing.assert_allclose(adjusted[key][field], target, atol=1e-7)
        else:
            np.testing.assert_allclose(adjusted[key], value, atol=1e-7)
    for name in ["boot.csv", "vocab/boot_joint.csv"]:
        replicates = pd.read_csv(run / "results" / name)
        if len(replicates) != 2000 or replicates.isna().any().any():
            raise ValueError(f"Incomplete joint bootstrap in {name}")
        parts = replicates[["qed_error", "estimand_misalignment", "chance_S_imbalance"]]
        np.testing.assert_allclose(parts.sum(axis=1), replicates.d_ate_misaligned, atol=1e-12)
    summary = json.loads((run / "results/vocab/summary.json").read_text())
    if summary["paired_failures"] or summary["paired"]["d_att"]["n"] != 2000:
        raise ValueError("Incomplete paired vocabulary bootstrap")

    audit = pd.read_csv(run / "results/discordant_decomposition.csv").set_index(["outcome", "estimand"])
    contrasts = reference.set_index(["outcome", "estimand"]).loc[audit.index]
    np.testing.assert_allclose(audit.original_gap, contrasts.point, atol=1e-12)
    np.testing.assert_allclose(audit.signed_residual + audit.prediction_mix, contrasts.point, atol=1e-12)
    for name in ["discordant_decomposition.csv", "discordant_intervals.csv", "discordant_shape.csv",
                 "discordant_simulation.csv", "discordant_retrained_gradients.csv"]:
        actual = pd.read_csv(run / "results" / name)
        expected = pd.read_csv(VALIDATION / "published" / name)
        pd.testing.assert_frame_equal(actual, expected, rtol=1e-7, atol=1e-7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rda", type=Path, help="WSCdata.rda from WSCdata 0.1.2")
    parser.add_argument("output", type=Path, help="New directory for this reproduction")
    args = parser.parse_args()
    if hashlib.sha256(args.rda.read_bytes()).hexdigest() != RDA_SHA256:
        parser.error("Input differs from the published WSCdata file")
    data = load_wsc_frame(args.rda)
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    (run / "data").mkdir()
    (run / "figures").mkdir()
    (run / "published").mkdir()
    shutil.copyfile(SOURCE / "published/math_primary_bootstrap.csv", run / "published/math_primary_bootstrap.csv")
    data.to_csv(run / "data/WSCdata.csv", index=False)
    scripts = sorted(SOURCE.glob("[0-9][0-9]_*.py"))
    for path in [SOURCE / "wsc_lib.py", *scripts]:
        shutil.copyfile(path, run / path.name)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for script in scripts:
        print(f"Running {script.name}", flush=True)
        subprocess.run([sys.executable, script.name], cwd=run, env=env, check=True)
    subprocess.run([sys.executable, str(VALIDATION / "simulation.py"),
                    str(run / "results/discordant_simulation.csv")], cwd=run, env=env, check=True)
    verify_exhibits(run)
    print("Verified six aligned contrasts, adjusted intervals, bootstrap identities, discordant validation, shape bounds, and simulations.")


if __name__ == "__main__":
    main()
