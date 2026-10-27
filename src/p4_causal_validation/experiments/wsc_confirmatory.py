"""Confirmatory WSC analysis. Refuses to run without a matching owner approval.

Implementation-preparation builds keep a second lock: a matching approval file
is necessary, and it is still not sufficient. This module does not fit mathPost.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from p4_causal_validation.experiments.approval import assert_owner_approved

REAL_WSC_EXECUTION_ENABLED = False


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "--config":
        raise SystemExit(
            "Usage: python -m p4_causal_validation.experiments.wsc_confirmatory --config configs/confirmatory_template.yaml"
        )
    config = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
    root = Path(__file__).resolve().parents[3]
    protocol = root / config["protocol_path"]
    approval = root / config["approval_path"]
    assert_owner_approved(protocol, approval)
    if not REAL_WSC_EXECUTION_ENABLED:
        raise SystemExit(
            "Refusing real WSC confirmatory execution. Approval was checked, but this build does not fit mathPost."
        )
    raise SystemExit("Real WSC execution flag was enabled without an authorized runner.")


if __name__ == "__main__":
    main()
