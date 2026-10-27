"""Exploratory WSC pilot. It does not run unless the owner sets an explicit flag."""

from __future__ import annotations

import sys

import yaml


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "--config":
        raise SystemExit("Usage: python -m p4_causal_validation.experiments.wsc_pilot --config configs/wsc_pilot.yaml")
    config = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
    if not config.get("owner_acknowledged_pilot"):
        raise SystemExit(
            "Refusing the WSC pilot. It is exploratory, requires owner acknowledgement, "
            "and cannot be used as confirmatory evidence."
        )
    raise SystemExit("Pilot acknowledgement was set, but this session stops before any WSC outcome comparison.")


if __name__ == "__main__":
    main()
