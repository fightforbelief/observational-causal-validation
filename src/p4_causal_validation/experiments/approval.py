"""Owner-approval gate. This module never writes an approval file."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def protocol_sha256(protocol_path: Path) -> str:
    return hashlib.sha256(protocol_path.read_bytes()).hexdigest()


def assert_owner_approved(protocol_path: Path, approval_path: Path) -> dict:
    if not approval_path.exists():
        raise SystemExit(
            "Refusing confirmatory WSC analysis: owner approval file is absent. "
            "Do not create one automatically."
        )
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    expected = protocol_sha256(protocol_path)
    if approval.get("protocol_sha256") != expected or approval.get("decision") != "approved":
        raise SystemExit("Refusing confirmatory WSC analysis: approval does not match the frozen protocol.")
    if not approval.get("approver"):
        raise SystemExit("Refusing confirmatory WSC analysis: approval has no named approver.")
    return approval
