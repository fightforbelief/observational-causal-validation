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


def assert_real_execution_authorized(protocol_path: Path, record_path: Path, approval: dict) -> dict:
    """Read the separate owner decision record. This does not weaken protocol approval."""
    if not record_path.exists():
        raise SystemExit(
            "Refusing real WSC confirmatory execution. Approval was checked, but this build does not fit mathPost."
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = protocol_sha256(protocol_path)
    if record.get("protocol_sha256") != expected:
        raise SystemExit("Refusing confirmatory WSC analysis: approval does not match the frozen protocol.")
    if not record.get("approver"):
        raise SystemExit("Refusing confirmatory WSC analysis: approval has no named approver.")
    if record.get("approver") != approval.get("approver"):
        raise SystemExit(
            "Refusing confirmatory WSC analysis: decision record approver does not match the approval file."
        )
    authorized = (
        record.get("joint_bootstrap") == "approved"
        and record.get("zero_failure_rule") == "approved"
        and record.get("real_primary_execution") == "authorized"
    )
    if not authorized:
        raise SystemExit(
            "Refusing real WSC confirmatory execution. Approval was checked, but this build does not fit mathPost."
        )
    return record
