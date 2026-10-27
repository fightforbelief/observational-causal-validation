import json
from pathlib import Path

import pytest

from p4_causal_validation.experiments.approval import assert_owner_approved, protocol_sha256
from p4_causal_validation.experiments.wsc_confirmatory import main as confirmatory_main
from p4_causal_validation.experiments.wsc_pilot import main as pilot_main


def test_confirmatory_refuses_without_approval(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("status: pending\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"protocol_path: {protocol.name}\napproval_path: missing.json\n",
        encoding="utf-8",
    )
    # The command resolves paths from the package root, so call the gate directly.
    with pytest.raises(SystemExit, match="approval file is absent"):
        assert_owner_approved(protocol, tmp_path / "missing.json")


def test_approval_must_match_protocol_hash(tmp_path):
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("primary: mathPost\n", encoding="utf-8")
    approval = tmp_path / "OWNER_APPROVAL.json"
    approval.write_text(
        json.dumps({"decision": "approved", "approver": "owner", "protocol_sha256": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="does not match"):
        assert_owner_approved(protocol, approval)
    approval.write_text(
        json.dumps(
            {
                "decision": "approved",
                "approver": "owner",
                "protocol_sha256": protocol_sha256(protocol),
            }
        ),
        encoding="utf-8",
    )
    assert assert_owner_approved(protocol, approval)["approver"] == "owner"


def test_commands_refuse_by_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["wsc_pilot", "--config", "configs/wsc_pilot.yaml"])
    with pytest.raises(SystemExit, match="Refusing the WSC pilot"):
        pilot_main()
