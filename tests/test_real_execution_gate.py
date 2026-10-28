"""Gate checks for owner approval and real-execution authorization.

These tests do not read WSCdata and are not confirmatory evidence.
TEST_FIXTURE_NOT_RESEARCH_EVIDENCE.
"""

import json
from pathlib import Path

import pandas as pd
import p4_causal_validation
import pytest

from p4_causal_validation.estimators.common import EstimateResult
from p4_causal_validation.experiments.approval import (
    assert_owner_approved,
    assert_real_execution_authorized,
    protocol_sha256,
)
from p4_causal_validation.experiments.primary_specification import (
    AUTHORIZED_REAL_SOURCE,
    ConfirmatoryNotAuthorized,
    assert_synthetic_fixture,
    randomized_math_preference_benchmark,
)
from p4_causal_validation.experiments.wsc_confirmatory import main, run_authorized_primary
from p4_causal_validation.inference.bootstrap import stratified_primary_contrast_bootstrap


def _protocol(tmp_path):
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("status: frozen\n", encoding="utf-8")
    return protocol


def _approval(tmp_path, protocol, **overrides):
    payload = {
        "approver": "Hanqing Yang",
        "decision": "approved",
        "protocol_sha256": protocol_sha256(protocol),
    }
    payload.update(overrides)
    path = tmp_path / "OWNER_APPROVAL.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _record(tmp_path, protocol, **overrides):
    payload = {
        "approver": "Hanqing Yang",
        "protocol_sha256": protocol_sha256(protocol),
        "joint_bootstrap": "approved",
        "zero_failure_rule": "approved",
        "real_primary_execution": "authorized",
    }
    payload.update(overrides)
    path = tmp_path / "OWNER_DECISION_RECORD.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _config(tmp_path, protocol, approval, record):
    path = tmp_path / "config.yaml"
    record_line = f"decision_record_path: '{record}'\n" if record is not None else ""
    path.write_text(
        f"protocol_path: '{protocol}'\napproval_path: '{approval}'\n{record_line}",
        encoding="utf-8",
    )
    return path


def test_missing_approval_file_refuses(tmp_path):
    protocol = _protocol(tmp_path)
    with pytest.raises(SystemExit, match="approval file is absent"):
        assert_owner_approved(protocol, tmp_path / "missing.json")


def test_frozen_hash_mismatch_refuses(tmp_path):
    protocol = _protocol(tmp_path)
    approval = _approval(tmp_path, protocol, protocol_sha256="0" * 64)
    with pytest.raises(SystemExit, match="does not match the frozen protocol"):
        assert_owner_approved(protocol, approval)


def test_missing_execution_authorization_refuses(tmp_path):
    protocol = _protocol(tmp_path)
    approval = assert_owner_approved(protocol, _approval(tmp_path, protocol))
    with pytest.raises(SystemExit, match="does not fit mathPost"):
        assert_real_execution_authorized(protocol, tmp_path / "missing_record.json", approval)


def test_unapproved_execution_scope_refuses(tmp_path):
    protocol = _protocol(tmp_path)
    approval = assert_owner_approved(protocol, _approval(tmp_path, protocol))
    record = _record(tmp_path, protocol, real_primary_execution="not_authorized")
    with pytest.raises(SystemExit, match="does not fit mathPost"):
        assert_real_execution_authorized(protocol, record, approval)


def test_all_gates_call_the_runner_without_fitting(monkeypatch, tmp_path):
    protocol = _protocol(tmp_path)
    approval = _approval(tmp_path, protocol)
    record = _record(tmp_path, protocol)
    config = _config(tmp_path, protocol, approval, record)
    called = {}

    def runner(*_args, **_kwargs):
        called["ran"] = True

    monkeypatch.setattr(
        "p4_causal_validation.experiments.wsc_confirmatory.run_authorized_primary",
        runner,
    )
    monkeypatch.setattr("sys.argv", ["wsc_confirmatory", "--config", str(config)])
    main()
    assert called["ran"] is True


def test_execution_flag_cannot_open_the_gate(monkeypatch, tmp_path):
    protocol = _protocol(tmp_path)
    approval = _approval(tmp_path, protocol)
    record = _record(tmp_path, protocol)
    config = _config(tmp_path, protocol, approval, record)
    monkeypatch.setattr(
        "p4_causal_validation.experiments.wsc_confirmatory.REAL_WSC_EXECUTION_ENABLED",
        True,
    )

    def runner(*_args, **_kwargs):
        raise AssertionError("The boolean flag must not start a fit.")

    monkeypatch.setattr(
        "p4_causal_validation.experiments.wsc_confirmatory.run_authorized_primary",
        runner,
    )
    monkeypatch.setattr("sys.argv", ["wsc_confirmatory", "--config", str(config)])
    with pytest.raises(SystemExit, match="must stay false"):
        main()


def _project_root() -> Path:
    current = Path(p4_causal_validation.__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError("project root not found")


def test_runner_refuses_before_data_when_implementation_hash_mismatches():
    root = _project_root()
    record = {"implementation_sha256": {"src/p4_causal_validation/inference/bootstrap.py": "0" * 64}}
    with pytest.raises(SystemExit, match="implementation hash"):
        run_authorized_primary(root, {}, {}, record)


def test_unmarked_frame_is_refused_and_authorized_marker_can_estimate_rct():
    frame = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "mathSel": [1, 1, 1, 1],
            "mathGrp": [1, 1, 0, 0],
            "mathPost": [1.0, 3.0, 0.0, 2.0],
        }
    )
    frame.attrs["source"] = "wsc_file"
    with pytest.raises(ConfirmatoryNotAuthorized):
        randomized_math_preference_benchmark(frame)
    frame.attrs["source"] = AUTHORIZED_REAL_SOURCE
    frame.attrs["real_execution_authorized"] = True
    with pytest.raises(ConfirmatoryNotAuthorized):
        assert_synthetic_fixture(frame)
    result = randomized_math_preference_benchmark(frame)
    assert result.estimand == "RCT_effect_prefer_math"
    assert result.n == 4 and result.n_treated == 2 and result.n_control == 2


def _authorized_cells():
    rows = []
    for row_id, preference, assignment in (
        *[(i, 1, 1) for i in range(4)],
        *[(i, 1, 0) for i in range(4, 8)],
        *[(i, 0, 1) for i in range(8, 10)],
        *[(i, 0, 0) for i in range(10, 14)],
    ):
        rows.append(
            {
                "row_id": row_id,
                "mathSel": preference,
                "mathGrp": assignment,
                "mathPost": float(row_id),
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["source"] = AUTHORIZED_REAL_SOURCE
    frame.attrs["real_execution_authorized"] = True
    return frame


def _fake_result(estimand: str) -> EstimateResult:
    return EstimateResult(
        estimand=estimand,
        estimator="gate_test_double",
        estimate=1.0,
        se=0.2,
        n=8,
        n_treated=4,
        n_control=4,
        inference="not an estimate",
        target_population="gate test",
        outcome="mathPost",
        diagnostics={
            "n_folds": 5,
            "seed": 20261026,
            "fold_assignment": "distinct_participant_id",
            "rows_deleted_for_overlap": 0,
        },
    )


def test_default_bootstrap_rejects_authorized_frame_and_authorized_path_keeps_marker(monkeypatch):
    frame = _authorized_cells()
    with pytest.raises(ConfirmatoryNotAuthorized):
        stratified_primary_contrast_bootstrap(frame, n_replicates=2, seed=1)
    seen = []

    def fake_rct(data):
        seen.append(data.attrs.get("source"))
        return _fake_result("RCT_effect_prefer_math")

    def fake_qed(data):
        seen.append(data.attrs.get("source"))
        return _fake_result("QED_concordant_ATT")

    monkeypatch.setattr(
        "p4_causal_validation.experiments.primary_specification.randomized_math_preference_benchmark",
        fake_rct,
    )
    monkeypatch.setattr(
        "p4_causal_validation.experiments.primary_specification.qed_concordant_att",
        fake_qed,
    )
    result = stratified_primary_contrast_bootstrap(
        frame,
        n_replicates=2,
        seed=1,
        authorized_real_run=True,
        failure_rule_approved=True,
    )
    assert seen
    assert set(seen) == {AUTHORIZED_REAL_SOURCE}
    assert result["not_a_wsc_result"] is False
    assert result["real_wsc_execution_authorized"] is True
    assert result["machine_approval_of_failure_rule"] is True
    assert result["zero_failure_rule_does_not_authorize_real_wsc"] is True
    assert result["confirmatory_interval_authorized"] is False
    assert result["interval_status"] == "diagnostic_only_zero_failure_rule_not_met"
