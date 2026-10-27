import pandas as pd
import pytest

from p4_causal_validation.data.estimands import subset_for_estimand
from p4_causal_validation.estimators.common import DataContractError
from p4_causal_validation.data.wsc import prepare_wsc


def _frame():
    return pd.DataFrame(
        {
            "mathSel": [1, 1, 0, 0],
            "mathGrp": [1, 0, 1, 0],
            "mathPost": [5, 2, 4, 1],
            "vocabPost": [1, 2, 3, 4],
        }
    )


def test_qed_cells_match_preference_assignment_definition():
    qed = subset_for_estimand(_frame(), "QED_concordant_ATE")
    assert qed["mathSel"].tolist() == [1, 0]
    assert qed["mathGrp"].tolist() == [1, 0]
    prefer = subset_for_estimand(_frame(), "RCT_effect_prefer_math")
    assert prefer["mathSel"].eq(1).all()
    assert prefer["mathGrp"].tolist() == [1, 0]


def test_unknown_estimand_errors():
    with pytest.raises(KeyError):
        subset_for_estimand(_frame(), "effect")


def test_adapter_rejects_potential_outcome_column(tmp_path):
    import rdata

    frame = _frame()
    frame["Y(1)"] = frame["mathPost"]
    path = tmp_path / "WSCdata.rda"
    # The rdata writer is not required; the guard is on column names after load.
    with pytest.raises(DataContractError):
        from p4_causal_validation.data.wsc import load_rda_frame

        class Fake:
            pass

        original = load_rda_frame
        try:
            import p4_causal_validation.data.wsc as module

            module.load_rda_frame = lambda path: frame
            module.prepare_wsc(path)
        finally:
            module.load_rda_frame = original
