"""One pre-specified matching estimator. The estimand is ATT, not a generic effect."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from p4_causal_validation.estimators.common import (
    EstimateResult,
    as_float_array,
    as_treatment,
    require_arms,
    validate_result,
)
from p4_causal_validation.estimators.weighting import fit_logit_propensity

CALIPER_SD_MULTIPLE = 0.2


def att_nearest_neighbor(
    y,
    treatment,
    covariates,
    *,
    estimand: str,
    outcome: str,
    target_population: str,
    random_state: int = 20261026,
) -> EstimateResult:
    """1:1 nearest-neighbor matching on the logit propensity, without replacement.

    Caliper is 0.2 times the standard deviation of the logit propensity, fixed
    before outcome comparison. Unmatched treated units are excluded, so the
    target population can shrink. The paired SE ignores uncertainty from fitting
    the propensity score.
    """
    if not estimand.startswith("ATT"):
        raise ValueError("This matching routine is an ATT estimator and requires an ATT estimand label.")
    outcome_values = as_float_array(y)
    assignment = as_treatment(treatment)
    features = np.asarray(covariates, dtype=float)
    n_treated, n_control = require_arms(assignment)
    propensity = fit_logit_propensity(assignment, features)
    propensity = np.clip(propensity, 1e-6, 1 - 1e-6)
    logit = np.log(propensity / (1 - propensity))
    caliper = CALIPER_SD_MULTIPLE * float(np.std(logit, ddof=1))
    treated_index = np.flatnonzero(assignment == 1)
    control_index = np.flatnonzero(assignment == 0)
    order = treated_index.copy()
    rng = np.random.default_rng(random_state)
    rng.shuffle(order)
    available = control_index.copy()
    pairs: list[tuple[int, int]] = []
    for treated in order:
        if available.size == 0:
            break
        neighbors = NearestNeighbors(n_neighbors=min(available.size, 1), metric="euclidean")
        neighbors.fit(logit[available].reshape(-1, 1))
        distance, local = neighbors.kneighbors(logit[[treated]].reshape(-1, 1))
        if float(distance[0, 0]) <= caliper:
            chosen = int(available[int(local[0, 0])])
            pairs.append((int(treated), chosen))
            available = available[available != chosen]
    if len(pairs) < 2:
        raise ValueError("Matching produced fewer than two pairs inside the pre-specified caliper.")
    treated_y = np.array([outcome_values[i] for i, _ in pairs])
    control_y = np.array([outcome_values[j] for _, j in pairs])
    differences = treated_y - control_y
    estimate = float(differences.mean())
    se = float(differences.std(ddof=1) / np.sqrt(differences.size))
    result = EstimateResult(
        estimand=estimand,
        estimator="nn_match_att_caliper_0.2sd",
        estimate=estimate,
        se=se,
        n=int(2 * len(pairs)),
        n_treated=int(len(pairs)),
        n_control=int(len(pairs)),
        inference="Paired difference SE among matched pairs; propensity-estimation uncertainty is not included",
        target_population=target_population + "; matched treated units inside the caliper",
        outcome=outcome,
        warnings=(
            "Matching changes the target population when treated units are discarded.",
            "This SE is not an Abadie-Imbens variance.",
        ),
        diagnostics={
            "n_treated_available": n_treated,
            "n_control_available": n_control,
            "n_pairs": int(len(pairs)),
            "n_unmatched_treated": int(n_treated - len(pairs)),
            "caliper": caliper,
            "caliper_rule": "0.2 * SD(logit propensity)",
        },
    )
    validate_result(result)
    return result
