"""Regression checks for the public endpoint-v2 geometry rule."""

from __future__ import annotations

import json
import numpy as np

from ligamd_pkoff.resources import bundled_path

from scripts.koff_ml.endpoint_two_metric import (
    TwoMetricEndpointSpec,
    detect_two_metric_endpoint,
    endpoint_spec_from_contract,
    two_metric_exit_mask,
)


def _trace(n: int = 240) -> dict[str, np.ndarray]:
    return {
        "pose__aligned_centroid_displacement_A": np.full(n, 4.0),
        "global__protein_min_heavy_distance_A": np.full(n, 3.0),
    }


def test_endpoint_is_first_persistent_onset_not_end_of_confirmation_tail() -> None:
    numeric = _trace()
    numeric["pose__aligned_centroid_displacement_A"][50:190] = 15.0
    numeric["global__protein_min_heavy_distance_A"][50:190] = 10.1
    result = detect_two_metric_endpoint(
        numeric, TwoMetricEndpointSpec(), frames1=np.arange(1, 241)
    )
    assert result.status == "FIRST_PERSISTENCE_CONFIRMED_TWO_METRIC_EXIT_ONSET"
    assert result.confirmed_exit_onset_index0 == 50
    assert result.confirmation_index0 == 149
    assert result.longest_run_frames == 140


def test_both_geometry_conditions_and_full_persistence_are_required() -> None:
    numeric = _trace()
    numeric["pose__aligned_centroid_displacement_A"][50:150] = 15.0
    numeric["global__protein_min_heavy_distance_A"][50:149] = 10.1  # only 99 joint frames
    mask = two_metric_exit_mask(numeric, TwoMetricEndpointSpec())
    assert int(mask.sum()) == 99
    result = detect_two_metric_endpoint(
        numeric, TwoMetricEndpointSpec(), frames1=np.arange(1, 241)
    )
    assert result.status == "TWO_METRIC_EXIT_OBSERVED_BUT_NOT_PERSISTENT"
    assert result.confirmed_exit_onset_index0 is None


def test_machine_contract_and_python_rule_cannot_drift() -> None:
    contract = json.loads(bundled_path("contracts/endpoint_v2.json").read_text())
    spec = TwoMetricEndpointSpec()
    conditions = {item["name"]: item for item in contract["frame_conditions_all_required"]}
    assert conditions["pocket_aligned_ligand_centroid_displacement"]["threshold_A"] == spec.displacement_min_A
    assert conditions["whole_protein_heavy_atom_clearance"]["threshold_A"] == spec.whole_protein_min_distance_strictly_greater_A
    assert contract["persistence"]["consecutive_saved_frames"] == spec.persistence_frames


def test_contract_parser_is_the_runtime_authority() -> None:
    contract = json.loads(bundled_path("contracts/endpoint_v2.json").read_text())
    contract["frame_conditions_all_required"][0]["threshold_A"] = 17.0
    contract["persistence"]["consecutive_saved_frames"] = 3
    spec = endpoint_spec_from_contract(contract)
    assert spec.displacement_min_A == 17.0
    assert spec.persistence_frames == 3
