"""Small offline smoke test for the public prediction/evaluation CLI.

Raw NetCDF featurization is covered by the existing shared-reference,
reference-feature, P512, coordinate-union and Core41 module tests.  This test
checks the new public hand-off: a feature receipt produced by ``featurize`` is
accepted by ``predict``, and the prediction can be compared by ``evaluate``
without feeding the label back into the model command.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import netcdf_file

from ligamd_pkoff.resources import bundled_path
from scripts.koff_ml import toolkit
from scripts.koff_ml.complete_static_geometric import LIGAND_FEATURES


REGISTRY = bundled_path("models/experimental_n31_registry_v1/model_registry.json")


def _midpoint_features(profile: dict[str, object]) -> dict[str, float]:
    ranges = profile["training_feature_ranges"]
    return {
        name: (float(bounds["minimum"]) + float(bounds["maximum"])) / 2.0
        for name, bounds in ranges.items()
    }


def test_predict_and_evaluate_public_receipt(tmp_path: Path) -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    profile = registry["profiles"]["combined30_p512_ridge"]
    all_values = _midpoint_features(profile)
    static = {name: all_values[name] for name in profile["input_feature_order"][:20]}
    dynamic = {name: all_values[name] for name in profile["input_feature_order"][20:]}
    features_path = tmp_path / "features.json"
    features_path.write_text(
        json.dumps(
            {
                "schema_version": "ligamd_pkoff_toolkit_features_v1.0",
                "status": "PASS_FEATURES_READY_FOR_EXPERIMENTAL_PREDICTION",
                "system_id": "synthetic-smoke-system",
                "condition_id": "synthetic-condition",
                "protein_target": profile["training_protein_targets"][0],
                "saved_frame_interval_ps": 1.0,
                "static20": static,
                "combined30": {**static, **dynamic},
            }
        ),
        encoding="utf-8",
    )
    prediction_path = tmp_path / "prediction.json"
    assert toolkit.main([
        "predict", "--features", str(features_path),
        "--model-id", "combined30_p512_ridge", "--output", str(prediction_path),
    ]) == 0
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert prediction["prediction_scope"] == "CAUTION_OUTSIDE_OBSERVED_RANGE"
    assert "training_cadence_not_reconstructed_for_frozen_N31" in prediction["scope_reasons"]
    assert isinstance(prediction["predicted_experimental_pKoff"], float)
    assert prediction["per_sample_confidence_interval"] is None

    labels_path = tmp_path / "labels.tsv"
    pd.DataFrame([
        {"system_id": "synthetic-smoke-system", "experimental_pKoff": 1.5}
    ]).to_csv(labels_path, sep="\t", index=False)
    evaluation_path = tmp_path / "evaluation.json"
    assert toolkit.main([
        "evaluate", "--predictions", str(prediction_path), "--labels", str(labels_path),
        "--output", str(evaluation_path),
    ]) == 0
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert evaluation["experimental_pKoff"] == 1.5
    assert evaluation["squared_error"] >= 0.0

    static_prediction_path = tmp_path / "static_prediction.json"
    assert toolkit.main([
        "predict", "--features", str(features_path),
        "--model-id", "static20_ridge", "--output", str(static_prediction_path),
    ]) == 0
    static_prediction = json.loads(static_prediction_path.read_text(encoding="utf-8"))
    assert static_prediction["prediction_scope"] == "WITHIN_OBSERVED_SCOPE"


def test_endpoint_contract_controls_the_public_runtime_predicate(tmp_path: Path) -> None:
    """A supplied contract must change the actual predicate, not only its receipt."""

    contract = json.loads(bundled_path("contracts/endpoint_v2.json").read_text(encoding="utf-8"))
    contract["frame_conditions_all_required"][0]["threshold_A"] = 17.0
    contract["persistence"]["consecutive_saved_frames"] = 3
    dense_trace = tmp_path / "dense.csv"
    pd.DataFrame(
        {
            "frame": np.arange(1, 6),
            "pose__aligned_centroid_displacement_A": [16.0] * 5,
            "global__protein_min_heavy_distance_A": [10.1] * 5,
        }
    ).to_csv(dense_trace, index=False)
    authority, result = toolkit._endpoint_authority(
        dense_trace=dense_trace,
        replica_id="r1",
        system_id="contract-test",
        cadence_ps=1.0,
        contract=contract,
    )
    assert authority["endpoint_spec"]["displacement_min_A"] == 17.0
    assert result.confirmed_exit_onset_index0 is None


def _pdb_line(record: str, serial: int, atom: str, residue: str, resid: int, xyz: tuple[float, float, float], element: str) -> str:
    x, y, z = xyz
    return (
        f"{record:<6}{serial:>5} {atom:<4} {residue:>3} A{resid:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2}\n"
    )


def _write_minimal_raw_fixture(root: Path) -> Path:
    """Create three tiny-but-real raw inputs for an offline coordinate smoke test."""

    pdb = root / "canonical.pdb"
    pdb.write_text(
        "CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n"
        + _pdb_line("ATOM", 1, "N", "ALA", 1, (0.0, 0.0, 0.0), "N")
        + _pdb_line("ATOM", 2, "CA", "ALA", 1, (1.5, 0.0, 0.0), "C")
        + _pdb_line("ATOM", 3, "C", "ALA", 1, (1.5, 1.5, 0.0), "C")
        + _pdb_line("HETATM", 4, "C1", "LIG", 501, (3.0, 0.0, 0.0), "C")
        + _pdb_line("HETATM", 5, "O1", "LIG", 501, (3.0, 1.0, 0.0), "O")
        + "END\n",
        encoding="utf-8",
    )
    topology = root / "canonical.parm7"
    topology.write_text("synthetic topology identity only\n", encoding="utf-8")
    frames = 620
    coordinates = np.zeros((frames, 5, 3), dtype=np.float32)
    coordinates[:, 0] = (0.0, 0.0, 0.0)
    coordinates[:, 1] = (1.5, 0.0, 0.0)
    coordinates[:, 2] = (1.5, 1.5, 0.0)
    coordinates[:, 3] = (3.0, 0.0, 0.0)
    coordinates[:, 4] = (3.0, 1.0, 0.0)
    # At source index 520, the ligand has left the original pocket and is
    # more than 10 Å from every protein atom for a full 100 saved frames.
    coordinates[520:, 3] = (30.0, 0.0, 0.0)
    coordinates[520:, 4] = (30.0, 1.0, 0.0)
    for replica in ("r1", "r2", "r3"):
        trajectory = root / f"{replica}.nc"
        with netcdf_file(str(trajectory), "w") as nc:
            nc.createDimension("frame", frames)
            nc.createDimension("atom", 5)
            nc.createDimension("spatial", 3)
            nc.createVariable("coordinates", "f", ("frame", "atom", "spatial"))[:] = coordinates
            nc.createVariable("cell_lengths", "f", ("frame", "spatial"))[:] = 100.0
            nc.createVariable("cell_angles", "f", ("frame", "spatial"))[:] = 90.0
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "ligamd_pkoff_toolkit_input_v1.0",
                "system_id": "synthetic-raw-system",
                "condition_id": "synthetic-condition",
                "protein_target": "synthetic-target",
                "saved_frame_interval_ps": 1.0,
                "canonical_bound_pdb": "canonical.pdb",
                "canonical_topology": "canonical.parm7",
                "ligand": {"resname": "LIG", "resid": "501", "chain": "A", "smiles": "CCO"},
                "replicas": [
                    {"replica_id": replica, "trajectory": f"{replica}.nc", "pdb": "canonical.pdb", "topology": "canonical.parm7"}
                    for replica in ("r1", "r2", "r3")
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_featurize_runs_full_three_replica_coordinate_path_with_synthetic_raw_input(tmp_path: Path, monkeypatch) -> None:
    manifest = _write_minimal_raw_fixture(tmp_path)
    # The production environment supplies RDKit through environment.yml.  This
    # local smoke test deliberately mocks only Static10 because the current
    # lightweight test interpreter does not install RDKit; the PBC/endpoint/
    # P512/Core41 calculation below is the real release implementation.
    monkeypatch.setattr(
        toolkit,
        "_compute_static10",
        lambda ligand: {name: float(index + 1) for index, name in enumerate(LIGAND_FEATURES)},
    )
    output = tmp_path / "features"
    assert toolkit.main([
        "featurize", "--manifest", str(manifest), "--output-dir", str(output),
    ]) == 0
    features = json.loads((output / "features.json").read_text(encoding="utf-8"))
    assert features["status"] == "PASS_FEATURES_READY_FOR_EXPERIMENTAL_PREDICTION"
    assert features["replica_count"] == 3
    assert features["saved_frame_interval_ps"] == 1.0
    assert len(features["static20"]) == 20
    assert len(features["dynamic10"]) == 10
    events = pd.read_csv(output / "replica_events.tsv", sep="\t")
    assert events["endpoint_onset_index0"].tolist() == [520, 520, 520]
    assert "p512_selected_frames" in events.columns
    assert (output / "trajectory_status.tsv").is_file()
    assert (output / "endpoint_events.tsv").is_file()
    system_features = pd.read_csv(output / "system_features.tsv", sep="\t")
    assert len(system_features) == 1
    assert len(system_features.columns) == 34
