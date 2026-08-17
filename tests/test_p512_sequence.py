"""Focused contract tests for the label-blind P512 sequence serializer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.koff_ml.p512_sequence import (
    CHANNEL_NAMES,
    P512SequenceError,
    fixed_shuffle_permutation,
    serialize_p512_system,
)


def _write_dense(path: Path, *, offset: float) -> None:
    fields = [
        "frame",
        "pocket_geometric_com_distance_A",
        "native_contact_fraction",
        "contact_formations",
        "contact_breaks",
        "pose__aligned_ligand_rmsd_A",
        "pose__aligned_centroid_displacement_A",
        "pose__ligand_internal_conformation_rmsd_A",
        "pose__pocket_alignment_rmsd_A",
        "global__protein_min_heavy_distance_A",
        "global__protein_contact_residue_count",
        "contact__A:1:ALA",
        "contact__A:2:GLY",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        previous = None
        for index in range(600):
            contacts = np.asarray((index < 255, index < 384), dtype=np.int8)
            if previous is None:
                formed = broken = 0
            else:
                formed = int(np.sum((contacts == 1) & (previous == 0)))
                broken = int(np.sum((contacts == 0) & (previous == 1)))
            progress = index / 599
            writer.writerow(
                [
                    index + 1,
                    3.0 + offset + 19.0 * progress,
                    float(np.mean(contacts)),
                    formed,
                    broken,
                    2.0 * progress,
                    18.0 * progress,
                    0.4 + progress,
                    0.05 + 0.2 * progress,
                    3.0 + 10.0 * progress,
                    int(np.sum(contacts)),
                    *contacts.tolist(),
                ]
            )
            previous = contacts


def _write_selection(
    path: Path, *, system_id: str, replica_id: str, include_other_methods: bool = False
) -> None:
    # Skip dense frame index 255, where one contact breaks, while retaining
    # index 256. This distinguishes the frozen dense-transition count from an
    # incorrect recomputation between adjacent nonconsecutive P512 rows.
    source_indices = [*range(255), *range(256, 512), 599]
    assert len(source_indices) == 512
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["system_id", "replica_id", "source_index0", "source_frame"]
        if include_other_methods:
            fieldnames.extend(("method", "selection_rank0"))
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        methods = ("U512", "E512_v2", "P512") if include_other_methods else (None,)
        for method in methods:
            for rank, index in enumerate(source_indices):
                row = {
                    "system_id": system_id,
                    "replica_id": replica_id,
                    "source_index0": index,
                    "source_frame": index + 1,
                }
                if method is not None:
                    row.update({"method": method, "selection_rank0": rank})
                writer.writerow(row)


def _manifest(tmp_path: Path, *, include_other_methods: bool = False) -> Path:
    routes = []
    for position, replica_id in enumerate(("replica_3", "replica_1", "replica_2")):
        dense = tmp_path / f"{replica_id}_dense.csv"
        selection = tmp_path / f"{replica_id}_selection.tsv"
        _write_dense(dense, offset=position / 10)
        _write_selection(
            selection,
            system_id="SYNTHETIC",
            replica_id=replica_id,
            include_other_methods=include_other_methods,
        )
        routes.append(
            {
                "replica_id": replica_id,
                "dense_trace": dense.name,
                "selected_frames": selection.name,
                "endpoint_onset_index0": 599,
                "endpoint_authority_id": f"endpoint::{replica_id}",
                "p512_sampler_authority_id": f"p512::{replica_id}",
                "saved_frame_interval_ps": 1.0,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "ligamd_p512_sequence_input_manifest_v1.0",
                "system_id": "SYNTHETIC",
                "trajectory_protocol_compatibility": "UNKNOWN",
                "routes": routes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_sequence_serializer_preserves_order_and_uses_fixed_row_shuffle(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, include_other_methods=True)
    payload_path = tmp_path / "sequence.npz"
    receipt_path = tmp_path / "sequence_receipt.json"
    receipt = serialize_p512_system(
        manifest_path=manifest,
        output_npz=payload_path,
        receipt_json=receipt_path,
    )

    with np.load(payload_path, allow_pickle=False) as payload:
        ordered = payload["ordered"]
        shuffled = payload["shuffled"]
        permutations = payload["permutation_index0"]
        assert ordered.shape == shuffled.shape == (3, 512, 11)
        assert ordered.dtype == shuffled.dtype == np.float32
        assert tuple(payload["channel_names"].tolist()) == CHANNEL_NAMES
        assert payload["replica_ids"].tolist() == ["replica_1", "replica_2", "replica_3"]
        expected_indices = np.asarray([*range(255), *range(256, 512), 599])
        assert np.array_equal(payload["source_index0"], np.tile(expected_indices, (3, 1)))
        assert np.all(np.diff(ordered[:, :, 0], axis=1) > 0)
        assert ordered[0, 255, 3] == pytest.approx(0.0)
        assert ordered[0, 255, 10] == pytest.approx(0.5)
        assert ordered[0, 383, 3] == pytest.approx(0.5)
        assert ordered[0, 383, 10] == pytest.approx(0.0)
        for row, replica_id in enumerate(payload["replica_ids"].tolist()):
            expected = fixed_shuffle_permutation(
                contract_id="P512_ORDERED_512x11_LABEL_BLIND_V1",
                seed=20260816,
                system_id="SYNTHETIC",
                replica_id=replica_id,
                length=512,
            )
            assert np.array_equal(permutations[row], expected)
            assert np.array_equal(shuffled[row], ordered[row, expected])

    assert receipt["status"] == "PASS_LABEL_BLIND_P512_SEQUENCE_READY_MODEL_RUN_NOT_AUTHORIZED"
    assert receipt["trajectory_protocol_compatibility"] == "UNKNOWN"
    assert receipt["labels_or_folds_read"] is False
    assert receipt["model_run_authorized"] is False
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["shape"] == [3, 512, 11]


def test_sequence_serializer_rejects_selection_identity_drift(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    selection = tmp_path / "replica_1_selection.tsv"
    text = selection.read_text(encoding="utf-8")
    selection.write_text(text.replace("SYNTHETIC", "WRONG", 1), encoding="utf-8")
    with pytest.raises(P512SequenceError, match="system_id mismatch"):
        serialize_p512_system(
            manifest_path=manifest,
            output_npz=tmp_path / "sequence.npz",
            receipt_json=tmp_path / "receipt.json",
        )


def test_sequence_serializer_wraps_dense_schema_failure(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    dense = tmp_path / "replica_1_dense.csv"
    text = dense.read_text(encoding="utf-8")
    text = text.replace("contact__A:1:ALA", "former_contact_a")
    text = text.replace("contact__A:2:GLY", "former_contact_b")
    dense.write_text(text, encoding="utf-8")
    with pytest.raises(P512SequenceError, match="replica_1: dense trace failed"):
        serialize_p512_system(
            manifest_path=manifest,
            output_npz=tmp_path / "sequence.npz",
            receipt_json=tmp_path / "receipt.json",
        )
