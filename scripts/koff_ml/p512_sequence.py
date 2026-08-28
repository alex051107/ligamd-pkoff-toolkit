"""Serialize frozen P512 identities into one label-blind sequence payload.

This developer utility deliberately starts after endpoint detection and P512
selection.  It reads the dense reference-feature table and the already-written
``p512_selected_frames.tsv`` for exactly three replicas, then writes ordered
and deterministically shuffled ``3 x 512 x 11`` tensors.  It never reads an
experimental label, fold assignment, prediction, loss, or model artifact.

The serializer does not scale channels.  A future supervised runner must fit
one scaler on each outer-training split and apply the same scaler to ordered
and shuffled arms.  This module also does not run a temporal model; producing a
valid tensor is an engineering milestone, not authorization to train one.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ligamd_pkoff.resources import bundled_path
from scripts.koff_ml.io import write_json
from scripts.koff_ml.p512_sampler import (
    P512SamplerError,
    P512Trace,
    load_p512_trace,
)


DEFAULT_SEQUENCE_CONTRACT = bundled_path("contracts/p512_sequence_v2.json")

MANIFEST_SCHEMA = "ligamd_p512_sequence_input_manifest_v1.0"
PROTOCOL_STATUSES = {"PASS", "SHIFT", "UNKNOWN"}

_SEQUENCE_CONTRACTS: dict[str, dict[str, str]] = {
    "P512_ORDERED_512x11_LABEL_BLIND_V1": {
        "schema_version": "ligamd_p512_sequence_contract_v1.0",
        "shuffle_mode": "ALL_ROWS",
        "payload_schema_version": "ligamd_p512_sequence_payload_v1.0",
        "receipt_schema_version": "ligamd_p512_sequence_receipt_v1.0",
    },
    "P512_ORDERED_512x11_ENDPOINT_PRESERVING_INTERIOR_SHUFFLE_V2": {
        "schema_version": "ligamd_p512_sequence_contract_v2.0",
        "shuffle_mode": "ENDPOINT_PRESERVING_INTERIOR_PERMUTATION",
        "payload_schema_version": "ligamd_p512_sequence_payload_v2.0",
        "receipt_schema_version": "ligamd_p512_sequence_receipt_v2.0",
    },
}

MANIFEST_FIELDS = {
    "schema_version",
    "system_id",
    "trajectory_protocol_compatibility",
    "routes",
}
ROUTE_FIELDS = {
    "replica_id",
    "dense_trace",
    "selected_frames",
    "endpoint_onset_index0",
    "endpoint_authority_id",
    "p512_sampler_authority_id",
    "dense_trace_receipt_id",
    "selected_identity_receipt_id",
    "saved_frame_interval_ps",
}

CHANNEL_NAMES = (
    "pocket_geometric_com_distance_A",
    "native_contact_fraction",
    "contact_formation_fraction",
    "contact_break_fraction",
    "pose__aligned_ligand_rmsd_A",
    "pose__aligned_centroid_displacement_A",
    "pose__ligand_internal_conformation_rmsd_A",
    "pose__pocket_alignment_rmsd_A",
    "global__protein_min_heavy_distance_A",
    "global__protein_contact_residue_count",
    "pocket_contact_fraction",
)


class P512SequenceError(RuntimeError):
    """Raised when a route violates the frozen sequence contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P512SequenceError(message)


def load_sequence_contract(path: Path = DEFAULT_SEQUENCE_CONTRACT) -> dict[str, Any]:
    """Load and narrowly validate the data-free sequence contract."""

    try:
        contract = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P512SequenceError(f"cannot read sequence contract: {path}") from exc
    contract_id = str(contract.get("contract_id", "")).strip()
    metadata = _SEQUENCE_CONTRACTS.get(contract_id)
    _require(metadata is not None, "unexpected sequence contract id")
    _require(contract.get("schema_version") == metadata["schema_version"], "unexpected sequence contract schema")
    shape = contract.get("shape", {})
    _require(
        (shape.get("replicas_per_system"), shape.get("frames_per_replica"), shape.get("channels_per_frame"))
        == (3, 512, 11),
        "sequence contract shape is not 3 x 512 x 11",
    )
    channels = contract.get("channels", [])
    names = tuple(item.get("name") for item in sorted(channels, key=lambda item: item.get("index0", -1)))
    _require(names == CHANNEL_NAMES, "sequence contract channel order differs from serializer")
    shuffled = contract.get("shuffled_control", {})
    _require(shuffled.get("seed") == 20260816, "unexpected shuffled-control seed")
    _require(shuffled.get("labels_or_folds_read") is False, "shuffle contract is not label-blind")
    shuffle_mode = str(shuffled.get("mode", "ALL_ROWS")).strip().upper()
    _require(shuffle_mode == metadata["shuffle_mode"], "unexpected shuffled-control mode")
    if shuffle_mode == "ENDPOINT_PRESERVING_INTERIOR_PERMUTATION":
        _require(shuffled.get("fixed_rank0") == [0, 511], "v2 shuffle must fix P512 ranks 0 and 511")
        _require(shuffled.get("interior_rank0_range") == [1, 510], "v2 shuffle interior ranks must be 1..510")
        _require(
            shuffled.get("require_nonidentity_interior_permutation") is True,
            "v2 shuffle must require a nonidentity interior permutation",
        )
    return contract


def _sequence_contract_metadata(contract: dict[str, Any]) -> dict[str, str]:
    """Return validated serialization metadata for a loaded sequence contract."""

    contract_id = str(contract.get("contract_id", "")).strip()
    metadata = _SEQUENCE_CONTRACTS.get(contract_id)
    _require(metadata is not None, "unexpected sequence contract id")
    return metadata


def _read_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P512SequenceError(f"cannot read sequence input manifest: {path}") from exc
    _require(isinstance(manifest, dict), "sequence input manifest must be an object")
    _require(set(manifest) == MANIFEST_FIELDS, f"manifest fields must be exactly {sorted(MANIFEST_FIELDS)}")
    _require(manifest["schema_version"] == MANIFEST_SCHEMA, "unexpected sequence input manifest schema")
    system_id = str(manifest["system_id"]).strip()
    _require(bool(system_id), "manifest system_id is blank")
    protocol = str(manifest["trajectory_protocol_compatibility"]).strip().upper()
    _require(protocol in PROTOCOL_STATUSES, "trajectory protocol compatibility must be PASS, SHIFT, or UNKNOWN")
    raw_routes = manifest["routes"]
    _require(isinstance(raw_routes, list) and len(raw_routes) == 3, "manifest must contain exactly three routes")
    routes: list[dict[str, Any]] = []
    for raw in raw_routes:
        _require(isinstance(raw, dict), "each route must be an object")
        unknown = set(raw) - ROUTE_FIELDS
        required = {
            "replica_id",
            "dense_trace",
            "selected_frames",
            "endpoint_onset_index0",
            "endpoint_authority_id",
            "p512_sampler_authority_id",
        }
        _require(not unknown and required <= set(raw), f"route fields are invalid: unknown={sorted(unknown)}, missing={sorted(required - set(raw))}")
        route = dict(raw)
        route["replica_id"] = str(route["replica_id"]).strip()
        _require(bool(route["replica_id"]), "replica_id is blank")
        for field in ("endpoint_authority_id", "p512_sampler_authority_id"):
            route[field] = str(route[field]).strip()
            _require(bool(route[field]), f"{route['replica_id']}: {field} is blank")
        try:
            route["endpoint_onset_index0"] = int(route["endpoint_onset_index0"])
        except (TypeError, ValueError) as exc:
            raise P512SequenceError(f"{route['replica_id']}: endpoint onset is not an integer") from exc
        if route.get("saved_frame_interval_ps") is not None:
            try:
                cadence = float(route["saved_frame_interval_ps"])
            except (TypeError, ValueError) as exc:
                raise P512SequenceError(f"{route['replica_id']}: saved-frame cadence is not numeric") from exc
            _require(np.isfinite(cadence) and cadence > 0, f"{route['replica_id']}: saved-frame cadence must be positive")
            route["saved_frame_interval_ps"] = cadence
        for field in ("dense_trace", "selected_frames"):
            candidate = Path(str(route[field]))
            route[field] = (candidate if candidate.is_absolute() else path.parent / candidate).resolve()
            _require(route[field].is_file(), f"{route['replica_id']}: missing {field}: {route[field]}")
        routes.append(route)
    routes.sort(key=lambda route: route["replica_id"])
    _require(len({route["replica_id"] for route in routes}) == 3, "replica IDs are not unique")
    _require(len({route["dense_trace"] for route in routes}) == 3, "dense trace paths are not unique")
    _require(len({route["selected_frames"] for route in routes}) == 3, "selected-frame paths are not unique")
    normalized = {
        "schema_version": MANIFEST_SCHEMA,
        "system_id": system_id,
        "trajectory_protocol_compatibility": protocol,
    }
    return normalized, routes


def _dense_frame_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        count = sum(1 for _ in handle) - 1
    _require(count > 0, f"dense trace has no data rows: {path}")
    return count


def _read_selection(path: Path, *, system_id: str, replica_id: str) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        rows = list(reader)
    required = {"system_id", "replica_id", "source_index0", "source_frame"}
    _require(required <= fields, f"{replica_id}: selected-frame schema is incomplete")
    if "method" in fields:
        rows = [row for row in rows if str(row.get("method", "")).strip() == "P512"]
        _require(rows, f"{replica_id}: multi-method selection table contains no P512 rows")
    _require(len(rows) == 512, f"{replica_id}: selected-frame row count is {len(rows)}, expected 512")
    _require({str(row["system_id"]).strip() for row in rows} == {system_id}, f"{replica_id}: selected-frame system_id mismatch")
    _require({str(row["replica_id"]).strip() for row in rows} == {replica_id}, f"{replica_id}: selected-frame replica_id mismatch")
    try:
        indices = np.asarray([int(row["source_index0"]) for row in rows], dtype=np.int64)
        frames = np.asarray([int(row["source_frame"]) for row in rows], dtype=np.int64)
        ranks = (
            [int(row["selection_rank0"]) for row in rows]
            if "selection_rank0" in fields
            else None
        )
    except (TypeError, ValueError) as exc:
        raise P512SequenceError(f"{replica_id}: selected-frame identities are not integers") from exc
    if ranks is not None:
        _require(ranks == list(range(512)), f"{replica_id}: P512 selection ranks are not exactly 0..511")
    _require(np.array_equal(frames, indices + 1), f"{replica_id}: source_frame is not source_index0 + 1")
    _require(indices[0] == 0 and np.all(np.diff(indices) > 0), f"{replica_id}: source indices are not strictly increasing from frame 0")
    return indices, frames


def _route_matrix(trace: P512Trace, indices: np.ndarray, *, replica_id: str) -> np.ndarray:
    n_contacts = len(trace.contact_columns)
    _require(n_contacts > 0, f"{replica_id}: trace contains no frozen pocket contacts")
    formed = np.asarray(trace.numeric["contact_formations"], dtype=float) / n_contacts
    broken = np.asarray(trace.numeric["contact_breaks"], dtype=float) / n_contacts
    _require(np.all((formed >= 0) & (formed <= 1)), f"{replica_id}: contact formation fraction leaves [0, 1]")
    _require(np.all((broken >= 0) & (broken <= 1)), f"{replica_id}: contact break fraction leaves [0, 1]")
    _require(formed[0] == 0 and broken[0] == 0, f"{replica_id}: frame-1 contact transitions are not zero")
    native = np.asarray(trace.numeric["native_contact_fraction"], dtype=float)
    pocket_fraction = np.mean(trace.contacts, axis=1, dtype=float)
    _require(np.all((native >= 0) & (native <= 1)), f"{replica_id}: native-contact fraction leaves [0, 1]")
    _require(np.all((pocket_fraction >= 0) & (pocket_fraction <= 1)), f"{replica_id}: pocket-contact fraction leaves [0, 1]")
    matrix = np.column_stack(
        (
            trace.numeric["pocket_geometric_com_distance_A"],
            native,
            formed,
            broken,
            trace.numeric["pose__aligned_ligand_rmsd_A"],
            trace.numeric["pose__aligned_centroid_displacement_A"],
            trace.numeric["pose__ligand_internal_conformation_rmsd_A"],
            trace.numeric["pose__pocket_alignment_rmsd_A"],
            trace.numeric["global__protein_min_heavy_distance_A"],
            trace.numeric["global__protein_contact_residue_count"],
            pocket_fraction,
        )
    )
    selected = np.asarray(matrix[indices], dtype=np.float32)
    _require(selected.shape == (512, 11), f"{replica_id}: selected matrix shape is {selected.shape}")
    _require(np.all(np.isfinite(selected)), f"{replica_id}: selected matrix contains non-finite values")
    nonnegative = selected[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    _require(np.all(nonnegative >= 0), f"{replica_id}: a nonnegative physical channel contains negative values")
    return selected


def _ranked_permutation(
    *, contract_id: str, seed: int, system_id: str, replica_id: str, ranks: range
) -> np.ndarray:
    """Return one deterministic, label-free ordering of the supplied ranks."""

    keyed: list[tuple[bytes, int]] = []
    for rank in ranks:
        payload = "\0".join((contract_id, str(seed), system_id, replica_id, str(rank))).encode("utf-8")
        keyed.append((hashlib.sha256(payload).digest(), rank))
    return np.asarray([rank for _, rank in sorted(keyed, key=lambda item: (item[0], item[1]))], dtype=np.int64)


def fixed_shuffle_permutation(
    *, contract_id: str, seed: int, system_id: str, replica_id: str, length: int
) -> np.ndarray:
    """Return the v1 deterministic permutation over every sequence row."""

    _require(length > 0, "fixed shuffle length must be positive")
    return _ranked_permutation(
        contract_id=contract_id,
        seed=seed,
        system_id=system_id,
        replica_id=replica_id,
        ranks=range(length),
    )


def endpoint_preserving_interior_shuffle_permutation(
    *, contract_id: str, seed: int, system_id: str, replica_id: str, length: int
) -> np.ndarray:
    """Return a v2 permutation that fixes first and endpoint rows.

    The input row multiset is unchanged. Rank 0 and rank ``length - 1`` stay
    at their original positions; only the interior rows are deterministically
    reordered. This prevents a final-state sequence encoder from comparing an
    endpoint-last ordered arm against an arbitrary-last shuffled arm.
    """

    _require(length >= 3, "endpoint-preserving interior shuffle requires at least three rows")
    interior = _ranked_permutation(
        contract_id=contract_id,
        seed=seed,
        system_id=system_id,
        replica_id=replica_id,
        ranks=range(1, length - 1),
    )
    _require(
        not np.array_equal(interior, np.arange(1, length - 1, dtype=np.int64)),
        "endpoint-preserving interior shuffle unexpectedly left every interior row in order",
    )
    return np.concatenate((np.asarray([0], dtype=np.int64), interior, np.asarray([length - 1], dtype=np.int64)))


def _permutation_sha256(permutation: np.ndarray) -> str:
    encoded = np.asarray(permutation, dtype="<i8").tobytes(order="C")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", suffix=".npz", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def serialize_p512_system(
    *,
    manifest_path: Path,
    output_npz: Path,
    receipt_json: Path,
    contract_path: Path = DEFAULT_SEQUENCE_CONTRACT,
) -> dict[str, Any]:
    """Write one three-replica ordered/shuffled payload and its receipt."""

    output_npz = Path(output_npz)
    receipt_json = Path(receipt_json)
    _require(not output_npz.exists(), f"refusing to overwrite sequence payload: {output_npz}")
    _require(not receipt_json.exists(), f"refusing to overwrite sequence receipt: {receipt_json}")
    contract = load_sequence_contract(Path(contract_path))
    manifest, routes = _read_manifest(Path(manifest_path))
    ordered_rows: list[np.ndarray] = []
    shuffled_rows: list[np.ndarray] = []
    indices_rows: list[np.ndarray] = []
    frames_rows: list[np.ndarray] = []
    permutation_rows: list[np.ndarray] = []
    route_receipts: list[dict[str, Any]] = []
    seed = int(contract["shuffled_control"]["seed"])
    contract_id = str(contract["contract_id"])
    contract_metadata = _sequence_contract_metadata(contract)
    shuffle_mode = contract_metadata["shuffle_mode"]
    for route in routes:
        replica_id = route["replica_id"]
        indices, frames = _read_selection(
            route["selected_frames"], system_id=manifest["system_id"], replica_id=replica_id
        )
        _require(indices[-1] == route["endpoint_onset_index0"], f"{replica_id}: final P512 frame is not the frozen endpoint onset")
        frame_count = _dense_frame_count(route["dense_trace"])
        _require(indices[-1] < frame_count, f"{replica_id}: selected source index exceeds dense trace")
        try:
            trace = load_p512_trace(route["dense_trace"], expected_frames=frame_count)
        except P512SamplerError as exc:
            raise P512SequenceError(f"{replica_id}: dense trace failed: {exc}") from exc
        ordered = _route_matrix(trace, indices, replica_id=replica_id)
        if shuffle_mode == "ALL_ROWS":
            permutation = fixed_shuffle_permutation(
                contract_id=contract_id,
                seed=seed,
                system_id=manifest["system_id"],
                replica_id=replica_id,
                length=512,
            )
        else:
            permutation = endpoint_preserving_interior_shuffle_permutation(
                contract_id=contract_id,
                seed=seed,
                system_id=manifest["system_id"],
                replica_id=replica_id,
                length=512,
            )
        _require(len(np.unique(permutation)) == 512, f"{replica_id}: shuffle permutation is not bijective")
        if shuffle_mode == "ENDPOINT_PRESERVING_INTERIOR_PERMUTATION":
            _require(
                permutation[0] == 0 and permutation[-1] == 511,
                f"{replica_id}: v2 shuffle did not preserve first and endpoint ranks",
            )
        shuffled = ordered[permutation]
        ordered_rows.append(ordered)
        shuffled_rows.append(shuffled)
        indices_rows.append(indices)
        frames_rows.append(frames)
        permutation_rows.append(permutation)
        route_receipt = {
            "replica_id": replica_id,
            "dense_trace": str(route["dense_trace"]),
            "selected_frames": str(route["selected_frames"]),
            "endpoint_onset_index0": int(route["endpoint_onset_index0"]),
            "endpoint_authority_id": route["endpoint_authority_id"],
            "p512_sampler_authority_id": route["p512_sampler_authority_id"],
            "dense_trace_receipt_id": route.get("dense_trace_receipt_id"),
            "selected_identity_receipt_id": route.get("selected_identity_receipt_id"),
            "saved_frame_interval_ps": route.get("saved_frame_interval_ps"),
            "permutation_sha256": _permutation_sha256(permutation),
        }
        if shuffle_mode == "ENDPOINT_PRESERVING_INTERIOR_PERMUTATION":
            route_receipt["fixed_rank0"] = [0, 511]
            route_receipt["interior_rank0_range"] = [1, 510]
        route_receipts.append(route_receipt)

    ordered_tensor = np.stack(ordered_rows).astype(np.float32, copy=False)
    shuffled_tensor = np.stack(shuffled_rows).astype(np.float32, copy=False)
    _require(ordered_tensor.shape == shuffled_tensor.shape == (3, 512, 11), "system tensor shape is not 3 x 512 x 11")
    _atomic_npz(
        output_npz,
        ordered=ordered_tensor,
        shuffled=shuffled_tensor,
        source_index0=np.stack(indices_rows),
        source_frame=np.stack(frames_rows),
        permutation_index0=np.stack(permutation_rows),
        channel_names=np.asarray(CHANNEL_NAMES, dtype="U64"),
        replica_ids=np.asarray([route["replica_id"] for route in routes], dtype="U64"),
    )
    receipt = {
        "schema_version": contract_metadata["receipt_schema_version"],
        "payload_schema_version": contract_metadata["payload_schema_version"],
        "status": "PASS_LABEL_BLIND_P512_SEQUENCE_READY_MODEL_RUN_NOT_AUTHORIZED",
        "contract_id": contract_id,
        "system_id": manifest["system_id"],
        "trajectory_protocol_compatibility": manifest["trajectory_protocol_compatibility"],
        "shape": [3, 512, 11],
        "dtype": "float32",
        "channel_names": list(CHANNEL_NAMES),
        "replica_axis_order": [route["replica_id"] for route in routes],
        "serializer_scales_values": False,
        "ordered_and_shuffled_row_budgets_identical": True,
        "shuffle_moves_all_channels_together": True,
        "permutation_seed": seed,
        "permutation_sha256_encoding": "SHA256 of little-endian signed-int64 permutation bytes in C order",
        "routes": route_receipts,
        "output_npz": str(output_npz.resolve()),
        "labels_or_folds_read": False,
        "endpoint_or_sampler_rerun": False,
        "model_run_authorized": False,
        "physical_koff_estimated": False,
        "claim_ceiling": "Label-blind P512 representation engineering only; no model, generalization, or physical-rate conclusion.",
    }
    if shuffle_mode == "ENDPOINT_PRESERVING_INTERIOR_PERMUTATION":
        receipt.update(
            {
                "shuffle_mode": shuffle_mode,
                "shuffle_preserves_boundary_rows": True,
                "fixed_rank0": [0, 511],
                "interior_rank0_range": [1, 510],
            }
        )
    write_json(receipt_json, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Three-route label-blind input manifest JSON")
    parser.add_argument("--output-npz", type=Path, required=True, help="Output ordered/shuffled tensor NPZ")
    parser.add_argument("--receipt-json", type=Path, required=True, help="Output serializer receipt JSON")
    parser.add_argument("--contract", type=Path, default=DEFAULT_SEQUENCE_CONTRACT, help="Frozen data-free sequence contract")
    args = parser.parse_args(argv)
    try:
        receipt = serialize_p512_system(
            manifest_path=args.manifest,
            output_npz=args.output_npz,
            receipt_json=args.receipt_json,
            contract_path=args.contract,
        )
    except P512SequenceError as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
