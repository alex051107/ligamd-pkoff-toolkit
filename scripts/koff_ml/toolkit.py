#!/usr/bin/env python3
"""A small command-line bridge from a LiGaMD trajectory to an experimental-pKoff model input.

The three subcommands deliberately correspond to three different scientific
questions instead of hiding a workflow behind one opaque command:

``featurize``
    Read one new system manifest, freeze one canonical structural reference,
    find the endpoint-v2 episode independently in exactly three replicas,
    take 512 real P512 frames from each episode, and return Static20,
    replica-level Dynamic10, and their three-replica arithmetic mean.

``predict``
    Apply one frozen *experimental* regression pipeline.  It returns a
    predicted experimental pKoff only when the required feature vector is
    present.  It never converts a trajectory duration into physical koff, and
    it never manufactures a per-sample confidence interval.

``evaluate``
    Compare an existing prediction table with an external label table.  This
    is intentionally separate from feature extraction so labels cannot steer
    the endpoint or frame selection.

The implementation reuses the release's shared-reference, PBC geometry,
P512 arclength, Core41, and fold-local-model components.  It adds routing and
receipts; it does not silently create a second set of molecular formulas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

if __package__ in {None, ""}:  # pragma: no cover - direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ligamd_pkoff.resources import bundled_path
from scripts.koff_ml.complete_static_geometric import (
    FEATURES as STATIC20_FEATURES,
    LIGAND_FEATURES,
    _geometry_from_reference,
)
from scripts.koff_ml.coordinate_shards import materialize_coordinate_union
from scripts.koff_ml.endpoint_two_metric import (
    detect_two_metric_endpoint,
    endpoint_spec_from_contract,
)
from scripts.koff_ml.io import sha256_file, write_json
from scripts.koff_ml.p512_sampler import load_p512_trace, p512_path_arclength_indices
from scripts.koff_ml.prediction_first_coordinate_features import (
    derive_coordinate_features,
)
from scripts.koff_ml.prediction_first_coordinate_materialization import (
    _validate_saved_atom_mapping,
)
from scripts.koff_ml.prediction_first_geometric_contact_v2 import (
    G50_TOTAL_CORE41_V2_FEATURE_NAMES,
    summarize_geometric_contact_replica_v2,
)
from scripts.koff_ml.reference_features import extract_reference_features, parse_pdb_atoms
from scripts.koff_ml.serialization_compat import install_legacy_reducer_alias
from scripts.koff_ml.shared_reference import (
    build_shared_reference_manifest,
    load_shared_reference,
    parse_pdb_bytes,
    write_shared_reference,
)
from scripts.koff_ml.typed_interactions import (
    CONTRACT_ID as TYPED_INTERACTION_CONTRACT_ID,
    STATUS as TYPED_INTERACTION_STATUS,
    TypedInteractionError,
    extract_p512_typed_interactions,
)


SCHEMA_VERSION = "ligamd_pkoff_toolkit_v1.0"
MANIFEST_SCHEMA = "ligamd_pkoff_toolkit_input_v1.0"
FEATURES_SCHEMA = "ligamd_pkoff_toolkit_features_v1.0"
PREDICTION_SCHEMA = "ligamd_pkoff_toolkit_prediction_v1.0"
ENDPOINT_CONTRACT_SCHEMA = "ligamd_endpoint_contract_v2.0"
P512_SAMPLER_CONTRACT_SCHEMA = "ligamd_p512_sampler_contract_v1.0"
DEFAULT_ENDPOINT_CONTRACT = bundled_path("contracts/endpoint_v2.json")
DEFAULT_P512_SAMPLER_CONTRACT = bundled_path("contracts/p512_sampler_v1.json")
DEFAULT_TYPED_INTERACTION_CONTRACT = bundled_path(
    "contracts/typed_interactions_core8_v1.json"
)
DEFAULT_MODEL_REGISTRY = bundled_path(
    "models/experimental_n31_registry_v1/model_registry.json"
)
DYNAMIC10_FEATURES = (
    "g50v2__pocket_com_distance_A__progress_000_050__mean",
    "g50v2__native_contact_fraction__progress_050_080__mean",
    "g50v2__ligand_pose_rmsd_A__progress_080_095__mean",
    "g50v2__ligand_pose_rmsd_A__progress_095_100__mean",
    "g50v2__global_min_heavy_distance_A__progress_095_100__mean",
    "g50v2__global_contact_fraction__progress_000_050__mean",
    "g50v2__global_contact_fraction__progress_080_095__mean",
    "g50v2__state_progress_occupancy__I",
    "g50v2__state_progress_occupancy__P_ONLY",
    "g50v2__ligand_internal_rmsd_A__progress_mean",
)


class ToolkitError(RuntimeError):
    """Raised for a missing input or a scientifically invalid workflow state."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ToolkitError(message)


def _resolve(manifest_path: Path, value: object, field: str) -> Path:
    require(isinstance(value, str) and value.strip(), f"manifest field {field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    require(path.is_file(), f"manifest field {field} does not exist: {path}")
    return path


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolkitError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(payload, Mapping), f"{path}: JSON root must be an object")
    return payload


def _read_manifest(path: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    raw = _read_json(path)
    require(raw.get("schema_version") == MANIFEST_SCHEMA, f"manifest schema must be {MANIFEST_SCHEMA}")
    required = {"system_id", "condition_id", "protein_target", "canonical_bound_pdb", "canonical_topology", "ligand", "replicas"}
    missing = sorted(required - set(raw))
    require(not missing, f"manifest is missing fields: {missing}")
    ligand = raw["ligand"]
    require(isinstance(ligand, Mapping), "manifest ligand must be an object")
    require(isinstance(ligand.get("resname"), str) and ligand["resname"].strip(), "ligand.resname is required")
    require(bool(ligand.get("smiles") or ligand.get("sdf")), "ligand requires either audited SMILES or an SDF path")
    resolved_ligand = dict(ligand)
    ligand_chain = resolved_ligand.get("chain")
    if ligand_chain is not None:
        require(isinstance(ligand_chain, str), "ligand.chain must be a string when supplied")
        # PDB readers represent a blank chain as ``_``.  Omission means no
        # chain filter; an explicitly blank chain means the blank PDB chain.
        resolved_ligand["chain"] = ligand_chain.strip() or "_"
    replicas = raw["replicas"]
    require(isinstance(replicas, list) and len(replicas) == 3, "manifest requires exactly three replicas")
    ids: list[str] = []
    resolved_replicas: list[dict[str, Any]] = []
    for item in replicas:
        require(isinstance(item, Mapping), "each replica must be an object")
        replica_id = str(item.get("replica_id", "")).strip()
        require(replica_id, "each replica needs replica_id")
        ids.append(replica_id)
        resolved_replicas.append({
            "replica_id": replica_id,
            "trajectory": _resolve(path, item.get("trajectory"), f"replica[{replica_id}].trajectory"),
            "pdb": _resolve(path, item.get("pdb"), f"replica[{replica_id}].pdb"),
            "topology": _resolve(path, item.get("topology"), f"replica[{replica_id}].topology"),
        })
    require(len(ids) == len(set(ids)), "replica_id values must be unique")
    trajectories = [item["trajectory"] for item in resolved_replicas]
    require(
        len(trajectories) == len(set(trajectories)),
        "replica trajectory paths must be distinct",
    )
    resolved = {
        "system_id": str(raw["system_id"]),
        "condition_id": str(raw["condition_id"]),
        "protein_target": str(raw["protein_target"]),
        "saved_frame_interval_ps": float(raw.get("saved_frame_interval_ps", 1.0)),
        "canonical_bound_pdb": _resolve(path, raw["canonical_bound_pdb"], "canonical_bound_pdb"),
        "canonical_topology": _resolve(path, raw["canonical_topology"], "canonical_topology"),
        "ligand": resolved_ligand,
        "replicas": resolved_replicas,
    }
    require(math.isfinite(resolved["saved_frame_interval_ps"]) and resolved["saved_frame_interval_ps"] > 0, "saved_frame_interval_ps must be finite and positive")
    if resolved["ligand"].get("sdf"):
        resolved["ligand"]["sdf_path"] = _resolve(path, resolved["ligand"]["sdf"], "ligand.sdf")
    return raw, resolved


def _compute_static10(ligand: Mapping[str, Any]) -> dict[str, float]:
    """Compute the audited ligand Static10 order with RDKit, only from chemistry input."""

    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except ImportError as exc:  # clear deployment requirement instead of a silent substitute
        raise ToolkitError("featurize needs RDKit for Static10; install the release environment including rdkit") from exc
    molecule = None
    source = ""
    if ligand.get("smiles"):
        source = str(ligand["smiles"])
        molecule = Chem.MolFromSmiles(source)
    elif ligand.get("sdf_path"):
        source = str(ligand["sdf_path"])
        supplier = Chem.SDMolSupplier(source, removeHs=False)
        molecule = next((item for item in supplier if item is not None), None)
    require(molecule is not None, f"RDKit could not parse ligand chemistry input: {source}")
    molecule = Chem.RemoveHs(molecule)
    values = (
        Descriptors.MolWt(molecule),
        Crippen.MolLogP(molecule),
        Lipinski.NumHDonors(molecule),
        Lipinski.NumHAcceptors(molecule),
        rdMolDescriptors.CalcTPSA(molecule),
        Descriptors.NumRotatableBonds(molecule),
        Lipinski.RingCount(molecule),
        Descriptors.HeavyAtomCount(molecule),
        Chem.GetFormalCharge(molecule),
        rdMolDescriptors.CalcFractionCSP3(molecule),
    )
    result = {name: float(value) for name, value in zip(LIGAND_FEATURES, values, strict=True)}
    require(all(math.isfinite(value) for value in result.values()), "RDKit emitted a non-finite Static10 value")
    return result


def _contract(path: Path) -> Mapping[str, Any]:
    payload = _read_json(path)
    try:
        endpoint_spec_from_contract(payload)
    except ValueError as exc:
        raise ToolkitError(str(exc)) from exc
    return payload


def _p512_sampler_contract(path: Path) -> Mapping[str, Any]:
    """Read the small sampler-only contract used by the public P512 route."""

    payload = _read_json(path)
    require(
        payload.get("schema_version") == P512_SAMPLER_CONTRACT_SCHEMA,
        f"P512 sampler contract schema must be {P512_SAMPLER_CONTRACT_SCHEMA}",
    )
    samplers = payload.get("samplers")
    require(isinstance(samplers, Mapping), "P512 sampler contract must contain samplers")
    shared = samplers.get("shared")
    p512 = samplers.get("P512")
    require(isinstance(shared, Mapping) and isinstance(p512, Mapping), "P512 sampler contract lacks shared or P512 settings")
    require(isinstance(samplers.get("budget"), int), "P512 sampler contract needs integer samplers.budget")
    for key in ("causal_trailing_mean_frames", "scale_prefix_frames"):
        require(isinstance(shared.get(key), int) and int(shared[key]) > 0, f"P512 sampler contract needs positive shared.{key}")
    require(int(samplers["budget"]) == 512, "bundled experimental models require P512 budget = 512")
    return payload


def _write_selection(path: Path, *, system_id: str, replica_id: str, indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("system_id", "replica_id", "source_index0", "source_frame"), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for index in indices:
            writer.writerow({"system_id": system_id, "replica_id": replica_id, "source_index0": int(index), "source_frame": int(index) + 1})


def _dynamic10_from_replica(
    *,
    selected_npz: Path,
    selected_indices0: np.ndarray,
    episode_frames: int,
    pdb_path: Path,
    shared_reference: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Recalculate Core41 from the selected raw coordinates, then name its G10 view."""

    arrays = np.load(selected_npz)
    source_indices = np.asarray(arrays["source_index0"], dtype=np.int64)
    require(np.array_equal(source_indices, selected_indices0), "coordinate union source indices differ from P512 selection")
    atoms = parse_pdb_atoms(pdb_path)
    block = derive_coordinate_features(
        arrays["coordinates_A"],
        arrays["cell_lengths_A"],
        arrays["cell_angles_degree"],
        atoms=atoms,
        shared_reference=shared_reference,
    )
    from scripts.koff_ml.g0_trace_summary_v2 import FROZEN_STATE_CRITERIA

    embedding = summarize_geometric_contact_replica_v2(
        block,
        source_indices0=source_indices,
        episode_frames=episode_frames,
        criteria=FROZEN_STATE_CRITERIA,
    ).as_mapping()
    require(set(DYNAMIC10_FEATURES) <= set(embedding), "Core41 no longer contains the frozen Dynamic10 view")
    return ({name: float(embedding[name]) for name in DYNAMIC10_FEATURES}, {name: float(embedding[name]) for name in G50_TOTAL_CORE41_V2_FEATURE_NAMES})


def _endpoint_authority(
    *, dense_trace: Path, replica_id: str, system_id: str, cadence_ps: float, contract: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Any]:
    table = pd.read_csv(dense_trace)
    required = {"frame", "pose__aligned_centroid_displacement_A", "global__protein_min_heavy_distance_A"}
    require(required <= set(table.columns), f"{replica_id}: dense trace is missing endpoint-v2 columns")
    frames1 = table["frame"].to_numpy(dtype=np.int64)
    numeric = {
        "pose__aligned_centroid_displacement_A": table["pose__aligned_centroid_displacement_A"].to_numpy(dtype=float),
        "global__protein_min_heavy_distance_A": table["global__protein_min_heavy_distance_A"].to_numpy(dtype=float),
    }
    try:
        spec = endpoint_spec_from_contract(contract)
    except ValueError as exc:
        raise ToolkitError(str(exc)) from exc
    result = detect_two_metric_endpoint(numeric, spec, frames1=frames1)
    authority = {
        "schema_version": "ligamd_endpoint_v2_authority_v1.0",
        "status": "PASS" if result.confirmed_exit_onset_index0 is not None else "FAIL_NO_PERSISTENCE_CONFIRMED_COMPLETE_EXIT",
        "system_id": system_id,
        "replica_id": replica_id,
        "contract_id": contract["contract_id"],
        "endpoint_spec": asdict(spec),
        "endpoint_result": asdict(result),
        "episode_policy": {
            "start_index0": 0,
            "end_index0": result.confirmed_exit_onset_index0,
            "end_is_first_persistence_confirmed_complete_exit_onset": True,
            "confirmation_tail_in_model_input": False,
            "post_exit_diffusion_in_model_input": False,
        },
        "saved_frame_interval_ps": cadence_ps,
        "label_blind": True,
        "experimental_pKoff_read": False,
    }
    return authority, result


def featurize(
    manifest_path: Path,
    output_dir: Path,
    endpoint_contract: Path = DEFAULT_ENDPOINT_CONTRACT,
    sampler_contract: Path = DEFAULT_P512_SAMPLER_CONTRACT,
    *,
    typed_interactions: bool = False,
) -> dict[str, Any]:
    """Produce Current30 and, when requested, an unselected typed challenger."""

    require(not output_dir.exists(), f"refusing to overwrite feature output: {output_dir}")
    raw_manifest, manifest = _read_manifest(manifest_path)
    if typed_interactions:
        require(
            manifest["ligand"].get("sdf_path") is not None,
            "--typed-interactions requires ligand.sdf in the input manifest",
        )
    contract = _contract(endpoint_contract)
    output_dir.mkdir(parents=True, exist_ok=False)
    canonical_atoms = parse_pdb_atoms(manifest["canonical_bound_pdb"])
    try:
        saved_atom_mapping_sha256 = _validate_saved_atom_mapping(
            manifest["canonical_bound_pdb"],
            manifest["canonical_topology"],
            len(canonical_atoms),
        )
    except ValueError as exc:
        raise ToolkitError(f"canonical PDB/topology atom mapping failed: {exc}") from exc
    write_json(output_dir / "input_manifest_receipt.json", {
        "schema_version": SCHEMA_VERSION,
        "manifest_schema": raw_manifest["schema_version"],
        "manifest_path": str(manifest_path.resolve()),
        "system_id": manifest["system_id"],
        "condition_id": manifest["condition_id"],
        "replica_ids": [item["replica_id"] for item in manifest["replicas"]],
        "canonical_saved_atom_mapping_sha256": saved_atom_mapping_sha256,
    })
    shared_path = output_dir / "shared_reference.json"
    shared = build_shared_reference_manifest(
        manifest["canonical_bound_pdb"],
        complex_id=manifest["system_id"],
        ligand_resname=str(manifest["ligand"]["resname"]),
        ligand_resid=(None if manifest["ligand"].get("resid") is None else str(manifest["ligand"]["resid"])),
        ligand_chain=(None if manifest["ligand"].get("chain") is None else str(manifest["ligand"]["chain"])),
        topology_path=manifest["canonical_topology"],
        contact_cutoff_A=4.5,
        pocket_cutoff_A=8.0,
    )
    write_shared_reference(shared_path, shared)
    protocol = _p512_sampler_contract(sampler_contract)
    replica_rows: list[dict[str, Any]] = []
    replicate_dynamic: list[dict[str, float]] = []
    replicate_typed: list[dict[str, float]] = []
    failures: list[str] = []
    for replica in manifest["replicas"]:
        replica_id = replica["replica_id"]
        replica_dir = output_dir / "replicas" / replica_id
        replica_dir.mkdir(parents=True)
        dense = replica_dir / "dense_reference_features.csv"
        gate = replica_dir / "dense_reference_gate.json"
        extract_reference_features(
            replica["trajectory"], replica["pdb"], dense, gate,
            ligand_resname=str(manifest["ligand"]["resname"]),
            ligand_resid=(None if manifest["ligand"].get("resid") is None else str(manifest["ligand"]["resid"])),
            ligand_chain=(None if manifest["ligand"].get("chain") is None else str(manifest["ligand"]["chain"])),
            contact_cutoff_A=4.5, pocket_cutoff_A=8.0, complex_id=manifest["system_id"],
            topology_path=replica["topology"], shared_manifest_path=shared_path,
        )
        authority, endpoint = _endpoint_authority(
            dense_trace=dense, replica_id=replica_id, system_id=manifest["system_id"],
            cadence_ps=manifest["saved_frame_interval_ps"], contract=contract,
        )
        authority_path = replica_dir / "endpoint_v2_authority.json"
        write_json(authority_path, authority)
        row = {
            "replica_id": replica_id,
            "dense_trace": str(dense.relative_to(output_dir)),
            "dense_gate": str(gate.relative_to(output_dir)),
            "endpoint_authority": str(authority_path.relative_to(output_dir)),
            "endpoint_status": authority["status"],
            "endpoint_onset_index0": endpoint.confirmed_exit_onset_index0,
            "longest_qualifying_candidate_run_frames": endpoint.longest_run_frames,
        }
        replica_rows.append(row)
        if endpoint.confirmed_exit_onset_index0 is None:
            failures.append(f"{replica_id}: endpoint-v2 persistence not confirmed")
            continue
        onset = int(endpoint.confirmed_exit_onset_index0)
        trace = load_p512_trace(dense, expected_frames=len(pd.read_csv(dense, usecols=["frame"])))
        indices, _, _ = p512_path_arclength_indices(trace, onset, protocol)
        require(len(indices) == 512 and len(np.unique(indices)) == 512, f"{replica_id}: P512 did not return 512 unique real frames")
        selection = replica_dir / "p512_selected_frames.tsv"
        _write_selection(selection, system_id=manifest["system_id"], replica_id=replica_id, indices=indices)
        atoms = parse_pdb_atoms(replica["pdb"])
        union_npz = replica_dir / "p512_coordinate_union.npz"
        union_map = replica_dir / "p512_coordinate_map.csv"
        union_manifest = replica_dir / "p512_coordinate_union_manifest.json"
        materialize_coordinate_union(
            trajectory_path=replica["trajectory"], selected_frames_path=selection,
            output_npz=union_npz, output_map_csv=union_map, output_manifest_json=union_manifest,
            expected_trajectory_sha256=sha256_file(replica["trajectory"]),
            expected_atoms=len(atoms), expected_frames=len(trace.frames1),
            expected_system_id=manifest["system_id"], expected_replica_id=replica_id,
            expected_source_frame_offset=1,
        )
        dynamic10, core41 = _dynamic10_from_replica(
            selected_npz=union_npz, selected_indices0=indices, episode_frames=onset + 1,
            pdb_path=replica["pdb"], shared_reference=shared,
        )
        replicate_dynamic.append(dynamic10)
        typed_result: Mapping[str, Any] | None = None
        if typed_interactions:
            try:
                typed_result = extract_p512_typed_interactions(
                    coordinate_union_npz=union_npz,
                    selected_indices0=indices,
                    episode_onset_index0=onset,
                    topology_path=replica["topology"],
                    pdb_path=replica["pdb"],
                    ligand_template_sdf=manifest["ligand"]["sdf_path"],
                    ligand_resname=str(manifest["ligand"]["resname"]),
                    contract_path=DEFAULT_TYPED_INTERACTION_CONTRACT,
                    system_id=manifest["system_id"],
                    replica_id=replica_id,
                    output_json=replica_dir / "typed_interactions.json",
                )
            except TypedInteractionError as exc:
                raise ToolkitError(f"{replica_id}: typed interactions failed: {exc}") from exc
            typed_values = typed_result.get("features")
            require(
                isinstance(typed_values, Mapping) and len(typed_values) == 32,
                f"{replica_id}: typed extractor did not return 32 features",
            )
            replicate_typed.append(
                {name: float(value) for name, value in typed_values.items()}
            )
        row.update({
            "p512_selected_frames": str(selection.relative_to(output_dir)),
            "p512_coordinate_union": str(union_npz.relative_to(output_dir)),
            "dynamic10": dynamic10,
            "core41_audit_only": core41,
        })
        if typed_result is not None:
            row["typed_interactions"] = str(
                (replica_dir / "typed_interactions.json").relative_to(output_dir)
            )
    events = pd.DataFrame(replica_rows)
    events.to_csv(output_dir / "replica_events.tsv", sep="\t", index=False, lineterminator="\n")
    status_columns = [
        "replica_id", "endpoint_status", "endpoint_onset_index0",
        "longest_qualifying_candidate_run_frames",
    ]
    events.reindex(columns=status_columns).to_csv(
        output_dir / "trajectory_status.tsv", sep="\t", index=False, lineterminator="\n"
    )
    endpoint_columns = [
        "replica_id", "endpoint_authority", "endpoint_status",
        "endpoint_onset_index0", "longest_qualifying_candidate_run_frames",
    ]
    events.reindex(columns=endpoint_columns).to_csv(
        output_dir / "endpoint_events.tsv", sep="\t", index=False, lineterminator="\n"
    )
    if failures or len(replicate_dynamic) != 3:
        payload = {
            "schema_version": FEATURES_SCHEMA,
            "status": "OUT_OF_SCOPE_NO_PREDICTION",
            "reason": "; ".join(failures) or "exactly three endpoint-PASS replicas were not produced",
            "system_id": manifest["system_id"],
            "replica_events": "replica_events.tsv",
            "endpoint_contract": str(endpoint_contract.resolve()),
        }
        write_json(output_dir / "features.json", payload)
        return payload
    dynamic_values = np.asarray([[row[name] for name in DYNAMIC10_FEATURES] for row in replicate_dynamic], dtype=float)
    dynamic10 = {name: float(value) for name, value in zip(DYNAMIC10_FEATURES, np.mean(dynamic_values, axis=0), strict=True)}
    parsed = parse_pdb_bytes(manifest["canonical_bound_pdb"].read_bytes())
    static20 = {**_compute_static10(manifest["ligand"]), **_geometry_from_reference(parsed, shared)}
    require(tuple(static20) == STATIC20_FEATURES, "Static20 feature order differs from frozen release contract")
    feature_payload = {
        "schema_version": FEATURES_SCHEMA,
        "status": "PASS_FEATURES_READY_FOR_EXPERIMENTAL_PREDICTION",
        "system_id": manifest["system_id"],
        "condition_id": manifest["condition_id"],
        "protein_target": manifest["protein_target"],
        "saved_frame_interval_ps": manifest["saved_frame_interval_ps"],
        "endpoint_contract": str(endpoint_contract.resolve()),
        "p512_sampler_contract": str(sampler_contract.resolve()),
        "endpoint_rule": contract["contract_id"],
        "endpoint_spec": asdict(endpoint_spec_from_contract(contract)),
        "sampler": "P512_multiblock_path_arclength",
        "sampler_budget": 512,
        "p512_sampler_contract_id": protocol["contract_id"],
        "p512_sampler_settings": protocol["samplers"],
        "replica_pooling": "arithmetic_mean_of_exactly_three_endpoint_PASS_replicas",
        "replica_count": 3,
        "static20": static20,
        "dynamic10": dynamic10,
        "combined30": {**static20, **dynamic10},
        "replica_events": "replica_events.tsv",
        "shared_reference": "shared_reference.json",
        "scientific_boundaries": {
            "model_status": "EXPERIMENTAL",
            "physical_koff_estimated": False,
            "sampler_selected": False,
            "representation_selected": False,
        },
    }
    if typed_interactions:
        require(
            len(replicate_typed) == 3,
            "typed interactions require exactly three endpoint-PASS replica vectors",
        )
        typed_names = tuple(replicate_typed[0])
        require(
            len(typed_names) == 32
            and all(tuple(values) == typed_names for values in replicate_typed),
            "typed interaction feature order differs across replicas",
        )
        typed_matrix = np.asarray(
            [[values[name] for name in typed_names] for values in replicate_typed],
            dtype=float,
        )
        pooled_typed = {
            name: float(value)
            for name, value in zip(
                typed_names, np.mean(typed_matrix, axis=0), strict=True
            )
        }
        typed_payload = {
            "schema_version": "ligamd_typed_interactions_system_v1.0",
            "contract_id": TYPED_INTERACTION_CONTRACT_ID,
            "status": TYPED_INTERACTION_STATUS,
            "system_id": manifest["system_id"],
            "representation": "GLOBAL_TYPED_FRACTION_STAGE",
            "sampler": "P512_multiblock_path_arclength",
            "replica_pooling": (
                "arithmetic_mean_of_exactly_three_endpoint_PASS_replicas"
            ),
            "replica_count": 3,
            "features": pooled_typed,
            "scientific_boundaries": {
                "experimental_pkoff_read": False,
                "representation_selected": False,
                "model_selected": False,
                "physical_koff_estimated": False,
            },
        }
        write_json(output_dir / "typed_interactions.json", typed_payload)
        pd.DataFrame(
            [{"system_id": manifest["system_id"], **pooled_typed}]
        ).to_csv(
            output_dir / "typed_interactions.tsv",
            sep="\t",
            index=False,
            lineterminator="\n",
        )
        feature_payload["typed_interactions"] = {
            "contract_id": TYPED_INTERACTION_CONTRACT_ID,
            "status": TYPED_INTERACTION_STATUS,
            "representation": "GLOBAL_TYPED_FRACTION_STAGE",
            "selected_for_prediction": False,
            "features": pooled_typed,
            "artifact": "typed_interactions.json",
        }
    write_json(output_dir / "features.json", feature_payload)
    pd.DataFrame([
        {
            "system_id": manifest["system_id"],
            "condition_id": manifest["condition_id"],
            "protein_target": manifest["protein_target"],
            "saved_frame_interval_ps": manifest["saved_frame_interval_ps"],
            **static20,
            **dynamic10,
        }
    ]).to_csv(output_dir / "system_features.tsv", sep="\t", index=False, lineterminator="\n")
    return feature_payload


def _prediction_feature_map(payload: Mapping[str, Any], profile: Mapping[str, Any]) -> Mapping[str, Any]:
    block = str(profile["feature_block"])
    if payload.get("status") != "PASS_FEATURES_READY_FOR_EXPERIMENTAL_PREDICTION":
        raise ToolkitError("feature payload is not endpoint-v2 PASS with exactly three replica vectors")
    if block == "Combined30":
        expected_endpoint = asdict(
            endpoint_spec_from_contract(_contract(DEFAULT_ENDPOINT_CONTRACT))
        )
        expected_sampler = _p512_sampler_contract(DEFAULT_P512_SAMPLER_CONTRACT)
        require(
            payload.get("endpoint_spec") == expected_endpoint,
            "Combined30 requires the bundled endpoint-v2 executable specification",
        )
        require(
            payload.get("p512_sampler_contract_id")
            == expected_sampler["contract_id"]
            and payload.get("p512_sampler_settings")
            == expected_sampler["samplers"],
            "Combined30 requires the bundled P512 executable specification",
        )
        require(
            payload.get("replica_pooling")
            == "arithmetic_mean_of_exactly_three_endpoint_PASS_replicas"
            and payload.get("replica_count") == 3,
            "Combined30 requires arithmetic pooling of exactly three endpoint-PASS replicas",
        )
    value = payload.get("static20") if block == "Static20" else payload.get("combined30")
    require(isinstance(value, Mapping), f"feature payload lacks {block} values")
    return value


def _scope(profile: Mapping[str, Any], values: Mapping[str, Any], payload: Mapping[str, Any]) -> tuple[str, list[str]]:
    missing: list[str] = []
    outside: list[str] = []
    for name in profile["input_feature_order"]:
        raw = values.get(name)
        try:
            number = float(raw)
        except (TypeError, ValueError):
            missing.append(name)
            continue
        if not math.isfinite(number):
            missing.append(name)
            continue
        limits = profile["training_feature_ranges"][name]
        if number < float(limits["minimum"]) or number > float(limits["maximum"]):
            outside.append(name)
    if missing:
        return "OUT_OF_SCOPE_NO_PREDICTION", ["missing_or_nonfinite=" + ",".join(missing)]
    target = str(payload.get("protein_target", ""))
    cadence_reasons: list[str] = []
    if str(profile.get("feature_block")) == "Combined30":
        cadence = payload.get("saved_frame_interval_ps")
        if cadence is None:
            cadence_reasons.append("missing_saved_frame_interval_ps")
        elif profile.get("training_saved_frame_interval_ps") is None:
            # The N31 registry did not preserve a single authoritative saved
            # cadence. Do not silently present a dynamic trajectory feature
            # vector as directly comparable when that metadata is missing.
            cadence_reasons.append("training_cadence_not_reconstructed_for_frozen_N31")
        elif not math.isclose(float(cadence), float(profile["training_saved_frame_interval_ps"]), rel_tol=0.0, abs_tol=1e-12):
            cadence_reasons.append("saved_frame_interval_ps_differs_from_training_contract")
    if target not in set(profile["training_protein_targets"]) or outside or cadence_reasons:
        reasons = ([] if target in set(profile["training_protein_targets"]) else [f"unseen_protein_target={target}"])
        if outside:
            reasons.append("outside_training_feature_range=" + ",".join(outside))
        reasons.extend(cadence_reasons)
        return "CAUTION_OUTSIDE_OBSERVED_RANGE", reasons
    return "WITHIN_OBSERVED_SCOPE", []


def predict(
    *,
    features_path: Path,
    registry_path: Path = DEFAULT_MODEL_REGISTRY,
    model_id: str,
    output_path: Path,
) -> Mapping[str, Any]:
    registry = _read_json(registry_path)
    require(registry.get("scientific_status") == "EXPERIMENTAL", "registry is not an experimental model registry")
    profiles = registry.get("profiles")
    require(isinstance(profiles, Mapping) and model_id in profiles, f"unknown model_id: {model_id}")
    profile = profiles[model_id]
    require(isinstance(profile, Mapping), "registry profile is malformed")
    payload = _read_json(features_path)
    try:
        values = _prediction_feature_map(payload, profile)
        status, reasons = _scope(profile, values, payload)
    except ToolkitError as exc:
        status, reasons, values = "OUT_OF_SCOPE_NO_PREDICTION", [str(exc)], {}
    result: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA,
        "system_id": payload.get("system_id"),
        "condition_id": payload.get("condition_id"),
        "protein_target": payload.get("protein_target"),
        "model_id": model_id,
        "model_status": "EXPERIMENTAL",
        "prediction_scope": status,
        "scope_reasons": reasons,
        "target_definition": "experimental_pKoff",
        "physical_koff_estimated": False,
        "per_sample_confidence_interval": None,
    }
    if status == "OUT_OF_SCOPE_NO_PREDICTION":
        result["predicted_experimental_pKoff"] = None
    else:
        artifact = (registry_path.parent / str(profile["artifact"])).resolve()
        require(artifact.is_file(), f"model artifact is missing: {artifact}")
        matrix = np.asarray([[float(values[name]) for name in profile["input_feature_order"]]], dtype=float)
        # The exported pipelines contain the release's fold-local reducer.  It
        # is loaded by a stable local module name before joblib unpickles the
        # estimator, so a fresh checkout does not depend on the original model
        # building process or an external workspace path.
        install_legacy_reducer_alias()
        estimator = joblib.load(artifact)
        predicted = float(np.asarray(estimator.predict(matrix), dtype=float)[0])
        require(math.isfinite(predicted), "model emitted a non-finite prediction")
        result["predicted_experimental_pKoff"] = predicted
    write_json(output_path, result)
    return result


def evaluate(*, predictions_path: Path, labels_path: Path, output_path: Path) -> Mapping[str, Any]:
    predictions = _read_json(predictions_path)
    labels = pd.read_csv(labels_path, sep="\t" if labels_path.suffix.lower() == ".tsv" else ",")
    require({"system_id", "experimental_pKoff"} <= set(labels.columns), "label table needs system_id and experimental_pKoff")
    system_id = str(predictions.get("system_id", ""))
    match = labels.loc[labels["system_id"].astype(str) == system_id]
    require(len(match) == 1, f"label table must contain exactly one row for {system_id}")
    value = predictions.get("predicted_experimental_pKoff")
    require(value is not None, "OUT_OF_SCOPE prediction cannot be evaluated")
    observed = float(match.iloc[0]["experimental_pKoff"])
    predicted = float(value)
    error = predicted - observed
    result = {
        "schema_version": "ligamd_pkoff_toolkit_evaluation_v1.0",
        "system_id": system_id,
        "experimental_pKoff": observed,
        "predicted_experimental_pKoff": predicted,
        "error_pred_minus_exp": error,
        "absolute_error": abs(error),
        "squared_error": error * error,
        "interpretation": "one held-out-like comparison, not a new validation cohort or a physical-koff calculation",
    }
    write_json(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    featurize_parser = commands.add_parser("featurize", help="build Current30 from a raw LiGaMD manifest")
    featurize_parser.add_argument("--manifest", type=Path, required=True)
    featurize_parser.add_argument("--output-dir", type=Path, required=True)
    featurize_parser.add_argument("--endpoint-contract", type=Path, default=DEFAULT_ENDPOINT_CONTRACT)
    featurize_parser.add_argument("--sampler-contract", type=Path, default=DEFAULT_P512_SAMPLER_CONTRACT)
    featurize_parser.add_argument(
        "--typed-interactions",
        action="store_true",
        help=(
            "also calculate the unselected CORE8 GLOBAL_TYPED_FRACTION_STAGE "
            "challenger from the existing P512 union"
        ),
    )
    predict_parser = commands.add_parser("predict", help="apply one experimental registry profile")
    predict_parser.add_argument("--features", type=Path, required=True)
    predict_parser.add_argument("--registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    predict_parser.add_argument("--model-id", default="combined30_p512_ridge")
    predict_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = commands.add_parser("evaluate", help="compare one prediction with an external pKoff table")
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--labels", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "featurize":
            result = featurize(
                args.manifest,
                args.output_dir,
                args.endpoint_contract,
                args.sampler_contract,
                typed_interactions=args.typed_interactions,
            )
        elif args.command == "predict":
            result = predict(features_path=args.features, registry_path=args.registry, model_id=args.model_id, output_path=args.output)
        else:
            result = evaluate(predictions_path=args.predictions, labels_path=args.labels, output_path=args.output)
    except (OSError, ValueError, ToolkitError) as exc:
        # A raw-coordinate failure should leave a readable receipt beside any
        # partial dense/reference files.  It is deliberately a terminal status:
        # no incomplete replica set is padded or converted into a prediction.
        if args.command == "featurize" and args.output_dir.exists():
            write_json(args.output_dir / "features.json", {
                "schema_version": FEATURES_SCHEMA,
                "status": "OUT_OF_SCOPE_NO_PREDICTION",
                "reason": str(exc),
                "endpoint_contract": str(args.endpoint_contract.resolve()),
            })
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "FAIL", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
