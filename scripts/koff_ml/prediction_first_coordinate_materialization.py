#!/usr/bin/env python3
"""Materialize frozen prediction-first U/E frame identities from Amber NetCDF.

This is an engineering bridge, not a sampler decision.  It converts the
prediction-first column names to the canonical coordinate-shard schema, binds
each replica to both its matching saved-solute PDB and its force-field parm7,
and then reuses :mod:`coordinate_shards` to extract a deduplicated coordinate
union without changing any method's declared frame budget.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .coordinate_shards import materialize_coordinate_union
from .io import sha256_file, write_json


SCHEMA_VERSION = "ligamd_prediction_first_coordinate_materialization_v1"
METHOD_PATTERN = re.compile(r"^[UER](\d+)_")
REQUIRED_IDENTITY_COLUMNS = {
    "system_id",
    "replica_id",
    "method",
    "selection_rank0",
    "source_frame_index0",
    "source_frame_1based",
    "normalized_progress_to_B_onset",
    "segment",
}
REQUIRED_REPLICA_COLUMNS = {
    "system_id",
    "replica_id",
    "trajectory_path",
    "trajectory_sha256",
    "coordinate_topology_pdb_path",
    "coordinate_topology_pdb_sha256",
    "forcefield_topology_parm7_path",
    "forcefield_topology_parm7_sha256",
    "expected_frames",
    "expected_saved_atoms",
    "lineage",
}


def _read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = list(reader.fieldnames or ())
        return fields, list(reader)


def _canonical_int(value: object, name: str, *, minimum: int = 0) -> int:
    text = str(value).strip()
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical integer, observed {value!r}") from exc
    if text != str(parsed) or parsed < minimum:
        raise ValueError(f"{name} must be a canonical integer >= {minimum}, observed {value!r}")
    return parsed


def _count_pdb_atoms(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")):
                count += 1
    if count < 1:
        raise ValueError(f"coordinate topology contains no ATOM/HETATM records: {path}")
    return count


def _pdb_atom_schema(path: Path) -> list[tuple[str, str]]:
    schema: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")):
                schema.append((line[12:16].strip(), line[17:21].strip()))
    if not schema or any(not atom or not residue for atom, residue in schema):
        raise ValueError(f"coordinate topology contains an invalid atom schema: {path}")
    return schema


def _parm7_flag_fields(path: Path, flag: str) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    marker = f"%FLAG {flag}"
    try:
        marker_index = lines.index(marker)
    except ValueError as exc:
        raise ValueError(f"parm7 lacks required flag {flag}: {path}") from exc
    if marker_index + 1 >= len(lines):
        raise ValueError(f"parm7 flag {flag} lacks a format line: {path}")
    format_line = lines[marker_index + 1].strip()
    match = re.fullmatch(r"%FORMAT\(\s*\d+([aAiI])([0-9]+)(?:\.[0-9]+)?\s*\)", format_line)
    if match is None:
        raise ValueError(f"unsupported parm7 format for {flag}: {format_line!r}")
    width = int(match.group(2))
    fields: list[str] = []
    for line in lines[marker_index + 2 :]:
        if line.startswith("%FLAG "):
            break
        fields.extend(line[offset : offset + width].strip() for offset in range(0, len(line), width))
    return [value for value in fields if value]


def _validate_saved_atom_mapping(pdb_path: Path, parm7_path: Path, expected_atoms: int) -> str:
    """Bind every saved NetCDF atom to the same ordered parm7 atom identity.

    AMBER writes the first ``ntwprt`` topology atoms to the trajectory.  The
    saved-solute PDB was generated from that same ordering.  Matching atom and
    residue names for every saved position therefore rejects a same-sized but
    semantically different topology before any structural features are built.
    """

    pdb_schema = _pdb_atom_schema(pdb_path)
    atom_names = _parm7_flag_fields(parm7_path, "ATOM_NAME")
    residue_labels = _parm7_flag_fields(parm7_path, "RESIDUE_LABEL")
    try:
        residue_pointers = [int(value) for value in _parm7_flag_fields(parm7_path, "RESIDUE_POINTER")]
        pointers = [int(value) for value in _parm7_flag_fields(parm7_path, "POINTERS")]
    except ValueError as exc:
        raise ValueError(f"parm7 contains non-integer topology pointers: {parm7_path}") from exc
    if not pointers or pointers[0] != len(atom_names):
        raise ValueError("parm7 NATOM does not match its ATOM_NAME field")
    if len(pdb_schema) != expected_atoms or len(atom_names) < expected_atoms:
        raise ValueError(
            f"saved atom mapping length mismatch: pdb={len(pdb_schema)} "
            f"parm7={len(atom_names)} expected={expected_atoms}"
        )
    if len(residue_labels) != len(residue_pointers) or not residue_pointers:
        raise ValueError("parm7 residue labels/pointers are incomplete")
    if residue_pointers[0] != 1 or any(
        right <= left for left, right in zip(residue_pointers, residue_pointers[1:])
    ):
        raise ValueError("parm7 RESIDUE_POINTER is not strictly increasing from atom 1")

    parm_schema: list[tuple[str, str]] = []
    residue_index = 0
    for atom_index1, atom_name in enumerate(atom_names[:expected_atoms], start=1):
        while (
            residue_index + 1 < len(residue_pointers)
            and residue_pointers[residue_index + 1] <= atom_index1
        ):
            residue_index += 1
        parm_schema.append((atom_name, residue_labels[residue_index]))
    for atom_index1, (pdb_identity, parm_identity) in enumerate(
        zip(pdb_schema, parm_schema), start=1
    ):
        if pdb_identity != parm_identity:
            raise ValueError(
                "saved-solute PDB / parm7 atom-order mismatch at saved atom "
                f"{atom_index1}: pdb={pdb_identity!r} parm7={parm_identity!r}"
            )
    payload = "\n".join(f"{atom}\t{residue}" for atom, residue in parm_schema).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _method_budget(method: str) -> int:
    match = METHOD_PATTERN.match(method)
    if match is None:
        raise ValueError(f"method does not encode a frozen budget: {method!r}")
    return int(match.group(1))


def canonicalize_selected_identities(
    source: Path,
    destination: Path,
    *,
    allowed_methods: set[str] | None = None,
) -> tuple[list[dict[str, str]], dict[tuple[str, str], list[dict[str, str]]]]:
    fields, rows = _read_table(source)
    missing = sorted(REQUIRED_IDENTITY_COLUMNS - set(fields))
    if missing:
        raise ValueError(f"selected identities are missing columns: {missing}")
    if allowed_methods is not None:
        observed_methods = {row.get("method", "").strip() for row in rows}
        missing_methods = sorted(allowed_methods - observed_methods)
        if missing_methods:
            raise ValueError(f"requested methods are absent: {missing_methods}")
        rows = [row for row in rows if row.get("method", "").strip() in allowed_methods]
    if not rows:
        raise ValueError("selected identities are empty")

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    replica_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    canonical: list[dict[str, str]] = []
    for row in rows:
        system = row["system_id"].strip()
        replica = row["replica_id"].strip()
        method = row["method"].strip()
        if not system or not replica or not method:
            raise ValueError("system_id, replica_id, and method must be non-empty")
        rank = _canonical_int(row["selection_rank0"], "selection_rank0")
        index0 = _canonical_int(row["source_frame_index0"], "source_frame_index0")
        frame1 = _canonical_int(row["source_frame_1based"], "source_frame_1based", minimum=1)
        if frame1 != index0 + 1:
            raise ValueError("source_frame_1based must equal source_frame_index0 + 1")
        try:
            progress = float(row["normalized_progress_to_B_onset"])
        except ValueError as exc:
            raise ValueError("normalized progress must be numeric") from exc
        if not np.isfinite(progress) or not 0.0 <= progress <= 1.0:
            raise ValueError("normalized progress must be finite and inside [0, 1]")
        converted = {
            **row,
            "system_id": system,
            "replica_id": replica,
            "method": method,
            "selection_rank0": str(rank),
            "source_index0": str(index0),
            "source_frame": str(frame1),
        }
        canonical.append(converted)
        groups.setdefault((system, replica, method), []).append(converted)
        replica_rows.setdefault((system, replica), []).append(converted)

    for (system, replica, method), group in groups.items():
        budget = _method_budget(method)
        ranks = [_canonical_int(row["selection_rank0"], "selection_rank0") for row in group]
        indices = np.asarray(
            [_canonical_int(row["source_index0"], "source_index0") for row in group],
            dtype=np.int64,
        )
        if len(group) != budget or ranks != list(range(budget)):
            raise ValueError(
                f"{system}/{replica}/{method} does not contain exact ranks 0..{budget - 1}"
            )
        if len(np.unique(indices)) != budget or np.any(np.diff(indices) <= 0):
            raise ValueError(f"{system}/{replica}/{method} indices are not unique/increasing")
        if int(indices[0]) != 0:
            raise ValueError(f"{system}/{replica}/{method} omits production frame 0")
        progress = np.asarray(
            [float(row["normalized_progress_to_B_onset"]) for row in group], dtype=float
        )
        if not np.isclose(progress[0], 0.0) or not np.isclose(progress[-1], 1.0):
            raise ValueError(f"{system}/{replica}/{method} omits a normalized endpoint")

    destination.parent.mkdir(parents=True, exist_ok=True)
    output_fields = [*fields]
    for name in ("source_index0", "source_frame"):
        if name not in output_fields:
            output_fields.append(name)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=output_fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in output_fields} for row in canonical)
    return canonical, replica_rows


def _load_replica_manifest(path: Path) -> list[dict[str, str]]:
    fields, rows = _read_table(path)
    missing = sorted(REQUIRED_REPLICA_COLUMNS - set(fields))
    if missing:
        raise ValueError(f"replica manifest is missing columns: {missing}")
    keys = [(row["system_id"].strip(), row["replica_id"].strip()) for row in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError("replica manifest is empty or has duplicate system/replica rows")
    return rows


def _file_binding(path_text: str, expected_sha256: str, description: str) -> Path:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"{description} SHA-256 mismatch: expected={expected_sha256} observed={observed}"
        )
    return path


def _require_sha256(path: Path, expected: str, description: str) -> str:
    normalized = str(expected).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{description} expected SHA-256 is not a 64-character hex digest")
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    observed = sha256_file(path)
    if observed != normalized:
        raise ValueError(
            f"{description} SHA-256 mismatch: expected={normalized} observed={observed}"
        )
    return observed


def run(
    *,
    selected_identities: Path,
    replica_manifest: Path,
    output_dir: Path,
    expected_selected_identities_sha256: str,
    expected_replica_manifest_sha256: str,
    only_key: tuple[str, str] | None = None,
    methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected_identities_sha256 = _require_sha256(
        selected_identities,
        expected_selected_identities_sha256,
        "frozen selected identities",
    )
    replica_manifest_sha256 = _require_sha256(
        replica_manifest,
        expected_replica_manifest_sha256,
        "frozen coordinate input manifest",
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent))
    try:
        bridge = partial / "selected_frame_bridge.tsv"
        allowed_methods = None if methods is None else set(methods)
        if allowed_methods is not None and len(allowed_methods) != len(methods):
            raise ValueError("--method values must be unique")
        _, selected_by_replica = canonicalize_selected_identities(
            selected_identities, bridge, allowed_methods=allowed_methods
        )
        manifest_rows = _load_replica_manifest(replica_manifest)
        if only_key is not None:
            manifest_rows = [
                row
                for row in manifest_rows
                if (row["system_id"].strip(), row["replica_id"].strip()) == only_key
            ]
            if len(manifest_rows) != 1:
                raise ValueError(f"requested replica key is absent or duplicated: {only_key}")

        expected_keys = {
            (row["system_id"].strip(), row["replica_id"].strip()) for row in manifest_rows
        }
        selected_keys = set(selected_by_replica)
        if only_key is None and selected_keys != expected_keys:
            raise ValueError(
                f"selection/replica key mismatch: selected_only={sorted(selected_keys - expected_keys)} "
                f"manifest_only={sorted(expected_keys - selected_keys)}"
            )
        if only_key is not None and only_key not in selected_keys:
            raise ValueError(f"selected identities do not contain requested replica: {only_key}")

        receipts: list[dict[str, Any]] = []
        for row in manifest_rows:
            system = row["system_id"].strip()
            replica = row["replica_id"].strip()
            expected_frames = _canonical_int(row["expected_frames"], "expected_frames", minimum=1)
            expected_atoms = _canonical_int(
                row["expected_saved_atoms"], "expected_saved_atoms", minimum=1
            )
            trajectory = _file_binding(
                row["trajectory_path"], row["trajectory_sha256"], "trajectory"
            )
            coordinate_topology = _file_binding(
                row["coordinate_topology_pdb_path"],
                row["coordinate_topology_pdb_sha256"],
                "saved-solute coordinate topology",
            )
            forcefield_topology = _file_binding(
                row["forcefield_topology_parm7_path"],
                row["forcefield_topology_parm7_sha256"],
                "force-field topology",
            )
            observed_pdb_atoms = _count_pdb_atoms(coordinate_topology)
            if observed_pdb_atoms != expected_atoms:
                raise ValueError(
                    f"{system}/{replica} coordinate-topology atoms={observed_pdb_atoms}, "
                    f"expected saved atoms={expected_atoms}"
                )
            saved_atom_schema_sha256 = _validate_saved_atom_mapping(
                coordinate_topology, forcefield_topology, expected_atoms
            )

            replica_dir = partial / "coordinates" / system / replica
            replica_dir.mkdir(parents=True)
            shard_manifest = materialize_coordinate_union(
                trajectory_path=trajectory,
                selected_frames_path=bridge,
                output_npz=replica_dir / "coordinate_union.npz",
                output_map_csv=replica_dir / "row_to_coordinate_union.csv",
                output_manifest_json=replica_dir / "coordinate_union_manifest.json",
                expected_trajectory_sha256=row["trajectory_sha256"],
                expected_atoms=expected_atoms,
                expected_frames=expected_frames,
                selection_filters={"system_id": system, "replica_id": replica},
                expected_system_id=system,
                expected_replica_id=replica,
                expected_source_frame_offset=1,
            )
            # The shard is built under an atomic partial directory that is
            # renamed at publication.  Replace the temporary absolute bridge
            # path with a locator that remains valid after that rename.
            shard_manifest["selected_frames"] = str(
                Path("../../..") / "selected_frame_bridge.tsv"
            )
            shard_manifest["selected_frames_locator_semantics"] = (
                "relative_to_coordinate_manifest_directory"
            )
            write_json(replica_dir / "coordinate_union_manifest.json", shard_manifest)
            receipts.append(
                {
                    "system_id": system,
                    "replica_id": replica,
                    "lineage": row["lineage"],
                    "trajectory_path": str(trajectory),
                    "trajectory_sha256": row["trajectory_sha256"],
                    "coordinate_topology_pdb_path": str(coordinate_topology),
                    "coordinate_topology_pdb_sha256": row[
                        "coordinate_topology_pdb_sha256"
                    ],
                    "forcefield_topology_parm7_path": str(forcefield_topology),
                    "forcefield_topology_parm7_sha256": row[
                        "forcefield_topology_parm7_sha256"
                    ],
                    "expected_frames": expected_frames,
                    "expected_saved_atoms": expected_atoms,
                    "saved_atom_mapping_semantics": (
                        "NetCDF atom position = first ntwprt parm7 atom position = "
                        "saved-solute PDB ATOM/HETATM position"
                    ),
                    "saved_atom_schema_sha256": saved_atom_schema_sha256,
                    "selection_row_count": shard_manifest["selection_row_count"],
                    "unique_coordinate_frame_count": shard_manifest[
                        "unique_coordinate_frame_count"
                    ],
                    "coordinate_array_shape": shard_manifest["coordinate_array_shape"],
                    "coordinate_union_sha256": shard_manifest["coordinate_shard_sha256"],
                    "coordinate_map_sha256": shard_manifest["coordinate_map_sha256"],
                    "coordinate_manifest_sha256": sha256_file(
                        replica_dir / "coordinate_union_manifest.json"
                    ),
                    "status": "PASS",
                }
            )

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS_COORDINATES_MATERIALIZED_ENGINEERING_ONLY",
            "scientific_claims": {
                "sampler_selected": False,
                "model_selected": False,
                "physical_koff_ready": False,
                "grouped_pkoff_comparison_run": False,
            },
            "source_selected_identities": str(selected_identities.resolve()),
            "source_selected_identities_sha256": selected_identities_sha256,
            "expected_selected_identities_sha256": expected_selected_identities_sha256,
            "source_replica_manifest": str(replica_manifest.resolve()),
            "source_replica_manifest_sha256": replica_manifest_sha256,
            "expected_replica_manifest_sha256": expected_replica_manifest_sha256,
            "selected_frame_bridge_sha256": sha256_file(bridge),
            "replica_count": len(receipts),
            "systems": sorted({row["system_id"] for row in receipts}),
            "methods": sorted(allowed_methods) if allowed_methods is not None else "ALL_IN_INPUT",
            "replicas": receipts,
        }
        write_json(partial / "run_manifest.json", manifest)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        partial.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _parse_key(text: str) -> tuple[str, str]:
    if "/" not in text:
        raise argparse.ArgumentTypeError("--only-key must be SYSTEM/REPLICA")
    system, replica = text.split("/", 1)
    if not system or not replica:
        raise argparse.ArgumentTypeError("--only-key must be SYSTEM/REPLICA")
    return system, replica


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-identities", type=Path, required=True)
    parser.add_argument("--replica-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-selected-identities-sha256", required=True)
    parser.add_argument("--expected-replica-manifest-sha256", required=True)
    parser.add_argument("--only-key", type=_parse_key)
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        help="Materialize only this exact method name; repeat for multiple methods",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run(
            selected_identities=args.selected_identities,
            replica_manifest=args.replica_manifest,
            output_dir=args.output_dir,
            expected_selected_identities_sha256=args.expected_selected_identities_sha256,
            expected_replica_manifest_sha256=args.expected_replica_manifest_sha256,
            only_key=args.only_key,
            methods=args.methods,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "FAIL", "reason": str(exc)}))
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
