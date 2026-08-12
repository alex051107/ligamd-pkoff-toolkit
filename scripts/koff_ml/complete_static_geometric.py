#!/usr/bin/env python3
"""Build the outcome-blind 20D complete-static geometric baseline.

The table produced here is deliberately independent of Production outcomes
and trajectories.  One complex contributes one row made from:

* ten precomputed ligand physicochemical descriptors; and
* ten fixed geometric descriptors of one audited, initially bound structure.

The geometric block is a deliberately modest contact/pocket proxy.  It is not
a chemistry-typed PLIF.  Every structural row is bound to an exact PDB, parm7,
``md.in`` and shared-reference SHA-256.  The ``timask1`` ligand selector in
``md.in`` must resolve one and only one PDB residue and must agree with the
shared reference.  No pKoff, sigma, outcome, replica, or trajectory field is
accepted by either input table.

The 20 fields have two sources. A separately audited canonical-SMILES step
supplies ten ligand physicochemical descriptors. One canonical bound complex
supplies ten initial-geometry and protein descriptors.

Together they ask how much molecular identity and the starting structure can
explain on their own. They do not describe what happened along a dissociation
path. The static block is therefore kept separate from trajectory-derived G10
and used as an ablation comparator.

Every structural measurement is bound to exact PDB, parm7, ``md.in`` and
shared-reference hashes. A solvated topology must never be forced onto the
atom axis of a stripped NetCDF or PDB. If that mapping gate fails, the row is
not written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import sha256_file, write_csv, write_json
from .prediction_first_coordinate_materialization import _validate_saved_atom_mapping
from .shared_reference import (
    STANDARD_AA,
    SharedReferenceGateError,
    build_shared_reference_manifest,
    canonical_manifest_bytes,
    load_shared_reference,
    parse_pdb_bytes,
    validate_shared_reference,
    write_shared_reference,
)


SCHEMA_VERSION = "ligamd-complete-static-geometric-v1.0.0"
CONTACT_CUTOFF_A = 4.5
POCKET_CUTOFF_A = 8.0
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TIMASK_ASSIGNMENT = re.compile(
    r"\btimask1\s*=\s*(['\"])([^'\"]+)\1", re.IGNORECASE
)
_SINGLE_RESIDUE_MASK = re.compile(r"^:([0-9]+[A-Za-z]?)$")


LIGAND_FEATURES = (
    "ligand_static__molecular_weight",
    "ligand_static__logP",
    "ligand_static__HBD",
    "ligand_static__HBA",
    "ligand_static__TPSA",
    "ligand_static__rotatable_bonds",
    "ligand_static__rings",
    "ligand_static__heavy_atoms",
    "ligand_static__formal_charge",
    "ligand_static__fraction_CSP3",
)

GEOMETRIC_FEATURES = (
    "initial_geometry__minimum_protein_ligand_heavy_distance_A",
    "initial_geometry__native_contact_residue_count",
    "initial_geometry__native_contact_residue_fraction_of_pocket",
    "initial_geometry__pocket_8A_residue_count",
    "initial_geometry__pocket_8A_heavy_atom_count",
    "initial_geometry__pocket_8A_radius_of_gyration_A",
    "initial_geometry__standard_protein_residue_count",
    "initial_geometry__pocket_hydrophobic_residue_fraction",
    "initial_geometry__pocket_polar_uncharged_residue_fraction",
    "initial_geometry__pocket_ionizable_residue_fraction",
)

FEATURES = LIGAND_FEATURES + GEOMETRIC_FEATURES

STRUCTURE_MANIFEST_FIELDS = (
    "complex_id",
    "canonical_bound_pdb_path",
    "canonical_bound_pdb_sha256",
    "topology_parm7_path",
    "topology_parm7_sha256",
    "md_in_path",
    "md_in_sha256",
    "shared_reference_path",
    "shared_reference_sha256",
)

HYDROPHOBIC_RESIDUES = frozenset(
    {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY"}
)
POLAR_UNCHARGED_RESIDUES = frozenset(
    {"ASN", "GLN", "SER", "THR", "CYS", "CYX", "TYR"}
)
IONIZABLE_RESIDUES = frozenset(
    {"ASP", "ASH", "GLU", "GLH", "ARG", "LYS", "LYN", "HIS", "HID", "HIE", "HIP", "CYM"}
)


class CompleteStaticGateError(RuntimeError):
    """Raised when an outcome-blind static row cannot be trusted."""


def _schema_payload() -> dict[str, object]:
    """Return the machine-readable meaning of the static20 representation.

    This payload fixes field order, cut-offs, residue classes, prohibited
    inputs, and scientific scope. Its canonical JSON hash is written into the
    receipt so a later user can tell whether two static tables use the same
    definition rather than merely the same column names.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "feature_count": len(FEATURES),
        "feature_order": list(FEATURES),
        "blocks": {
            "ligand_physicochemical": list(LIGAND_FEATURES),
            "initial_geometric_contact_pocket_proxy": list(GEOMETRIC_FEATURES),
        },
        "geometry": {
            "source": "one exact audited canonical bound PDB and its shared structural reference",
            "contact_cutoff_A": CONTACT_CUTOFF_A,
            "pocket_cutoff_A": POCKET_CUTOFF_A,
            "distance_convention": "minimum-image heavy-atom distance frozen by shared_reference",
            "pocket_radius_of_gyration": "unweighted RMS distance of all 8 A pocket heavy atoms from their PBC-unwrapped geometric centroid",
            "native_contact_fraction_denominator": "all 8 A pocket residues",
            "pocket_chemistry_fraction_denominator": "all 8 A pocket residues",
            "residue_classes": {
                "hydrophobic": sorted(HYDROPHOBIC_RESIDUES),
                "polar_uncharged": sorted(POLAR_UNCHARGED_RESIDUES),
                "ionizable": sorted(IONIZABLE_RESIDUES),
            },
        },
        "input_contract": {
            "one_row_per_complex": True,
            "ligand_is_uniquely_parsed_from_md_in_timask1": True,
            "exact_PDB_topology_md_in_hashes_required": True,
            "PDB_parm7_saved_atom_order_mapping_required": True,
            "shared_reference": "exact prebound hash or deterministic build from those exact inputs",
            "forbidden_sources": [
                "pKoff or koff labels",
                "sigma or boost metadata",
                "Production outcomes or dissociation classifications",
                "replica identities",
                "trajectory coordinates or time series",
            ],
        },
        "scientific_scope": {
            "typed_PLIF": False,
            "description": "fixed initial geometric contact/pocket proxy; not chemistry-typed PLIF",
            "sampler_or_model_selected": False,
        },
    }


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_expected_sha(path: Path, expected: str, role: str) -> str:
    expected = str(expected).strip().lower()
    if _HEX_SHA256.fullmatch(expected) is None:
        raise CompleteStaticGateError(f"{role} expected SHA-256 is malformed")
    observed = sha256_file(path)
    if observed != expected:
        raise CompleteStaticGateError(
            f"{role} SHA-256 mismatch: expected={expected}, observed={observed}, path={path}"
        )
    return observed


def _read_exact_csv(path: Path, expected_fields: Sequence[str], role: str) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = list(reader.fieldnames or ())
        if fields != list(expected_fields):
            raise CompleteStaticGateError(
                f"{role} must contain exactly the frozen columns in order; "
                f"expected={list(expected_fields)!r}, observed={fields!r}"
            )
        rows = list(reader)
    if not rows:
        raise CompleteStaticGateError(f"{role} is empty: {path}")
    return rows


def _read_expected_complex_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        ids.append(value)
    if not ids:
        raise CompleteStaticGateError("expected-complex-id file is empty")
    if len(ids) != len(set(ids)):
        raise CompleteStaticGateError("expected-complex-id file contains duplicates")
    if ids != sorted(ids):
        raise CompleteStaticGateError("expected-complex-id file must be sorted for deterministic output")
    return ids


def _index_unique(rows: Sequence[Mapping[str, str]], role: str) -> dict[str, Mapping[str, str]]:
    indexed: dict[str, Mapping[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        complex_id = str(row.get("complex_id", "")).strip()
        if not complex_id:
            raise CompleteStaticGateError(f"{role} row {row_number} has empty complex_id")
        if complex_id in indexed:
            raise CompleteStaticGateError(f"{role} contains duplicate complex_id={complex_id}")
        indexed[complex_id] = row
    return indexed


def _resolve(manifest_path: Path, value: str, field: str) -> Path:
    text = str(value).strip()
    if not text:
        raise CompleteStaticGateError(f"structure manifest has empty {field}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise CompleteStaticGateError(f"{field} does not exist: {path}")
    return path


def parse_unique_timask1_residue(md_in_path: str | Path) -> str:
    """Return a single AMBER residue identity from the only active timask1.

    Complex masks, ranges, multiple assignments, and selectors ambiguous in
    the canonical PDB fail closed.  This intentionally supports the project's
    actual ``timask1=':NNN'`` convention rather than guessing AMBER masks.
    """

    active_lines = []
    for raw_line in Path(md_in_path).read_text(encoding="utf-8", errors="strict").splitlines():
        # AMBER namelist comments begin with !.  Project timask values do not
        # contain !, so removing the suffix is unambiguous here.
        active_lines.append(raw_line.split("!", 1)[0])
    matches = _TIMASK_ASSIGNMENT.findall("\n".join(active_lines))
    if len(matches) != 1:
        raise CompleteStaticGateError(
            f"md.in must contain exactly one active quoted timask1 assignment; observed={len(matches)}"
        )
    mask = matches[0][1].strip()
    residue = _SINGLE_RESIDUE_MASK.fullmatch(mask)
    if residue is None:
        raise CompleteStaticGateError(
            f"timask1 must be one exact residue selector like ':511'; observed={mask!r}"
        )
    return residue.group(1)


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompleteStaticGateError(f"shared reference {name} must be a JSON object")
    return value


def _validate_and_resolve_ligand(
    *, parsed_pdb: object, shared_reference: Mapping[str, object], timask_resid: str
) -> tuple[str, str, str]:
    atoms = parsed_pdb.atoms  # type: ignore[attr-defined]
    matching_keys = {atom.residue_key for atom in atoms if atom.resid == timask_resid}
    if len(matching_keys) != 1:
        raise CompleteStaticGateError(
            f"timask1 :{timask_resid} resolves {len(matching_keys)} PDB residues; exactly one is required"
        )
    observed_key = next(iter(matching_keys))
    ligand = _as_mapping(shared_reference.get("ligand"), "ligand")
    resolved = _as_mapping(ligand.get("resolved_residue"), "ligand.resolved_residue")
    expected_key = (
        str(resolved.get("chain", "")),
        str(resolved.get("resid", "")),
        str(resolved.get("resname", "")),
    )
    if observed_key != expected_key:
        raise CompleteStaticGateError(
            f"md.in timask1 ligand disagrees with shared reference: md.in={observed_key}, shared={expected_key}"
        )
    selected_heavy = sorted(
        atom.index0 for atom in atoms if atom.residue_key == observed_key and atom.is_heavy
    )
    frozen_heavy = ligand.get("heavy_atom_indices0")
    if selected_heavy != frozen_heavy:
        raise CompleteStaticGateError(
            "timask1-selected ligand heavy atoms disagree with shared-reference atom correspondence"
        )
    return observed_key


def _resolve_timask_key(parsed_pdb: object, timask_resid: str) -> tuple[str, str, str]:
    atoms = parsed_pdb.atoms  # type: ignore[attr-defined]
    matching_keys = {atom.residue_key for atom in atoms if atom.resid == timask_resid}
    if len(matching_keys) != 1:
        raise CompleteStaticGateError(
            f"timask1 :{timask_resid} resolves {len(matching_keys)} PDB residues; exactly one is required"
        )
    return next(iter(matching_keys))


def _geometry_from_reference(
    parsed_pdb: object, shared_reference: Mapping[str, object]
) -> dict[str, float]:
    """Calculate the ten initial-geometry/protein fields from one bound reference.

    The fields are the ligand-to-pocket minimum heavy-atom distance; number of
    native-contact residues at 4.5 Å; native-contact fraction of the 8 Å
    pocket; number of pocket residues; number of pocket heavy atoms; unweighted
    RMS radius of pocket heavy atoms around their PBC-unwrapped geometric
    centroid; number of standard protein residues in the canonical PDB; and
    hydrophobic, polar-uncharged, and ionisable residue fractions in the
    pocket.

    The three composition fields describe residue classes in a starting pocket.
    They do not claim that a hydrogen bond, salt bridge, or pi interaction was
    observed during a trajectory.
    """

    definitions = _as_mapping(shared_reference.get("definitions"), "definitions")
    try:
        contact_cutoff = float(definitions["contact_cutoff_A"])
        pocket_cutoff = float(definitions["pocket_cutoff_A"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CompleteStaticGateError("shared-reference cutoffs are missing or nonnumeric") from exc
    if not math.isclose(contact_cutoff, CONTACT_CUTOFF_A, abs_tol=1e-12):
        raise CompleteStaticGateError(
            f"shared-reference contact cutoff must be frozen at {CONTACT_CUTOFF_A:g} A"
        )
    if not math.isclose(pocket_cutoff, POCKET_CUTOFF_A, abs_tol=1e-12):
        raise CompleteStaticGateError(
            f"shared-reference pocket cutoff must be frozen at {POCKET_CUTOFF_A:g} A"
        )

    pocket = _as_mapping(shared_reference.get("pocket"), "pocket")
    residues = pocket.get("residues")
    if not isinstance(residues, list) or not residues:
        raise CompleteStaticGateError("shared-reference pocket has no residues")
    pocket_rows = [_as_mapping(row, "pocket residue") for row in residues]
    pocket_count = len(pocket_rows)
    native_count = sum(bool(row.get("native_contact")) for row in pocket_rows)
    distances = np.asarray(
        [float(row["canonical_min_ligand_heavy_distance_A"]) for row in pocket_rows],
        dtype=float,
    )
    if not np.all(np.isfinite(distances)):
        raise CompleteStaticGateError("shared-reference pocket distances contain NaN/inf")

    pocket_resnames = [str(row.get("resname", "")).upper() for row in pocket_rows]
    unknown = sorted(set(pocket_resnames) - STANDARD_AA)
    if unknown:
        raise CompleteStaticGateError(f"pocket contains nonstandard protein residues: {unknown}")
    if any(
        residue not in HYDROPHOBIC_RESIDUES
        and residue not in POLAR_UNCHARGED_RESIDUES
        and residue not in IONIZABLE_RESIDUES
        for residue in pocket_resnames
    ):
        raise CompleteStaticGateError("frozen residue chemistry classes are not exhaustive")

    pocket_indices = pocket.get("heavy_atom_indices0")
    if not isinstance(pocket_indices, list) or not pocket_indices:
        raise CompleteStaticGateError("shared-reference pocket has no heavy-atom indices")
    pose = _as_mapping(shared_reference.get("pose_reference"), "pose_reference")
    centered = np.asarray(pose.get("pocket_centered_coordinates_A"), dtype=float)
    if centered.shape != (len(pocket_indices), 3) or not np.all(np.isfinite(centered)):
        raise CompleteStaticGateError("shared-reference centered pocket coordinates are invalid")
    # The shared reference already places all pocket heavy atoms in one PBC
    # image and centres them at the geometric centroid. The RMS radius can
    # therefore be calculated directly without mass weighting.
    radius_of_gyration = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))

    atoms = parsed_pdb.atoms  # type: ignore[attr-defined]
    protein_residue_keys = {
        atom.residue_key
        for atom in atoms
        if atom.record == "ATOM" and atom.residue in STANDARD_AA
    }
    if not protein_residue_keys:
        raise CompleteStaticGateError("canonical bound PDB has no standard protein residues")

    return {
        GEOMETRIC_FEATURES[0]: float(np.min(distances)),
        GEOMETRIC_FEATURES[1]: float(native_count),
        GEOMETRIC_FEATURES[2]: float(native_count / pocket_count),
        GEOMETRIC_FEATURES[3]: float(pocket_count),
        GEOMETRIC_FEATURES[4]: float(len(pocket_indices)),
        GEOMETRIC_FEATURES[5]: radius_of_gyration,
        GEOMETRIC_FEATURES[6]: float(len(protein_residue_keys)),
        GEOMETRIC_FEATURES[7]: float(
            sum(name in HYDROPHOBIC_RESIDUES for name in pocket_resnames) / pocket_count
        ),
        GEOMETRIC_FEATURES[8]: float(
            sum(name in POLAR_UNCHARGED_RESIDUES for name in pocket_resnames) / pocket_count
        ),
        GEOMETRIC_FEATURES[9]: float(
            sum(name in IONIZABLE_RESIDUES for name in pocket_resnames) / pocket_count
        ),
    }


def _parse_ligand_features(row: Mapping[str, str], complex_id: str) -> dict[str, float]:
    """Read one audited ligand-static10 row and reject missing or non-finite values."""

    output: dict[str, float] = {}
    for feature in LIGAND_FEATURES:
        try:
            value = float(row[feature])
        except (KeyError, TypeError, ValueError) as exc:
            raise CompleteStaticGateError(
                f"{complex_id} ligand feature {feature} is missing or nonnumeric"
            ) from exc
        if not math.isfinite(value):
            raise CompleteStaticGateError(f"{complex_id} ligand feature {feature} is NaN/inf")
        output[feature] = value
    return output


def build_complete_static_geometric(
    *,
    structure_manifest: str | Path,
    expected_structure_manifest_sha256: str,
    ligand_features: str | Path,
    expected_ligand_features_sha256: str,
    expected_complex_ids: str | Path,
    expected_complex_ids_sha256: str,
    output_dir: str | Path,
) -> dict[str, object]:
    """Write one static20 row per complex together with complete provenance.

    The function reads exact-column inputs, verifies the expected complex set,
    checks the relevant structural-file hashes, resolves exactly one ``timask1``
    ligand, validates the saved-atom mapping and shared reference, calculates
    geometry10, and joins it to ligand static10.

    It accepts no replica or trajectory table. A dissociation outcome, episode
    length, or pKoff value therefore cannot influence the static fields.
    """

    structure_path = Path(structure_manifest).resolve()
    ligand_path = Path(ligand_features).resolve()
    expected_ids_path = Path(expected_complex_ids).resolve()
    output = Path(output_dir).resolve()
    source_hashes = {
        "structure_manifest_sha256": _require_expected_sha(
            structure_path, expected_structure_manifest_sha256, "structure manifest"
        ),
        "ligand_features_sha256": _require_expected_sha(
            ligand_path, expected_ligand_features_sha256, "ligand feature table"
        ),
        "expected_complex_ids_sha256": _require_expected_sha(
            expected_ids_path, expected_complex_ids_sha256, "expected complex IDs"
        ),
    }

    expected_ids = _read_expected_complex_ids(expected_ids_path)
    structures = _index_unique(
        _read_exact_csv(structure_path, STRUCTURE_MANIFEST_FIELDS, "structure manifest"),
        "structure manifest",
    )
    ligand_rows = _index_unique(
        _read_exact_csv(
            ligand_path, ("complex_id",) + LIGAND_FEATURES, "ligand feature table"
        ),
        "ligand feature table",
    )
    expected_set = set(expected_ids)
    for name, observed in (("structure manifest", set(structures)), ("ligand feature table", set(ligand_rows))):
        if observed != expected_set:
            raise CompleteStaticGateError(
                f"{name} complex coverage mismatch: missing={sorted(expected_set - observed)}, "
                f"extra={sorted(observed - expected_set)}"
            )

    schema = _schema_payload()
    schema_sha256 = _canonical_json_sha256(schema)
    feature_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    generated_references: dict[str, Mapping[str, object]] = {}
    for complex_id in expected_ids:
        row = structures[complex_id]
        paths = {
            "pdb": _resolve(structure_path, row["canonical_bound_pdb_path"], "canonical_bound_pdb_path"),
            "topology": _resolve(structure_path, row["topology_parm7_path"], "topology_parm7_path"),
            "md_in": _resolve(structure_path, row["md_in_path"], "md_in_path"),
        }
        observed_hashes = {
            "canonical_bound_pdb_sha256": _require_expected_sha(
                paths["pdb"], row["canonical_bound_pdb_sha256"], f"{complex_id} canonical PDB"
            ),
            "topology_parm7_sha256": _require_expected_sha(
                paths["topology"], row["topology_parm7_sha256"], f"{complex_id} topology"
            ),
            "md_in_sha256": _require_expected_sha(
                paths["md_in"], row["md_in_sha256"], f"{complex_id} md.in"
            ),
        }
        parsed = parse_pdb_bytes(paths["pdb"].read_bytes())
        try:
            atom_mapping_sha256 = _validate_saved_atom_mapping(
                paths["pdb"], paths["topology"], len(parsed.atoms)
            )
        except ValueError as exc:
            raise CompleteStaticGateError(
                f"{complex_id} saved-solute PDB / parm7 atom mapping failed: {exc}"
            ) from exc
        observed_hashes["pdb_parm7_atom_mapping_sha256"] = atom_mapping_sha256
        timask_resid = parse_unique_timask1_residue(paths["md_in"])
        timask_key = _resolve_timask_key(parsed, timask_resid)
        shared_path_text = str(row["shared_reference_path"]).strip()
        shared_sha_text = str(row["shared_reference_sha256"]).strip()
        if bool(shared_path_text) != bool(shared_sha_text):
            raise CompleteStaticGateError(
                f"{complex_id} shared_reference_path and shared_reference_sha256 must both be set or both be blank"
            )
        if shared_path_text:
            shared_path = _resolve(structure_path, shared_path_text, "shared_reference_path")
            observed_hashes["shared_reference_sha256"] = _require_expected_sha(
                shared_path, shared_sha_text, f"{complex_id} shared reference"
            )
            shared = load_shared_reference(shared_path)
            shared_source = "PREBOUND_EXACT_HASH"
        else:
            shared = build_shared_reference_manifest(
                paths["pdb"],
                complex_id=complex_id,
                topology_path=paths["topology"],
                ligand_resname=timask_key[2],
                ligand_resid=timask_key[1],
                ligand_chain=timask_key[0],
                contact_cutoff_A=CONTACT_CUTOFF_A,
                pocket_cutoff_A=POCKET_CUTOFF_A,
            )
            observed_hashes["shared_reference_sha256"] = hashlib.sha256(
                canonical_manifest_bytes(shared)
            ).hexdigest()
            generated_references[complex_id] = shared
            shared_source = "DETERMINISTIC_BUILD_FROM_EXACT_STATIC_INPUTS"
        try:
            validation = validate_shared_reference(
                shared,
                complex_id=complex_id,
                pdb_path=paths["pdb"],
                topology_path=paths["topology"],
            )
        except SharedReferenceGateError as exc:
            raise CompleteStaticGateError(f"{complex_id} shared-reference context failed: {exc}") from exc
        provenance = _as_mapping(shared.get("provenance"), "provenance")
        if observed_hashes["canonical_bound_pdb_sha256"] != provenance.get("canonical_pdb_sha256"):
            raise CompleteStaticGateError(
                f"{complex_id} PDB is schema-compatible but not the exact canonical bound structure"
            )

        ligand_key = _validate_and_resolve_ligand(
            parsed_pdb=parsed, shared_reference=shared, timask_resid=timask_resid
        )
        features: dict[str, object] = {"complex_id": complex_id}
        features.update(_parse_ligand_features(ligand_rows[complex_id], complex_id))
        features.update(_geometry_from_reference(parsed, shared))
        values = np.asarray([features[name] for name in FEATURES], dtype=float)
        if values.shape != (20,) or not np.all(np.isfinite(values)):
            raise CompleteStaticGateError(f"{complex_id} does not produce exactly 20 finite features")
        feature_rows.append(features)
        audit_rows.append(
            {
                "complex_id": complex_id,
                **observed_hashes,
                "canonical_pdb_atom_schema_sha256": validation["pdb_atom_schema_sha256"],
                "feature_schema_sha256": schema_sha256,
                "timask1_residue": timask_resid,
                "resolved_ligand_chain": ligand_key[0],
                "resolved_ligand_resid": ligand_key[1],
                "resolved_ligand_resname": ligand_key[2],
                "shared_reference_source": shared_source,
                "status": "PASS_OUTCOME_BLIND_GEOMETRIC_STATIC",
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / "complete_static_geometric_20d.csv"
    audit_path = output / "complete_static_geometric_provenance.csv"
    schema_path = output / "complete_static_geometric_schema.json"
    gate_path = output / "complete_static_geometric_gate.json"
    generated_reference_hashes: dict[str, str] = {}
    for complex_id, shared in generated_references.items():
        reference_path = output / "shared_references" / f"{complex_id}.json"
        write_shared_reference(reference_path, shared)
        generated_reference_hashes[complex_id] = sha256_file(reference_path)
        expected_hash = next(
            str(row["shared_reference_sha256"])
            for row in audit_rows
            if row["complex_id"] == complex_id
        )
        if generated_reference_hashes[complex_id] != expected_hash:
            raise CompleteStaticGateError(
                f"{complex_id} generated shared-reference bytes drifted before publication"
            )
    write_csv(feature_path, feature_rows, fieldnames=("complex_id",) + FEATURES)
    write_csv(
        audit_path,
        audit_rows,
        fieldnames=(
            "complex_id",
            "canonical_bound_pdb_sha256",
            "topology_parm7_sha256",
            "md_in_sha256",
            "shared_reference_sha256",
            "pdb_parm7_atom_mapping_sha256",
            "canonical_pdb_atom_schema_sha256",
            "feature_schema_sha256",
            "timask1_residue",
            "resolved_ligand_chain",
            "resolved_ligand_resid",
            "resolved_ligand_resname",
            "shared_reference_source",
            "status",
        ),
    )
    write_json(schema_path, schema)
    if _canonical_json_sha256(json.loads(schema_path.read_text(encoding="utf-8"))) != schema_sha256:
        raise CompleteStaticGateError("written feature schema does not match its frozen semantic hash")
    gate: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_COMPLETE_STATIC_GEOMETRIC_20D",
        "rows": len(feature_rows),
        "expected_rows": len(expected_ids),
        "complex_ids": expected_ids,
        "feature_count": len(FEATURES),
        "feature_schema_sha256": schema_sha256,
        "input_hashes": source_hashes,
        "output_hashes": {
            "complete_static_geometric_20d.csv": sha256_file(feature_path),
            "complete_static_geometric_provenance.csv": sha256_file(audit_path),
            "complete_static_geometric_schema.json": sha256_file(schema_path),
        },
        "generated_shared_reference_hashes": generated_reference_hashes,
        "pdb_parm7_atom_mapping_sha256": {
            str(row["complex_id"]): str(row["pdb_parm7_atom_mapping_sha256"])
            for row in audit_rows
        },
        "outcome_blind": True,
        "coordinate_source": "one canonical bound PDB per complex; no trajectory",
        "representation": "10 ligand physicochemical + 10 initial geometric contact/pocket features",
        "typed_PLIF": False,
        "scientific_claims": {
            "static_input_engineering_ready": True,
            "sampler_selected": False,
            "model_selected": False,
            "physical_koff_estimated": False,
        },
    }
    write_json(gate_path, gate)
    return gate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.koff_ml.complete_static_geometric",
        description="Build one outcome-blind 20D complete-static row per complex",
    )
    parser.add_argument("--structure-manifest", required=True)
    parser.add_argument("--expected-structure-manifest-sha256", required=True)
    parser.add_argument("--ligand-features", required=True)
    parser.add_argument("--expected-ligand-features-sha256", required=True)
    parser.add_argument("--expected-complex-ids", required=True)
    parser.add_argument("--expected-complex-ids-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        gate = build_complete_static_geometric(
            structure_manifest=args.structure_manifest,
            expected_structure_manifest_sha256=args.expected_structure_manifest_sha256,
            ligand_features=args.ligand_features,
            expected_ligand_features_sha256=args.expected_ligand_features_sha256,
            expected_complex_ids=args.expected_complex_ids,
            expected_complex_ids_sha256=args.expected_complex_ids_sha256,
            output_dir=args.output_dir,
        )
    except (CompleteStaticGateError, SharedReferenceGateError, OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
