"""Build and validate one deterministic structural reference per complex.

The reference is deliberately separate from per-replica feature extraction.
It freezes, from one audited canonical PDB, the ligand atom correspondence,
initial pocket/native-contact vocabulary, and pocket-aligned pose coordinates
that every replica of the same complex must reuse.

Two different PDB hashes have intentionally different jobs:

* the full canonical-PDB hash binds the actual reference coordinates;
* the coordinate-independent atom-schema hash gates replica compatibility.

The builder requires a verified orthorhombic ``CRYST1`` record and performs all
pocket/contact selection with the minimum-image convention.  Ambiguous ligand
selection, missing PBC metadata, topology drift, or atom-schema drift fails
closed.  No wall-clock field is written, so identical inputs produce identical
JSON bytes when written with :func:`write_shared_reference`.

Think of the manifest as the common ruler for every trajectory in one complex.
The topology and PDB atom order say what each coordinate represents.  Canonical
ligand heavy atoms fix atom correspondence.  Protein residues within the pocket
cutoff of any ligand heavy atom form the frozen pocket; the subset within the
contact cutoff forms the native-contact set.  PBC-consistent pocket-centred
coordinates then provide the pose-alignment reference.  The manifest is built
once from an audited bound structure.  If each replica rebuilt its own pocket,
contact fractions would no longer mean the same thing across replicas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import atomic_text_writer, sha256_file


SCHEMA_VERSION = "ligamd-shared-structural-reference-v1.0.0"
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROUND_DECIMALS = 8

STANDARD_AA = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYM", "CYX", "GLN", "GLU",
        "GLH", "GLY", "HID", "HIE", "HIP", "HIS", "ILE", "LEU", "LYS", "LYN",
        "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    }
)


class SharedReferenceGateError(RuntimeError):
    """Raised whenever a shared reference cannot be trusted or reused."""


@dataclass(frozen=True)
class PDBAtom:
    """One ordered PDB atom and the fields needed for schema validation."""

    index0: int
    record: str
    name: str
    residue: str
    chain: str
    resid: str
    element: str
    coordinates_A: tuple[float, float, float]

    @property
    def residue_key(self) -> tuple[str, str, str]:
        return self.chain, self.resid, self.residue

    @property
    def is_heavy(self) -> bool:
        return self.element.upper() != "H"


@dataclass(frozen=True)
class ParsedCanonicalPDB:
    atoms: tuple[PDBAtom, ...]
    box_lengths_A: tuple[float, float, float]
    box_angles_deg: tuple[float, float, float]
    atom_schema_sha256: str


def _infer_element(atom_name: str) -> str:
    letters = "".join(ch for ch in atom_name if ch.isalpha()).upper()
    if not letters:
        return ""
    stripped = atom_name.strip().upper()
    if letters[:2] in {"CL", "BR", "NA", "MG", "ZN", "FE"} and stripped[:2] == letters[:2]:
        return letters[:2]
    return letters[0]


def _schema_identity(line: str) -> str:
    """Match the coordinate-independent hash used by ``manifest_builder``."""

    return "|".join(
        (
            line[0:6].strip(),
            line[12:16].strip(),
            line[17:20].strip(),
            line[21:22].strip(),
            line[22:27].strip(),
            line[76:78].strip() if len(line) >= 78 else "",
        )
    )


def pdb_atom_schema_sha256_bytes(pdb_bytes: bytes) -> str:
    """Hash ordered atom identity while excluding coordinates.

    This intentionally mirrors ``manifest_builder._pdb_atom_schema_sha256`` so
    existing replica-manifest hashes can be compared without translation.
    Parsing/ambiguity gates are applied separately by :func:`parse_pdb_bytes`.
    """

    try:
        text = pdb_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SharedReferenceGateError("PDB is not valid UTF-8/ASCII text") from exc
    digest = hashlib.sha256()
    count = 0
    for line in text.splitlines():
        if line[0:6].strip().upper() not in {"ATOM", "HETATM"}:
            continue
        digest.update((_schema_identity(line) + "\n").encode("utf-8"))
        count += 1
    if count == 0:
        raise SharedReferenceGateError("PDB contains no ATOM/HETATM records")
    return digest.hexdigest()


def pdb_atom_schema_sha256(path: str | Path) -> str:
    """File wrapper for :func:`pdb_atom_schema_sha256_bytes`."""

    return pdb_atom_schema_sha256_bytes(Path(path).read_bytes())


def _parse_cryst1(line: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    try:
        lengths = tuple(float(line[start:stop]) for start, stop in ((6, 15), (15, 24), (24, 33)))
        angles = tuple(float(line[start:stop]) for start, stop in ((33, 40), (40, 47), (47, 54)))
    except (TypeError, ValueError) as exc:
        raise SharedReferenceGateError(f"Malformed CRYST1 record: {line.rstrip()}") from exc
    if not all(math.isfinite(value) and value > 0 for value in lengths):
        raise SharedReferenceGateError(f"CRYST1 cell lengths must be finite and positive: {lengths}")
    if not all(math.isfinite(value) for value in angles):
        raise SharedReferenceGateError(f"CRYST1 cell angles must be finite: {angles}")
    if not all(abs(value - 90.0) <= 1e-2 for value in angles):
        raise SharedReferenceGateError(
            f"Only verified orthorhombic CRYST1 boxes are supported; observed angles={angles}"
        )
    return lengths, angles


def parse_pdb_bytes(pdb_bytes: bytes) -> ParsedCanonicalPDB:
    """Pure, fail-closed parser for a canonical single-model PDB."""

    try:
        text = pdb_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SharedReferenceGateError("PDB is not valid UTF-8/ASCII text") from exc

    atoms: list[PDBAtom] = []
    cryst1: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    seen_model = False
    model_closed = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        record = line[0:6].strip().upper()
        if record == "CRYST1":
            cryst1.append(_parse_cryst1(line))
            continue
        if record == "MODEL":
            if seen_model or atoms or model_closed:
                raise SharedReferenceGateError("Canonical PDB must contain at most one model")
            seen_model = True
            continue
        if record == "ENDMDL":
            if not seen_model or model_closed:
                raise SharedReferenceGateError("Unexpected ENDMDL in canonical PDB")
            model_closed = True
            continue
        if record not in {"ATOM", "HETATM"}:
            continue
        if model_closed:
            raise SharedReferenceGateError("Canonical PDB contains atoms after ENDMDL/multiple models")
        if len(line) < 54:
            raise SharedReferenceGateError(f"Short ATOM/HETATM record at line {line_number}")
        altloc = line[16:17].strip()
        if altloc:
            raise SharedReferenceGateError(
                f"Unresolved alternate location '{altloc}' at PDB line {line_number}; canonicalize first"
            )
        name = line[12:16].strip()
        residue = line[17:20].strip().upper()
        chain = line[21:22].strip() or "_"
        resid = (line[22:26].strip() + line[26:27].strip()) or "?"
        element_field = line[76:78].strip().upper() if len(line) >= 78 else ""
        element = element_field or _infer_element(name)
        if not name or not residue or not element:
            raise SharedReferenceGateError(f"Incomplete atom identity at PDB line {line_number}")
        try:
            xyz = tuple(float(line[start:stop]) for start, stop in ((30, 38), (38, 46), (46, 54)))
        except ValueError as exc:
            raise SharedReferenceGateError(f"Invalid coordinates at PDB line {line_number}") from exc
        if not all(math.isfinite(value) for value in xyz):
            raise SharedReferenceGateError(f"Non-finite coordinates at PDB line {line_number}")
        atoms.append(
            PDBAtom(
                index0=len(atoms),
                record=record,
                name=name,
                residue=residue,
                chain=chain,
                resid=resid,
                element=element,
                coordinates_A=xyz,  # type: ignore[arg-type]
            )
        )
    if not atoms:
        raise SharedReferenceGateError("PDB contains no ATOM/HETATM records")
    if len(cryst1) != 1:
        raise SharedReferenceGateError(
            f"Canonical PDB must contain exactly one verified CRYST1 record; found {len(cryst1)}"
        )
    return ParsedCanonicalPDB(
        atoms=tuple(atoms),
        box_lengths_A=cryst1[0][0],
        box_angles_deg=cryst1[0][1],
        atom_schema_sha256=pdb_atom_schema_sha256_bytes(pdb_bytes),
    )


def _normalise_chain(chain: str | None) -> str | None:
    if chain is None:
        return None
    return chain.strip() or "_"


def _resolve_topology_sha256(
    *, topology_path: str | Path | None = None, topology_sha256: str | None = None
) -> str:
    supplied = topology_sha256.lower() if topology_sha256 is not None else None
    if supplied is not None and _HEX_SHA256.fullmatch(supplied) is None:
        raise SharedReferenceGateError("topology_sha256 must be exactly 64 lowercase hexadecimal characters")
    computed = sha256_file(topology_path) if topology_path is not None else None
    if supplied is None and computed is None:
        raise SharedReferenceGateError("Either topology_path or topology_sha256 is required")
    if supplied is not None and computed is not None and supplied != computed:
        raise SharedReferenceGateError(
            f"Provided topology hash does not match file: provided={supplied}, computed={computed}"
        )
    return supplied or str(computed)


def _minimum_image(delta: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return delta - lengths * np.rint(delta / lengths)


def _round_float(value: float) -> float:
    rounded = float(np.round(value, _ROUND_DECIMALS))
    return 0.0 if rounded == 0.0 else rounded


def _round_coordinates(values: np.ndarray) -> list[list[float]]:
    return [[_round_float(float(value)) for value in row] for row in values]


def _residue_record(key: tuple[str, str, str]) -> dict[str, str]:
    return {"chain": key[0], "resid": key[1], "resname": key[2]}


def _residue_key_string(key: tuple[str, str, str]) -> str:
    return f"{key[0]}:{key[1]}:{key[2]}"


def _select_ligand(
    atoms: Sequence[PDBAtom],
    *,
    ligand_resname: str,
    ligand_resid: str | None,
    ligand_chain: str | None,
) -> tuple[tuple[str, str, str], list[PDBAtom]]:
    if not ligand_resname.strip():
        raise SharedReferenceGateError("Explicit ligand_resname is required")
    resname = ligand_resname.strip().upper()
    chain = _normalise_chain(ligand_chain)
    matches = [atom for atom in atoms if atom.residue == resname]
    if ligand_resid is not None:
        matches = [atom for atom in matches if atom.resid == str(ligand_resid).strip()]
    if chain is not None:
        matches = [atom for atom in matches if atom.chain == chain]
    keys = {atom.residue_key for atom in matches}
    if not matches:
        raise SharedReferenceGateError(
            f"Ligand selector matched zero atoms: resname={resname}, resid={ligand_resid}, chain={chain}"
        )
    if len(keys) != 1:
        raise SharedReferenceGateError(
            f"Ligand selector is ambiguous ({len(keys)} residues); specify ligand_resid and ligand_chain"
        )
    heavy = [atom for atom in matches if atom.is_heavy]
    if not heavy:
        raise SharedReferenceGateError("Selected ligand contains no identifiable heavy atoms")
    return next(iter(keys)), heavy


def build_shared_reference_manifest_from_bytes(
    pdb_bytes: bytes,
    *,
    complex_id: str,
    topology_sha256: str,
    ligand_resname: str,
    ligand_resid: str | None = None,
    ligand_chain: str | None = None,
    contact_cutoff_A: float = 4.5,
    pocket_cutoff_A: float = 8.0,
) -> dict[str, object]:
    """Build one deterministic structural reference from canonical PDB bytes.

    The pure core reads neither a trajectory nor a label.  It resolves one
    ligand and its heavy atoms, measures every protein residue with a
    minimum-image distance under the canonical `CRYST1` box, freezes the pocket
    and native-contact subsets, and stores PBC-consistent pocket-centred
    coordinates for later pose alignment.  The topology hash, full PDB hash,
    and coordinate-independent atom-schema hash are saved with the definitions.

    Identical inputs produce identical JSON semantics.  An ambiguous ligand,
    unsupported periodic cell, or atom-mapping drift raises
    :class:`SharedReferenceGateError` instead of producing a reference whose
    meaning could change between replicas.
    """

    if not isinstance(complex_id, str) or not complex_id.strip():
        raise SharedReferenceGateError("complex_id must be a non-empty string")
    topology_hash = _resolve_topology_sha256(topology_sha256=topology_sha256)
    if not math.isfinite(contact_cutoff_A) or contact_cutoff_A <= 0:
        raise SharedReferenceGateError("contact_cutoff_A must be finite and positive")
    if not math.isfinite(pocket_cutoff_A) or pocket_cutoff_A <= 0:
        raise SharedReferenceGateError("pocket_cutoff_A must be finite and positive")
    if contact_cutoff_A > pocket_cutoff_A:
        raise SharedReferenceGateError("contact_cutoff_A cannot exceed pocket_cutoff_A")

    parsed = parse_pdb_bytes(pdb_bytes)
    box = np.asarray(parsed.box_lengths_A, dtype=float)
    if pocket_cutoff_A > float(np.min(box)) / 2.0:
        raise SharedReferenceGateError(
            "pocket_cutoff_A exceeds half the shortest CRYST1 length; minimum-image selection is ambiguous"
        )
    ligand_key, ligand_atoms = _select_ligand(
        parsed.atoms,
        ligand_resname=ligand_resname,
        ligand_resid=ligand_resid,
        ligand_chain=ligand_chain,
    )
    ligand_indices = np.asarray([atom.index0 for atom in ligand_atoms], dtype=np.int64)
    coordinates = np.asarray([atom.coordinates_A for atom in parsed.atoms], dtype=float)
    ligand_coordinates = coordinates[ligand_indices]

    protein_by_residue: dict[tuple[str, str, str], list[PDBAtom]] = {}
    for atom in parsed.atoms:
        if atom.record == "ATOM" and atom.residue in STANDARD_AA and atom.is_heavy:
            protein_by_residue.setdefault(atom.residue_key, []).append(atom)
    if not protein_by_residue:
        raise SharedReferenceGateError("No standard-amino-acid protein heavy atoms were identified")

    pocket_rows: list[dict[str, object]] = []
    pocket_indices_list: list[int] = []
    native_keys: list[str] = []
    for key in sorted(protein_by_residue):
        residue_atoms = protein_by_residue[key]
        indices = np.asarray([atom.index0 for atom in residue_atoms], dtype=np.int64)
        residue_coordinates = coordinates[indices]
        delta = residue_coordinates[:, None, :] - ligand_coordinates[None, :, :]
        distance = float(np.sqrt(np.min(np.sum(_minimum_image(delta, box) ** 2, axis=-1))))
        if distance > pocket_cutoff_A:
            continue
        is_native = distance <= contact_cutoff_A
        key_string = _residue_key_string(key)
        pocket_rows.append(
            {
                **_residue_record(key),
                "residue_key": key_string,
                "heavy_atom_indices0": indices.tolist(),
                "canonical_min_ligand_heavy_distance_A": _round_float(distance),
                "native_contact": bool(is_native),
            }
        )
        pocket_indices_list.extend(int(value) for value in indices)
        if is_native:
            native_keys.append(key_string)
    if not pocket_rows:
        raise SharedReferenceGateError(
            f"No protein residue is within the {pocket_cutoff_A:g} A pocket cutoff"
        )
    if not native_keys:
        raise SharedReferenceGateError(
            f"Canonical PDB has zero native residue contacts at {contact_cutoff_A:g} A"
        )

    pocket_indices = np.asarray(sorted(set(pocket_indices_list)), dtype=np.int64)
    pocket_raw = coordinates[pocket_indices]
    pocket_anchor = pocket_raw[0]
    pocket_unwrapped = pocket_anchor + _minimum_image(pocket_raw - pocket_anchor, box)
    pocket_center = np.mean(pocket_unwrapped, axis=0)
    pocket_centered = pocket_unwrapped - pocket_center

    ligand_raw = coordinates[ligand_indices]
    ligand_anchor_raw = ligand_raw[0]
    ligand_anchor = pocket_center + _minimum_image(ligand_anchor_raw - pocket_center, box)
    ligand_unwrapped = ligand_anchor + _minimum_image(ligand_raw - ligand_anchor_raw, box)
    ligand_centered = ligand_unwrapped - pocket_center
    if np.linalg.matrix_rank(pocket_centered, tol=1e-8) < 2:
        raise SharedReferenceGateError(
            "Pocket heavy-atom geometry is rank-deficient; Kabsch pose alignment is not auditable"
        )

    ligand_records = [
        {
            "index0": atom.index0,
            "record": atom.record,
            "name": atom.name,
            "resname": atom.residue,
            "chain": atom.chain,
            "resid": atom.resid,
            "element": atom.element,
        }
        for atom in ligand_atoms
    ]
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "complex_id": complex_id.strip(),
        "provenance": {
            "canonical_pdb_sha256": hashlib.sha256(pdb_bytes).hexdigest(),
            "canonical_pdb_atom_schema_sha256": parsed.atom_schema_sha256,
            "canonical_pdb_atom_count": len(parsed.atoms),
            "topology_sha256": topology_hash,
            "atom_schema_hash_algorithm": "manifest_builder_ordered_identity_v1",
        },
        "periodic_box": {
            "source": "canonical_PDB_CRYST1",
            "lengths_A": [_round_float(value) for value in parsed.box_lengths_A],
            "angles_deg": [_round_float(value) for value in parsed.box_angles_deg],
            "minimum_image_convention": "orthorhombic_delta_minus_L_rint_delta_over_L",
            "coordinates_interpreted_under_pbc": True,
        },
        "definitions": {
            "contact_cutoff_A": _round_float(contact_cutoff_A),
            "pocket_cutoff_A": _round_float(pocket_cutoff_A),
            "protein_scope": "standard_amino_acid_ATOM_heavy_atoms",
            "pocket_definition": "canonical_PDB_residue_minimum_heavy_atom_distance_to_ligand_under_PBC",
            "native_contact_definition": "canonical_PDB_pocket_residue_with_minimum_heavy_atom_distance_at_or_below_contact_cutoff",
        },
        "ligand": {
            "selector": {
                "resname": ligand_resname.strip().upper(),
                "resid": None if ligand_resid is None else str(ligand_resid).strip(),
                "chain": _normalise_chain(ligand_chain),
            },
            "resolved_residue": _residue_record(ligand_key),
            "heavy_atom_indices0": ligand_indices.tolist(),
            "heavy_atoms": ligand_records,
        },
        "pocket": {
            "residues": pocket_rows,
            "heavy_atom_indices0": pocket_indices.tolist(),
        },
        "native_contacts": {
            "residue_keys": native_keys,
            "denominator_residue_count": len(native_keys),
        },
        "pose_reference": {
            "source": "canonical_PDB_coordinates",
            "alignment_atom_indices0": pocket_indices.tolist(),
            "ligand_atom_indices0": ligand_indices.tolist(),
            "pocket_centered_coordinates_A": _round_coordinates(pocket_centered),
            "ligand_pocket_centered_coordinates_A": _round_coordinates(ligand_centered),
            "centering": "PBC_unwrapped_canonical_pocket_geometric_centroid",
            "pbc_unwrapping": "pocket_around_first_pocket_atom_then_ligand_around_pocket_centroid",
            "atom_correspondence": "zero_based_canonical_PDB_and_topology_order",
            "global_translation_rotation_invariant_after_Kabsch": True,
        },
    }
    validate_manifest_structure(manifest)
    return manifest


def build_shared_reference_manifest(
    canonical_pdb_path: str | Path,
    *,
    complex_id: str,
    ligand_resname: str,
    ligand_resid: str | None = None,
    ligand_chain: str | None = None,
    topology_path: str | Path | None = None,
    topology_sha256: str | None = None,
    contact_cutoff_A: float = 4.5,
    pocket_cutoff_A: float = 8.0,
) -> dict[str, object]:
    """File API for building a shared reference manifest."""

    topology_hash = _resolve_topology_sha256(
        topology_path=topology_path, topology_sha256=topology_sha256
    )
    return build_shared_reference_manifest_from_bytes(
        Path(canonical_pdb_path).read_bytes(),
        complex_id=complex_id,
        topology_sha256=topology_hash,
        ligand_resname=ligand_resname,
        ligand_resid=ligand_resid,
        ligand_chain=ligand_chain,
        contact_cutoff_A=contact_cutoff_A,
        pocket_cutoff_A=pocket_cutoff_A,
    )


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SharedReferenceGateError(f"{name} must be a JSON object")
    return value


def _require_indices(value: object, name: str, atom_count: int) -> list[int]:
    if not isinstance(value, list) or not value:
        raise SharedReferenceGateError(f"{name} must be a non-empty JSON array")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise SharedReferenceGateError(f"{name} must contain integer indices")
    indices = [int(item) for item in value]
    if indices != sorted(set(indices)):
        raise SharedReferenceGateError(f"{name} must be sorted and unique")
    if indices[0] < 0 or indices[-1] >= atom_count:
        raise SharedReferenceGateError(f"{name} contains an out-of-range atom index")
    return indices


def _require_coordinates(value: object, name: str, rows: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SharedReferenceGateError(f"{name} must be numeric") from exc
    if array.shape != (rows, 3):
        raise SharedReferenceGateError(f"{name} shape is {array.shape}; expected ({rows}, 3)")
    if not np.all(np.isfinite(array)):
        raise SharedReferenceGateError(f"{name} contains NaN/inf")
    return array


def _contains_created_at(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(key == "created_at" or _contains_created_at(child) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_created_at(child) for child in value)
    return False


def validate_manifest_structure(manifest: Mapping[str, object]) -> None:
    """Validate schema and internal invariants without consulting external files."""

    if _contains_created_at(manifest):
        raise SharedReferenceGateError("Shared reference must not contain created_at; deterministic bytes are required")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SharedReferenceGateError(
            f"Unsupported shared-reference schema_version={manifest.get('schema_version')!r}"
        )
    complex_id = manifest.get("complex_id")
    if not isinstance(complex_id, str) or not complex_id.strip():
        raise SharedReferenceGateError("Manifest complex_id must be a non-empty string")
    provenance = _require_mapping(manifest.get("provenance"), "provenance")
    for field in ("canonical_pdb_sha256", "canonical_pdb_atom_schema_sha256", "topology_sha256"):
        value = provenance.get(field)
        if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
            raise SharedReferenceGateError(f"provenance.{field} is not a lowercase SHA-256")
    atom_count = provenance.get("canonical_pdb_atom_count")
    if isinstance(atom_count, bool) or not isinstance(atom_count, int) or atom_count <= 0:
        raise SharedReferenceGateError("provenance.canonical_pdb_atom_count must be a positive integer")

    periodic_box = _require_mapping(manifest.get("periodic_box"), "periodic_box")
    try:
        lengths = np.asarray(periodic_box.get("lengths_A"), dtype=float)
        angles = np.asarray(periodic_box.get("angles_deg"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise SharedReferenceGateError("periodic_box lengths/angles must be numeric") from exc
    if lengths.shape != (3,) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise SharedReferenceGateError("periodic_box.lengths_A must contain three finite positive values")
    if angles.shape != (3,) or not np.allclose(angles, 90.0, atol=1e-2):
        raise SharedReferenceGateError("periodic_box.angles_deg must describe an orthorhombic box")
    if periodic_box.get("coordinates_interpreted_under_pbc") is not True:
        raise SharedReferenceGateError("periodic_box must explicitly enable PBC interpretation")

    definitions = _require_mapping(manifest.get("definitions"), "definitions")
    try:
        contact_cutoff = float(definitions["contact_cutoff_A"])
        pocket_cutoff = float(definitions["pocket_cutoff_A"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SharedReferenceGateError("definitions must contain numeric contact/pocket cutoffs") from exc
    if not (math.isfinite(contact_cutoff) and 0 < contact_cutoff <= pocket_cutoff):
        raise SharedReferenceGateError("Manifest contact/pocket cutoffs are invalid")
    if pocket_cutoff > float(np.min(lengths)) / 2.0:
        raise SharedReferenceGateError("Manifest pocket cutoff exceeds the minimum-image uniqueness limit")

    ligand = _require_mapping(manifest.get("ligand"), "ligand")
    ligand_indices = _require_indices(ligand.get("heavy_atom_indices0"), "ligand.heavy_atom_indices0", atom_count)
    ligand_atoms = ligand.get("heavy_atoms")
    if not isinstance(ligand_atoms, list) or len(ligand_atoms) != len(ligand_indices):
        raise SharedReferenceGateError("ligand.heavy_atoms must match ligand heavy-atom indices")
    ligand_record_indices: list[int] = []
    for ordinal, row in enumerate(ligand_atoms):
        atom_row = _require_mapping(row, f"ligand.heavy_atoms[{ordinal}]")
        index = atom_row.get("index0")
        if isinstance(index, bool) or not isinstance(index, int):
            raise SharedReferenceGateError("Each ligand atom record requires integer index0")
        ligand_record_indices.append(index)
        if not all(isinstance(atom_row.get(field), str) and atom_row.get(field) for field in ("name", "resname", "chain", "resid", "element")):
            raise SharedReferenceGateError("Each ligand atom record requires complete atom identity")
        if str(atom_row["element"]).upper() == "H":
            raise SharedReferenceGateError("ligand.heavy_atoms unexpectedly contains hydrogen")
    if ligand_record_indices != ligand_indices:
        raise SharedReferenceGateError("ligand.heavy_atoms order/index values do not match heavy_atom_indices0")

    pocket = _require_mapping(manifest.get("pocket"), "pocket")
    pocket_indices = _require_indices(pocket.get("heavy_atom_indices0"), "pocket.heavy_atom_indices0", atom_count)
    if set(ligand_indices) & set(pocket_indices):
        raise SharedReferenceGateError("Ligand and protein-pocket atom indices overlap")
    residues = pocket.get("residues")
    if not isinstance(residues, list) or not residues:
        raise SharedReferenceGateError("pocket.residues must be non-empty")
    residue_keys: list[str] = []
    residue_identity_tuples: list[tuple[str, str, str]] = []
    native_from_rows: list[str] = []
    residue_index_union: list[int] = []
    for ordinal, row in enumerate(residues):
        residue = _require_mapping(row, f"pocket.residues[{ordinal}]")
        key = residue.get("residue_key")
        if not isinstance(key, str) or not key:
            raise SharedReferenceGateError("Each pocket residue requires residue_key")
        expected_key = f"{residue.get('chain')}:{residue.get('resid')}:{residue.get('resname')}"
        if key != expected_key:
            raise SharedReferenceGateError(f"Pocket residue key is inconsistent: {key} != {expected_key}")
        residue_keys.append(key)
        residue_identity_tuples.append(
            (str(residue.get("chain")), str(residue.get("resid")), str(residue.get("resname")))
        )
        indices = _require_indices(
            residue.get("heavy_atom_indices0"),
            f"pocket.residues[{ordinal}].heavy_atom_indices0",
            atom_count,
        )
        residue_index_union.extend(indices)
        try:
            distance = float(residue["canonical_min_ligand_heavy_distance_A"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SharedReferenceGateError("Pocket residue canonical distance must be numeric") from exc
        if not math.isfinite(distance) or distance < 0 or distance > pocket_cutoff + 1e-7:
            raise SharedReferenceGateError("Pocket residue canonical distance is outside the pocket cutoff")
        native = residue.get("native_contact")
        if not isinstance(native, bool):
            raise SharedReferenceGateError("Pocket residue native_contact must be boolean")
        if native != (distance <= contact_cutoff + 1e-7):
            raise SharedReferenceGateError("Pocket residue native flag disagrees with its canonical distance")
        if native:
            native_from_rows.append(key)
    if residue_identity_tuples != sorted(set(residue_identity_tuples)):
        raise SharedReferenceGateError("Pocket residues must be sorted and unique")
    if sorted(residue_index_union) != pocket_indices or len(set(residue_index_union)) != len(residue_index_union):
        raise SharedReferenceGateError("Pocket heavy-atom indices do not equal the disjoint residue-index union")

    native = _require_mapping(manifest.get("native_contacts"), "native_contacts")
    native_keys = native.get("residue_keys")
    if not isinstance(native_keys, list) or not native_keys or any(not isinstance(key, str) for key in native_keys):
        raise SharedReferenceGateError("native_contacts.residue_keys must be a non-empty string array")
    if native_keys != native_from_rows:
        raise SharedReferenceGateError("native-contact vocabulary disagrees with pocket residue flags")
    if native.get("denominator_residue_count") != len(native_keys):
        raise SharedReferenceGateError("native-contact denominator does not match the frozen vocabulary")

    pose = _require_mapping(manifest.get("pose_reference"), "pose_reference")
    alignment_indices = _require_indices(
        pose.get("alignment_atom_indices0"), "pose_reference.alignment_atom_indices0", atom_count
    )
    pose_ligand_indices = _require_indices(
        pose.get("ligand_atom_indices0"), "pose_reference.ligand_atom_indices0", atom_count
    )
    if alignment_indices != pocket_indices or pose_ligand_indices != ligand_indices:
        raise SharedReferenceGateError("Pose atom correspondence disagrees with frozen pocket/ligand indices")
    pocket_coordinates = _require_coordinates(
        pose.get("pocket_centered_coordinates_A"),
        "pose_reference.pocket_centered_coordinates_A",
        len(pocket_indices),
    )
    _require_coordinates(
        pose.get("ligand_pocket_centered_coordinates_A"),
        "pose_reference.ligand_pocket_centered_coordinates_A",
        len(ligand_indices),
    )
    if not np.allclose(np.mean(pocket_coordinates, axis=0), 0.0, atol=2e-8):
        raise SharedReferenceGateError("Pose pocket coordinates are not centered on their geometric centroid")
    if np.linalg.matrix_rank(pocket_coordinates, tol=1e-8) < 2:
        raise SharedReferenceGateError("Pose pocket reference is rank-deficient")


def canonical_manifest_bytes(manifest: Mapping[str, object]) -> bytes:
    """Return the canonical on-disk JSON bytes after structural validation."""

    validate_manifest_structure(manifest)
    return (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_shared_reference(path: str | Path, manifest: Mapping[str, object]) -> None:
    """Atomically write deterministic JSON (no path/time-dependent fields)."""

    payload = canonical_manifest_bytes(manifest).decode("utf-8")
    with atomic_text_writer(path) as handle:
        handle.write(payload)


def load_shared_reference(path: str | Path) -> dict[str, object]:
    """Load a manifest and fail closed on malformed JSON or schema drift."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SharedReferenceGateError(f"Cannot load shared reference {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SharedReferenceGateError("Shared-reference JSON root must be an object")
    validate_manifest_structure(payload)
    return payload


def validate_shared_reference(
    manifest: Mapping[str, object],
    *,
    complex_id: str,
    pdb_path: str | Path,
    topology_path: str | Path | None = None,
    topology_sha256: str | None = None,
) -> dict[str, object]:
    """Validate a replica context against a previously frozen reference.

    Replica coordinates may differ from the canonical pose.  Therefore this
    gate compares the exact topology hash and coordinate-independent ordered
    PDB atom schema, while the manifest's full canonical-PDB hash continues to
    bind the reference coordinates themselves.
    """

    validate_manifest_structure(manifest)
    expected_complex = str(manifest["complex_id"])
    if complex_id != expected_complex:
        raise SharedReferenceGateError(
            f"Complex mismatch: manifest={expected_complex}, requested={complex_id}"
        )
    observed_topology = _resolve_topology_sha256(
        topology_path=topology_path, topology_sha256=topology_sha256
    )
    provenance = _require_mapping(manifest["provenance"], "provenance")
    expected_topology = str(provenance["topology_sha256"])
    if observed_topology != expected_topology:
        raise SharedReferenceGateError(
            f"Topology hash mismatch: manifest={expected_topology}, observed={observed_topology}"
        )
    pdb_bytes = Path(pdb_path).read_bytes()
    # A matching hash alone is not enough.  Parse the PDB as well so malformed
    # or ambiguous atom records cannot pass just because a hashing convention
    # happened to accept them.  The replica may have moved coordinates and box
    # lengths, but it must preserve the atom identities used by the reference.
    parsed = parse_pdb_bytes(pdb_bytes)
    expected_schema = str(provenance["canonical_pdb_atom_schema_sha256"])
    if parsed.atom_schema_sha256 != expected_schema:
        raise SharedReferenceGateError(
            f"PDB atom-schema mismatch: manifest={expected_schema}, observed={parsed.atom_schema_sha256}"
        )
    if len(parsed.atoms) != int(provenance["canonical_pdb_atom_count"]):
        raise SharedReferenceGateError(
            f"PDB atom-count mismatch: manifest={provenance['canonical_pdb_atom_count']}, observed={len(parsed.atoms)}"
        )
    return {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "complex_id": expected_complex,
        "topology_sha256": observed_topology,
        "pdb_atom_schema_sha256": parsed.atom_schema_sha256,
        "pdb_atom_count": len(parsed.atoms),
    }


def load_and_validate_shared_reference(
    manifest_path: str | Path,
    *,
    complex_id: str,
    pdb_path: str | Path,
    topology_path: str | Path | None = None,
    topology_sha256: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Convenience API returning both the manifest and its validation receipt."""

    manifest = load_shared_reference(manifest_path)
    receipt = validate_shared_reference(
        manifest,
        complex_id=complex_id,
        pdb_path=pdb_path,
        topology_path=topology_path,
        topology_sha256=topology_sha256,
    )
    return manifest, receipt


def _add_topology_source(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--topology", help="topology file whose SHA-256 is computed")
    source.add_argument("--topology-sha256", help="pre-audited lowercase SHA-256")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.koff_ml.shared_reference",
        description="Build/validate a deterministic per-complex LiGaMD structural reference",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="freeze ligand, pocket, native contacts, and pose reference")
    build.add_argument("--complex-id", required=True)
    build.add_argument("--canonical-pdb", required=True)
    build.add_argument("--ligand-resname", required=True)
    build.add_argument("--ligand-resid")
    build.add_argument("--ligand-chain")
    build.add_argument("--contact-cutoff", type=float, default=4.5)
    build.add_argument("--pocket-cutoff", type=float, default=8.0)
    build.add_argument("--output", required=True)
    _add_topology_source(build)

    validate = commands.add_parser("validate", help="fail closed on complex/topology/PDB-schema drift")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--complex-id", required=True)
    validate.add_argument("--pdb", required=True)
    _add_topology_source(validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        topology_path = getattr(args, "topology", None)
        topology_sha = getattr(args, "topology_sha256", None)
        if args.command == "build":
            manifest = build_shared_reference_manifest(
                args.canonical_pdb,
                complex_id=args.complex_id,
                ligand_resname=args.ligand_resname,
                ligand_resid=args.ligand_resid,
                ligand_chain=args.ligand_chain,
                topology_path=topology_path,
                topology_sha256=topology_sha,
                contact_cutoff_A=args.contact_cutoff,
                pocket_cutoff_A=args.pocket_cutoff,
            )
            write_shared_reference(args.output, manifest)
            receipt = {
                "status": "PASS",
                "command": "build",
                "complex_id": manifest["complex_id"],
                "manifest_sha256": sha256_file(args.output),
                "output": str(Path(args.output).resolve()),
            }
        else:
            manifest, receipt = load_and_validate_shared_reference(
                args.manifest,
                complex_id=args.complex_id,
                pdb_path=args.pdb,
                topology_path=topology_path,
                topology_sha256=topology_sha,
            )
            receipt = {
                **receipt,
                "command": "validate",
                "manifest_sha256": sha256_file(args.manifest),
            }
    except (SharedReferenceGateError, OSError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``main`` in tests
    raise SystemExit(main())
