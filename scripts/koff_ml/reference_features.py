"""Turn one Amber NetCDF trajectory into a dense, auditable geometry trace.

The output is a table with one row per saved coordinate frame.  It is the
bridge between raw molecular-dynamics coordinates and the later frame samplers:
the sampler may choose only rows that already exist in this table.  The module
does not infer a kinetic rate, classify a chemical interaction, or read an
experimental pKoff label.

For a formal comparison across replicas, every trajectory is measured against
the same outcome-blind ``shared_reference.json``.  That file freezes the ligand
atom order, the pocket residues, the native-contact vocabulary, and the pose
reference before any result is considered.  A frame-1-local reference is
available only as an explicitly marked exploratory option because it is not
comparable across replicas.

The dense trace records geometry that a PDB-plus-coordinate file can support
without guessing chemistry:

* ligand-to-pocket geometric-centroid distance;
* residue-level heavy-atom contact bits and native-contact fraction;
* contact formation and loss relative to the previous saved frame;
* pocket-aligned ligand pose and ligand-internal RMSD;
* whole-protein minimum distance and contacted-residue count.

These residue contacts are proximity measurements, not a typed interaction
fingerprint.  Hydrogen bonds, pi stacking, salt bridges, and water bridges need
chemical typing that this extractor does not assume.  The program stops rather
than silently produce a trace when the ligand selector is ambiguous, the PDB
atom order does not match the NetCDF atom axis, or periodic-box information is
missing or non-orthorhombic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .io import atomic_text_writer, sha256_file, write_json
from .shared_reference import (
    SharedReferenceGateError,
    load_and_validate_shared_reference,
)


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYM", "CYX", "GLN", "GLU", "GLH", "GLY",
    "HID", "HIE", "HIP", "HIS", "ILE", "LEU", "LYS", "LYN", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
}


class ReferenceFeatureGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class PDBAtom:
    index: int
    record: str
    name: str
    residue: str
    chain: str
    resid: str
    element: str

    @property
    def residue_key(self) -> tuple[str, str, str]:
        return self.chain, self.resid, self.residue

    @property
    def is_heavy(self) -> bool:
        return self.element.upper() != "H"


def _infer_element(atom_name: str) -> str:
    letters = "".join(ch for ch in atom_name if ch.isalpha()).upper()
    if not letters:
        return ""
    # Amber atom names may begin with a digit; common two-letter elements need
    # explicit handling while CA in proteins is alpha-carbon, not calcium.
    if letters[:2] in {"CL", "BR", "NA", "MG", "ZN", "FE"} and atom_name.strip().upper()[:2] == letters[:2]:
        return letters[:2]
    return letters[0]


def parse_pdb_atoms(path: str | Path) -> list[PDBAtom]:
    """Read atom identities in PDB order, which must match the NetCDF axis.

    The extractor needs only identity fields needed to select atoms and residues;
    it does not read charges or bond types.  ``PDBAtom.index`` is zero-based and
    is used directly to slice the trajectory coordinate array.
    """

    atoms: list[PDBAtom] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[0:6].strip().upper()
            if record not in {"ATOM", "HETATM"}:
                continue
            name = line[12:16].strip()
            residue = line[17:20].strip().upper()
            chain = line[21:22].strip() or "_"
            resid = (line[22:26].strip() + line[26:27].strip()) or "?"
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            atoms.append(PDBAtom(len(atoms), record, name, residue, chain, resid, element or _infer_element(name)))
    if not atoms:
        raise ReferenceFeatureGateError(f"PDB contains no ATOM/HETATM records: {path}")
    return atoms


def select_ligand_indices(
    atoms: list[PDBAtom],
    *,
    ligand_resname: str,
    ligand_resid: str | None = None,
    ligand_chain: str | None = None,
) -> np.ndarray:
    """Return zero-based heavy-atom indices for one explicitly named ligand.

    Guessing that the last residue is a ligand can silently select an ion,
    cofactor, or the wrong copy of a repeated ligand.  The caller therefore
    supplies a residue name, and must also provide residue number or chain when
    that name occurs in more than one residue.
    """

    if not ligand_resname:
        raise ReferenceFeatureGateError("Explicit --ligand-resname is required; automatic last-residue guessing is forbidden")
    resname = ligand_resname.upper()
    matches = [atom for atom in atoms if atom.residue == resname]
    if ligand_resid is not None:
        matches = [atom for atom in matches if atom.resid == str(ligand_resid)]
    if ligand_chain is not None:
        matches = [atom for atom in matches if atom.chain == ligand_chain]
    residue_keys = {atom.residue_key for atom in matches}
    if not matches:
        raise ReferenceFeatureGateError(
            f"Ligand selector matched zero atoms: resname={resname}, resid={ligand_resid}, chain={ligand_chain}"
        )
    if len(residue_keys) != 1:
        raise ReferenceFeatureGateError(
            f"Ligand selector is ambiguous ({len(residue_keys)} residues); add --ligand-resid/--ligand-chain"
        )
    heavy = [atom.index for atom in matches if atom.is_heavy]
    if not heavy:
        raise ReferenceFeatureGateError("Selected ligand has no identifiable heavy atoms")
    return np.asarray(heavy, dtype=np.int64)


def _minimum_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Apply the orthorhombic minimum-image convention to an xyz displacement.

    The operation is ``delta - box * round(delta / box)``.  It keeps each atom
    pair in the nearest periodic image, so a molecule split across a box edge is
    not mistaken for a very long distance.  It is valid only after
    :func:`_validate_box` has established 90-degree cell angles.
    """

    return delta - box * np.rint(delta / box)


def _validate_box(lengths: np.ndarray, angles: np.ndarray | None) -> None:
    """Require finite positive box lengths and verified orthorhombic PBC."""

    if lengths is None or lengths.shape[-1] != 3 or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ReferenceFeatureGateError("Valid per-frame cell_lengths are required for fail-closed PBC handling")
    if angles is None:
        raise ReferenceFeatureGateError("cell_angles are missing; orthorhombic minimum-image handling cannot be verified")
    if angles.shape[-1] != 3 or not np.all(np.isfinite(angles)):
        raise ReferenceFeatureGateError("cell_angles are invalid")
    if not np.allclose(angles, 90.0, atol=1e-2):
        raise ReferenceFeatureGateError("Only verified orthorhombic boxes are supported; non-90-degree cell angles found")


def _residue_label(key: tuple[str, str, str]) -> str:
    chain, resid, residue = key
    clean = lambda value: "".join(ch if ch.isalnum() else "_" for ch in value)
    return f"{clean(chain)}_{clean(resid)}_{clean(residue)}"


def _residue_min_distance_matrix(
    coordinates: np.ndarray,
    boxes: np.ndarray,
    ligand_indices: np.ndarray,
    residues: list[np.ndarray],
) -> np.ndarray:
    """Compute the nearest ligand--residue heavy-atom distance in every frame.

    For frame ``t`` and protein residue ``r`` the value is
    ``min(||MIC(x_residue - x_ligand)||)`` over every heavy-atom pair.  The
    returned matrix has shape ``(frames, residues)``.  Contacts, global minimum
    distance, and whole-protein contact count all come from this one matrix so
    that they cannot drift because of different distance implementations.
    """

    ligand = coordinates[:, ligand_indices, :]
    result = np.zeros((len(coordinates), len(residues)), dtype=float)
    for column, indices in enumerate(residues):
        residue = coordinates[:, indices, :]
        delta = residue[:, :, None, :] - ligand[:, None, :, :]
        delta = _minimum_image(delta, boxes[:, None, None, :])
        squared = np.sum(delta * delta, axis=-1)
        result[:, column] = np.sqrt(np.min(squared, axis=(1, 2)))
    return result


def _contact_matrix(
    coordinates: np.ndarray,
    boxes: np.ndarray,
    ligand_indices: np.ndarray,
    residues: list[np.ndarray],
    cutoff: float,
) -> np.ndarray:
    """Convert nearest residue distances to a Boolean proximity-contact matrix."""

    return _residue_min_distance_matrix(coordinates, boxes, ligand_indices, residues) <= cutoff


def _geometric_com_distance(
    coordinates: np.ndarray,
    boxes: np.ndarray,
    ligand_indices: np.ndarray,
    pocket_indices: np.ndarray,
) -> np.ndarray:
    """Measure ligand-centroid distance from the frozen-pocket centroid under PBC.

    First the ligand is unwrapped around its first heavy atom, then its
    geometric (not mass-weighted) centroid is calculated.  Each frozen-pocket
    atom is placed in the ligand centroid's nearest periodic image.  The norm of
    the mean pocket displacement answers one specific question: how far is the
    ligand as a whole from the original pocket?  It is not an interaction energy
    and it is not a single atom-pair distance.
    """

    ligand = coordinates[:, ligand_indices, :]
    anchor = ligand[:, :1, :]
    ligand_unwrapped = anchor + _minimum_image(ligand - anchor, boxes[:, None, :])
    ligand_center = np.mean(ligand_unwrapped, axis=1)
    pocket = coordinates[:, pocket_indices, :]
    pocket_delta = _minimum_image(pocket - ligand_center[:, None, :], boxes[:, None, :])
    pocket_center_delta = np.mean(pocket_delta, axis=1)
    return np.linalg.norm(pocket_center_delta, axis=1)


def _unwrap_pocket_and_ligand(
    coordinates: np.ndarray,
    boxes: np.ndarray,
    ligand_indices: np.ndarray,
    pocket_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one PBC-consistent pocket/ligand image per frame.

    The pocket is unwrapped around its first heavy atom.  The ligand is first
    unwrapped internally and then placed in the minimum image around the
    pocket centroid.  This uses no future frame or outcome information.
    """

    pocket_raw = coordinates[:, pocket_indices, :]
    pocket_anchor = pocket_raw[:, :1, :]
    pocket = pocket_anchor + _minimum_image(pocket_raw - pocket_anchor, boxes[:, None, :])
    pocket_center = np.mean(pocket, axis=1)

    ligand_raw = coordinates[:, ligand_indices, :]
    ligand_anchor_raw = ligand_raw[:, :1, :]
    ligand_anchor = pocket_center[:, None, :] + _minimum_image(
        ligand_anchor_raw - pocket_center[:, None, :], boxes[:, None, :]
    )
    ligand = ligand_anchor + _minimum_image(ligand_raw - ligand_anchor_raw, boxes[:, None, :])
    return pocket, ligand


def _proper_kabsch_rotations(mobile_centered: np.ndarray, reference_centered: np.ndarray) -> np.ndarray:
    """Return the proper Kabsch rotation from each mobile pocket to the reference.

    Kabsch alignment uses an SVD to minimise least-squares error between pocket
    heavy atoms.  An SVD can otherwise return a reflection; when that happens we
    flip one singular vector so the rotation has determinant ``+1``.  A mirror
    image must not be treated as a physically valid protein alignment.
    """

    covariance = np.einsum("fpi,pj->fij", mobile_centered, reference_centered, optimize=True)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    reflected = np.linalg.det(rotation) < 0
    if np.any(reflected):
        left = left.copy()
        left[reflected, :, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def _aligned_ligand_pose_features(
    coordinates: np.ndarray,
    boxes: np.ndarray,
    ligand_indices: np.ndarray,
    pocket_indices: np.ndarray,
    reference_pocket_centered: np.ndarray,
    reference_ligand_centered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align each frame to the frozen pocket and calculate pose-invariant outputs.

    The first Kabsch fit uses pocket heavy atoms and applies the same rigid-body
    rotation to the ligand.  It yields a ligand *pose* RMSD, which includes
    ligand movement and rotation relative to the pocket as well as internal
    deformation.  A second fit aligns the ligand to itself and yields internal
    conformational RMSD only.  Those values answer different questions and must
    not be interchanged.

    The return order is aligned ligand RMSD, aligned centroid displacement,
    ligand-internal RMSD, pocket-alignment RMSD, and per-ligand-heavy-atom
    displacement.  Atom correspondence remains the fixed PDB/trajectory order.
    """

    pocket, ligand = _unwrap_pocket_and_ligand(coordinates, boxes, ligand_indices, pocket_indices)
    mobile_center = np.mean(pocket, axis=1)
    pocket_centered = pocket - mobile_center[:, None, :]
    rotations = _proper_kabsch_rotations(pocket_centered, reference_pocket_centered)
    aligned_pocket = np.einsum("fpi,fij->fpj", pocket_centered, rotations, optimize=True)
    aligned_ligand = np.einsum(
        "fpi,fij->fpj", ligand - mobile_center[:, None, :], rotations, optimize=True
    )
    displacement = aligned_ligand - reference_ligand_centered[None, :, :]
    per_atom_displacement = np.linalg.norm(displacement, axis=2)
    aligned_rmsd = np.sqrt(np.mean(np.sum(displacement * displacement, axis=2), axis=1))
    centroid_displacement = np.linalg.norm(
        np.mean(aligned_ligand, axis=1) - np.mean(reference_ligand_centered, axis=0), axis=1
    )
    pocket_rmsd = np.sqrt(
        np.mean(np.sum((aligned_pocket - reference_pocket_centered[None, :, :]) ** 2, axis=2), axis=1)
    )

    ligand_mobile_centered = aligned_ligand - np.mean(aligned_ligand, axis=1)[:, None, :]
    ligand_reference_centered = (
        reference_ligand_centered - np.mean(reference_ligand_centered, axis=0)[None, :]
    )
    ligand_rotations = _proper_kabsch_rotations(ligand_mobile_centered, ligand_reference_centered)
    self_aligned = np.einsum(
        "fpi,fij->fpj", ligand_mobile_centered, ligand_rotations, optimize=True
    )
    internal_rmsd = np.sqrt(
        np.mean(np.sum((self_aligned - ligand_reference_centered[None, :, :]) ** 2, axis=2), axis=1)
    )
    return aligned_rmsd, centroid_displacement, internal_rmsd, pocket_rmsd, per_atom_displacement


def _extract_reference_features_impl(
    trajectory_path: str | Path,
    pdb_path: str | Path,
    output_csv: str | Path,
    gate_json: str | Path,
    *,
    ligand_resname: str,
    ligand_resid: str | None = None,
    ligand_chain: str | None = None,
    contact_cutoff_A: float = 4.5,
    pocket_cutoff_A: float = 8.0,
    chunk_frames: int = 250,
    max_frames: int | None = None,
    complex_id: str | None = None,
    topology_path: str | Path | None = None,
    shared_manifest_path: str | Path | None = None,
    exploratory_replica_local_reference: bool = False,
) -> dict[str, object]:
    """Extract one dense frame-level reference table from a full production NetCDF.

    The formal path validates the PDB, NetCDF, topology, and shared reference;
    reads the frozen ligand, pocket, native contacts, and pose reference; then
    streams the NetCDF in chunks.  Each chunk yields a whole-protein
    residue-distance matrix, from which pocket/global contacts, centroid
    distance, pose RMSD, and internal RMSD are derived.  The function writes a
    CSV plus a gate JSON that records the input identities and measurement rules.

    ``max_frames`` is only for explicitly marked smoke or exploratory work.  A
    formal complete-exit check needs the full production trajectory because the
    frames after a candidate onset are needed to confirm persistence.
    """

    try:
        from scipy.io import netcdf_file
    except ImportError as exc:
        raise ReferenceFeatureGateError("scipy.io.netcdf_file is unavailable") from exc
    trajectory_path = Path(trajectory_path)
    pdb_path = Path(pdb_path)
    output_csv = Path(output_csv)
    gate_json = Path(gate_json)
    if shared_manifest_path is None and not exploratory_replica_local_reference:
        raise ReferenceFeatureGateError(
            "Formal extraction requires --shared-manifest, --complex-id, and --topology. "
            "Use --exploratory-replica-local-reference only for explicitly non-comparable smoke work."
        )
    if shared_manifest_path is not None and exploratory_replica_local_reference:
        raise ReferenceFeatureGateError(
            "--shared-manifest and --exploratory-replica-local-reference are mutually exclusive"
        )
    if shared_manifest_path is not None and (not complex_id or topology_path is None):
        raise ReferenceFeatureGateError(
            "--shared-manifest requires non-empty --complex-id and --topology"
        )
    atoms = parse_pdb_atoms(pdb_path)
    ligand_indices = select_ligand_indices(
        atoms,
        ligand_resname=ligand_resname,
        ligand_resid=ligand_resid,
        ligand_chain=ligand_chain,
    )
    protein_by_residue: dict[tuple[str, str, str], list[int]] = {}
    for atom in atoms:
        if atom.record == "ATOM" and atom.residue in STANDARD_AA and atom.is_heavy:
            protein_by_residue.setdefault(atom.residue_key, []).append(atom.index)
    if not protein_by_residue:
        raise ReferenceFeatureGateError("No standard protein heavy atoms were identified in sys-protein.pdb")
    shared_manifest: dict[str, object] | None = None
    shared_manifest_sha256: str | None = None
    shared_validation: dict[str, object] | None = None
    if shared_manifest_path is not None:
        try:
            shared_manifest, shared_validation = load_and_validate_shared_reference(
                shared_manifest_path,
                complex_id=str(complex_id),
                pdb_path=pdb_path,
                topology_path=topology_path,
            )
        except SharedReferenceGateError as exc:
            raise ReferenceFeatureGateError(f"Shared-reference gate failed: {exc}") from exc
        shared_manifest_sha256 = sha256_file(shared_manifest_path)
    with netcdf_file(str(trajectory_path), "r", mmap=True) as nc:
        required_variables = {"coordinates", "cell_lengths", "cell_angles"}
        missing_variables = sorted(required_variables - set(nc.variables))
        if missing_variables:
            raise ReferenceFeatureGateError(
                "Amber NetCDF is missing required variable(s): " + ", ".join(missing_variables)
            )
        coord_var = nc.variables["coordinates"]
        n_frames, n_atoms, spatial = coord_var.shape
        if spatial != 3:
            raise ReferenceFeatureGateError(f"coordinates last dimension is {spatial}, expected 3")
        if n_atoms != len(atoms):
            raise ReferenceFeatureGateError(f"Atom-count mismatch: trajectory={n_atoms}, PDB={len(atoms)}")
        lengths_var = nc.variables["cell_lengths"]
        angles_var = nc.variables["cell_angles"]
        first_coords = np.asarray(coord_var[0:1], dtype=float)
        first_boxes = np.asarray(lengths_var[0:1], dtype=float)
        first_angles = np.asarray(angles_var[0:1], dtype=float)
        _validate_box(first_boxes, first_angles)

        residue_keys = sorted(protein_by_residue)
        residue_arrays = [np.asarray(protein_by_residue[key], dtype=np.int64) for key in residue_keys]
        if shared_manifest is not None:
            definitions = shared_manifest["definitions"]
            manifest_contact_cutoff = float(definitions["contact_cutoff_A"])
            manifest_pocket_cutoff = float(definitions["pocket_cutoff_A"])
            if not np.isclose(contact_cutoff_A, manifest_contact_cutoff, atol=1e-12):
                raise ReferenceFeatureGateError(
                    f"contact cutoff differs from shared manifest: CLI={contact_cutoff_A}, "
                    f"manifest={manifest_contact_cutoff}"
                )
            if not np.isclose(pocket_cutoff_A, manifest_pocket_cutoff, atol=1e-12):
                raise ReferenceFeatureGateError(
                    f"pocket cutoff differs from shared manifest: CLI={pocket_cutoff_A}, "
                    f"manifest={manifest_pocket_cutoff}"
                )
            manifest_ligand_indices = np.asarray(
                shared_manifest["ligand"]["heavy_atom_indices0"], dtype=np.int64
            )
            if not np.array_equal(ligand_indices, manifest_ligand_indices):
                raise ReferenceFeatureGateError(
                    "Resolved ligand heavy-atom indices differ from the shared manifest"
                )
            pocket_rows = shared_manifest["pocket"]["residues"]
            pocket_keys = [
                (str(row["chain"]), str(row["resid"]), str(row["resname"]))
                for row in pocket_rows
            ]
            missing_keys = [key for key in pocket_keys if key not in protein_by_residue]
            if missing_keys:
                raise ReferenceFeatureGateError(
                    "Shared pocket residue(s) are absent from replica PDB: "
                    + ",".join(map(str, missing_keys))
                )
            pocket_residues = [
                np.asarray(protein_by_residue[key], dtype=np.int64) for key in pocket_keys
            ]
            for key, indices, row in zip(pocket_keys, pocket_residues, pocket_rows):
                expected = np.asarray(row["heavy_atom_indices0"], dtype=np.int64)
                if not np.array_equal(indices, expected):
                    raise ReferenceFeatureGateError(
                        f"Shared pocket atom indices differ for residue {key}"
                    )
            pocket_indices = np.asarray(
                shared_manifest["pocket"]["heavy_atom_indices0"], dtype=np.int64
            )
            native_key_set = set(shared_manifest["native_contacts"]["residue_keys"])
            native = np.asarray(
                [f"{key[0]}:{key[1]}:{key[2]}" in native_key_set for key in pocket_keys],
                dtype=bool,
            )
            pose_reference = shared_manifest["pose_reference"]
            reference_pocket_centered = np.asarray(
                pose_reference["pocket_centered_coordinates_A"], dtype=float
            )
            reference_ligand_centered = np.asarray(
                pose_reference["ligand_pocket_centered_coordinates_A"], dtype=float
            )
            reference_mode = "shared_outcome_blind_canonical_structure"
        else:
            # Explicit exploratory fallback: define vocabulary/reference from
            # this replica's frame 1.  The gate below marks the resulting table
            # as non-comparable across replicas.
            pocket_mask = _contact_matrix(
                first_coords,
                first_boxes,
                ligand_indices,
                residue_arrays,
                pocket_cutoff_A,
            )[0]
            pocket_keys = [key for key, keep in zip(residue_keys, pocket_mask) if keep]
            pocket_residues = [indices for indices, keep in zip(residue_arrays, pocket_mask) if keep]
            if not pocket_residues:
                raise ReferenceFeatureGateError(
                    f"No protein residue is within pocket cutoff {pocket_cutoff_A} A in frame 1"
                )
            pocket_indices = np.unique(np.concatenate(pocket_residues))
            native = _contact_matrix(
                first_coords,
                first_boxes,
                ligand_indices,
                pocket_residues,
                contact_cutoff_A,
            )[0]
            if not np.any(native):
                raise ReferenceFeatureGateError(
                    f"Frame 1 has zero native contacts at {contact_cutoff_A} A; "
                    "native-contact fraction is undefined"
                )
            reference_pocket, reference_ligand = _unwrap_pocket_and_ligand(
                first_coords,
                first_boxes,
                ligand_indices,
                pocket_indices,
            )
            reference_pocket_center = np.mean(reference_pocket[0], axis=0)
            reference_pocket_centered = reference_pocket[0] - reference_pocket_center
            reference_ligand_centered = reference_ligand[0] - reference_pocket_center
            reference_mode = "exploratory_replica_frame1"
        if np.linalg.matrix_rank(reference_pocket_centered, tol=1e-8) < 2:
            raise ReferenceFeatureGateError(
                "Pocket heavy-atom geometry is rank-deficient; a unique audited rotational alignment cannot be established"
            )
        ligand_atom_labels = []
        for ordinal, atom_index in enumerate(ligand_indices):
            atom_name = "".join(ch if ch.isalnum() else "_" for ch in atoms[int(atom_index)].name)
            ligand_atom_labels.append(f"{ordinal:03d}_{atom_name or 'UNK'}")
        total = min(n_frames, int(max_frames)) if max_frames is not None else n_frames
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
        ] + [f"contact__{_residue_label(key)}" for key in pocket_keys] + [
            f"pose_proxy__{_residue_label(key)}__min_heavy_distance_A" for key in pocket_keys
        ] + [
            f"pose__ligand_atom_{label}__aligned_displacement_A" for label in ligand_atom_labels
        ]
        feature_schema_sha256 = hashlib.sha256(
            ("\n".join(fields) + "\n").encode("utf-8")
        ).hexdigest()
        previous: np.ndarray | None = None
        residue_key_to_position = {key: position for position, key in enumerate(residue_keys)}
        pocket_positions = np.asarray(
            [residue_key_to_position[key] for key in pocket_keys], dtype=np.int64
        )
        with atomic_text_writer(output_csv, newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            for start in range(0, total, chunk_frames):
                stop = min(total, start + chunk_frames)
                # Copy one chunk at a time so a long trajectory does not occupy
                # memory as one giant array.  Every calculation still uses that
                # frame's own cell lengths and angles.
                coordinates = np.asarray(coord_var[start:stop], dtype=float)
                boxes = np.asarray(lengths_var[start:stop], dtype=float)
                angles = np.asarray(angles_var[start:stop], dtype=float)
                _validate_box(boxes, angles)
                # One full-protein residue-distance pass supplies two
                # independent bulk-unbound gates, while the frozen pocket
                # subset retains the original contact/pose schema.  These two
                # scalars distinguish pocket exit from true loss of all
                # protein-surface contacts without storing a full route-wide
                # contact matrix for every 1-ps frame.
                all_residue_distances = _residue_min_distance_matrix(
                    coordinates, boxes, ligand_indices, residue_arrays
                )
                residue_distances = all_residue_distances[:, pocket_positions]
                contacts = residue_distances <= contact_cutoff_A
                global_min_distance = np.min(all_residue_distances, axis=1)
                global_contact_count = np.sum(
                    all_residue_distances <= contact_cutoff_A, axis=1
                )
                com_distance = _geometric_com_distance(coordinates, boxes, ligand_indices, pocket_indices)
                native_fraction = np.mean(contacts[:, native], axis=1)
                (
                    aligned_pose_rmsd,
                    aligned_centroid_displacement,
                    internal_conformation_rmsd,
                    pocket_alignment_rmsd,
                    per_atom_pose_displacement,
                ) = _aligned_ligand_pose_features(
                    coordinates,
                    boxes,
                    ligand_indices,
                    pocket_indices,
                    reference_pocket_centered,
                    reference_ligand_centered,
                )
                for local in range(len(coordinates)):
                    current = contacts[local]
                    if previous is None:
                        formed = broken = 0
                    else:
                        formed = int(np.sum(current & ~previous))
                        broken = int(np.sum(~current & previous))
                    writer.writerow(
                        [
                            start + local + 1,
                            f"{com_distance[local]:.8g}",
                            f"{native_fraction[local]:.8g}",
                            formed,
                            broken,
                            f"{aligned_pose_rmsd[local]:.8g}",
                            f"{aligned_centroid_displacement[local]:.8g}",
                            f"{internal_conformation_rmsd[local]:.8g}",
                            f"{pocket_alignment_rmsd[local]:.8g}",
                            f"{global_min_distance[local]:.8g}",
                            int(global_contact_count[local]),
                            *current.astype(np.int8).tolist(),
                            *[f"{value:.8g}" for value in residue_distances[local]],
                            *[f"{value:.8g}" for value in per_atom_pose_displacement[local]],
                        ]
                    )
                    previous = current.copy()
        # scipy's mmap-backed variable objects keep references to the mapping
        # and otherwise emit a misleading close warning even though every slice
        # above was copied. The file is read-only and all needed arrays are now
        # detached, so release variable objects before the context closes.
        nc.variables.clear()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Cannot close a netcdf_file opened with mmap=True.*", category=RuntimeWarning)
            nc.close()
    input_hashes = {
        str(trajectory_path.resolve()): sha256_file(trajectory_path),
        str(pdb_path.resolve()): sha256_file(pdb_path),
    }
    if topology_path is not None:
        input_hashes[str(Path(topology_path).resolve())] = sha256_file(topology_path)
    if shared_manifest_path is not None:
        input_hashes[str(Path(shared_manifest_path).resolve())] = str(shared_manifest_sha256)
    gate = {
        "status": "PASS" if shared_manifest is not None else "EXPLORATORY_LOCAL_VOCABULARY",
        "trajectory": str(trajectory_path.resolve()),
        "pdb": str(pdb_path.resolve()),
        "input_sha256": input_hashes,
        "formal_cross_replica_comparison_allowed": shared_manifest is not None,
        "shared_reference": {
            "path": None if shared_manifest_path is None else str(Path(shared_manifest_path).resolve()),
            "sha256": shared_manifest_sha256,
            "validation": shared_validation,
            "mode": reference_mode,
        },
        "feature_schema_sha256": feature_schema_sha256,
        "frames_extracted": int(total),
        "atoms": int(len(atoms)),
        "ligand_heavy_atoms": int(len(ligand_indices)),
        "pocket_residues": [_residue_label(key) for key in pocket_keys],
        "native_contact_residues": [_residue_label(key) for key, flag in zip(pocket_keys, native) if flag],
        "pbc": "orthorhombic minimum image verified from per-frame cell_lengths/cell_angles",
        "feature_scope": "geometric residue contacts; NOT a chemistry-typed PLIF",
        "contact_feature_name": "residue_contact_vector/contact_proxy",
        "bulk_unbound_gate_features": [
            "global__protein_min_heavy_distance_A",
            "global__protein_contact_residue_count",
        ],
        "pose_feature_name": "pocket-Kabsch-aligned ligand pose displacement vector plus invariant RMSD summaries",
        "pose_definition": (
            "For each frame, protein pocket heavy atoms are minimum-image unwrapped and least-squares Kabsch aligned "
            "to the frozen reference pocket. Ligand atoms are PBC-unwrapped around that pocket, transformed by the same rotation, "
            "and compared atom-by-atom (fixed PDB order) with the frozen reference ligand. Emitted pose__ values are RMSDs, "
            "centroid displacement, and per-atom displacement magnitudes; all are invariant to global translation and rotation."
        ),
        "pose_proxy_feature_name": "pose_proxy__ residue minimum-heavy-atom distances retained under an explicit proxy name",
        "pose_reference_frame": 1 if shared_manifest is None else None,
        "pose_reference_mode": reference_mode,
        "pose_alignment_atoms": "frozen pocket standard-amino-acid heavy atoms",
        "pose_atom_correspondence": "fixed sys-protein.pdb / trajectory atom order",
        "pose_global_translation_rotation_invariant": True,
        "contact_cutoff_A": float(contact_cutoff_A),
        "pocket_cutoff_A": float(pocket_cutoff_A),
    }
    write_json(gate_json, gate)
    return gate


def extract_reference_features(*args, **kwargs) -> dict[str, object]:
    """Public wrapper that suppresses scipy's known mmap close warning."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Cannot close a netcdf_file opened with mmap=True.*", category=RuntimeWarning
        )
        return _extract_reference_features_impl(*args, **kwargs)


def fail_closed_gate(gate_json: str | Path, error: Exception, inputs: Iterable[str | Path] = ()) -> dict[str, object]:
    """Record an auditable failure instead of leaving apparently usable output."""

    gate = {
        "status": "FAIL",
        "reason": str(error),
        "inputs": [str(Path(value).resolve()) for value in inputs],
        "output_features_valid": False,
    }
    write_json(gate_json, gate)
    return gate


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line interface for dense trace extraction."""

    parser = argparse.ArgumentParser(
        description=(
            "Measure every saved frame of one Amber NetCDF trajectory against a "
            "formal shared structural reference."
        )
    )
    parser.add_argument("--trajectory", type=Path, required=True, help="Amber NetCDF production trajectory")
    parser.add_argument("--pdb", type=Path, required=True, help="saved-solute PDB in the NetCDF atom order")
    parser.add_argument("--output-csv", type=Path, required=True, help="new dense frame-level CSV")
    parser.add_argument("--gate-json", type=Path, required=True, help="new provenance and pass/fail JSON")
    parser.add_argument("--ligand-resname", required=True, help="ligand residue name; never guessed automatically")
    parser.add_argument("--ligand-resid", help="ligand residue number when the residue name is not unique")
    parser.add_argument("--ligand-chain", help="ligand chain when the residue name is not unique")
    parser.add_argument("--contact-cutoff-A", type=float, default=4.5, help="heavy-atom proximity cutoff in angstrom")
    parser.add_argument("--pocket-cutoff-A", type=float, default=8.0, help="canonical pocket cutoff in angstrom")
    parser.add_argument("--chunk-frames", type=int, default=250, help="number of NetCDF frames held in memory at once")
    parser.add_argument("--max-frames", type=int, help="smoke/exploratory limit; do not use for formal endpoint work")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shared-manifest", type=Path, help="formal outcome-blind shared_reference.json")
    mode.add_argument(
        "--exploratory-replica-local-reference",
        action="store_true",
        help="build a frame-1-local vocabulary; output is explicitly not comparable across replicas",
    )
    parser.add_argument("--complex-id", help="identifier stored in the formal shared reference")
    parser.add_argument("--topology", type=Path, help="topology whose identity is frozen by the formal shared reference")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public CLI and always leave a readable gate JSON on known failures."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.shared_manifest is not None and (not args.complex_id or args.topology is None):
        parser.error("--shared-manifest requires both --complex-id and --topology")
    if args.exploratory_replica_local_reference and (args.complex_id or args.topology is not None):
        parser.error("--complex-id/--topology belong to formal shared-reference extraction, not exploratory mode")
    try:
        gate = extract_reference_features(
            args.trajectory,
            args.pdb,
            args.output_csv,
            args.gate_json,
            ligand_resname=args.ligand_resname,
            ligand_resid=args.ligand_resid,
            ligand_chain=args.ligand_chain,
            contact_cutoff_A=args.contact_cutoff_A,
            pocket_cutoff_A=args.pocket_cutoff_A,
            chunk_frames=args.chunk_frames,
            max_frames=args.max_frames,
            complex_id=args.complex_id,
            topology_path=args.topology,
            shared_manifest_path=args.shared_manifest,
            exploratory_replica_local_reference=args.exploratory_replica_local_reference,
        )
    except (OSError, ValueError, ReferenceFeatureGateError) as exc:
        gate = fail_closed_gate(args.gate_json, exc, inputs=(args.trajectory, args.pdb))
        print(json.dumps(gate, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
