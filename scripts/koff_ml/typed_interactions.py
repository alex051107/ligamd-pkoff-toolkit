"""Optional CORE8 interaction summaries from an existing P512 coordinate union.

This module is deliberately downstream of endpoint detection and P512 frame
selection.  It does not read a trajectory, choose frames, use labels, or add
features to a prediction model.  For one endpoint-PASS replica it calculates
eight frozen ProLIF interaction classes over four normalized episode stages.
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.koff_ml.io import write_json
from scripts.koff_ml.reference_features import STANDARD_AA


SCHEMA_VERSION = "ligamd_typed_interactions_p512_v1.0"
STATUS = "ENGINEERING_CHALLENGER_NOT_SELECTED"
CONTRACT_SCHEMA = "ligamd_typed_interactions_core8_contract_v1.0"
CONTRACT_ID = "CORE8_GLOBAL_TYPED_FRACTION_STAGE_v1"
PROGRESS_BINS = (
    ("progress_000_050", 0.00, 0.50),
    ("progress_050_080", 0.50, 0.80),
    ("progress_080_095", 0.80, 0.95),
    ("progress_095_100", 0.95, 1.00),
)


class TypedInteractionError(RuntimeError):
    """Raised when an opt-in typed interaction vector cannot be trusted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TypedInteractionError(message)


def feature_names(interaction_order: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        f"typed8__global_active_residue_fraction__{kind}__{label}__mean"
        for kind in interaction_order
        for label, _, _ in PROGRESS_BINS
    )


def cell_matrix(
    lengths_A: Sequence[float], angles_degree: Sequence[float]
) -> np.ndarray:
    """Return a row-vector triclinic cell matrix from lengths and angles."""

    lengths = np.asarray(lengths_A, dtype=float)
    angles = np.asarray(angles_degree, dtype=float)
    require(
        lengths.shape == (3,) and angles.shape == (3,),
        "cell requires three lengths and three angles",
    )
    require(
        np.all(np.isfinite(lengths)) and np.all(lengths > 0.0),
        "cell lengths are invalid",
    )
    require(
        np.all(np.isfinite(angles))
        and np.all((angles > 0.0) & (angles < 180.0)),
        "cell angles are invalid",
    )
    alpha, beta, gamma = np.deg2rad(angles)
    sin_gamma = math.sin(float(gamma))
    require(abs(sin_gamma) > 1e-8, "cell gamma is singular")
    a = np.array([lengths[0], 0.0, 0.0], dtype=float)
    b = np.array(
        [lengths[1] * math.cos(float(gamma)), lengths[1] * sin_gamma, 0.0],
        dtype=float,
    )
    cx = lengths[2] * math.cos(float(beta))
    cy = lengths[2] * (
        math.cos(float(alpha))
        - math.cos(float(beta)) * math.cos(float(gamma))
    ) / sin_gamma
    cz2 = lengths[2] ** 2 - cx**2 - cy**2
    require(cz2 > 1e-8, "cell has a non-positive third-vector height")
    box = np.vstack((a, b, np.array([cx, cy, math.sqrt(float(cz2))])))
    require(abs(float(np.linalg.det(box))) > 1e-8, "cell matrix is singular")
    return box


def minimum_image(displacement: np.ndarray, box: np.ndarray) -> np.ndarray:
    values = np.asarray(displacement, dtype=float)
    fractional = values @ np.linalg.inv(np.asarray(box, dtype=float))
    fractional -= np.rint(fractional)
    return fractional @ box


def _adjacency(
    n_atoms: int, bond_indices: np.ndarray
) -> tuple[tuple[int, ...], ...]:
    neighbours: list[list[int]] = [[] for _ in range(n_atoms)]
    for left, right in np.asarray(bond_indices, dtype=int):
        require(
            0 <= left < n_atoms and 0 <= right < n_atoms,
            "bond index lies outside the saved coordinates",
        )
        neighbours[left].append(right)
        neighbours[right].append(left)
    return tuple(tuple(sorted(values)) for values in neighbours)


def reconstruct_whole_group(
    raw_coordinates_A: np.ndarray,
    atom_indices0: Sequence[int],
    adjacency: Sequence[Sequence[int]],
    box: np.ndarray,
) -> np.ndarray:
    """Reconstruct one bonded residue or ligand across periodic boundaries."""

    raw = np.asarray(raw_coordinates_A, dtype=float)
    indices = tuple(map(int, atom_indices0))
    require(
        bool(indices) and len(indices) == len(set(indices)),
        "group indices are empty or duplicated",
    )
    allowed = set(indices)
    local = {atom: position for position, atom in enumerate(indices)}
    whole = np.full((len(indices), 3), np.nan, dtype=float)
    root = indices[0]
    whole[local[root]] = raw[root]
    seen = {root}
    queue: deque[int] = deque([root])
    while queue:
        parent = queue.popleft()
        for child in adjacency[parent]:
            if child not in allowed or child in seen:
                continue
            whole[local[child]] = whole[local[parent]] + minimum_image(
                raw[child] - raw[parent], box
            )
            seen.add(child)
            queue.append(child)
    require(seen == allowed, "residue or ligand bond graph is disconnected")
    require(np.all(np.isfinite(whole)), "group reconstruction produced NaN/inf")
    return whole


def _exact_image_shift(
    moving_heavy_A: np.ndarray,
    reference_heavy_A: np.ndarray,
    box: np.ndarray,
) -> np.ndarray:
    translations = np.asarray(
        [
            i * box[0] + j * box[1] + k * box[2]
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
        ],
        dtype=float,
    )
    distances = []
    for shift in translations:
        delta = (moving_heavy_A + shift)[:, None, :] - reference_heavy_A[None, :, :]
        distances.append(float(np.min(np.linalg.norm(delta, axis=2))))
    return translations[int(np.argmin(distances))].copy()


def image_frame_to_ligand(
    raw_coordinates_A: np.ndarray,
    *,
    box: np.ndarray,
    adjacency: Sequence[Sequence[int]],
    ligand_indices0: Sequence[int],
    ligand_heavy_indices0: Sequence[int],
    protein_residue_indices0: Sequence[Sequence[int]],
    protein_residue_heavy_indices0: Sequence[Sequence[int]],
    intragroup_bonds: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, float]:
    """Make each group whole and image every protein residue to the ligand."""

    raw = np.asarray(raw_coordinates_A, dtype=float)
    imaged = raw.copy()
    ligand_indices = tuple(map(int, ligand_indices0))
    ligand_whole = reconstruct_whole_group(raw, ligand_indices, adjacency, box)
    imaged[np.asarray(ligand_indices, dtype=int)] = ligand_whole
    ligand_heavy_coordinates = imaged[np.asarray(ligand_heavy_indices0, dtype=int)]
    for group, heavy in zip(
        protein_residue_indices0,
        protein_residue_heavy_indices0,
        strict=True,
    ):
        group_indices = tuple(map(int, group))
        whole = reconstruct_whole_group(raw, group_indices, adjacency, box)
        heavy_local = [group_indices.index(int(index)) for index in heavy]
        shift = _exact_image_shift(
            whole[heavy_local], ligand_heavy_coordinates, box
        )
        imaged[np.asarray(group_indices, dtype=int)] = whole + shift
    max_bond = max(
        (
            float(np.linalg.norm(imaged[left] - imaged[right]))
            for left, right in intragroup_bonds
        ),
        default=0.0,
    )
    require(
        max_bond <= 2.5,
        f"PBC reconstruction left an intragroup bond of {max_bond:.6f} A",
    )
    return imaged, max_bond


def voronoi_interval_mean(
    normalized_progress: Sequence[float],
    values: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Average selected-frame values using support on normalized progress."""

    progress = np.asarray(normalized_progress, dtype=float)
    matrix = np.asarray(values, dtype=float)
    require(
        progress.ndim == 1
        and matrix.ndim == 2
        and len(progress) == len(matrix)
        and len(progress) >= 2,
        "Voronoi aggregation requires aligned frames and values",
    )
    require(
        np.all(np.diff(progress) > 0.0),
        "selected normalized progress is not strictly increasing",
    )
    require(
        abs(float(progress[0])) <= 1e-12
        and abs(float(progress[-1]) - 1.0) <= 1e-12,
        "selected frames do not span normalized progress [0, 1]",
    )
    left = np.r_[0.0, (progress[:-1] + progress[1:]) / 2.0]
    right = np.r_[(progress[:-1] + progress[1:]) / 2.0, 1.0]
    overlap = np.maximum(
        0.0, np.minimum(right, upper) - np.maximum(left, lower)
    )
    require(
        abs(float(overlap.sum()) - (upper - lower)) <= 1e-10,
        "Voronoi supports do not cover the requested stage",
    )
    result = overlap @ matrix / (upper - lower)
    require(np.all(np.isfinite(result)), "stage aggregation produced NaN/inf")
    return np.asarray(result, dtype=float)


def summarize_stage_fractions(
    *,
    source_indices0: Sequence[int],
    episode_onset_index0: int,
    frame_fractions: np.ndarray,
    interaction_order: Sequence[str],
) -> dict[str, float]:
    """Return the frozen 8-interaction by 4-stage feature dictionary."""

    sources = np.asarray(source_indices0, dtype=np.int64)
    require(episode_onset_index0 > 0, "episode onset must be positive")
    require(
        frame_fractions.shape == (len(sources), len(interaction_order)),
        "frame fraction matrix shape differs from interaction order",
    )
    progress = sources / float(episode_onset_index0)
    values: dict[str, float] = {}
    for type_index, kind in enumerate(interaction_order):
        for label, lower, upper in PROGRESS_BINS:
            name = (
                "typed8__global_active_residue_fraction__"
                f"{kind}__{label}__mean"
            )
            values[name] = float(
                voronoi_interval_mean(
                    progress,
                    frame_fractions[:, [type_index]],
                    lower,
                    upper,
                )[0]
            )
    require(tuple(values) == feature_names(interaction_order), "typed feature order drifted")
    return values


def _read_contract(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TypedInteractionError(f"cannot read typed interaction contract {path}: {exc}") from exc
    require(
        isinstance(payload, dict)
        and payload.get("schema_version") == CONTRACT_SCHEMA,
        f"typed interaction contract schema must be {CONTRACT_SCHEMA}",
    )
    require(
        payload.get("contract_id") == CONTRACT_ID,
        f"typed interaction contract id must be {CONTRACT_ID}",
    )
    order = payload.get("interaction_order")
    require(
        isinstance(order, list) and len(order) == 8 and len(set(order)) == 8,
        "typed interaction contract requires eight unique interactions",
    )
    return payload


def _optional_dependencies() -> dict[str, Any]:
    try:
        import MDAnalysis as mda
        import prolif as plf
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        from rdkit.Geometry import Point3D
    except ImportError as exc:
        raise TypedInteractionError(
            "typed interactions require the optional 'typed' dependencies: "
            "install ligamd-pkoff-toolkit[typed]"
        ) from exc
    return {
        "mda": mda,
        "plf": plf,
        "Chem": Chem,
        "descriptors": rdMolDescriptors,
        "Point3D": Point3D,
    }


def _molecule_signature(molecule: Any, chem: Any, descriptors: Any) -> dict[str, Any]:
    heavy = chem.RemoveHs(molecule)
    return {
        "heavy_atom_count": int(heavy.GetNumAtoms()),
        "formula": str(descriptors.CalcMolFormula(molecule)),
        "formal_charge": int(
            sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
        ),
        "aromatic_atom_count": int(
            sum(atom.GetIsAromatic() for atom in molecule.GetAtoms())
        ),
        "heavy_bond_count": int(
            sum(
                molecule.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() > 1
                and molecule.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() > 1
                for bond in molecule.GetBonds()
            )
        ),
        "canonical_heavy_smiles": str(
            chem.MolToSmiles(heavy, isomericSmiles=False)
        ),
        "explicit_hydrogen_count": int(
            sum(atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms())
        ),
    }


def _load_template(path: Path, chem: Any) -> Any:
    require(path.suffix.lower() in {".sdf", ".sd"}, "typed ligand template must be SDF")
    molecules = [
        molecule
        for molecule in chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
        if molecule is not None
    ]
    require(len(molecules) == 1, "ligand SDF must contain one sanitized molecule")
    return molecules[0]


def _validate_ligand_template(
    topology_molecule: Any,
    template: Any,
    chem: Any,
    descriptors: Any,
) -> dict[str, Any]:
    observed = _molecule_signature(topology_molecule, chem, descriptors)
    expected = _molecule_signature(template, chem, descriptors)
    fields = (
        "heavy_atom_count",
        "formula",
        "formal_charge",
        "aromatic_atom_count",
        "heavy_bond_count",
        "canonical_heavy_smiles",
    )
    differences = {
        field: {"topology": observed[field], "template": expected[field]}
        for field in fields
        if observed[field] != expected[field]
    }
    require(
        not differences,
        f"topology-derived ligand chemistry differs from SDF: {differences}",
    )
    require(
        observed["explicit_hydrogen_count"] > 0,
        "topology-derived ligand has no explicit hydrogens",
    )
    return {"status": "PASS", "topology": observed, "template": expected}


def _validate_cached_mapping(molecule: Any) -> None:
    parent = [
        int(atom.GetIntProp("_MDAnalysis_index"))
        for atom in molecule.GetAtoms()
    ]
    cached = [
        int(atom.GetIntProp("_MDAnalysis_index"))
        for residue in molecule.residues.values()
        for atom in residue.GetAtoms()
    ]
    expected = list(range(molecule.GetNumAtoms()))
    require(
        sorted(parent) == sorted(cached) == expected,
        "ProLIF parent/cached atom mapping has duplicates or gaps",
    )


def _set_positions(
    molecule: Any,
    local_to_saved: np.ndarray,
    positions_A: np.ndarray,
    point3d: Any,
) -> None:
    def update(container: Any) -> None:
        conformer = container.GetConformer()
        for atom in container.GetAtoms():
            local = int(atom.GetIntProp("_MDAnalysis_index"))
            saved = int(local_to_saved[local])
            x, y, z = map(float, positions_A[saved])
            conformer.SetAtomPosition(atom.GetIdx(), point3d(x, y, z))

    update(molecule)
    for residue in molecule.residues.values():
        update(residue)


def _residue_label(name: Any, number: Any, chain: Any = None) -> str:
    suffix = "" if chain in {None, "", "0", 0} else f":{chain}"
    return f"{str(name).strip()}:{int(number)}{suffix}"


def _ifp_bits(
    ifp: Mapping[Any, np.ndarray], interaction_order: Sequence[str]
) -> dict[tuple[str, str], tuple[int, ...]]:
    result: dict[tuple[str, str], tuple[int, ...]] = {}
    for (ligand, protein), bits in ifp.items():
        key = (
            _residue_label(ligand.name, ligand.number, ligand.chain),
            _residue_label(protein.name, protein.number, protein.chain),
        )
        require(key not in result, f"duplicate ligand/protein residue key: {key}")
        values = tuple(int(value) for value in np.asarray(bits, dtype=bool))
        require(
            len(values) == len(interaction_order),
            "interaction bit order or length drifted",
        )
        result[key] = values
    return result


def extract_p512_typed_interactions(
    *,
    coordinate_union_npz: Path,
    selected_indices0: Sequence[int],
    episode_onset_index0: int,
    topology_path: Path,
    pdb_path: Path,
    ligand_template_sdf: Path,
    ligand_resname: str,
    contract_path: Path,
    system_id: str,
    replica_id: str,
    output_json: Path,
) -> dict[str, Any]:
    """Calculate the label-blind GLOBAL_TYPED_FRACTION_STAGE challenger."""

    require(not output_json.exists(), "refusing to overwrite typed output")
    contract = _read_contract(contract_path)
    dependencies = _optional_dependencies()
    mda = dependencies["mda"]
    plf = dependencies["plf"]
    chem = dependencies["Chem"]
    descriptors = dependencies["descriptors"]
    point3d = dependencies["Point3D"]
    engine = contract.get("engine")
    require(isinstance(engine, Mapping), "typed contract lacks engine settings")
    require(
        str(plf.__version__) == str(engine.get("version")),
        "installed ProLIF version differs from the typed contract",
    )
    interaction_order = tuple(map(str, contract["interaction_order"]))

    with np.load(coordinate_union_npz, allow_pickle=False) as arrays:
        sources = np.asarray(arrays["source_index0"], dtype=np.int64)
        atoms = np.asarray(arrays["atom_index0"], dtype=np.int64)
        coordinates = np.asarray(arrays["coordinates_A"], dtype=float)
        lengths = np.asarray(arrays["cell_lengths_A"], dtype=float)
        angles = np.asarray(arrays["cell_angles_degree"], dtype=float)
    expected_sources = np.asarray(selected_indices0, dtype=np.int64)
    require(
        np.array_equal(sources, expected_sources)
        and len(sources) == 512
        and np.all(np.diff(sources) > 0)
        and int(sources[0]) == 0
        and int(sources[-1]) == episode_onset_index0,
        "coordinate union does not match the accepted P512 episode",
    )
    require(
        coordinates.shape == (512, len(atoms), 3)
        and lengths.shape == angles.shape == (512, 3)
        and np.array_equal(atoms, np.arange(len(atoms)))
        and np.all(np.isfinite(coordinates))
        and np.all(np.isfinite(lengths))
        and np.all(np.isfinite(angles)),
        "coordinate union shapes, atom order, or finite-value gate failed",
    )

    topology = mda.Universe(str(topology_path))
    n_saved = coordinates.shape[1]
    require(len(topology.atoms) >= n_saved, "topology has fewer atoms than P512 union")
    solute = mda.Merge(topology.atoms[:n_saved])
    pdb = mda.Universe(str(pdb_path))
    require(len(pdb.atoms) == n_saved, "PDB atom count differs from P512 union")
    require(
        np.array_equal(solute.atoms.names, pdb.atoms.names)
        and np.array_equal(solute.atoms.resnames, pdb.atoms.resnames),
        "first N topology atom names/resnames differ from the PDB",
    )
    ligand_residues = [
        residue
        for residue in solute.residues
        if str(residue.resname).strip() == ligand_resname
    ]
    require(
        len(ligand_residues) == 1,
        f"expected exactly one ligand residue named {ligand_resname}",
    )
    protein_residues = [
        residue
        for residue in solute.residues
        if str(residue.resname).strip() in STANDARD_AA
    ]
    require(bool(protein_residues), "no standard-amino-acid protein residues found")
    ligand_ag = ligand_residues[0].atoms
    protein_ag = solute.atoms[
        np.concatenate(
            [np.asarray(residue.atoms.indices, dtype=int) for residue in protein_residues]
        )
    ]
    ligand_indices = np.asarray(ligand_ag.indices, dtype=int)
    protein_indices = np.asarray(protein_ag.indices, dtype=int)
    elements = np.asarray(solute.atoms.elements, dtype=object)
    ligand_heavy = ligand_indices[elements[ligand_indices] != "H"]
    residue_groups = [
        np.asarray(residue.atoms.indices, dtype=int) for residue in protein_residues
    ]
    residue_heavy = [group[elements[group] != "H"] for group in residue_groups]
    require(
        len(ligand_heavy) > 0 and all(len(group) > 0 for group in residue_heavy),
        "ligand or protein heavy-atom selection is empty",
    )
    bonds = np.asarray(solute.bonds.indices, dtype=int)
    adjacency = _adjacency(n_saved, bonds)
    membership: dict[int, int] = {}
    for group_index, group in enumerate([ligand_indices, *residue_groups]):
        for atom in group:
            membership[int(atom)] = group_index
    intragroup_bonds = [
        (int(left), int(right))
        for left, right in bonds
        if membership.get(int(left)) == membership.get(int(right))
        and int(left) in membership
    ]

    imaged0, max_bond = image_frame_to_ligand(
        coordinates[0],
        box=cell_matrix(lengths[0], angles[0]),
        adjacency=adjacency,
        ligand_indices0=ligand_indices,
        ligand_heavy_indices0=ligand_heavy,
        protein_residue_indices0=residue_groups,
        protein_residue_heavy_indices0=residue_heavy,
        intragroup_bonds=intragroup_bonds,
    )
    solute.load_new(imaged0[np.newaxis, :, :], order="fac")
    ligand_molecule = plf.Molecule.from_mda(ligand_ag)
    protein_molecule = plf.Molecule.from_mda(protein_ag)
    _validate_cached_mapping(ligand_molecule)
    _validate_cached_mapping(protein_molecule)
    require(
        sum(atom.GetAtomicNum() == 1 for atom in protein_molecule.GetAtoms()) > 0,
        "topology-derived protein has no explicit hydrogens",
    )
    chemistry = _validate_ligand_template(
        ligand_molecule,
        _load_template(ligand_template_sdf, chem),
        chem,
        descriptors,
    )
    fingerprint = plf.Fingerprint(
        interactions=list(interaction_order),
        parameters=contract["parameters"],
        count=bool(engine["count"]),
        vicinity_cutoff=float(engine["vicinity_cutoff_A"]),
        use_segid=bool(engine["use_segid"]),
        implicit_hydrogens=bool(engine["implicit_hydrogens"]),
    )
    require(
        tuple(fingerprint.interactions) == interaction_order,
        "ProLIF interaction order drifted",
    )

    frame_fractions = np.zeros((512, len(interaction_order)), dtype=float)
    for frame_index in range(512):
        imaged, frame_max_bond = image_frame_to_ligand(
            coordinates[frame_index],
            box=cell_matrix(lengths[frame_index], angles[frame_index]),
            adjacency=adjacency,
            ligand_indices0=ligand_indices,
            ligand_heavy_indices0=ligand_heavy,
            protein_residue_indices0=residue_groups,
            protein_residue_heavy_indices0=residue_heavy,
            intragroup_bonds=intragroup_bonds,
        )
        max_bond = max(max_bond, frame_max_bond)
        _set_positions(ligand_molecule, ligand_indices, imaged, point3d)
        _set_positions(protein_molecule, protein_indices, imaged, point3d)
        bits = _ifp_bits(
            fingerprint.generate(
                ligand_molecule,
                protein_molecule,
                residues=None,
                metadata=False,
            ),
            interaction_order,
        )
        if bits:
            counts = np.sum(np.asarray(list(bits.values()), dtype=float), axis=0)
            require(
                np.all(counts <= len(protein_residues)),
                "typed residue count exceeds the standard-protein denominator",
            )
            frame_fractions[frame_index] = counts / float(len(protein_residues))

    features = summarize_stage_fractions(
        source_indices0=sources,
        episode_onset_index0=episode_onset_index0,
        frame_fractions=frame_fractions,
        interaction_order=interaction_order,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "status": STATUS,
        "system_id": system_id,
        "replica_id": replica_id,
        "representation": "GLOBAL_TYPED_FRACTION_STAGE",
        "sampler": "P512_multiblock_path_arclength",
        "sampler_budget": 512,
        "interaction_order": list(interaction_order),
        "standard_protein_residue_count": len(protein_residues),
        "max_intragroup_bond_A": max_bond,
        "chemistry_validation": chemistry,
        "features": features,
        "scientific_boundaries": {
            "experimental_pkoff_read": False,
            "representation_selected": False,
            "model_selected": False,
            "physical_koff_estimated": False,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, result)
    return result
