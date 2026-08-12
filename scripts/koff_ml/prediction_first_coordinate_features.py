#!/usr/bin/env python3
"""Build a true coordinate-derived geometric baseline from frozen U/E shards.

The input NPZ files contain real saved-solute coordinates selected from the
original Amber NetCDF trajectories.  This module reuses the geometry/PBC/pose
implementation in :mod:`reference_features` and the frozen, outcome-blind
``shared_reference`` for each complex.  Every selected-frame value is compared
with the matching row of the full-frame reference table before publication.

The output is deliberately a low-capacity geometric baseline.  It is not a
chemistry-typed PLIF, a GNN representation, a selected pKoff model, or an
estimate of physical koff.

本文件位于“选帧”和“trajectory-level aggregation”之间：输入是 sampler
已经选出的真实 NetCDF 坐标，输出是这些帧共享的一套几何通道。不同 sampler
必须调用同一个 :func:`derive_coordinate_features`，否则下游 MAE 的差异无法
归因于取帧方法。

重要边界：

* ``StructuralContext`` 只由 PDB + shared reference 决定，不读取 pKoff；
* ``CoordinateFeatureBlock`` 的第一维始终对应真实 selected frames；
* 状态 A/I/P_ONLY/B 是 operational bins，不是经过 TICA/MSM 验证的
  metastable states；
* 本文件中的 ``_linear_reconstruction`` 服务于历史 33D baseline。当前
  Core41/G10-v2 正式路径使用 ``normalized_progress_grid_v1`` 的
  previous-observation/left-hold，不应把两种重建语义混称为同一算法。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .io import sha256_file, write_json
from .reference_features import (
    STANDARD_AA,
    PDBAtom,
    _aligned_ligand_pose_features,
    _geometric_com_distance,
    _residue_label,
    _residue_min_distance_matrix,
    _validate_box,
    parse_pdb_atoms,
)
from .shared_reference import load_and_validate_shared_reference
from .state_labels import STATE_ORDER, operational_state_labels
from .trajectory_summary_math import temporal_voronoi_weights


SCHEMA_VERSION = "ligamd_prediction_first_coordinate_geometric_baseline_v1.0"
METHOD_PATTERN = re.compile(r"^[UE](\d+)_")
DEFAULT_ATOL = 5e-5
DEFAULT_RTOL = 5e-6

# The sixth scalar is a residue fraction rather than a raw contact count so
# systems with different protein sizes share the same fixed-dimensional scale.
PATH_SCALARS = (
    "pocket_geometric_com_distance_A",
    "native_contact_fraction",
    "pose__aligned_ligand_rmsd_A",
    "pose__aligned_centroid_displacement_A",
    "global__protein_min_heavy_distance_A",
    "global__protein_contact_residue_fraction",
)

REFERENCE_NUMERIC_FIELDS = (
    "pocket_geometric_com_distance_A",
    "native_contact_fraction",
    "pose__aligned_ligand_rmsd_A",
    "pose__aligned_centroid_displacement_A",
    "pose__ligand_internal_conformation_rmsd_A",
    "pose__pocket_alignment_rmsd_A",
    "global__protein_min_heavy_distance_A",
    "global__protein_contact_residue_count",
)


@dataclass(frozen=True)
class StructuralContext:
    """从 shared reference 冻结的原子集合与几何定义。

    ``protein_residues`` 保存每个标准蛋白 residue 的 heavy-atom indices；
    ``pocket_positions`` 把 frozen pocket residue 映射回 whole-protein
    residue-distance matrix；``native_mask`` 再标出 canonical bound
    reference 中冻结的 native subset。这里的 canonical reference 不等同于
    production NetCDF 的第 0 帧；production frame 0 只是 episode 的第一个
    真实坐标观测。
    """

    ligand_indices: np.ndarray
    protein_residue_keys: tuple[tuple[str, str, str], ...]
    protein_residues: tuple[np.ndarray, ...]
    pocket_keys: tuple[tuple[str, str, str], ...]
    pocket_positions: np.ndarray
    pocket_indices: np.ndarray
    native_mask: np.ndarray
    reference_pocket_centered: np.ndarray
    reference_ligand_centered: np.ndarray
    contact_cutoff_A: float


@dataclass(frozen=True)
class CoordinateFeatureBlock:
    """同一组 selected real frames 上的公共逐帧特征容器。

    ``numeric`` 保存 distance/RMSD/fraction 等一维轨迹；
    ``pocket_contacts`` 和 ``global_contacts`` 是 frame×residue 布尔矩阵；
    ``per_ligand_atom_displacement_A`` 保留 pocket-aligned ligand 位移向量，
    用于与 dense reference table 做逐原子交叉验证。
    """

    numeric: Mapping[str, np.ndarray]
    pocket_contacts: np.ndarray
    global_contacts: np.ndarray
    per_ligand_atom_displacement_A: np.ndarray
    pocket_contact_labels: tuple[str, ...]
    global_contact_labels: tuple[str, ...]


def _require_sha256(path: Path, expected: str, description: str) -> str:
    normalized = str(expected).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"{description} expected SHA-256 is not a lowercase 64-character digest")
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    observed = sha256_file(path)
    if observed != normalized:
        raise ValueError(
            f"{description} SHA-256 mismatch: expected={normalized} observed={observed}"
        )
    return observed


def _hash_bound_to_path(hashes: Mapping[str, object], path: Path) -> str | None:
    """Find a gate hash using exact or resolved path identity."""

    candidates = {str(path), str(path.resolve())}
    for raw_path, digest in hashes.items():
        try:
            resolved = str(Path(raw_path).resolve())
        except (OSError, RuntimeError):
            resolved = str(raw_path)
        if str(raw_path) in candidates or resolved in candidates:
            return str(digest)
    return None


def _method_budget(method: str) -> int:
    match = METHOD_PATTERN.match(str(method))
    if match is None:
        raise ValueError(f"method does not encode a frozen U/E budget: {method!r}")
    return int(match.group(1))


def _build_structural_context(
    atoms: Sequence[PDBAtom], shared_reference: Mapping[str, object]
) -> StructuralContext:
    """把 PDB atom order 与 shared-reference vocabulary 映射成 NumPy indices。

    这里不重新定义 pocket/native contacts。它只验证 frozen residue 是否仍
    存在于当前 saved-solute PDB，并建立
    ``whole-protein residues → pocket subset → native subset``。
    """

    protein_by_residue: dict[tuple[str, str, str], list[int]] = {}
    for atom in atoms:
        if atom.record == "ATOM" and atom.residue in STANDARD_AA and atom.is_heavy:
            protein_by_residue.setdefault(atom.residue_key, []).append(atom.index)
    if not protein_by_residue:
        raise ValueError("saved-solute PDB contains no standard protein heavy-atom residues")

    protein_keys = tuple(sorted(protein_by_residue))
    protein_residues = tuple(
        np.asarray(protein_by_residue[key], dtype=np.int64) for key in protein_keys
    )
    positions = {key: index for index, key in enumerate(protein_keys)}

    pocket_rows = shared_reference["pocket"]["residues"]  # type: ignore[index]
    pocket_keys = tuple(
        (str(row["chain"]), str(row["resid"]), str(row["resname"]))
        for row in pocket_rows
    )
    missing = [key for key in pocket_keys if key not in positions]
    if missing:
        raise ValueError(f"shared-reference pocket residues are absent from PDB: {missing}")
    pocket_positions = np.asarray([positions[key] for key in pocket_keys], dtype=np.int64)
    native_key_set = set(shared_reference["native_contacts"]["residue_keys"])  # type: ignore[index]
    native_mask = np.asarray(
        [f"{key[0]}:{key[1]}:{key[2]}" in native_key_set for key in pocket_keys],
        dtype=bool,
    )
    if not np.any(native_mask):
        raise ValueError("shared reference has no native-contact residues")

    pose_reference = shared_reference["pose_reference"]  # type: ignore[index]
    ligand_indices = np.asarray(
        shared_reference["ligand"]["heavy_atom_indices0"], dtype=np.int64  # type: ignore[index]
    )
    pocket_indices = np.asarray(
        shared_reference["pocket"]["heavy_atom_indices0"], dtype=np.int64  # type: ignore[index]
    )
    definitions = shared_reference["definitions"]  # type: ignore[index]
    return StructuralContext(
        ligand_indices=ligand_indices,
        protein_residue_keys=protein_keys,
        protein_residues=protein_residues,
        pocket_keys=pocket_keys,
        pocket_positions=pocket_positions,
        pocket_indices=pocket_indices,
        native_mask=native_mask,
        reference_pocket_centered=np.asarray(
            pose_reference["pocket_centered_coordinates_A"], dtype=float
        ),
        reference_ligand_centered=np.asarray(
            pose_reference["ligand_pocket_centered_coordinates_A"], dtype=float
        ),
        contact_cutoff_A=float(definitions["contact_cutoff_A"]),
    )


def derive_coordinate_features(
    coordinates_A: np.ndarray,
    cell_lengths_A: np.ndarray,
    cell_angles_degree: np.ndarray,
    *,
    atoms: Sequence[PDBAtom],
    shared_reference: Mapping[str, object],
) -> CoordinateFeatureBlock:
    """对 sampler 选出的真实坐标计算公共逐帧几何特征。

    输入 shape：

    - ``coordinates_A``: ``(n_selected_frames, n_atoms, 3)``；
    - ``cell_lengths_A`` / ``cell_angles_degree``: 每帧周期盒；
    - ``atoms``: 与 NetCDF atom axis 完全同序的 PDB atoms。

    主要公式：

    - residue contact: ``min heavy-atom distance <= contact_cutoff``；
    - native fraction: 当前接触的 frozen-native residues 比例；
    - global contact fraction: 当前接触 residues / 全部标准蛋白 residues；
    - pose/internal RMSD: 见 ``reference_features._aligned_ligand_pose_features``。

    返回值仍是 frame-level block；此处不做 progress 分箱、三 replica 平均或
    任何与 experimental pKoff 有关的特征选择。
    """

    coordinates = np.asarray(coordinates_A, dtype=float)
    boxes = np.asarray(cell_lengths_A, dtype=float)
    angles = np.asarray(cell_angles_degree, dtype=float)
    if coordinates.ndim != 3 or coordinates.shape[2] != 3 or coordinates.shape[1] != len(atoms):
        raise ValueError(
            f"coordinate shape {coordinates.shape} does not match PDB atom count {len(atoms)}"
        )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinate array contains NaN/inf")
    _validate_box(boxes, angles)
    if len(boxes) != len(coordinates) or len(angles) != len(coordinates):
        raise ValueError("coordinate and periodic-box frame counts differ")

    context = _build_structural_context(atoms, shared_reference)
    # 先计算 whole-protein residue distance matrix，再通过 pocket_positions
    # 切出 pocket view。这样 global 与 pocket contacts 共享完全相同的 PBC
    # 和 heavy-atom distance implementation。
    all_residue_distances = _residue_min_distance_matrix(
        coordinates,
        boxes,
        context.ligand_indices,
        list(context.protein_residues),
    )
    pocket_distances = all_residue_distances[:, context.pocket_positions]
    pocket_contacts = pocket_distances <= context.contact_cutoff_A
    global_contacts = all_residue_distances <= context.contact_cutoff_A
    global_contact_count = np.sum(global_contacts, axis=1).astype(float)
    global_contact_fraction = global_contact_count / len(context.protein_residues)
    (
        aligned_pose_rmsd,
        aligned_centroid_displacement,
        internal_conformation_rmsd,
        pocket_alignment_rmsd,
        per_atom_displacement,
    ) = _aligned_ligand_pose_features(
        coordinates,
        boxes,
        context.ligand_indices,
        context.pocket_indices,
        context.reference_pocket_centered,
        context.reference_ligand_centered,
    )
    numeric: dict[str, np.ndarray] = {
        "pocket_geometric_com_distance_A": _geometric_com_distance(
            coordinates,
            boxes,
            context.ligand_indices,
            context.pocket_indices,
        ),
        "native_contact_fraction": np.mean(
            pocket_contacts[:, context.native_mask], axis=1
        ),
        "pose__aligned_ligand_rmsd_A": aligned_pose_rmsd,
        "pose__aligned_centroid_displacement_A": aligned_centroid_displacement,
        "pose__ligand_internal_conformation_rmsd_A": internal_conformation_rmsd,
        "pose__pocket_alignment_rmsd_A": pocket_alignment_rmsd,
        "global__protein_min_heavy_distance_A": np.min(all_residue_distances, axis=1),
        "global__protein_contact_residue_count": global_contact_count,
        "global__protein_contact_residue_fraction": global_contact_fraction,
    }
    if not all(np.all(np.isfinite(values)) for values in numeric.values()):
        raise ValueError("coordinate-derived numeric features contain NaN/inf")
    return CoordinateFeatureBlock(
        numeric=numeric,
        pocket_contacts=pocket_contacts,
        global_contacts=global_contacts,
        per_ligand_atom_displacement_A=per_atom_displacement,
        pocket_contact_labels=tuple(_residue_label(key) for key in context.pocket_keys),
        global_contact_labels=tuple(_residue_label(key) for key in context.protein_residue_keys),
    )


def cross_validate_against_reference(
    block: CoordinateFeatureBlock,
    reference_rows: pd.DataFrame,
    *,
    atoms: Sequence[PDBAtom],
    shared_reference: Mapping[str, object],
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict[str, Any]:
    """将 selected-coordinate 重算值与 dense full-frame table 对行核对。

    Sampler shard 中保存的 ``source_indices0`` 应当指向 dense reference 的
    同一真实帧。本函数检查：

    1. 所有 numeric channels 在 ``atol/rtol`` 内一致；
    2. 每个 frozen-pocket residue contact bit 完全一致；
    3. 每个 ligand heavy atom 的 aligned displacement 一致。

    任一项不一致通常意味着 atom mapping、PBC、frame identity 或 reference
    漂移，必须停止，而不能用近似值继续训练。
    """

    if len(reference_rows) != len(block.pocket_contacts):
        raise ValueError("coordinate/reference selected-frame row counts differ")
    errors: dict[str, float] = {}
    for name in REFERENCE_NUMERIC_FIELDS:
        if name not in reference_rows:
            raise ValueError(f"reference table lacks cross-validation field {name}")
        observed = np.asarray(block.numeric[name], dtype=float)
        expected = reference_rows[name].to_numpy(float)
        if not np.all(np.isfinite(expected)):
            raise ValueError(f"reference field {name} contains NaN/inf")
        errors[name] = float(np.max(np.abs(observed - expected), initial=0.0))
        if not np.allclose(observed, expected, atol=atol, rtol=rtol):
            index = int(np.argmax(np.abs(observed - expected)))
            raise ValueError(
                f"coordinate/reference numeric mismatch for {name} at selected row {index}: "
                f"coordinate={observed[index]:.10g} reference={expected[index]:.10g} "
                f"abs_error={abs(observed[index]-expected[index]):.6g}"
            )

    expected_contact_columns = [f"contact__{label}" for label in block.pocket_contact_labels]
    missing = sorted(set(expected_contact_columns) - set(reference_rows.columns))
    if missing:
        raise ValueError(f"reference table lacks shared pocket contact columns: {missing}")
    reference_contacts = reference_rows[expected_contact_columns].to_numpy(np.int8)
    contact_mismatches = int(
        np.count_nonzero(reference_contacts != block.pocket_contacts.astype(np.int8))
    )
    if contact_mismatches:
        raise ValueError(
            f"coordinate/reference pocket contact vector mismatch: {contact_mismatches} bits"
        )

    ligand_indices = np.asarray(
        shared_reference["ligand"]["heavy_atom_indices0"], dtype=np.int64  # type: ignore[index]
    )
    ligand_columns: list[str] = []
    for ordinal, atom_index in enumerate(ligand_indices):
        atom_name = "".join(
            character if character.isalnum() else "_" for character in atoms[int(atom_index)].name
        )
        ligand_columns.append(
            f"pose__ligand_atom_{ordinal:03d}_{atom_name or 'UNK'}__aligned_displacement_A"
        )
    missing_ligand = sorted(set(ligand_columns) - set(reference_rows.columns))
    if missing_ligand:
        raise ValueError(f"reference table lacks ligand-pose vector columns: {missing_ligand}")
    reference_pose = reference_rows[ligand_columns].to_numpy(float)
    pose_error = float(
        np.max(
            np.abs(reference_pose - block.per_ligand_atom_displacement_A),
            initial=0.0,
        )
    )
    if not np.allclose(
        reference_pose,
        block.per_ligand_atom_displacement_A,
        atol=atol,
        rtol=rtol,
    ):
        raise ValueError(
            f"coordinate/reference per-ligand-atom pose vector mismatch; max_abs_error={pose_error:.6g}"
        )
    return {
        "status": "PASS",
        "numeric_max_abs_error": errors,
        "pocket_contact_bit_mismatches": contact_mismatches,
        "ligand_pose_vector_max_abs_error_A": pose_error,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def coordinate_state_labels(
    block: CoordinateFeatureBlock, criteria: Mapping[str, float]
) -> np.ndarray:
    """按冻结阈值把每个 selected frame 标为 A/I/P_ONLY/B。

    定义（当前 contract 数值由 ``g0_trace_summary_v2`` 冻结）：

    - A: ``d_pocket<=6 Å`` 且 ``Q_native>=0.50``；
    - PocketExit: ``d_pocket>=15 Å``、``Q_native<=0.05`` 且 pocket contacts=0；
    - B: PocketExit，再加 whole-protein contacts=0 和 global min distance>=6 Å；
    - P_ONLY: PocketExit 但尚未满足 B；
    - I: 其余帧。

    赋值顺序先 I，再 P_ONLY，再 B，最后 A，因此最终 precedence 是
    ``A > B > P_ONLY > I``。这些是工程 operational states，不应解释成
    平衡态或 Markov states。
    """

    distance = np.asarray(block.numeric["pocket_geometric_com_distance_A"])
    native = np.asarray(block.numeric["native_contact_fraction"])
    global_min = np.asarray(block.numeric["global__protein_min_heavy_distance_A"])
    global_count = np.asarray(block.numeric["global__protein_contact_residue_count"])
    pocket_count = np.sum(block.pocket_contacts, axis=1)
    bound = (
        (distance <= float(criteria["bound_max_pocket_com_A"]))
        & (native >= float(criteria["bound_min_native_contact_fraction"]))
    )
    pocket_exit = (
        (distance >= float(criteria["pocket_exit_min_pocket_com_A"]))
        & (native <= float(criteria["pocket_exit_max_native_contact_fraction"]))
        & (pocket_count <= float(criteria["pocket_exit_max_initial_contacts"]))
    )
    bulk = (
        pocket_exit
        & (global_count <= float(criteria["bulk_unbound_max_global_contacts"]))
        & (global_min >= float(criteria["bulk_unbound_min_global_distance_A"]))
    )
    labels = np.full(len(distance), "I", dtype="<U6")
    labels[pocket_exit] = "P_ONLY"
    labels[bulk] = "B"
    labels[bound] = "A"
    return labels


def _linear_reconstruction(values: np.ndarray, indices: np.ndarray, n_frames: int) -> np.ndarray:
    """历史 33D baseline 的线性插值重建；不是 Core41/G10-v2 正式语义。

    它在整数 source-frame grid 上用 ``np.interp`` 填充 selected observations
    之间的值。当前 G10-v2 为避免制造不存在的坐标状态，改用
    ``normalized_progress_grid_v1.left_hold_on_normalized_grid``。
    """

    matrix = np.asarray(values, dtype=float)
    selected = np.asarray(indices, dtype=np.int64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if len(matrix) != len(selected) or len(selected) < 2:
        raise ValueError("reconstruction requires at least two aligned selected frames")
    if selected[0] != 0 or selected[-1] != n_frames - 1 or np.any(np.diff(selected) <= 0):
        raise ValueError("selected indices must be increasing and span both episode endpoints")
    query = np.arange(n_frames, dtype=float)
    return np.column_stack(
        [np.interp(query, selected.astype(float), matrix[:, column]) for column in range(matrix.shape[1])]
    )


def summarize_coordinate_replica(
    block: CoordinateFeatureBlock,
    *,
    selected_rows0: np.ndarray,
    source_indices0: np.ndarray,
    episode_frames: int,
    criteria: Mapping[str, float],
) -> dict[str, float]:
    """把 selected-frame block 汇总成历史 frozen 33D replica baseline。

    保留该函数是为了重放早期实验和做回归测试。新 Current30/G10-v2 的
    authoritative aggregation 位于
    :func:`prediction_first_geometric_contact_v2.summarize_geometric_contact_replica_v2`。
    """

    rows = np.asarray(selected_rows0, dtype=np.int64)
    indices = np.asarray(source_indices0, dtype=np.int64)
    if len(rows) != len(indices) or rows.min(initial=0) < 0 or rows.max(initial=-1) >= len(
        block.pocket_contacts
    ):
        raise ValueError("selected coordinate-union row identities are invalid")
    scalar = np.column_stack([np.asarray(block.numeric[name])[rows] for name in PATH_SCALARS])
    reconstruction = _linear_reconstruction(scalar, indices, int(episode_frames))
    normalized_time = np.arange(episode_frames, dtype=float) / max(1, episode_frames - 1)
    output: dict[str, float] = {}
    for bin_index, (left, right) in enumerate(
        ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0)), start=1
    ):
        mask = (normalized_time >= left) & (
            (normalized_time <= right) if bin_index == 3 else (normalized_time < right)
        )
        means = np.mean(reconstruction[mask], axis=0)
        for name, value in zip(PATH_SCALARS, means, strict=True):
            output[f"timebin{bin_index}__{name}__mean"] = float(value)
    for name, value in zip(PATH_SCALARS, np.std(reconstruction, axis=0, ddof=0), strict=True):
        output[f"global__{name}__sd"] = float(value)

    weights = temporal_voronoi_weights(indices, int(episode_frames))
    labels = coordinate_state_labels(block, criteria)[rows]
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        raise ValueError("temporal support weights are empty")
    for state in STATE_ORDER:
        output[f"state_occupancy__{state}"] = float(
            np.sum(weights[labels == state]) / total_weight
        )

    residue_occupancy = np.average(
        block.global_contacts[rows].astype(float), axis=0, weights=weights
    )
    output["global_residue_contact_occupancy__mean"] = float(np.mean(residue_occupancy))
    output["global_residue_contact_occupancy__sd"] = float(
        np.std(residue_occupancy, ddof=0)
    )
    output["global_residue_contact_occupancy__max"] = float(np.max(residue_occupancy))

    internal = np.asarray(
        block.numeric["pose__ligand_internal_conformation_rmsd_A"]
    )[rows]
    internal_mean = float(np.average(internal, weights=weights))
    internal_sd = float(np.sqrt(np.average((internal - internal_mean) ** 2, weights=weights)))
    output["ligand_internal_rmsd_A__weighted_mean"] = internal_mean
    output["ligand_internal_rmsd_A__weighted_sd"] = internal_sd
    if len(output) != 33 or not np.all(np.isfinite(list(output.values()))):
        raise RuntimeError(
            f"coordinate-derived replica baseline must contain 33 finite features; observed={len(output)}"
        )
    return output


def _load_coordinate_run(
    coordinate_root: Path,
    expected_run_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    manifest_path = coordinate_root / "run_manifest.json"
    _require_sha256(
        manifest_path,
        expected_run_manifest_sha256,
        "frozen coordinate materialization run manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_COORDINATES_MATERIALIZED_ENGINEERING_ONLY":
        raise ValueError("coordinate materialization run is not a formal engineering PASS")
    bridge = coordinate_root / "selected_frame_bridge.tsv"
    if sha256_file(bridge) != manifest.get("selected_frame_bridge_sha256"):
        raise ValueError("selected-frame bridge hash differs from coordinate run manifest")
    receipts: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in manifest.get("replicas", []):
        receipt = dict(raw)
        key = (str(receipt["system_id"]), str(receipt["replica_id"]))
        if key in receipts:
            raise ValueError(f"duplicate coordinate receipt for {key}")
        receipts[key] = receipt
    if not receipts:
        raise ValueError("coordinate run manifest contains no replica receipts")
    return manifest, receipts


def _write_tsv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, sep="\t", index=False, lineterminator="\n")


def _bits(row: np.ndarray) -> str:
    return "".join("1" if value else "0" for value in np.asarray(row, dtype=bool))


def _integer_array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<i8").tobytes(order="C")
    ).hexdigest()


def run(
    *,
    coordinate_root: Path,
    input_manifest: Path,
    sampler_protocol: Path,
    output_dir: Path,
    expected_coordinate_run_manifest_sha256: str,
    expected_input_manifest_sha256: str,
    expected_sampler_protocol_sha256: str,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict[str, Any]:
    """Retired private-campaign wrapper retained only for import compatibility.

    Public users should call :func:`derive_coordinate_features` through
    ``ligamd-pkoff featurize``.  The old wrapper expected campaign manifests
    and provenance records that deliberately are not distributed with this
    toolkit, so it fails before reading a trajectory instead of leaving a
    confusing ``NameError`` deep in the historical code path.
    """

    raise RuntimeError(
        "the historical coordinate-campaign runner is not distributed; "
        "use 'ligamd-pkoff featurize' for the public workflow"
    )

    if not (np.isfinite(atol) and np.isfinite(rtol) and atol >= 0 and rtol >= 0):
        raise ValueError("cross-validation tolerances must be finite and non-negative")
    input_sha = _require_sha256(
        input_manifest, expected_input_manifest_sha256, "frozen successful-replica manifest"
    )
    protocol_sha = _require_sha256(
        sampler_protocol, expected_sampler_protocol_sha256, "frozen sampler protocol"
    )
    coordinate_run, coordinate_receipts = _load_coordinate_run(
        coordinate_root, expected_coordinate_run_manifest_sha256
    )
    protocol = json.loads(sampler_protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_RESULTS":
        raise ValueError("sampler protocol must be FROZEN_BEFORE_RESULTS")
    criteria = protocol.get("kinetics_criteria")
    kinetics_protocol_sha = protocol.get("kinetics_protocol_sha256")
    if not isinstance(criteria, dict) or not isinstance(kinetics_protocol_sha, str):
        raise ValueError("sampler protocol lacks frozen kinetics criteria/protocol hash")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    inputs = _load_inputs(input_manifest)
    input_keys = {(record.system_id, record.replica_id) for record in inputs}
    if input_keys != set(coordinate_receipts):
        raise ValueError(
            f"coordinate/input replica mismatch: coordinate_only={sorted(set(coordinate_receipts)-input_keys)} "
            f"input_only={sorted(input_keys-set(coordinate_receipts))}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent))
    try:
        selected_rows: list[dict[str, Any]] = []
        replica_rows: list[dict[str, Any]] = []
        validation_rows: list[dict[str, Any]] = []
        vocabulary_receipts: list[dict[str, Any]] = []
        feature_order: list[str] | None = None

        for record in inputs:
            reference, _, boundary, gate, validation = _validate_and_load_replica(
                record,
                expected_kinetics_protocol_sha256=kinetics_protocol_sha,
            )
            key = (record.system_id, record.replica_id)
            receipt = coordinate_receipts[key]
            expected_receipt_values = {
                "trajectory_path": str(record.trajectory),
                "forcefield_topology_parm7_path": str(record.topology),
                "expected_frames": int(record.expected_frames),
            }
            for field, expected in expected_receipt_values.items():
                if receipt.get(field) != expected:
                    raise ValueError(
                        f"coordinate/input {field} mismatch for {record.system_id}/{record.replica_id}: "
                        f"coordinate={receipt.get(field)!r} input={expected!r}"
                    )
            gate_hashes = gate.get("input_sha256")
            if not isinstance(gate_hashes, dict):
                raise ValueError("reference gate lacks input_sha256 provenance")
            for path, digest, label in (
                (record.trajectory, receipt["trajectory_sha256"], "trajectory"),
                (record.topology, receipt["forcefield_topology_parm7_sha256"], "parm7"),
                (
                    Path(receipt["coordinate_topology_pdb_path"]),
                    receipt["coordinate_topology_pdb_sha256"],
                    "saved-solute PDB",
                ),
            ):
                if _hash_bound_to_path(gate_hashes, path) != digest:
                    raise ValueError(
                        f"coordinate/reference {label} provenance mismatch for "
                        f"{record.system_id}/{record.replica_id}: {path}"
                    )

            shared_info = gate.get("shared_reference")
            if not isinstance(shared_info, dict) or shared_info.get("mode") != (
                "shared_outcome_blind_canonical_structure"
            ):
                raise ValueError("reference gate is not bound to a shared outcome-blind reference")
            shared_path = Path(str(shared_info.get("path", "")))
            shared_sha = str(shared_info.get("sha256", ""))
            _require_sha256(shared_path, shared_sha, "shared structural reference")
            if _hash_bound_to_path(gate_hashes, shared_path) != shared_sha:
                raise ValueError("reference gate does not hash-bind the shared structural reference")
            ledger = validation["ledger"]
            if ledger.get("input_hashes", {}).get("shared_reference_sha256") != shared_sha:
                raise ValueError("kinetics ledger shared-reference hash differs from feature gate")

            pdb_path = Path(receipt["coordinate_topology_pdb_path"])
            _require_sha256(
                pdb_path,
                receipt["coordinate_topology_pdb_sha256"],
                "saved-solute coordinate PDB",
            )
            shared_reference, shared_validation = load_and_validate_shared_reference(
                shared_path,
                complex_id=record.system_id,
                pdb_path=pdb_path,
                topology_path=record.topology,
            )
            atoms = parse_pdb_atoms(pdb_path)

            replica_dir = coordinate_root / "coordinates" / record.system_id / record.replica_id
            npz_path = replica_dir / "coordinate_union.npz"
            map_path = replica_dir / "row_to_coordinate_union.csv"
            shard_manifest_path = replica_dir / "coordinate_union_manifest.json"
            for path, field, label in (
                (npz_path, "coordinate_union_sha256", "coordinate union"),
                (map_path, "coordinate_map_sha256", "coordinate map"),
                (shard_manifest_path, "coordinate_manifest_sha256", "coordinate manifest"),
            ):
                _require_sha256(path, str(receipt[field]), label)
            shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
            if shard_manifest.get("status") != "PASS":
                raise ValueError(f"coordinate shard is not PASS for {record.system_id}/{record.replica_id}")
            expected_shard_values = {
                "coordinate_shard_sha256": receipt["coordinate_union_sha256"],
                "coordinate_map_sha256": receipt["coordinate_map_sha256"],
                "trajectory": receipt["trajectory_path"],
                "trajectory_sha256": receipt["trajectory_sha256"],
                "expected_system_id": record.system_id,
                "expected_replica_id": record.replica_id,
                "expected_frames": int(record.expected_frames),
                "expected_atoms": int(receipt["expected_saved_atoms"]),
                "saved_atom_count": int(receipt["expected_saved_atoms"]),
                "selected_frames_sha256": coordinate_run["selected_frame_bridge_sha256"],
            }
            for field, expected in expected_shard_values.items():
                if shard_manifest.get(field) != expected:
                    raise ValueError(
                        f"coordinate shard manifest {field} mismatch for "
                        f"{record.system_id}/{record.replica_id}: "
                        f"observed={shard_manifest.get(field)!r} expected={expected!r}"
                    )
            mapping = pd.read_csv(
                map_path, dtype={"system_id": str, "replica_id": str, "method": str}
            )
            required_map = {
                "system_id",
                "replica_id",
                "method",
                "selection_rank0",
                "source_index0",
                "source_frame",
                "coordinate_union_index0",
                "normalized_progress_to_B_onset",
            }
            if missing := required_map.difference(mapping.columns):
                raise ValueError(f"coordinate map lacks columns: {sorted(missing)}")
            if set(mapping["system_id"]) != {record.system_id} or set(mapping["replica_id"]) != {
                record.replica_id
            }:
                raise ValueError("coordinate map system/replica identity mismatch")

            with np.load(npz_path, allow_pickle=False) as arrays:
                source_index0 = np.asarray(arrays["source_index0"], dtype=np.int64)
                source_frame = np.asarray(arrays["source_frame"], dtype=np.int64)
                atom_index0 = np.asarray(arrays["atom_index0"], dtype=np.int64)
                coordinates = np.asarray(arrays["coordinates_A"], dtype=float)
                boxes = np.asarray(arrays["cell_lengths_A"], dtype=float)
                angles = np.asarray(arrays["cell_angles_degree"], dtype=float)
            if not np.array_equal(atom_index0, np.arange(len(atoms), dtype=np.int64)):
                raise ValueError("coordinate union is not the complete ordered saved-solute atom set")
            if not np.array_equal(source_frame, source_index0 + 1):
                raise ValueError("coordinate union source-frame identity drifted from NetCDF index+1")
            if len(source_index0) != len(np.unique(source_index0)) or np.any(
                np.diff(source_index0) <= 0
            ):
                raise ValueError("coordinate union source indices are not unique/increasing")
            if int(source_index0[-1]) > boundary.onset_index0:
                raise ValueError("coordinate union includes a post-B-onset frame")
            expected_shape = tuple(int(value) for value in shard_manifest["coordinate_array_shape"])
            if coordinates.shape != expected_shape or expected_shape != (
                len(source_index0),
                len(atoms),
                3,
            ):
                raise ValueError("coordinate union array shape differs from its frozen manifest/PDB")
            if _integer_array_sha256(source_index0) != shard_manifest.get(
                "union_source_index_sha256"
            ):
                raise ValueError("coordinate union source-index digest mismatch")
            if _integer_array_sha256(source_frame) != shard_manifest.get(
                "union_source_frame_sha256"
            ):
                raise ValueError("coordinate union source-frame digest mismatch")
            source_pair_sha = hashlib.sha256(
                np.column_stack((source_index0, source_frame))
                .astype("<i8", copy=False)
                .tobytes(order="C")
            ).hexdigest()
            if source_pair_sha != shard_manifest.get("union_source_identity_sha256"):
                raise ValueError("coordinate union paired source-identity digest mismatch")

            block = derive_coordinate_features(
                coordinates,
                boxes,
                angles,
                atoms=atoms,
                shared_reference=shared_reference,
            )
            reference_rows = reference.iloc[source_index0].reset_index(drop=True)
            cross_validation = cross_validate_against_reference(
                block,
                reference_rows,
                atoms=atoms,
                shared_reference=shared_reference,
                atol=atol,
                rtol=rtol,
            )
            coordinate_labels = coordinate_state_labels(block, criteria)
            reference_labels = operational_state_labels(reference_rows, criteria)
            if not np.array_equal(coordinate_labels, reference_labels):
                raise ValueError(
                    f"coordinate/reference operational state mismatch for "
                    f"{record.system_id}/{record.replica_id}"
                )
            validation_rows.append(
                {
                    "system_id": record.system_id,
                    "replica_id": record.replica_id,
                    "union_frame_count": len(source_index0),
                    "shared_reference_sha256": shared_sha,
                    "max_numeric_abs_error": max(
                        cross_validation["numeric_max_abs_error"].values(), default=0.0
                    ),
                    "pocket_contact_bit_mismatches": cross_validation[
                        "pocket_contact_bit_mismatches"
                    ],
                    "ligand_pose_vector_max_abs_error_A": cross_validation[
                        "ligand_pose_vector_max_abs_error_A"
                    ],
                    "operational_state_mismatches": 0,
                    "status": "PASS",
                }
            )
            vocabulary_receipts.append(
                {
                    "system_id": record.system_id,
                    "replica_id": record.replica_id,
                    "protein_residue_count": len(block.global_contact_labels),
                    "pocket_residue_count": len(block.pocket_contact_labels),
                    "global_contact_vector_order": list(block.global_contact_labels),
                    "pocket_contact_vector_order": list(block.pocket_contact_labels),
                    "shared_reference_validation": shared_validation,
                }
            )

            union_lookup = {int(source): row for row, source in enumerate(source_index0)}
            for method, group in mapping.groupby("method", sort=True):
                ordered = group.sort_values("selection_rank0").reset_index(drop=True)
                budget = _method_budget(str(method))
                ranks = ordered["selection_rank0"].to_numpy(np.int64)
                selected_source = ordered["source_index0"].to_numpy(np.int64)
                selected_union = ordered["coordinate_union_index0"].to_numpy(np.int64)
                if (
                    len(ordered) != budget
                    or not np.array_equal(ranks, np.arange(budget))
                    or len(np.unique(selected_source)) != budget
                    or np.any(np.diff(selected_source) <= 0)
                    or selected_source[0] != 0
                    or selected_source[-1] != boundary.onset_index0
                ):
                    raise ValueError(
                        f"{record.system_id}/{record.replica_id}/{method} is not an exact endpoint-bound B={budget} view"
                    )
                expected_union = np.asarray(
                    [union_lookup[int(source)] for source in selected_source], dtype=np.int64
                )
                if not np.array_equal(selected_union, expected_union):
                    raise ValueError("coordinate map union row identity is inconsistent with NPZ")

                summary = summarize_coordinate_replica(
                    block,
                    selected_rows0=selected_union,
                    source_indices0=selected_source,
                    episode_frames=boundary.onset_index0 + 1,
                    criteria=criteria,
                )
                if feature_order is None:
                    feature_order = list(summary)
                elif list(summary) != feature_order:
                    raise ValueError("33D coordinate-derived feature schema drifted between replicas")
                replica_rows.append(
                    {
                        "system_id": record.system_id,
                        "replica_id": record.replica_id,
                        "method": method,
                        "budget": budget,
                        "B_onset_index0": boundary.onset_index0,
                        **summary,
                    }
                )

                labels = coordinate_labels
                for row_position, identity in ordered.iterrows():
                    union_row = int(identity["coordinate_union_index0"])
                    output_row: dict[str, Any] = {
                        "system_id": record.system_id,
                        "replica_id": record.replica_id,
                        "method": method,
                        "budget": budget,
                        "selection_rank0": int(identity["selection_rank0"]),
                        "source_frame_index0": int(identity["source_index0"]),
                        "source_frame_1based": int(identity["source_frame"]),
                        "coordinate_union_index0": union_row,
                        "normalized_progress_to_B_onset": float(
                            identity["normalized_progress_to_B_onset"]
                        ),
                        "coordinate_derived_state": str(labels[union_row]),
                        "protein_residue_count": len(block.global_contact_labels),
                        "pocket_contact_bits": _bits(block.pocket_contacts[union_row]),
                        "global_protein_contact_bits": _bits(block.global_contacts[union_row]),
                    }
                    output_row.update(
                        {name: float(values[union_row]) for name, values in block.numeric.items()}
                    )
                    selected_rows.append(output_row)

        if feature_order is None:
            raise RuntimeError("no coordinate-derived replica baselines were produced")
        replica_table = pd.DataFrame(replica_rows)
        selected_table = pd.DataFrame(selected_rows)
        validation_table = pd.DataFrame(validation_rows)
        system_rows: list[dict[str, Any]] = []
        for (system, method, budget), group in replica_table.groupby(
            ["system_id", "method", "budget"], sort=True
        ):
            row: dict[str, Any] = {
                "system_id": system,
                "method": method,
                "budget": int(budget),
                "replica_count_observed": int(len(group)),
                "three_replica_ready": bool(len(group) == 3),
            }
            for name in feature_order:
                values = group[name].to_numpy(float)
                row[f"replica_mean__{name}"] = float(np.mean(values))
                row[f"replica_sd__{name}"] = float(np.std(values, ddof=0))
            system_rows.append(row)
        system_table = pd.DataFrame(system_rows)
        dynamic_columns = [
            name
            for name in system_table.columns
            if name.startswith(("replica_mean__", "replica_sd__"))
        ]
        if len(feature_order) != 33 or len(dynamic_columns) != 66:
            raise RuntimeError("replica/system geometric baseline dimensions are not 33/66")
        if not np.all(np.isfinite(replica_table[feature_order].to_numpy(float))) or not np.all(
            np.isfinite(system_table[dynamic_columns].to_numpy(float))
        ):
            raise ValueError("coordinate-derived baseline contains NaN/inf")

        selected_path = partial / "selected_frame_coordinate_features.tsv"
        replica_path = partial / "replica_geometric_baseline_33d.tsv"
        system_path = partial / "system_geometric_baseline_66d.tsv"
        validation_path = partial / "coordinate_reference_cross_validation.tsv"
        vocabulary_path = partial / "replica_contact_vocabularies.json"
        _write_tsv(selected_path, selected_table)
        _write_tsv(replica_path, replica_table)
        _write_tsv(system_path, system_table)
        _write_tsv(validation_path, validation_table)
        write_json(vocabulary_path, {"replicas": vocabulary_receipts})

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS_COORDINATE_DERIVED_GEOMETRIC_BASELINE_ENGINEERING",
            "scientific_scope": {
                "representation": "low-capacity geometric baseline from real selected coordinates",
                "chemistry_typed_plif": False,
                "gnn_representation": False,
                "sampler_selected": False,
                "model_selected": False,
                "physical_koff_estimated": False,
                "grouped_pkoff_comparison_run": False,
            },
            "inputs": {
                "coordinate_root": str(coordinate_root.resolve()),
                "coordinate_run_manifest_sha256": expected_coordinate_run_manifest_sha256,
                "coordinate_selected_identities_sha256": coordinate_run.get(
                    "source_selected_identities_sha256"
                ),
                "successful_replica_manifest": str(input_manifest.resolve()),
                "successful_replica_manifest_sha256": input_sha,
                "sampler_protocol": str(sampler_protocol.resolve()),
                "sampler_protocol_sha256": protocol_sha,
                "kinetics_protocol_sha256": kinetics_protocol_sha,
            },
            "coordinate_reference_cross_validation": {
                "status": "PASS_ALL_SELECTED_UNION_FRAMES",
                "absolute_tolerance": float(atol),
                "relative_tolerance": float(rtol),
                "channels": [
                    *REFERENCE_NUMERIC_FIELDS,
                    "shared-pocket residue contact vector",
                    "per-ligand-atom aligned displacement vector",
                    "operational A/I/P_ONLY/B state",
                ],
                "global_contact_vector_note": (
                    "the full reference table stores the exact all-protein contact count, not the "
                    "all-residue bit vector; the coordinate-derived vector is therefore checked by "
                    "the reused canonical geometry implementation and its exact count, while the "
                    "shared-pocket bit vector is compared bit-for-bit"
                ),
            },
            "replica_feature_definition": {
                "dimension": 33,
                "ordered_names": feature_order,
                "blocks": {
                    "three_normalized_time_bin_means": 18,
                    "six_global_path_standard_deviations": 6,
                    "coordinate_derived_state_occupancies": 4,
                    "all_protein_residue_contact_occupancy_distribution": 3,
                    "ligand_internal_rmsd_weighted_mean_sd": 2,
                },
                "global_contact_size_control": (
                    "per-frame contact count divided by total standard-protein residue count; "
                    "contact occupancy summarized across residues by mean/population-SD/max"
                ),
            },
            "system_feature_definition": {
                "dimension": 66,
                "aggregation": "per-system replica population mean and population SD for each 33D feature",
            },
            "row_counts": {
                "selected_frame_rows": int(len(selected_table)),
                "replica_rows": int(len(replica_table)),
                "system_rows": int(len(system_table)),
            },
            "outputs": {},
        }
        for path in (selected_path, replica_path, system_path, validation_path, vocabulary_path):
            manifest["outputs"][path.name] = sha256_file(path)
        write_json(partial / "run_manifest.json", manifest)
        partial.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinate-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--sampler-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-coordinate-run-manifest-sha256", required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--expected-sampler-protocol-sha256", required=True)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run(
            coordinate_root=args.coordinate_root,
            input_manifest=args.input_manifest,
            sampler_protocol=args.sampler_protocol,
            output_dir=args.output_dir,
            expected_coordinate_run_manifest_sha256=args.expected_coordinate_run_manifest_sha256,
            expected_input_manifest_sha256=args.expected_input_manifest_sha256,
            expected_sampler_protocol_sha256=args.expected_sampler_protocol_sha256,
            atol=args.atol,
            rtol=args.rtol,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "FAIL", "reason": str(exc)}))
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
