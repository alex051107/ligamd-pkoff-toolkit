"""Materialize deduplicated source coordinates for frozen sampler selections.

The sampler budget is defined by rows in ``selected_frames.csv``.  Different
methods and budgets often select the same source frame, so this module stores
one full-solute coordinate copy per unique source frame and writes an explicit
row-to-shard map.  Deduplication changes storage only; it never changes a
method's declared coordinate budget.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np

from .io import sha256_file, write_csv, write_json


SCHEMA_VERSION = "ligamd_coordinate_union_v1.1"


def _selection_identity(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _read_selection_rows(
    path: str | Path,
    selection_filters: Mapping[str, str] | None = None,
    *,
    expected_system_id: str | None = None,
    expected_replica_id: str | None = None,
    expected_source_frame_offset: int | None = None,
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    selection_path = Path(path)
    delimiter = "\t" if selection_path.suffix.lower() == ".tsv" else ","
    with selection_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        filters = {str(key): str(value) for key, value in (selection_filters or {}).items()}
        missing_filter_columns = sorted(set(filters) - set(fieldnames))
        if missing_filter_columns:
            raise ValueError(
                "selection filters reference missing columns: "
                + ", ".join(missing_filter_columns)
            )
        rows = [
            row
            for row in reader
            if all(str(row.get(key, "")) == value for key, value in filters.items())
        ]
    if not rows:
        raise ValueError("selected-frame table/filter result is empty")
    for column, expected in (
        ("system_id", expected_system_id),
        ("replica_id", expected_replica_id),
    ):
        if column not in rows[0]:
            continue
        if expected is None or not str(expected):
            raise ValueError(
                f"selected-frame table contains {column}; an explicit expected_{column} is required"
            )
        observed = {str(row.get(column, "")) for row in rows}
        if observed != {str(expected)}:
            raise ValueError(
                f"selected-frame rows do not bind exactly one expected {column}: "
                f"expected={expected!r} observed={sorted(observed)!r}"
            )
    if "source_index0" not in rows[0]:
        raise ValueError("selected-frame table must contain source_index0")
    raw_values: list[int] = []
    try:
        for row in rows:
            value = int(row["source_index0"])
            if str(value) != str(row["source_index0"]).strip():
                raise ValueError
            raw_values.append(value)
        raw = np.asarray(raw_values, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("source_index0 must contain canonical exact integers") from exc
    if np.any(raw < 0):
        raise ValueError("source_index0 cannot be negative")
    union_indices = np.unique(raw)
    if "source_frame" in rows[0]:
        source_frame_by_index: dict[int, int] = {}
        for row, index in zip(rows, raw):
            try:
                source_frame = int(row["source_frame"])
            except (TypeError, ValueError) as exc:
                raise ValueError("source_frame must contain exact integers") from exc
            if str(source_frame) != str(row["source_frame"]).strip():
                raise ValueError("source_frame must contain canonical exact integers")
            previous = source_frame_by_index.setdefault(int(index), source_frame)
            if previous != source_frame:
                raise ValueError("one source_index0 maps to multiple source_frame identities")
        union_source_frames = np.asarray(
            [source_frame_by_index[int(index)] for index in union_indices],
            dtype=np.int64,
        )
        if np.any(np.diff(union_source_frames) <= 0):
            raise ValueError("source_frame identities must increase with source_index0")
    else:
        if expected_source_frame_offset is not None:
            raise ValueError(
                "source_frame is required when an expected source-frame offset is frozen"
            )
        union_source_frames = union_indices + 1
    if expected_source_frame_offset is not None:
        expected_source_frames = union_indices + int(expected_source_frame_offset)
        if not np.array_equal(union_source_frames, expected_source_frames):
            raise ValueError(
                "source_frame is not the frozen NetCDF-frame identity: expected "
                f"source_frame=source_index0+{int(expected_source_frame_offset)}"
            )
    return rows, union_indices, union_source_frames


def _contiguous_runs(indices: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return ``(source_start, source_stop_exclusive, out_start, out_stop)``."""

    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    edges = np.r_[0, breaks, len(idx)]
    return [
        (int(idx[left]), int(idx[right - 1] + 1), int(left), int(right))
        for left, right in zip(edges[:-1], edges[1:])
    ]


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def materialize_coordinate_union(
    *,
    trajectory_path: str | Path,
    selected_frames_path: str | Path,
    output_npz: str | Path,
    output_map_csv: str | Path,
    output_manifest_json: str | Path,
    expected_trajectory_sha256: str,
    expected_atoms: int,
    expected_frames: int,
    atom_indices0: Sequence[int] | None = None,
    selection_filters: Mapping[str, str] | None = None,
    expected_system_id: str | None = None,
    expected_replica_id: str | None = None,
    expected_source_frame_offset: int | None = None,
) -> dict[str, object]:
    """Extract all unique selected frames from an Amber NetCDF trajectory."""

    try:
        from scipy.io import netcdf_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required to materialize Amber NetCDF coordinates") from exc

    trajectory = Path(trajectory_path)
    selection = Path(selected_frames_path)
    output_npz = Path(output_npz)
    output_map_csv = Path(output_map_csv)
    output_manifest_json = Path(output_manifest_json)
    observed_trajectory_sha = sha256_file(trajectory)
    if observed_trajectory_sha != expected_trajectory_sha256:
        raise ValueError(
            "trajectory SHA-256 differs from the frozen manifest: "
            f"expected={expected_trajectory_sha256} observed={observed_trajectory_sha}"
        )
    filters = {str(key): str(value) for key, value in (selection_filters or {}).items()}
    rows, union_indices, union_source_frames = _read_selection_rows(
        selection,
        filters,
        expected_system_id=expected_system_id,
        expected_replica_id=expected_replica_id,
        expected_source_frame_offset=expected_source_frame_offset,
    )
    if len(union_indices) == 0:
        raise ValueError("selection union is empty")

    # Use a read-only memory map so a ~4 GB Production NetCDF is not copied in
    # full merely to extract a ~512--2,048-frame coordinate union.  Every
    # scattered slice is explicitly copied before the file is closed, and all
    # external netcdf_variable references are deleted before context exit.
    nc = netcdf_file(str(trajectory), "r", mmap=True)
    coordinates = None
    cell_lengths = None
    cell_angles = None
    block = None
    try:
        required = {"coordinates", "cell_lengths", "cell_angles"}
        missing = sorted(required - set(nc.variables))
        if missing:
            raise ValueError("trajectory lacks required variables: " + ", ".join(missing))
        coordinates = nc.variables["coordinates"]
        cell_lengths = nc.variables["cell_lengths"]
        cell_angles = nc.variables["cell_angles"]
        shape = tuple(map(int, coordinates.shape))
        if shape != (int(expected_frames), int(expected_atoms), 3):
            raise ValueError(
                f"trajectory shape differs from frozen manifest: observed={shape}, "
                f"expected={(expected_frames, expected_atoms, 3)}"
            )
        if int(union_indices[-1]) >= shape[0]:
            raise ValueError("selected source index exceeds trajectory length")
        if atom_indices0 is None:
            atom_indices = np.arange(shape[1], dtype=np.int64)
            atom_scope = "ALL_SAVED_SOLUTE_ATOMS"
        else:
            atom_indices = np.unique(np.asarray(atom_indices0, dtype=np.int64))
            if len(atom_indices) == 0 or atom_indices[0] < 0 or atom_indices[-1] >= shape[1]:
                raise ValueError("atom_indices0 are empty or outside the saved trajectory")
            atom_scope = "EXPLICIT_ATOM_SUBSET"
        coord_out = np.empty((len(union_indices), len(atom_indices), 3), dtype=np.float32)
        lengths_out = np.empty((len(union_indices), 3), dtype=np.float32)
        angles_out = np.empty((len(union_indices), 3), dtype=np.float32)
        for source_start, source_stop, out_start, out_stop in _contiguous_runs(union_indices):
            block = np.array(
                coordinates[source_start:source_stop], dtype=np.float32, copy=True
            )
            coord_out[out_start:out_stop] = block[:, atom_indices, :]
            lengths_out[out_start:out_stop] = np.array(
                cell_lengths[source_start:source_stop], dtype=np.float32, copy=True
            )
            angles_out[out_start:out_stop] = np.array(
                cell_angles[source_start:source_stop], dtype=np.float32, copy=True
            )
        if not (
            np.all(np.isfinite(coord_out))
            and np.all(np.isfinite(lengths_out))
            and np.all(np.isfinite(angles_out))
        ):
            raise ValueError("selected coordinates or periodic-box values contain non-finite data")
    finally:
        # Clear every external netcdf_variable or mmap-backed slice before
        # netcdf_file.close() checks whether the memory map can be released.
        block = None
        coordinates = None
        cell_lengths = None
        cell_angles = None
        nc.close()
    _atomic_npz(
        output_npz,
        {
            "source_index0": union_indices.astype("<i8", copy=False),
            "source_frame": union_source_frames.astype("<i8", copy=False),
            "atom_index0": atom_indices.astype("<i8", copy=False),
            "coordinates_A": coord_out,
            "cell_lengths_A": lengths_out,
            "cell_angles_degree": angles_out,
        },
    )
    union_position = {int(source): int(position) for position, source in enumerate(union_indices)}
    mapped_rows: list[dict[str, object]] = []
    for row in rows:
        source = int(row["source_index0"])
        mapped_rows.append(
            {
                **row,
                "coordinate_union_index0": union_position[source],
                "coordinate_shard": output_npz.name,
            }
        )
    write_csv(output_map_csv, mapped_rows)
    arrays_sha = sha256_file(output_npz)
    map_sha = sha256_file(output_map_csv)
    manifest_parent = output_manifest_json.resolve().parent
    shard_locator = os.path.relpath(output_npz.resolve(), manifest_parent)
    map_locator = os.path.relpath(output_map_csv.resolve(), manifest_parent)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "trajectory": str(trajectory.resolve()),
        "selected_frames": str(selection.resolve()),
        "coordinate_shard": shard_locator,
        "coordinate_map": map_locator,
        "output_locator_semantics": "relative_to_coordinate_manifest_directory",
        "trajectory_sha256": observed_trajectory_sha,
        "selected_frames_sha256": sha256_file(selection),
        "selection_filters": filters,
        "expected_system_id": expected_system_id,
        "expected_replica_id": expected_replica_id,
        "expected_frames": int(expected_frames),
        "expected_atoms": int(expected_atoms),
        "expected_source_frame_offset": expected_source_frame_offset,
        "source_frame_semantics": (
            "copied from the frozen selected-frame table"
            if "source_frame" in rows[0]
            else "legacy source_index0 plus one"
        ),
        "coordinate_shard_sha256": arrays_sha,
        "coordinate_map_sha256": map_sha,
        "union_source_index_sha256": _selection_identity(union_indices),
        "union_source_frame_sha256": _selection_identity(union_source_frames),
        "union_source_identity_sha256": hashlib.sha256(
            np.column_stack((union_indices, union_source_frames))
            .astype("<i8", copy=False)
            .tobytes(order="C")
        ).hexdigest(),
        "selection_row_count": len(rows),
        "unique_coordinate_frame_count": int(len(union_indices)),
        "saved_atom_count": int(len(atom_indices)),
        "coordinate_array_shape": [int(len(union_indices)), int(len(atom_indices)), 3],
        "atom_scope": atom_scope,
        "coordinate_dtype": "float32",
        "raw_coordinate_semantics": (
            "source Amber coordinates and per-frame periodic box; no alignment, unwrapping, "
            "or outcome-dependent transformation"
        ),
        "budget_semantics": (
            "each method/view/budget retains its own exact selected-frame rows; union "
            "deduplication is a storage optimization only"
        ),
    }
    write_json(output_manifest_json, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--selected-frames", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-map", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--expected-trajectory-sha256", required=True)
    parser.add_argument("--expected-atoms", required=True, type=int)
    parser.add_argument("--expected-frames", required=True, type=int)
    parser.add_argument("--expected-system-id")
    parser.add_argument("--expected-replica-id")
    parser.add_argument(
        "--expected-source-frame-offset",
        type=int,
        help=(
            "Require source_frame=source_index0+OFFSET. The frozen two-system "
            "Production panel uses OFFSET=1."
        ),
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="Filter a shared selected-frame TSV/CSV before materializing this trajectory",
    )
    return parser


def _parse_filters(values: Sequence[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"invalid --where filter {item!r}; expected COLUMN=VALUE")
        key, value = item.split("=", 1)
        if not key or key in filters:
            raise ValueError(f"empty or duplicate --where column: {key!r}")
        filters[key] = value
    return filters


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        filters = _parse_filters(args.where)
        manifest = materialize_coordinate_union(
            trajectory_path=args.trajectory,
            selected_frames_path=args.selected_frames,
            output_npz=args.output_npz,
            output_map_csv=args.output_map,
            output_manifest_json=args.output_manifest,
            expected_trajectory_sha256=args.expected_trajectory_sha256,
            expected_atoms=args.expected_atoms,
            expected_frames=args.expected_frames,
            selection_filters=filters,
            expected_system_id=args.expected_system_id,
            expected_replica_id=args.expected_replica_id,
            expected_source_frame_offset=args.expected_source_frame_offset,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        failure = {"schema_version": SCHEMA_VERSION, "status": "FAIL", "reason": str(exc)}
        write_json(args.output_manifest, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
