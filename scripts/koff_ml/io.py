"""Small, dependency-light readers and writers used by the benchmark CLI."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@contextmanager
def atomic_text_writer(path: str | Path, *, newline: str | None = None):
    """Write a same-directory temporary file and replace only on clean exit."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline=newline) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def read_cpptraj_series(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a one-value cpptraj ``*.dat`` file.

    Blank/comment/header lines are ignored.  The first numeric column is the
    frame identifier and the last numeric column is the value.  This handles
    both ``#Frame value`` and whitespace-only cpptraj output.
    """

    frames: list[int] = []
    values: list[float] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "@")):
                continue
            fields = stripped.replace(",", " ").split()
            try:
                nums = [float(x) for x in fields]
            except ValueError:
                continue
            if len(nums) == 1:
                frames.append(len(frames) + 1)
                values.append(nums[0])
            elif len(nums) >= 2:
                frames.append(int(nums[0]))
                values.append(nums[-1])
    if not values:
        raise ValueError(f"No numeric rows found in cpptraj file: {path}")
    return np.asarray(frames, dtype=np.int64), np.asarray(values, dtype=float)


def read_feature_csv(path: str | Path) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    """Read a rectangular feature CSV without requiring pandas.

    Numeric columns become the returned feature matrix.  Columns that contain
    non-numeric values are returned as metadata.  ``frame`` is metadata and is
    not silently included as a predictive feature.
    """

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Feature table is empty: {path}")
    columns = list(rows[0])
    numeric: list[str] = []
    metadata: dict[str, np.ndarray] = {}
    for col in columns:
        if col.lower() in {"frame", "replica", "complex", "complex_id", "outcome", "label"}:
            metadata[col] = np.asarray([r.get(col, "") for r in rows], dtype=object)
            continue
        try:
            values = np.asarray([float(r[col]) for r in rows], dtype=float)
        except (TypeError, ValueError):
            metadata[col] = np.asarray([r.get(col, "") for r in rows], dtype=object)
        else:
            if np.all(np.isfinite(values)):
                numeric.append(col)
            else:
                metadata[col] = np.asarray([r.get(col, "") for r in rows], dtype=object)
    if not numeric:
        raise ValueError(f"No finite numeric feature columns in: {path}")
    matrix = np.asarray([[float(r[c]) for c in numeric] for r in rows], dtype=float)
    return matrix, numeric, metadata


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, object]],
    *,
    fieldnames: Iterable[str] | None = None,
) -> None:
    rows = list(rows)
    target = Path(path)
    fields: list[str] = [] if fieldnames is None else list(fieldnames)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        with atomic_text_writer(target):
            pass
        return
    with atomic_text_writer(target, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    with atomic_text_writer(target) as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()
