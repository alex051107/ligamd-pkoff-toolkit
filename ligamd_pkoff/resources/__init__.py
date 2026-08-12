"""Bundled public contracts and frozen experimental baseline artifacts."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def bundled_path(relative_path: str) -> Path:
    """Return one checked-in public resource from a normal wheel installation.

    ``pip`` installs wheels as directories, so the JSON contracts and frozen
    joblib artifacts remain beside the Python package. The helper fails loudly
    when a distribution omits an asset instead of falling back to a source
    checkout or an unrelated working directory.
    """

    candidate = files("ligamd_pkoff").joinpath("resources", relative_path)
    path = Path(candidate)
    if not path.is_file():
        raise FileNotFoundError(f"bundled toolkit resource is missing: {relative_path}")
    return path
