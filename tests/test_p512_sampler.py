"""Focused checks for the public, label-free P512 frame selector."""

from __future__ import annotations

import json

import numpy as np

from ligamd_pkoff.resources import bundled_path
from scripts.koff_ml.p512_sampler import P512Trace, p512_path_arclength_indices


def _trace(n_frames: int = 620) -> P512Trace:
    progress = np.linspace(0.0, 1.0, n_frames)
    numeric = {
        "pocket_geometric_com_distance_A": 3.0 + 19.0 * progress,
        "native_contact_fraction": 1.0 - progress,
        "contact_formations": np.zeros(n_frames),
        "contact_breaks": 5.0 * progress,
        "pose__aligned_ligand_rmsd_A": 2.0 * progress**2,
        "pose__aligned_centroid_displacement_A": 18.0 * progress,
        "pose__ligand_internal_conformation_rmsd_A": 0.4 + progress,
        "pose__pocket_alignment_rmsd_A": 0.05 + 0.2 * progress,
        "global__protein_min_heavy_distance_A": 3.0 + 10.0 * progress,
        "global__protein_contact_residue_count": 15.0 * (1.0 - progress),
    }
    contacts = np.column_stack((progress < 0.4, progress < 0.7)).astype(np.float32)
    return P512Trace(
        frames1=np.arange(1, n_frames + 1, dtype=np.int64),
        numeric=numeric,
        contacts=contacts,
        contact_columns=("contact__A:1:ALA", "contact__A:2:GLY"),
    )


def test_p512_returns_exact_real_episode_frames_deterministically() -> None:
    contract = json.loads(bundled_path("contracts/p512_sampler_v1.json").read_text())
    trace = _trace()
    first, first_roles, first_arc = p512_path_arclength_indices(trace, 619, contract)
    second, second_roles, second_arc = p512_path_arclength_indices(trace, 619, contract)
    assert len(first) == 512
    assert first[0] == 0 and first[-1] == 619
    assert np.all(np.diff(first) > 0)
    assert len(np.unique(first)) == 512
    assert first_roles[0] == first_roles[619] == "forced_endpoint"
    assert np.array_equal(first, second)
    assert first_roles == second_roles
    assert np.array_equal(first_arc, second_arc)
