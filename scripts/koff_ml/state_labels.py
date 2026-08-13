"""Shared operational state labels for trajectory summaries.

The four labels are bookkeeping categories, not validated equilibrium or
Markov states.  They help a feature summariser distinguish a bound-like frame,
an ambiguous intermediate frame, a frame that left the original pocket while
remaining associated with the protein surface, and a frame that is clear of
the original protein contact universe under a supplied criteria dictionary.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


STATE_ORDER = ("A", "I", "P_ONLY", "B")


def operational_state_labels_from_channels(
    *,
    pocket_distance_A: np.ndarray,
    native_contact_fraction: np.ndarray,
    global_min_distance_A: np.ndarray,
    global_contact_count: np.ndarray,
    initial_contact_count: np.ndarray,
    criteria: Mapping[str, float],
) -> np.ndarray:
    """Apply the frozen A/I/P_ONLY/B precedence to numeric frame channels.

    This is deliberately a small, representation-independent predicate.  The
    dense-reference route and the selected-coordinate route first obtain the
    same five physical quantities in different containers, then call this
    function.  Keeping the threshold logic in one place prevents the two
    bookkeeping implementations from silently drifting apart.
    """

    distance = np.asarray(pocket_distance_A, dtype=float)
    native = np.asarray(native_contact_fraction, dtype=float)
    global_min = np.asarray(global_min_distance_A, dtype=float)
    global_contacts = np.asarray(global_contact_count, dtype=float)
    initial_contacts = np.asarray(initial_contact_count, dtype=float)
    lengths = {
        len(distance),
        len(native),
        len(global_min),
        len(global_contacts),
        len(initial_contacts),
    }
    if len(lengths) != 1:
        raise ValueError("operational-state channels must have one shared frame count")

    bound = (
        (distance <= float(criteria["bound_max_pocket_com_A"]))
        & (native >= float(criteria["bound_min_native_contact_fraction"]))
    )
    pocket_exit = (
        (distance >= float(criteria["pocket_exit_min_pocket_com_A"]))
        & (native <= float(criteria["pocket_exit_max_native_contact_fraction"]))
        & (initial_contacts <= float(criteria["pocket_exit_max_initial_contacts"]))
    )
    bulk = (
        pocket_exit
        & (global_contacts <= float(criteria["bulk_unbound_max_global_contacts"]))
        & (global_min >= float(criteria["bulk_unbound_min_global_distance_A"]))
    )
    labels = np.full(len(distance), "I", dtype="<U6")
    labels[pocket_exit] = "P_ONLY"
    labels[bulk] = "B"
    labels[bound] = "A"
    return labels


def operational_state_labels(
    reference: pd.DataFrame, criteria: Mapping[str, float]
) -> np.ndarray:
    """Label dense reference rows with the documented A/I/P_ONLY/B precedence."""

    distance = reference["pocket_geometric_com_distance_A"].to_numpy(float)
    native = reference["native_contact_fraction"].to_numpy(float)
    global_min = reference["global__protein_min_heavy_distance_A"].to_numpy(float)
    global_contacts = reference["global__protein_contact_residue_count"].to_numpy(float)
    contact_columns = [name for name in reference if name.startswith("contact__")]
    initial_contacts = (
        reference[contact_columns].to_numpy(float).sum(axis=1)
        if contact_columns
        else np.zeros(len(reference), dtype=float)
    )
    return operational_state_labels_from_channels(
        pocket_distance_A=distance,
        native_contact_fraction=native,
        global_min_distance_A=global_min,
        global_contact_count=global_contacts,
        initial_contact_count=initial_contacts,
        criteria=criteria,
    )
