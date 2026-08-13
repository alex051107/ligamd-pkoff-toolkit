from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.koff_ml.prediction_first_coordinate_features import (
    CoordinateFeatureBlock,
    coordinate_state_labels,
)
from scripts.koff_ml.state_labels import operational_state_labels


def test_dense_and_selected_coordinate_state_labels_share_one_precedence_rule() -> None:
    """The same physical channels must produce the same bookkeeping labels."""

    criteria = {
        "bound_max_pocket_com_A": 6.0,
        "bound_min_native_contact_fraction": 0.50,
        "pocket_exit_min_pocket_com_A": 15.0,
        "pocket_exit_max_native_contact_fraction": 0.05,
        "pocket_exit_max_initial_contacts": 0.0,
        "bulk_unbound_max_global_contacts": 0.0,
        "bulk_unbound_min_global_distance_A": 6.0,
    }
    distance = np.asarray([5.0, 16.0, 16.0, 10.0])
    native = np.asarray([0.60, 0.00, 0.00, 0.20])
    global_min = np.asarray([2.0, 5.0, 6.0, 4.0])
    global_count = np.asarray([4.0, 1.0, 0.0, 1.0])
    initial_count = np.asarray([1.0, 0.0, 0.0, 1.0])
    dense = pd.DataFrame(
        {
            "pocket_geometric_com_distance_A": distance,
            "native_contact_fraction": native,
            "global__protein_min_heavy_distance_A": global_min,
            "global__protein_contact_residue_count": global_count,
            "contact__initial_residue": initial_count,
        }
    )
    block = CoordinateFeatureBlock(
        numeric={
            "pocket_geometric_com_distance_A": distance,
            "native_contact_fraction": native,
            "global__protein_min_heavy_distance_A": global_min,
            "global__protein_contact_residue_count": global_count,
        },
        pocket_contacts=initial_count.astype(bool).reshape(-1, 1),
        global_contacts=np.zeros((4, 0), dtype=bool),
        per_ligand_atom_displacement_A=np.zeros((4, 1)),
        pocket_contact_labels=("A:1:RES",),
        global_contact_labels=(),
    )

    expected = np.asarray(["A", "P_ONLY", "B", "I"])
    assert np.array_equal(operational_state_labels(dense, criteria), expected)
    assert np.array_equal(coordinate_state_labels(block, criteria), expected)
