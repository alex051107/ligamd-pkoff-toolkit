from __future__ import annotations

import numpy as np

from scripts.koff_ml.typed_interactions import (
    _adjacency,
    cell_matrix,
    feature_names,
    reconstruct_whole_group,
    summarize_stage_fractions,
)


def test_p512_typed_core_reconstructs_pbc_and_emits_exactly_8_by_4() -> None:
    box = cell_matrix((10.0, 10.0, 10.0), (90.0, 90.0, 90.0))
    raw = np.asarray([[9.8, 0.0, 0.0], [0.2, 0.0, 0.0]])
    whole = reconstruct_whole_group(
        raw,
        (0, 1),
        _adjacency(2, np.asarray([[0, 1]], dtype=int)),
        box,
    )
    assert np.isclose(np.linalg.norm(whole[1] - whole[0]), 0.4)

    order = (
        "Hydrophobic",
        "HBAcceptor",
        "HBDonor",
        "PiStacking",
        "Anionic",
        "Cationic",
        "CationPi",
        "PiCation",
    )
    sources = np.asarray([0, 25, 50, 75, 100])
    constants = np.linspace(0.1, 0.8, 8)
    features = summarize_stage_fractions(
        source_indices0=sources,
        episode_onset_index0=100,
        frame_fractions=np.tile(constants, (len(sources), 1)),
        interaction_order=order,
    )
    assert tuple(features) == feature_names(order)
    assert len(features) == 32
    for type_index, kind in enumerate(order):
        values = [value for name, value in features.items() if f"__{kind}__" in name]
        assert len(values) == 4
        assert np.allclose(values, constants[type_index])
