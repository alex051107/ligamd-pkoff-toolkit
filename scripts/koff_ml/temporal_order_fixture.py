"""Data-free order-positive fixture for a future temporal smoke runner.

The fixture deliberately has the same 512-frame row multiset for every
synthetic system and the same first and final rows. Its only target signal is
whether the 510 interior rows occur as state A followed by state B, or in the
opposite order. It is an engineering fixture: it contains no trajectory,
experimental label, fold assignment, or scientific result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.koff_ml.p512_sequence import endpoint_preserving_interior_shuffle_permutation


FIXTURE_ID = "SYNTHETIC_ORDER_POSITIVE_P512_INTERIOR_V1"
FIXTURE_SYSTEMS_PER_CLASS = 16
FIXTURE_REPLICAS = 3
FIXTURE_FRAMES = 512
FIXTURE_CHANNELS = 11
FIXTURE_INTERIOR_HALF = 255
FIXTURE_STATE_SEPARATION = 0.40
FIXTURE_SHUFFLED_ABSOLUTE_SCORE_MAX = 0.20


@dataclass(frozen=True)
class TemporalOrderFixture:
    """One fully deterministic order-positive synthetic tensor collection."""

    system_ids: tuple[str, ...]
    replica_ids: tuple[str, ...]
    targets: np.ndarray
    ordered: np.ndarray


def make_order_positive_fixture() -> TemporalOrderFixture:
    """Construct target +/-1 tensors whose only signal is interior order."""

    state_a = np.linspace(0.05, 0.55, FIXTURE_CHANNELS, dtype=np.float32)
    state_b = np.linspace(0.45, 0.95, FIXTURE_CHANNELS, dtype=np.float32)
    boundary = np.linspace(0.20, 0.80, FIXTURE_CHANNELS, dtype=np.float32)
    targets = np.asarray([-1.0, 1.0] * FIXTURE_SYSTEMS_PER_CLASS, dtype=np.float32)
    sequences: list[np.ndarray] = []
    for target in targets:
        first, second = (state_a, state_b) if target > 0 else (state_b, state_a)
        route = np.vstack(
            (
                boundary,
                np.repeat(first[None, :], FIXTURE_INTERIOR_HALF, axis=0),
                np.repeat(second[None, :], FIXTURE_INTERIOR_HALF, axis=0),
                boundary,
            )
        ).astype(np.float32, copy=False)
        sequences.append(np.repeat(route[None, :, :], FIXTURE_REPLICAS, axis=0))
    ordered = np.stack(sequences).astype(np.float32, copy=False)
    return TemporalOrderFixture(
        system_ids=tuple(f"SYNTHETIC_ORDER_{index:02d}" for index in range(len(targets))),
        replica_ids=("replica_1", "replica_2", "replica_3"),
        targets=targets,
        ordered=ordered,
    )


def endpoint_preserving_shuffled_fixture(
    fixture: TemporalOrderFixture,
    *,
    contract_id: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the v2 matched null independently to each fixture route."""

    ordered = fixture.ordered
    shuffled = np.empty_like(ordered)
    permutations = np.empty(
        (len(fixture.system_ids), FIXTURE_REPLICAS, FIXTURE_FRAMES),
        dtype=np.int64,
    )
    for system_index, system_id in enumerate(fixture.system_ids):
        for replica_index, replica_id in enumerate(fixture.replica_ids):
            permutation = endpoint_preserving_interior_shuffle_permutation(
                contract_id=contract_id,
                seed=seed,
                system_id=system_id,
                replica_id=replica_id,
                length=FIXTURE_FRAMES,
            )
            permutations[system_index, replica_index] = permutation
            shuffled[system_index, replica_index] = ordered[system_index, replica_index, permutation]
    return shuffled, permutations


def interior_transition_score(tensors: np.ndarray) -> np.ndarray:
    """Return a deterministic order oracle used only to validate the fixture."""

    if tensors.ndim != 4 or tensors.shape[1:] != (FIXTURE_REPLICAS, FIXTURE_FRAMES, FIXTURE_CHANNELS):
        raise ValueError(f"fixture tensor shape must be (n, 3, 512, 11), got {tensors.shape}")
    first_half = tensors[:, :, 1 : 1 + FIXTURE_INTERIOR_HALF, 0].mean(axis=(1, 2))
    second_half = tensors[:, :, 1 + FIXTURE_INTERIOR_HALF : -1, 0].mean(axis=(1, 2))
    return (second_half - first_half) / FIXTURE_STATE_SEPARATION


def validate_order_positive_fixture(
    fixture: TemporalOrderFixture,
    *,
    contract_id: str,
    seed: int,
) -> dict[str, object]:
    """Fail closed if the fixture no longer isolates interior order."""

    ordered = fixture.ordered
    expected_shape = (
        FIXTURE_SYSTEMS_PER_CLASS * 2,
        FIXTURE_REPLICAS,
        FIXTURE_FRAMES,
        FIXTURE_CHANNELS,
    )
    if ordered.shape != expected_shape or ordered.dtype != np.float32 or not np.isfinite(ordered).all():
        raise ValueError("order-positive fixture tensor is not finite float32 [32, 3, 512, 11]")
    if not np.array_equal(ordered[:, :, 0], ordered[:, :, -1]):
        raise ValueError("order-positive fixture first and final boundary rows differ")
    if not np.allclose(interior_transition_score(ordered), fixture.targets, atol=1e-6, rtol=0.0):
        raise ValueError("fixture targets are not encoded only by ordered interior direction")

    shuffled, permutations = endpoint_preserving_shuffled_fixture(
        fixture,
        contract_id=contract_id,
        seed=seed,
    )
    for system_index in range(len(fixture.system_ids)):
        for replica_index in range(FIXTURE_REPLICAS):
            permutation = permutations[system_index, replica_index]
            if permutation[0] != 0 or permutation[-1] != FIXTURE_FRAMES - 1:
                raise ValueError("fixture shuffle did not preserve boundary ranks")
            if not np.array_equal(np.sort(permutation[1:-1]), np.arange(1, FIXTURE_FRAMES - 1)):
                raise ValueError("fixture shuffle is not an interior-row bijection")
            if not np.array_equal(
                shuffled[system_index, replica_index],
                ordered[system_index, replica_index, permutation],
            ):
                raise ValueError("fixture shuffle did not move all channels as one row")
    shuffled_scores = interior_transition_score(shuffled)
    if float(np.max(np.abs(shuffled_scores))) > FIXTURE_SHUFFLED_ABSOLUTE_SCORE_MAX:
        raise ValueError("fixture shuffled null retains too much deterministic transition direction")
    return {
        "fixture_id": FIXTURE_ID,
        "system_count": len(fixture.system_ids),
        "shape": list(ordered.shape[1:]),
        "targets_only_interior_order": True,
        "first_and_endpoint_rows_fixed": True,
        "max_abs_shuffled_transition_score": float(np.max(np.abs(shuffled_scores))),
    }
