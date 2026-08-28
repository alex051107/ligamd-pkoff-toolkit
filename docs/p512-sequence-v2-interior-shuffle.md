# P512 sequence v2: hold boundary frames fixed

`p512_sequence_v2.json` is the current data-free contract for the developer-only
ordered-versus-shuffled sequence representation. It replaces v1 for new work.
The historical v1 contract remains in the package so a prior payload can be
reproduced exactly, but it is no longer the matched control for a new temporal
smoke.

## The input, calculation, and output

The input is one already-approved P512 route: 512 real selected frames, in
strictly increasing source-index order, with the first P512 frame at rank 0 and
the frozen endpoint onset at rank 511. The serializer reads no label, fold,
prediction, residual, or loss.

The ordered arm `O` retains all 512 rows. The matched shuffled arm `H` uses the
same rows and the same 11 values per row, but it applies one deterministic,
label-free permutation to ranks 1 through 510. It keeps `H[0] = O[0]` and
`H[511] = O[511]`. Every channel in a row moves together.

The output remains one `3 x 512 x 11` tensor per system plus the permutation
and a receipt. A v2 receipt records that boundary rows were preserved. The
serializer still does not scale channels or train a model.

## Why the boundary rule changed

The proposed sequence encoder pools its final hidden state. Under an all-row
shuffle, the ordered arm always ended at the endpoint frame while the shuffled
arm ended at an arbitrary frame. That changes more than interior order: it
lets a final-state encoder use endpoint placement as an arm-specific shortcut.

Holding both boundary rows fixed removes that shortcut. The comparison can then
ask the narrower question: does the order of the same 510 interior P512 frames
help under one frozen sequence protocol?

## Engineering fixture

`scripts/koff_ml/temporal_order_fixture.py` provides a deterministic synthetic
fixture with 32 systems and three replicas per system. Every synthetic system
has the same row multiset and identical first and final rows. Its target is
encoded only by whether the interior state pattern is A-then-B or B-then-A.

The fixture validates the v2 permutation, row integrity, fixed boundaries, and
loss of the simple transition-direction oracle under the shuffled arm. A future
temporal runner must also pass its predeclared fixture loss/update checks before
it reads any N31 label. Passing this fixture verifies software wiring only; it
does not show that trajectory order predicts experimental pKoff.

## Boundaries

This contract does not change endpoint detection, P512 selection, replica
pooling, Static20, Dynamic10, experimental labels, or a bundled model. It does
not make P512 a uniform physical-time grid, estimate physical `koff`, select a
representation, or authorize a GRU run.
