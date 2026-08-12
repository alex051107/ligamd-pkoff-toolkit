# Model-development roadmap

The four bundled models are a controlled starting point. They do not justify a
large search across model families. The next scientific question is narrower:
does trajectory information improve experimental-pKoff prediction beyond the
ligand and bound-structure controls when the comparison is made on new ligand
chemistries?

For the meanings of pKoff, exact-ligand groups, grouped MAE, and the grouped
Dummy baseline, read the [model card](../MODEL_CARD.md). P512 is the
path-arclength sampler that selects 512 real saved frames from each replica;
its definition is in the [episode and sampler explanation](endpoint-and-sampler.md).
For literature context, read the [method evidence ledger](method-evidence.md).

## What the frozen N31 result can support

`combined30_p512_ridge` has the lowest point estimate, a group-equal MAE of
0.8182. That is useful discovery evidence. It does not select a final model.
Its paired bootstrap interval against the grouped Dummy median crosses zero.
The direct paired Dynamic10 comparison in
[paired_dynamic_increment.tsv](../ligamd_pkoff/resources/models/experimental_n31_registry_v1/paired_dynamic_increment.tsv)
also crosses zero for both Ridge and Random Forest.

Several explanations remain plausible, and this result cannot choose among
them by itself.

1. Dynamic10 may contain route information, yet the effect may be small beside
   the current number of independent ligand groups.
2. A small Dynamic10 summary may miss interaction chemistry that matters only
   in some target and ligand settings.
3. Assay conditions, constructs, identity alignment, or trajectory cadence may
   contribute more variation than the dynamic signal.
4. Static20 may already capture much of the variation through ligand and
   starting-pocket descriptors.

Increasing model capacity cannot resolve these alternatives on its own.

## A predeclared prospective decision rule

The project proposes a prospective set of **at least 15 new independent
exact-ligand groups**. “New” means the audited chemical identity was absent
from N31 and absent from every other prospective group. Feature and endpoint
receipts are frozen before experimental labels are connected. The four existing
model files are then applied once, without re-tuning.

The project would regard Combined30 as a candidate for recommendation only if
the same-family Static20 comparison meets every condition below.

| Check | Predeclared project condition |
| --- | --- |
| Mean improvement | Combined30 lowers paired group-equal MAE by at least 0.10 pKoff. |
| Breadth | More than half of new exact-ligand groups improve. |
| Uncertainty | The upper end of the paired 95% bootstrap interval is below zero. |
| Robustness | No major target family is consistently harmed and no one group dominates the mean. |

These are **project decision criteria**, not numbers proven or required by any
paper. They make a future recommendation falsifiable before the prospective
labels are seen. The number 15 and the 0.10-pKoff target should be revisited
when a new panel's assay precision, group structure, and available independent
coverage are known.

For this comparison, group-equal MAE first gives every exact-ligand group one
average absolute error, regardless of its row count. A paired bootstrap then
resamples those group-level errors with replacement and recomputes
`Combined30 MAE − Static20 MAE`. An upper interval bound below zero means that
the observed dynamic advantage stayed favourable over those resamples. It does
not prove a universal effect on every future protein or assay.

## If the prospective comparison does not support Dynamic10

Run one representation experiment, not a broad model search. Keep the
endpoint-v2 contract, P512, exact-three-replica arithmetic pooling, systems,
fold definitions, and classical model families fixed. Compare:

```text
Static20
Static20 plus Dynamic10
Static20 plus Dynamic10 plus one typed-interaction block
```

Typed interaction fingerprints are literature-motivated as a challenger. They
must be compared with an otherwise matched untyped control, because a richer
chemical name for a contact does not by itself demonstrate useful predictive
information. See the specific typed-fingerprint citation in the
[method evidence ledger](method-evidence.md).

## When to compare more model families

Only after a representation shows a stable prospective increment should one
limited family comparison be opened. Ridge, ElasticNet, Random Forest, and
RBF-SVR should use the same groups, outer folds, training-only preprocessing,
and uncertainty procedure. The question then becomes whether the chosen input
needs a different functional form, not whether a larger model can fit the same
discovery panel more closely.

## When neural or deep models become reasonable

Neural networks, sequence models, LoRA-style adaptation, and other
high-capacity methods are not forbidden. The present evidence simply does not
make them the most informative next experiment. They remain closed until all
of the following are true.

- A dynamic representation has passed the prospective comparison.
- A controlled classical-family comparison has been completed.
- The development set has grown beyond the current 27 independent exact-ligand
  groups while leaving enough untouched groups for external evaluation.
- A meaningful residual-error pattern remains after the classical baselines.

This sequence ties a capacity decision to evidence rather than to the hope
that a larger model will repair an unclear representation.
