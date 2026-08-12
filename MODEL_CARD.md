# Model card for the bundled N31 experimental pKoff baselines

## What the models predict

Each bundled pipeline maps a feature vector to an **experimental pKoff** value.
That is the label reported by an experimental assay and used as the target in
supervised learning. It is different from a physical `koff` inferred from the
length of a LiGaMD trajectory. This repository does not make the latter claim.

When an assay reports a dissociation-rate constant in s<sup>−1</sup>, the
conventional label is

`pKoff = −log10(koff / 1 s<sup>−1</sup>)`.

The division by `1 s<sup>−1</sup>` makes the logarithm dimensionless. A
reported pKoff or pKoff MAE is therefore in log<sub>10</sub> units, not in
seconds or simulation frames. A label sourced in another rate unit must be
converted to s<sup>−1</sup> before this definition is applied. The trajectory
feature route does not calculate this assay label from its elapsed simulation
time.

## What an exact-ligand group means

The frozen N31 panel has one immutable `exact_ligand_group` field for every
system. Its purpose is to stop the same audited ligand chemistry from appearing
both in the training rows and in the test rows of one outer fold. It is not a
system identifier, a PDB identifier, a protein target label, or a judgement
that two molecules have the same pharmacology.

The public release intentionally omits campaign-specific group keys and the
private identity-reconciliation table. It therefore cannot recreate the N31
grouping from names alone. At the data-free level, a new supervised panel
should document all three of these items before any split is made.

1. A versioned molecular standardisation policy, including its treatment of
   salts, protonation, tautomers, and stereochemistry.
2. A reproducible chemical-identity representation produced by that policy,
   such as a canonical isomeric SMILES or InChIKey.
3. An immutable group key derived from that audited representation and saved in
   the label-panel metadata.

The bundled runtime does not compute this key. It predicts from a feature
receipt; grouping belongs to training and evaluation data management.

## How the training and test rows were separated

The frozen N31 engineering panel contains 31 systems. Its 31 rows represent
27 exact-ligand groups because several rows share one audited, standardized
ligand chemical identity. In outer validation, one entire exact-ligand group is
held out at a time. The inner model-choice procedure uses the same type of
grouped split within the remaining training rows.

That arrangement answers a constrained question. It asks whether a model can
predict a held-out ligand chemistry group within the observed mixed-target
panel. It does not prove performance for an unseen protein family, a new assay
condition, or a different trajectory-saving cadence.

The outer split creates the reported prediction errors. One exact-ligand group
is held out, a model is fitted using all remaining groups, and the held-out
group is predicted without being used for preprocessing or model choice. The
inner split happens only within that remaining training material. It selects a
small hyperparameter setting for a model family without looking at the outer
held-out group. This distinction matters because choosing a setting after
seeing the final group would make its error look better than a real new-group
prediction error.

## Included profiles

| Profile | Feature block | Regression family | Role |
| --- | --- | --- | --- |
| `static20_ridge` | Static20 | Ridge | low-capacity static comparator |
| `static20_random_forest` | Static20 | Random Forest | nonlinear static comparator |
| `combined30_p512_ridge` | Static20 plus Dynamic10 | Ridge | low-capacity dynamic candidate |
| `combined30_p512_random_forest` | Static20 plus Dynamic10 | Random Forest | nonlinear dynamic candidate |

The associated `model_scoreboard.tsv` reports group-equal MAE, system-equal
MAE, squared-error summaries, correlations, and paired bootstrap comparisons
with the grouped Dummy median. It contains no system-level labels.

The grouped Dummy median has no molecular features. In each outer fold, it
takes the median experimental pKoff among the remaining training rows and uses
that one number for every system in the held-out ligand group. It is refitted
inside every outer fold. Reusing the same grouped Dummy predictions across
profiles makes the comparison ask one simple question: did the molecular input
improve on a training-only central-value guess?

MAE is the mean absolute difference between a predicted and experimental
pKoff. A group-equal MAE first averages errors within each exact-ligand group,
then averages the group values so a chemistry represented by several systems
does not dominate the result. A paired bootstrap repeatedly resamples the same
groups and recomputes the difference between two methods. Its 95% interval
describes how much that comparison changes across group-level resamples. It is
not a per-system confidence interval and it does not turn an experimental
baseline into a clinically validated predictor.

## Current interpretation

`combined30_p512_ridge` has the lowest N31 group-equal MAE, 0.8182. Its point
estimate is better than the grouped Dummy baseline by 0.1475 pKoff. The 5,000
draw paired bootstrap interval is `[-0.2805, +0.0067]`, so the available panel
does not support a stable model-selection statement.

The more direct question is whether Dynamic10 adds value over Static20 when the
model family is fixed. The public aggregate table reports:

| Paired comparison | Combined30 minus Static20 MAE | Combined improves | 95% paired bootstrap interval |
| --- | ---: | ---: | --- |
| Ridge | -0.0663 | 12 of 27 groups | [-0.2297, +0.0327] |
| Random Forest | -0.0225 | 16 of 27 groups | [-0.0765, +0.0347] |

Negative values favor Combined30. Both intervals cross zero. The two model
families therefore offer suggestive but conflicting directionality: Ridge has
a larger mean improvement, while Random Forest improves more groups but by a
smaller average amount. Neither is enough to select a representation.

Ridge is a linear regression with a penalty that discourages one unstable
feature coefficient from becoming excessively large. Random Forest averages
many decision trees, so it can express nonlinear thresholds and feature
interactions. They are included as different, limited tests of the same input
representation. A lower point estimate from one family does not demonstrate
that the feature representation itself has been established.

## Applicability checks

The prediction command checks that required features are finite, that they lie
inside the N31 training ranges, and that the literal `protein_target` string
appears in the registry. These checks are warnings about applicability. They
are not a calibrated uncertainty interval or proof of external validity.

Dynamic profiles have one additional warning. The frozen N31 registry did not
preserve a single, authoritative saved-frame cadence. The public code marks
Combined30 predictions as cautionary until the training cadence is recovered
or a new registry is fitted with cadence metadata. Static20 profiles do not
use trajectory cadence.

## Known limitations

- The endpoint-v2 thresholds and P512 budget are project-specific engineering
  contracts. They are not universal definitions of biochemical dissociation.
- The model uses exactly three endpoint-passing replicas, pooled by arithmetic
  mean. It does not represent replica disagreement as an uncertainty interval.
- Protein target labels in N31 contain naming aliases. The existing N31 folds
  and scores are frozen and are not retroactively changed by alias cleanup.
- The models are `EXPERIMENTAL`. They are supplied for reproducible baseline
  comparison and prospective evaluation, not clinical, regulatory, or
  production decision making.

## When a more complex model becomes reasonable

The next question is not whether a neural network can lower the current MAE on
the same N31 panel. The project has predeclared a prospective check using at
least 15 new independent exact-ligand groups. The four supplied models should
be applied once without re-tuning. This is a project decision for a useful
future decision boundary, not a threshold that any cited paper proves to be
universal. Only a stable prospective Dynamic10 increment would justify a
single typed-interaction representation comparison. Neural or other
high-capacity methods remain closed until the representation and new
independent coverage pass those checks. The complete decision sequence is in
[docs/model-development-roadmap.md](docs/model-development-roadmap.md).
