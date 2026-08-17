# Current evidence and status for the plan review

The scientific target is assay-derived experimental
`pKoff = -log10(koff / 1 s^-1)`. LiGaMD supplies trajectory-derived inputs. The
project has not estimated a physical `koff` from biased trajectory time.

The table below tells the reviewer both what was observed and how far the
public repository can verify it.

| Evidence | Cohort and comparison | Aggregate result | Current decision | Public verification level |
| --- | --- | --- | --- | --- |
| Frozen public Ridge pair | N31, 27 exact-ligand groups, Combined30 minus Static20 | `-0.0663` group-equal MAE; 12/27 groups improve; 95% paired interval `[-0.2297, +0.0327]` | Directionally favorable, not selected | Public aggregate table and bundled files |
| Frozen public Random Forest pair | N31, Combined30 minus Static20 | `-0.0225`; 16/27 improve; interval `[-0.0765, +0.0347]` | Directionally favorable, not selected | Public aggregate table and bundled files |
| Matched XGBoost feature-block check | N31, Combined30 minus Static20 | `+0.0371`; 10/27 improve; interval `[-0.0007, +0.0733]` | Does not preserve the favorable direction | Private aggregate receipt only |
| Bounded classical extension | N31, Elastic Net / PLS / RBF Kernel Ridge | Dynamic increment `-0.0104`, `+0.0051`, and `+0.0331`; only the RBF-KRR interval excludes zero, in the unfavorable direction | Common tabular-family extension closed for this representation | Private aggregate receipts only |
| Development-exposed cohort sensitivity | N33-SE, 29 groups, Ridge Combined30 minus Static20 | `-0.0340`; 14/29 improve; interval `[-0.0908, +0.0143]` | Directionally favorable, still unstable; not independent validation | Private aggregate receipt only |
| Matched typed representation | N21, typed4-stagePC1 minus matched untyped4 | `+0.1050`; 8/18 improve; interval `[-0.1035, +0.4137]`; leave-one-group direction reversal | This typed4 definition closed | Private aggregate receipt only |
| Full typed block | N31, Current30 plus typed32 residual minus Current30 | `-0.00670`; 11/27 improve; interval `[-0.0995, +0.0793]`; leave-one-group reversal | Did not pass its promotion rule | Private aggregate receipt only |
| External static prior plus Dynamic10 | N31, frozen corrected BiCoA plus Dynamic10 residual minus frozen BiCoA | `-0.0358`; 12/27 improve; interval `[-0.2469, +0.1806]`; descriptor-valid sensitivity changes to `+0.0082` | Exploratory only; input overlap and descriptor fallback limit interpretation | Private aggregate receipt only |
| Fixed output-head LoRA | N31, rank-1 LoRA plus Dynamic10 minus non-LoRA plus Dynamic10 | `+0.2328`; interval `[+0.0268, +0.4855]` | This implementation is closed | Private aggregate receipt only |
| Protein-family stress | Four main observed families, family-equal MAE | Dummy `1.0628`; Static20 Ridge `1.1159`; Combined30 Ridge `1.1047` | Unseen-family transfer is not established | Private aggregate receipt only |
| Ordered P512 payload feasibility | 31 systems, 3 replicas each, 512 real frames, 11 fixed channels | Existing assets can form `31 x 3 x 512 x 11`; tensor has not been serialized and no temporal model has run | Label-blind serialization is feasible | Private route audit summarized here; raw assets excluded |
| Collaborator candidate inventory | Private workbook, candidate and simulation-status roster | 131 rows; 92 marked Finished; 62 outside N31; zero currently satisfy every admission gate | Candidate source, not a training table | Private aggregate intake statement only |

Negative deltas favor the candidate named first. An interval that crosses zero
does not establish a stable increment. These results also do not establish that
Dynamic10 has no information. Several low-capacity comparisons have favorable
point estimates, but the direction is not stable across model families,
independent groups, or sensitivity checks.

The public Random Forest row above is the authority shipped in the bundled
registry. A later private matched feature-block diagnostic reported a nearby
Combined30-minus-Static20 value of `-0.0218` rather than `-0.0225`, because the
two tables use different saved OOF authorities. This packet does not average or
silently merge them. Both lead to the same bounded statement: a small favorable
point estimate whose paired interval crosses zero.

## What has already been tried

The project has already tested linear shrinkage, sparse linear regression,
latent components, tree ensembles, boosting, and a smooth RBF kernel on the
current summarized representation. It has also tested two predeclared typed
interaction representations, a frozen external static prior, one fixed
output-head LoRA configuration, and a family-level stress test.

The unresolved question is therefore not simply whether another common
tabular learner can lower N31 error. The two remaining scientific gaps are:

1. whether the present Combined30 increment persists on qualified ligand
   groups that did not enter N31 development; and
2. whether the ordered P512 frame sequence contains useful information that
   ten Dynamic10 summary values discarded.

## Candidate-data boundary

The private collaborator workbook already contains experimental values, so a
future comparison based on it is called a **locked independent-group
evaluation**, not a pristine prospective acquisition. A system becomes
eligible only after separate checks for experimental-label authority, exact
simulation-to-assay ligand identity, three endpoint-passing replicas, finite
P512/Current30 features, and method-development exposure.

The word `Finished` in that workbook is a collaborator workflow status. It is
not accepted as a synonym for the public endpoint, three-replica feature
readiness, or supervised-model admission.

The revised admission ledger also carries one trajectory-protocol gate. It
records whether saved-frame cadence, LiGaMD boost/sigma settings, and the
production protocol are compatible with the frozen N31 authority. `UNKNOWN`
is not silently treated as `PASS`. This matters because an acquisition shift
could otherwise be mistaken for a failure of Dynamic10 or protein-family
transfer.

## Current claim ceiling

The public repository supports a reproducible trajectory-to-feature route and
four experimental baseline predictors. It does not currently support a final
model choice, a universal endpoint or sampler, unseen-protein-family
generalization, stable Dynamic10 improvement, or physical-rate estimation.
