# Define an episode, then select 512 real frames

This explanation covers two operations that must remain separate.

1. **Endpoint-v2** decides whether a production replica contains a sustained,
   geometry-defined complete exit and, if so, where the analysable episode ends.
2. **P512** decides which 512 already-saved coordinate frames represent that
   fixed episode.

Neither operation reads an experimental pKoff label, a model error, a system
name, or a sampler comparison result. Endpoint-v2 does not estimate physical
koff. P512 does not decide whether a trajectory dissociated.

## Inputs and the frozen structural reference

The input manifest supplies one canonical bound PDB, a matching topology, and
three production NetCDF replicas. Before processing a replica, the toolkit
builds one shared structural reference from the canonical bound structure. It
freezes the following items.

1. The ligand heavy atoms and their order on the coordinate axis.
2. The standard-amino-acid protein atoms.
3. The **pocket**, defined as standard protein residues whose canonical
   minimum ligand-heavy-atom distance is at most 8 A.
4. The **native-contact set**, the subset of frozen pocket residues whose
   canonical minimum ligand-heavy-atom distance is at most 4.5 A.
5. The pocket heavy atoms used to align each frame back to the canonical bound
   reference.

The point of freezing these sets is comparability. If each replica redefined
its pocket after the ligand had moved, a contact fraction could change because
the denominator changed instead of because the ligand moved. The construction
and fail-closed checks are in
[shared_reference.py](../scripts/koff_ml/shared_reference.py). The input
fields are described in the [input-manifest reference](input-manifest.md).

## Endpoint-v2, the rule that ends an episode

For each saved frame, the toolkit asks two different geometry questions.

| Measurement | A frame satisfies the condition when | Why this separate condition is needed |
| --- | --- | --- |
| Pocket-aligned ligand-centroid displacement | After Kabsch alignment of the frozen pocket to the bound reference, the ligand centroid is at least 15 A from its bound-reference position. | A ligand can leave the original pocket but move to another protein surface. This measurement establishes displacement from the original bound pose. |
| Whole-protein heavy-atom clearance | The smallest periodic-boundary-aware distance between any ligand heavy atom and any standard-protein heavy atom is **strictly greater than 10 A**. | A ligand can be far from the original pocket and still touch the protein surface. This measurement tests that residual physical proximity directly. |

The conditions are joined by **AND**. A geometrically clear frame is not enough
by itself. Both conditions must be true for 100 consecutive **saved** frames.
The first frame of the first qualifying run is the **endpoint onset**.

~~~text
production frame 0                      first stable-clear frame
bound-to-exit path ------------------------------------o==== 99 confirming frames ===>
selected episode: frame 0 through o, inclusive
~~~

The confirmation tail proves that the onset was persistent. It is read by the
endpoint check but excluded from P512 selection and from feature summaries.
Later solvent diffusion is also excluded. This stops a simulation that simply
ran longer after exit from contributing more model input than a shorter,
otherwise similar episode.

The numerical contract is
[endpoint_v2.json](../ligamd_pkoff/resources/contracts/endpoint_v2.json).
The contract parser and the first-persistent-run calculation are
[endpoint_two_metric.py](../scripts/koff_ml/endpoint_two_metric.py#L44-L232).
The operational A, I, P_ONLY, and B labels used for Dynamic10 occupancies are
documented in the [feature reference](feature-definitions.md#the-a-i-p_only-and-b-bookkeeping-labels).
They are **not** extra endpoint-v2 conditions.

## P512, a path-change representation of the already-fixed episode

Once endpoint-v2 has supplied an onset index, P512 receives only the saved
frames from production frame 0 through that onset, inclusive. It returns 512
unique zero-based source indices. Each index points to a coordinate frame that
already exists in the NetCDF trajectory.

P512 gives more representation budget to sections where the measured path
changes more, while still retaining the first and endpoint-onset frames. It
does not equate a large change with a physical rate. It is a way to preserve
geometry, contact, and pose changes within a fixed 512-frame storage budget.

### The three blocks it measures

| Block | Per-frame channels | What a change in this block means |
| --- | --- | --- |
| Separation | pocket geometric-centre distance, pocket-aligned ligand-centroid displacement, global minimum ligand/protein heavy-atom distance | The ligand moves relative to the original pocket or any protein surface. |
| Contact | native-contact fraction, number of globally contacted protein residues, one binary column for each frozen pocket contact | Initial contacts are lost, gained, or rearranged. |
| Pose | pocket-aligned ligand RMSD, ligand internal RMSD, pocket-alignment RMSD | The ligand pose, ligand conformation, or local pocket alignment changes. |

The dense per-frame channels are produced by
[reference_features.py](../scripts/koff_ml/reference_features.py) and are
validated by [load_p512_trace](../scripts/koff_ml/p512_sampler.py#L49-L81).

### The operational algorithm

Let x(t,j) be channel j at saved frame t, from frame 0 through the endpoint
onset. The following is the exact v1 procedure in plain language.

1. **Smooth using five past-and-current observations.** For every channel,
   replace x(t,j) by the mean from max(0, t - 4) through t. The procedure never
   looks ahead to a later frame. It therefore dampens one-frame noise without
   using future trajectory information.
2. **Give every channel its own early-path scale.** In the first 128 smoothed
   frames, calculate first differences. Use 1.4826 times the median absolute
   deviation of those differences as the scale. If that value is at or below
   1e-8, use the standard deviation. If that is also at or below 1e-8, use 1.
   This prevents a channel measured in large numerical units from dominating
   merely because of its unit, and it prevents a constant channel from causing
   division by zero.
3. **Give the three blocks equal opportunity to contribute.** Divide each
   scaled block by the square root of its number of columns. The contact block
   can contain many frozen contact bits, so this block-size normalization keeps
   it from overwhelming the separation and pose blocks solely because it has
   more columns.
4. **Measure cumulative path length.** For adjacent frames, calculate the
   Euclidean length of the combined, normalized block change. Add these step
   lengths from the beginning of the episode to frame t to obtain cumulative
   path length L(t).
5. **Place 512 equally spaced targets on that cumulative length.** The first
   frame and endpoint-onset frame are forced. For each interior target, choose
   the nearest saved frame in L(t). An exact tie chooses the earlier source
   frame.
6. **Resolve duplicate targets deterministically.** Very calm intervals can
   make several targets choose the same real source frame. Until exactly 512
   unique indices exist, split the largest remaining time gap at its integer
   midpoint. If several gaps have equal size, choose the earlier gap.

With normalized block vector z(t), P512 uses

~~~text
step(t) = EuclideanNorm(z(t) - z(t - 1))
L(0) = 0
L(t) = sum(step(i) for i = 1 through t)
interior targets = equally spaced values between 0 and L(endpoint)
~~~

The source implementation is
[p512_sampler.py](../scripts/koff_ml/p512_sampler.py#L84-L216). The frozen
settings are in
[p512_sampler_v1.json](../ligamd_pkoff/resources/contracts/p512_sampler_v1.json).
The following pseudocode mirrors the code without hiding the tie and fallback
rules.

~~~python
episode = dense_trace[0 : endpoint_onset + 1]
blocks = make_separation_contact_pose_blocks(episode)

for block in blocks:
    smoothed = trailing_mean(block, width=5)       # current and past only
    scale = MAD_of_first_differences(smoothed[:128])
    scale = SD_fallback_then_one(scale)
    block = (smoothed / scale) / sqrt(block.width)

step = sqrt(sum(sum(diff(block, axis=0) ** 2, axis=1) for block in blocks))
cumulative = concatenate([0, cumsum(step)])
selected = {0, endpoint_onset}

for target in interior_equal_targets(cumulative[-1], count=510):
    selected.add(nearest_saved_frame(cumulative, target, tie="earlier"))

selected = fill_largest_time_gap_until_512(selected)
~~~

P512 fails rather than inventing data if the episode contains fewer than 512
real saved frames. It does not interpolate coordinates, define an endpoint,
read a label, or select a regression model. The 5-frame, 128-frame, and
512-frame settings are project contracts, not universal biochemical constants.
The [method-evidence page](method-evidence.md) separates the literature idea
of path geometry from this project's particular implementation.
