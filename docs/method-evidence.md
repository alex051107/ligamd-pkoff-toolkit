# Method evidence ledger and project choices

This page records where the public workflow borrowed an idea and where the
workflow made its own engineering choice. A citation supports only the claim
in its row. It does not validate the whole pipeline, the N31 result, or a
numerical threshold that the paper did not test.

The public repository does not redistribute source PDFs. The **source locator**
column therefore gives a page, section, or figure only when that locator was
verified from an available source record. When it says that no locator is
retained, the limitation is deliberate. It is better to cite a DOI without a
made-up page number than to imply a source check that did not happen.

| Topic | Paper and DOI | Source locator available to this release | Narrow paper-supported idea | Project use and explicit boundary |
| --- | --- | --- | --- | --- |
| Trajectory interaction information with regression | Kokh et al. (2019), [doi:10.3389/fmolb.2019.00036](https://doi.org/10.3389/fmolb.2019.00036) | p. 1, Abstract in the publicly accessible Frontiers PDF | Interaction information from τRAMD dissociation trajectories can be used as input to regression analysis of residence-time-related outcomes. | Contact and state summaries motivate Dynamic10. The paper does not prove the public endpoint-v2 rule, P512, experimental pKoff target, or a Dynamic10 benefit here. |
| Ligand dissociation and interaction-fingerprint workflow | Kokh et al. (2020), [doi:10.1063/5.0019088](https://doi.org/10.1063/5.0019088) | Article title and accepted-manuscript metadata were available; no verified section, page, or figure is retained in this release | A workflow can analyse ligand dissociation trajectories with interaction fingerprints. | Per-frame distance, pose, and contact measurements are used before summary. A τRAMD workflow does not validate a LiGaMD endpoint or an assay-label predictor. |
| Chemically typed interaction fingerprints | Bouysset and Fiorucci (2021), [doi:10.1186/s13321-021-00548-6](https://doi.org/10.1186/s13321-021-00548-6) | No page or figure locator is retained in the public source record | Protein–ligand interaction fingerprints can encode named interaction types rather than only an untyped proximity event. | The optional CORE8 route materializes one engineering challenger. It is not part of Current30 or a predictor, and it must be compared with a matched-capacity untyped block before any benefit claim. |
| Static molecular and complex descriptors | Xu et al. (2024), [doi:10.1021/jacsau.4c00123](https://doi.org/10.1021/jacsau.4c00123) | p. 1635, “Feature Extraction Based on Complex Structures” in the article record | Ligand descriptors and structure-derived contact-shell descriptors can be used in HSP90 kinetic prediction. | Static20 is retained as a control so a dynamic feature is tested against chemistry and bound-structure information. The paper does not prescribe this repository's Static20 fields or split policy. |
| LiGaMD kinetic context | Wang and Miao (2024), [doi:10.1021/acs.jctc.4c00502](https://doi.org/10.1021/acs.jctc.4c00502) | Bibliographic record identifies pp. 5829–5841; no verified section or figure is retained in the public source record | LiGaMD3 addresses ligand binding thermodynamics and kinetics under its own repeated-sampling and reweighting framework. | This toolkit uses LiGaMD paths as descriptors for an experimental label. It does not recover a physical `koff` from a single path or from simulated elapsed time. |
| Multidimensional path progress | Branduardi, Gervasio, and Parrinello (2007), [doi:10.1063/1.2432340](https://doi.org/10.1063/1.2432340) | Bibliographic record identifies J. Chem. Phys. 126, 054103; no verified section or figure is retained in the public source record | A molecular configuration can be described by progress relative to a path in a multidimensional space. | P512 uses several measured channels to allocate frames across observed path change. It is a project algorithm; the paper does not prescribe its blocks, scales, frame budget, or gap-filling rule. |
| Sampling dynamic molecular representations | Liu et al. (2026), [doi:10.1371/journal.pcbi.1014515](https://doi.org/10.1371/journal.pcbi.1014515) | Introduction and Fig. 3 on the PLOS article page | Dynamic molecular-property workflows can compare ways of sampling and representing molecular conformations. | Pose and internal-conformation channels are measured as possible dynamic information. The paper does not identify P512 as optimal for LiGaMD or for experimental pKoff prediction. |

## What came from the project rather than a paper

The following items are explicit project contracts. They are published so that
another researcher can use them, challenge them, or replace them with a new
documented contract. They should not be described as literature defaults.

| Project choice | Why it is written down | What it does not establish |
| --- | --- | --- |
| Two-metric endpoint-v2, 15 Å aligned centroid displacement plus greater than 10 Å whole-protein heavy-atom clearance for 100 saved frames | It gives a label-blind and reproducible episode boundary that rejects a ligand merely moving across the original pocket while still contacting the protein. | A universal biochemical definition of dissociation or zero interaction energy. |
| 512 real P512 frames per replica | It holds the coordinate budget fixed while spreading selected frames across observed multichannel path change. | A universal optimal number of frames or a physical time reweighting. |
| Current30, including Static20 plus Dynamic10 | It creates a small, inspectable engineering representation in which dynamic information can be compared with static controls. | A selected final scientific feature set or a complete mechanism. |
| Three endpoint-passing replicas pooled by arithmetic mean | It keeps the model input at one vector per system while preserving per-replica receipts. | A statistical uncertainty model for replica disagreement. |
| Fold-local constant/correlation reduction and scaling | It prevents the held-out exact-ligand group from directing feature preprocessing. | A claim that a particular correlation cutoff or reducer is chemically correct. |

## The literature boundary in one sentence

The papers make the measurements and comparisons less arbitrary. They do not
turn the current small, frozen N31 engineering panel into proof that a sampler,
representation, or model has been scientifically selected. The appropriate
next claim is set out in the [model-development roadmap](model-development-roadmap.md).
