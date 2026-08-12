# LiGaMD pKoff collaboration rules

## Keep the two products separate

This workspace contains two related but different products.

1. **Trajectory Analysis Toolkit** turns a validated LiGaMD production trajectory into an episode, sampled real coordinate frames, quality-control records, and feature vectors.
2. **Experimental pKoff Predictor** is an experimental supervised-learning branch. It predicts an assay-derived `pKoff` label. It does not estimate a physical dissociation rate unless that claim has separately been demonstrated.

Do not describe the predictor as a finished scientific model when the evidence only supports an engineering or discovery result.

## Public-repository synchronization

For any bounded change that the user has authorized for public release, treat a small, reviewable pull request as the normal handoff rather than repeatedly recalculating hashes.

1. Work on a `codex/<short-topic>` branch in the public repository.
2. Keep one coherent change set per pull request. Update the README, changelog, and a short decision note whenever the public interface, a scientific claim, or a model artifact changes.
3. Run the smallest check that covers the changed risk. Do not repeat hash checks for ordinary code or documentation edits. Use a release-boundary check only when raw-data exclusions, contracts, or bundled model files change.
4. Stage explicit paths, never a blanket `git add -A` from a mixed worktree.
5. Never publish raw trajectories, topology files, private assay tables, source PDFs, credentials, local absolute paths, or campaign-specific identity reconciliations without explicit user approval.
6. Before opening the pull request, verify that the accompanying documentation states what the method does, what it does not establish, and where a new user should begin.

An agent cannot create a real daily schedule by writing this file. During every active workday that changes the public repository, however, the agent must either open/update a PR or record in the delivery note why no public change was appropriate.

## Writing for a first reader

Write as though the reader has not seen the internal project history. Introduce a technical term before using an abbreviation or internal label. For each scientific operation, explain the input, the computation, the output, why that output is needed next, and the evidence boundary. Use plain, human language without overstating what a small cohort or a model comparison proves.
