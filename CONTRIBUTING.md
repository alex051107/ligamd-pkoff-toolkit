# Contributing to the LiGaMD pKoff Toolkit

This repository is intended to be useful to a new group member as well as to
the original project. A contribution should make one part of the scientific
route easier to reproduce, easier to inspect, or safer to use.

## The two boundaries that must stay visible

The trajectory-analysis toolkit and the experimental-pKoff predictors have
different responsibilities. A change to coordinate handling, an endpoint
contract, or a feature calculation can affect all downstream model inputs. A
change to an experimental model registry can affect only the bounded model
claim recorded in its model card. Please do not silently turn one boundary
into the other.

The public repository also excludes raw trajectories, topologies, production
logs, individual assay labels, full out-of-fold prediction tables, internal
identity records, and source PDFs. If a proposed change needs any of those
materials, keep them in an authorized private location and publish only the
aggregate result that may be shared.

## A normal update

1. Start a focused branch named `codex/<short-topic>`.
2. Change the code, contract, documentation, or test that belongs to one
   bounded question.
3. Explain the scientific consequence in the pull request. A reader should be
   able to tell whether the change affects coordinates, features, model scope,
   or wording only.
4. Run the smallest check that covers that change. The standard public check
   is `python -m pytest -q tests` followed by
   `python tools/verify_public_release.py` when a release boundary changes.
5. Open a pull request. Do not update model scores, rewrite a contract, or
   publish a new model artifact without a separate evidence note and review.

Routine documentation and local code changes do not need a new raw-data hash
or a re-run of the N31 matrix. Use an additional integrity check only when
there is a concrete reason to suspect data transfer or artifact corruption.

## Writing and review

Write for a reader who has not attended a project meeting. When a scientific
term first appears, say what enters the calculation, what the code calculates,
what it returns, and what it does not prove. Keep project-specific values such
as 15 Å, 10 Å, 100 saved frames, and 512 selected frames attached to their
contract rather than presenting them as universal biochemical constants.
