# Data notice

This is a code-and-contract release. It is designed so a researcher can run
the same calculation on data they are authorized to use.

## Included

- reusable source code and focused tests
- the endpoint-v2 and P512 sampler contracts
- a synthetic three-replica campaign generator
- four `EXPERIMENTAL` N31 model files
- a model registry, aggregate scorecard, and aggregate dynamic-increment table

## Not included

- raw production trajectories, topology files, restart files, or job logs
- canonical structures from the original campaigns
- per-system experimental pKoff values, ligand identifiers, InChIKeys, or
  out-of-fold prediction rows
- internal identity reconciliation records and assay-source locators
- source PDFs, slide decks, PyMOL images, or other third-party material
- credentials, machine-local paths, and historical Git history

The absence of a data file is deliberate. An aggregate model scorecard can
document the current result without redistributing a campaign-specific assay
table. If you need to reproduce a particular published result, obtain the
underlying structures, trajectories, and experimental labels from their
rightful owners or public sources, then use this toolkit to process them.

The MIT License applies to the code in this repository. It does not grant
rights to data or materials that are not included.
