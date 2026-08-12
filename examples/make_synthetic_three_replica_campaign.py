#!/usr/bin/env python3
"""Create a tiny three-replica LiGaMD-shaped campaign for a local smoke run.

The generated coordinates are deliberately artificial. They exist only to
exercise the public input schema and the complete geometric route. All three
replicas begin with a two-heavy-atom ligand in a small alanine pocket. At saved
frame 521 the ligand moves far enough away to satisfy the public endpoint-v2
geometry rule for the final 100 saved frames.

This is not a molecular simulation, an HSP90 example, or a model benchmark.
Use your own validated Amber production data for science.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file


def _pdb_line(
    record: str,
    serial: int,
    atom: str,
    residue: str,
    resid: int,
    xyz: tuple[float, float, float],
    element: str,
) -> str:
    x, y, z = xyz
    return (
        f"{record:<6}{serial:>5} {atom:<4} {residue:>3} A{resid:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2}\n"
    )


def build(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing example directory: {output_dir}")
    output_dir.mkdir(parents=True)
    pdb = output_dir / "canonical.pdb"
    pdb.write_text(
        "CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n"
        + _pdb_line("ATOM", 1, "N", "ALA", 1, (0.0, 0.0, 0.0), "N")
        + _pdb_line("ATOM", 2, "CA", "ALA", 1, (1.5, 0.0, 0.0), "C")
        + _pdb_line("ATOM", 3, "C", "ALA", 1, (1.5, 1.5, 0.0), "C")
        + _pdb_line("HETATM", 4, "C1", "LIG", 501, (3.0, 0.0, 0.0), "C")
        + _pdb_line("HETATM", 5, "O1", "LIG", 501, (3.0, 1.0, 0.0), "O")
        + "END\n",
        encoding="utf-8",
    )
    topology = output_dir / "canonical.parm7"
    topology.write_text("synthetic topology identity placeholder\n", encoding="utf-8")

    frames = 620
    coordinates = np.zeros((frames, 5, 3), dtype=np.float32)
    coordinates[:, 0] = (0.0, 0.0, 0.0)
    coordinates[:, 1] = (1.5, 0.0, 0.0)
    coordinates[:, 2] = (1.5, 1.5, 0.0)
    coordinates[:, 3] = (3.0, 0.0, 0.0)
    coordinates[:, 4] = (3.0, 1.0, 0.0)
    coordinates[520:, 3] = (30.0, 0.0, 0.0)
    coordinates[520:, 4] = (30.0, 1.0, 0.0)
    for replica_id in ("r1", "r2", "r3"):
        trajectory = output_dir / f"{replica_id}.nc"
        with netcdf_file(str(trajectory), "w") as nc:
            nc.createDimension("frame", frames)
            nc.createDimension("atom", 5)
            nc.createDimension("spatial", 3)
            nc.createVariable("coordinates", "f", ("frame", "atom", "spatial"))[:] = coordinates
            nc.createVariable("cell_lengths", "f", ("frame", "spatial"))[:] = 100.0
            nc.createVariable("cell_angles", "f", ("frame", "spatial"))[:] = 90.0
    manifest = output_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "ligamd_pkoff_toolkit_input_v1.0",
                "system_id": "synthetic-raw-system",
                "condition_id": "synthetic-condition",
                "protein_target": "synthetic-target",
                "saved_frame_interval_ps": 1.0,
                "canonical_bound_pdb": "canonical.pdb",
                "canonical_topology": "canonical.parm7",
                "ligand": {"resname": "LIG", "resid": "501", "chain": "A", "smiles": "CCO"},
                "replicas": [
                    {
                        "replica_id": replica_id,
                        "trajectory": f"{replica_id}.nc",
                        "pdb": "canonical.pdb",
                        "topology": "canonical.parm7",
                    }
                    for replica_id in ("r1", "r2", "r3")
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
