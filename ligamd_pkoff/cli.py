"""Expose the stable command-line entry point without hiding the scientific code."""

from __future__ import annotations

from scripts.koff_ml.toolkit import main

__all__ = ("main",)


if __name__ == "__main__":  # pragma: no cover - exercised by the console entry point
    raise SystemExit(main())
