"""Compatibility entry point for the leakage-safe grouped splitter.

The legacy implementation shuffled individual filenames and is intentionally
retired. Use explicit manifests so missing provenance cannot be hidden.
"""

import runpy
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).with_name("split-dataset-grouped.py")
    runpy.run_path(str(script), run_name="__main__")
