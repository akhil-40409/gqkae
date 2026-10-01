#!/usr/bin/env python3
"""Random / SA / QAOA / GQE / GQKAE on random SK spin glasses.

    python examples/run_compare.py                    # laptop profile
    python examples/run_compare.py --profile smoke
    python examples/run_compare.py --profile nano --n 18
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gqe.cli import main

if __name__ == "__main__":
    main(["compare", *(sys.argv[1:] or ["--profile", "laptop"])])
