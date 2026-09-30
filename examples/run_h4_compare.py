#!/usr/bin/env python3
"""Fig. 4(a)/5(a)-style H₄ comparison: classical + VQE + GQE + GQKAE.

Default: nano (Sol-friendly). Override the profile::

    python examples/run_h4_compare.py
    python examples/run_h4_compare.py --profile laptop --n-points 3
    python examples/run_h4_compare.py --profile smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gqkae.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:] or ["compare", "--profile", "nano"]
    if argv[0] != "compare" and not argv[0].startswith("-"):
        pass
    elif argv[0] != "compare":
        argv = ["compare", *argv]
    main(argv)
