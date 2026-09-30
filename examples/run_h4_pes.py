#!/usr/bin/env python3
"""Reproduce a Fig. 4a-style H₄ potential-energy surface with GQKAE.

Laptop defaults (fewer shots / iters than the paper). Override via CLI flags::

    python examples/run_h4_pes.py
    python examples/run_h4_pes.py --n-points 5 --n-iters 15 --shots 512
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gqkae.cli import main

if __name__ == "__main__":
    # default args if none provided
    argv = sys.argv[1:] or [
        "pes",
        "--n-points",
        "5",
        "--n-iters",
        "20",
        "--group-size",
        "4",
        "--shots",
        "512",
        "--out-dir",
        "runs/h4_pes",
    ]
    if argv[0] != "pes" and not argv[0].startswith("-"):
        pass
    elif argv[0] != "pes":
        argv = ["pes", *argv]
    main(argv)
