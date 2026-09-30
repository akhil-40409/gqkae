"""Head-to-head on random SK instances: exact, random, SA, QAOA, GQE, GQKAE.

Every method is scored the same way: the lowest energy of any bitstring it
produced, compared with the brute-force ground state. Random sampling gets the
same shot budget as one generator run, which is the baseline that matters.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gqkae.baselines import qaoa, random_search, simulated_annealing
from gqkae.ising import IsingProblem, random_sk
from gqkae.profiles import get_profile, train_config
from gqkae.train import train

METHODS = ("random", "SA", "QAOA", "GQE", "GQKAE")
TOL = 1e-9


def run_comparison(
    *,
    profile: str = "laptop",
    out_dir: str = "runs/laptop",
    n: int | None = None,
    n_instances: int | None = None,
    n_seeds: int | None = None,
    skip: tuple[str, ...] = (),
) -> dict[str, Any]:
    cfg = get_profile(profile)
    for k, v in {"n": n, "n_instances": n_instances, "n_seeds": n_seeds}.items():
        if v is not None:
            cfg[k] = v
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for inst in range(int(cfg["n_instances"])):
        model = random_sk(int(cfg["n"]), seed=inst)
        problem = IsingProblem(model, cvar_alpha=float(cfg["cvar_alpha"]))
        e0, _ = model.ground_state()
        row: dict[str, Any] = {"instance": inst, "exact": e0, "vocab": len(problem.pool)}
        print(f"\n=== SK n={model.n} instance {inst}: E0={e0:.6f}  vocab={len(problem.pool)} ===")

        budget = cfg["n_iters"] * cfg["group_size"] * cfg["shots"]
        seeds = range(int(cfg["n_seeds"]))

        def record(name: str, energies: list[float], t0: float, **extra) -> None:
            row[name] = {
                "best": energies,
                "gap": [e - e0 for e in energies],
                "hit_rate": float(np.mean([e - e0 < TOL for e in energies])),
                "wall_s": time.perf_counter() - t0,
                **extra,
            }
            print(
                f"  {name:6s} mean gap={np.mean(row[name]['gap']):.4f}  "
                f"hit={row[name]['hit_rate']:.0%}  ({row[name]['wall_s']:.1f}s)"
            )

        if "random" not in skip:
            t0 = time.perf_counter()
            record("random", [random_search(problem, budget, seed=s) for s in seeds], t0, shots=budget)
        if "SA" not in skip:
            t0 = time.perf_counter()
            es = [simulated_annealing(model, sweeps=int(cfg["sa_sweeps"]), seed=s) for s in seeds]
            record("SA", es, t0)
        if "QAOA" not in skip:
            t0 = time.perf_counter()
            res = [
                qaoa(problem, p=int(cfg["qaoa_p"]), maxiter=int(cfg["qaoa_maxiter"]), shots=budget, seed=s)
                for s in seeds
            ]
            record("QAOA", [r.best_energy for r in res], t0, expected=[r.expected_energy for r in res])
        for name in ("GQE", "GQKAE"):
            if name in skip:
                continue
            t0 = time.perf_counter()
            results = [
                train(
                    problem,
                    train_config(
                        cfg,
                        backbone=name.lower(),
                        seed=s + 17 * inst,
                        out_dir=str(out / name.lower() / f"inst{inst}_seed{s}"),
                    ),
                )
                for s in seeds
            ]
            record(name, [r.best_energy for r in results], t0, n_params=results[0].n_params, shots=budget)
        rows.append(row)

    payload = {"profile": profile, "config": cfg, "instances": rows}
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    _write_summary(out, payload)
    return payload


def _write_summary(out: Path, payload: dict[str, Any]) -> None:
    cfg = payload["config"]
    methods = [m for m in METHODS if any(m in r for r in payload["instances"])]
    budget = cfg["n_iters"] * cfg["group_size"] * cfg["shots"]
    lines = [
        f"# SK spin glass, n={cfg['n']} ({payload['profile']})",
        "",
        f"- {cfg['n_instances']} instances × {cfg['n_seeds']} seeds; state space 2^{cfg['n']} = {2 ** cfg['n']}.",
        f"- Shot budget per generator run (and for random): {budget}.",
        "- Cell = mean (best energy − exact ground energy) / hit rate.",
        "",
        "| instance | E0 | " + " | ".join(methods) + " |",
        "|---:|---:|" + "---:|" * len(methods),
    ]
    for r in payload["instances"]:
        cells = [
            f"{np.mean(r[m]['gap']):.4f} / {r[m]['hit_rate']:.0%}" if m in r else "—" for m in methods
        ]
        lines.append(f"| {r['instance']} | {r['exact']:.4f} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'summary.md'}")
