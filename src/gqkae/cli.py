"""CLI: train on one Ising/QUBO instance, or run the SK comparison."""

from __future__ import annotations

import argparse

from gqkae.compare import METHODS, run_comparison
from gqkae.ising import IsingProblem, bits_to_str, index_to_bits, load_qubo, random_sk
from gqkae.profiles import PROFILES
from gqkae.train import TrainConfig, train


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="GQE / GQKAE for Ising and QUBO problems")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train on one instance")
    src = t.add_mutually_exclusive_group()
    src.add_argument("--qubo", type=str, help=".npy (Q) or .json ({Q,const} or {h,J,offset})")
    src.add_argument("--sk", type=int, default=12, metavar="N", help="random SK instance with N spins")
    t.add_argument("--instance-seed", type=int, default=0)
    t.add_argument("--cvar-alpha", type=float, default=0.1)
    t.add_argument("--n-iters", type=int, default=40)
    t.add_argument("--group-size", type=int, default=8)
    t.add_argument("--shots", type=int, default=256)
    t.add_argument("--seq-len", type=int, default=20)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--backbone", choices=["gqkae", "gqe"], default="gqkae")
    t.add_argument("--d-model", type=int, default=32)
    t.add_argument("--n-layers", type=int, default=2)
    t.add_argument("--out-dir", type=str, default="runs/train")
    t.add_argument("--no-exact", action="store_true", help="skip brute force (needed past ~22 vars)")

    c = sub.add_parser("compare", help="random / SA / QAOA / GQE / GQKAE on random SK instances")
    c.add_argument("--profile", choices=sorted(PROFILES), default="laptop")
    c.add_argument("--n", type=int, default=None)
    c.add_argument("--n-instances", type=int, default=None)
    c.add_argument("--n-seeds", type=int, default=None)
    c.add_argument("--out-dir", type=str, default=None)
    c.add_argument("--skip", nargs="*", choices=METHODS, default=[])

    args = p.parse_args(argv)

    if args.cmd == "train":
        model = load_qubo(args.qubo) if args.qubo else random_sk(args.sk, seed=args.instance_seed)
        problem = IsingProblem(model, cvar_alpha=args.cvar_alpha)
        cfg = TrainConfig(
            n_iters=args.n_iters,
            group_size=args.group_size,
            shots=args.shots,
            seq_len=args.seq_len,
            seed=args.seed,
            backbone=args.backbone,
            d_model=args.d_model,
            n_layers=args.n_layers,
            out_dir=args.out_dir,
        )
        res = train(problem, cfg)
        bits = bits_to_str(index_to_bits(res.best_index, model.n))
        line = f"{res.backbone} best E={res.best_energy:.6f}  x={bits}  params={res.n_params}"
        if not args.no_exact:
            e0, _ = model.ground_state()
            line += f"  exact={e0:.6f}  gap={res.best_energy - e0:.2e}"
        print(line)
        return

    if args.cmd == "compare":
        run_comparison(
            profile=args.profile,
            out_dir=args.out_dir or f"runs/{args.profile}",
            n=args.n,
            n_instances=args.n_instances,
            n_seeds=args.n_seeds,
            skip=tuple(args.skip),
        )


if __name__ == "__main__":
    main()
