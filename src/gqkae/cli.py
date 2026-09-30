"""CLI: train one geometry, scan a PES, or run the multi-method comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gqkae.compare import run_comparison
from gqkae.molecule import reference_energies
from gqkae.profiles import PROFILES
from gqkae.train import TrainConfig, train_geometry


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="GQKAE / GQE for linear H₄ (8 qubits)")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train at one bond length")
    t.add_argument("--R", type=float, default=1.0, help="H–H spacing (Å)")
    t.add_argument("--n-iters", type=int, default=40)
    t.add_argument("--group-size", type=int, default=8)
    t.add_argument("--shots", type=int, default=2048)
    t.add_argument("--seq-len", type=int, default=20)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--backbone", choices=["gqkae", "gqe"], default="gqkae")
    t.add_argument("--d-model", type=int, default=32)
    t.add_argument("--n-layers", type=int, default=2)
    t.add_argument("--out-dir", type=str, default="runs/h4")

    s = sub.add_parser("pes", help="Bond-length scan → Fig. 4a-style GQKAE curve")
    s.add_argument("--r-min", type=float, default=0.8)
    s.add_argument("--r-max", type=float, default=2.2)
    s.add_argument("--n-points", type=int, default=7)
    s.add_argument("--n-iters", type=int, default=25)
    s.add_argument("--group-size", type=int, default=6)
    s.add_argument("--shots", type=int, default=1024)
    s.add_argument("--seq-len", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--backbone", choices=["gqkae", "gqe"], default="gqkae")
    s.add_argument("--out-dir", type=str, default="runs/h4_pes")

    c = sub.add_parser("compare", help="HF/CCSD/CASCI + VQE + GQE + GQKAE PES")
    c.add_argument("--profile", choices=sorted(PROFILES), default="nano")
    c.add_argument("--r-min", type=float, default=None)
    c.add_argument("--r-max", type=float, default=None)
    c.add_argument("--n-points", type=int, default=None)
    c.add_argument("--n-seeds", type=int, default=None)
    c.add_argument("--out-dir", type=str, default=None)
    c.add_argument("--skip-vqe", action="store_true")
    c.add_argument("--skip-gqe", action="store_true")
    c.add_argument("--skip-gqkae", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "train":
        cfg = TrainConfig(
            bond_length=args.R,
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
        res = train_geometry(cfg)
        print(
            f"{res.backbone} best={res.best_energy:.8f}  CASCI={res.casci_energy:.8f}  "
            f"err={1e3*(res.best_energy-res.casci_energy):.3f} mHa  params={res.n_params}"
        )
        return

    if args.cmd == "pes":
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        Rs = np.linspace(args.r_min, args.r_max, args.n_points)
        refs = reference_energies(Rs)
        energies = []
        for i, R in enumerate(Rs):
            cfg = TrainConfig(
                bond_length=float(R),
                n_iters=args.n_iters,
                group_size=args.group_size,
                shots=args.shots,
                seq_len=args.seq_len,
                seed=args.seed + i,
                backbone=args.backbone,
                out_dir=str(out / f"R_{R:.3f}"),
            )
            res = train_geometry(cfg)
            energies.append(res.best_energy)
            print(
                f"R={R:.3f}  {args.backbone.upper()}={res.best_energy:.6f}  "
                f"CASCI={res.casci_energy:.6f}  "
                f"Δ={1e3*(res.best_energy-res.casci_energy):.2f} mHa"
            )
        payload = {
            "R": Rs.tolist(),
            "HF": refs["HF"].tolist(),
            "CCSD": refs["CCSD"].tolist(),
            "CASCI": refs["CASCI"].tolist(),
            args.backbone.upper(): energies,
        }
        import json

        with open(out / "pes.json", "w") as f:
            json.dump(payload, f, indent=2)

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(Rs, refs["HF"], ":", color="0.45", label="HF")
            ax.plot(Rs, refs["CCSD"], "--", color="0.45", label="CCSD")
            ax.plot(Rs, refs["CASCI"], "-", color="k", label="CASCI")
            marker = "s" if args.backbone == "gqkae" else "o"
            color = "#6b3fa0" if args.backbone == "gqkae" else "#2e8b57"
            ax.plot(Rs, energies, marker, color=color, label=args.backbone.upper())
            ax.set_xlabel("Bond length R (Å)")
            ax.set_ylabel("Energy (Ha)")
            ax.set_title(r"H$_4$ (8 qubits) — Fig. 4a style PES")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "fig4a_h4.png", dpi=160)
            print(f"wrote {out / 'fig4a_h4.png'}")
        except Exception as exc:  # noqa: BLE001
            print(f"plot skipped: {exc}")
        return

    if args.cmd == "compare":
        defaults = {
            "smoke": "runs/h4_smoke",
            "laptop": "runs/h4_laptop",
            "nano": "runs/h4_nano",
            "paperish": "runs/h4_paperish",
        }
        out_dir = args.out_dir or defaults[args.profile]
        run_comparison(
            profile=args.profile,
            out_dir=out_dir,
            r_min=args.r_min,
            r_max=args.r_max,
            n_points=args.n_points,
            n_seeds=args.n_seeds,
            skip_vqe=args.skip_vqe,
            skip_gqe=args.skip_gqe,
            skip_gqkae=args.skip_gqkae,
        )


if __name__ == "__main__":
    main()
