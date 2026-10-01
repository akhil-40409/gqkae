"""Fig. 4(a)/5(a)-style H₄ comparison: HF, CCSD, CASCI, VQE, GQE, GQKAE."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gqe.molecule import H4System, reference_energies
from gqe.profiles import get_profile
from gqe.train import TrainConfig, train_geometry
from gqe.vqe import run_vqe

CHEMICAL_ACCURACY_MHA = 1.6


def _train_kwargs(profile: dict[str, Any], bond_length: float, seed: int, out_dir: str, backbone: str) -> TrainConfig:
    return TrainConfig(
        bond_length=bond_length,
        basis=profile["basis"],
        n_iters=profile["n_iters"],
        group_size=profile["group_size"],
        seq_len=profile["seq_len"],
        shots=profile["shots"],
        d_max=profile["d_max"],
        d_model=profile["d_model"],
        d_latent=profile["d_latent"],
        d_ff=profile["d_ff"],
        n_layers=profile["n_layers"],
        n_heads=profile["n_heads"],
        backbone=backbone,  # type: ignore[arg-type]
        lr=profile["lr"],
        weight_decay=profile["weight_decay"],
        policy_updates=profile["policy_updates"],
        clip_eps=profile["clip_eps"],
        temperature=profile["temperature"],
        repetition_penalty=profile["repetition_penalty"],
        seed=seed,
        out_dir=out_dir,
        op_angle=profile["op_angle"],
    )


def run_comparison(
    *,
    profile: str = "nano",
    out_dir: str = "runs/h4_nano",
    r_min: float | None = None,
    r_max: float | None = None,
    n_points: int | None = None,
    n_seeds: int | None = None,
    skip_vqe: bool = False,
    skip_gqe: bool = False,
    skip_gqkae: bool = False,
) -> dict[str, Any]:
    """Scan R; write pes.json, PES/error figures, and summary.md."""
    cfg = get_profile(profile)
    if r_min is not None:
        cfg["r_min"] = r_min
    if r_max is not None:
        cfg["r_max"] = r_max
    if n_points is not None:
        cfg["n_points"] = n_points
    if n_seeds is not None:
        cfg["n_seeds"] = n_seeds

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Rs = np.linspace(float(cfg["r_min"]), float(cfg["r_max"]), int(cfg["n_points"]))
    refs = reference_energies(Rs, basis=cfg["basis"])

    vqe_E = []
    vqe_evals = []
    gqe_mean, gqe_std, gqe_params, gqe_time = [], [], [], []
    gqkae_mean, gqkae_std, gqkae_params, gqkae_time = [], [], [], []

    for i, R in enumerate(Rs):
        r = float(R)
        print(f"\n=== R = {r:.3f} Å ===")
        sys = H4System(bond_length=r, basis=cfg["basis"])
        print(f"  HF={sys.hf_energy:.8f}  CCSD={refs['CCSD'][i]:.8f}  CASCI={sys.casci_energy():.8f}")

        if skip_vqe:
            vqe_E.append(float("nan"))
            vqe_evals.append(0)
        else:
            t0 = time.perf_counter()
            vqe = run_vqe(sys, maxiter=int(cfg["vqe_maxiter"]))
            print(
                f"  VQE={vqe.energy:.8f}  Δ={1e3 * (vqe.energy - vqe.casci_energy):.3f} mHa  "
                f"evals={vqe.n_evals}  ({time.perf_counter() - t0:.1f}s)"
            )
            vqe_E.append(vqe.energy)
            vqe_evals.append(vqe.n_evals)

        def _run_gen(backbone: str) -> tuple[list[float], int, float]:
            energies = []
            n_params = 0
            t0 = time.perf_counter()
            for s in range(int(cfg["n_seeds"])):
                train_cfg = _train_kwargs(
                    cfg,
                    r,
                    seed=s + 17 * i,
                    out_dir=str(out / backbone / f"R_{r:.3f}_seed{s}"),
                    backbone=backbone,
                )
                res = train_geometry(train_cfg)
                energies.append(res.best_energy)
                n_params = res.n_params
            elapsed = time.perf_counter() - t0
            return energies, n_params, elapsed

        if skip_gqe:
            gqe_mean.append(float("nan"))
            gqe_std.append(float("nan"))
            gqe_params.append(0)
            gqe_time.append(0.0)
        else:
            es, npar, elapsed = _run_gen("gqe")
            gqe_mean.append(float(np.mean(es)))
            gqe_std.append(float(np.std(es)) if len(es) > 1 else 0.0)
            gqe_params.append(npar)
            gqe_time.append(elapsed)
            print(
                f"  GQE={gqe_mean[-1]:.8f} ± {gqe_std[-1]:.2e}  "
                f"Δ={1e3 * (gqe_mean[-1] - refs['CASCI'][i]):.3f} mHa  "
                f"params={npar}  ({elapsed:.1f}s)"
            )

        if skip_gqkae:
            gqkae_mean.append(float("nan"))
            gqkae_std.append(float("nan"))
            gqkae_params.append(0)
            gqkae_time.append(0.0)
        else:
            es, npar, elapsed = _run_gen("gqkae")
            gqkae_mean.append(float(np.mean(es)))
            gqkae_std.append(float(np.std(es)) if len(es) > 1 else 0.0)
            gqkae_params.append(npar)
            gqkae_time.append(elapsed)
            print(
                f"  GQKAE={gqkae_mean[-1]:.8f} ± {gqkae_std[-1]:.2e}  "
                f"Δ={1e3 * (gqkae_mean[-1] - refs['CASCI'][i]):.3f} mHa  "
                f"params={npar}  ({elapsed:.1f}s)"
            )

    payload: dict[str, Any] = {
        "profile": profile,
        "config": {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in cfg.items()},
        "R": Rs.tolist(),
        "HF": refs["HF"].tolist(),
        "CCSD": refs["CCSD"].tolist(),
        "CASCI": refs["CASCI"].tolist(),
        "VQE": vqe_E,
        "VQE_evals": vqe_evals,
        "GQE_mean": gqe_mean,
        "GQE_std": gqe_std,
        "GQE_params": gqe_params,
        "GQE_wall_s": gqe_time,
        "GQKAE_mean": gqkae_mean,
        "GQKAE_std": gqkae_std,
        "GQKAE_params": gqkae_params,
        "GQKAE_wall_s": gqkae_time,
        "chemical_accuracy_mHa": CHEMICAL_ACCURACY_MHA,
        "notes": (
            "Nano/laptop models are much smaller than the paper GPT-2 HQKANsformer. "
            "Param-count ratios here are not the paper Table II 66% claim. "
            "CASCI is exact FCI in the (4e,4o) active space."
        ),
    }
    with open(out / "pes.json", "w") as f:
        json.dump(payload, f, indent=2)

    _write_plots(out, payload)
    _write_summary(out, payload)
    return payload


def _err_mHa(e, casci):
    e = np.asarray(e, dtype=float)
    casci = np.asarray(casci, dtype=float)
    return np.abs(e - casci) * 1e3


def _write_plots(out: Path, payload: dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"plot skipped: {exc}")
        return

    Rs = np.asarray(payload["R"])
    casci = np.asarray(payload["CASCI"])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(Rs, payload["HF"], ":", color="0.45", label="HF")
    ax.plot(Rs, payload["CCSD"], "--", color="0.45", label="CCSD")
    ax.plot(Rs, casci, "-", color="k", lw=2, label="CASCI")
    if np.isfinite(payload["VQE"]).any():
        ax.plot(Rs, payload["VQE"], "^", color="#2a6fbb", ms=7, label="VQE")
    if np.isfinite(payload["GQE_mean"]).any():
        ax.errorbar(
            Rs,
            payload["GQE_mean"],
            yerr=payload["GQE_std"],
            fmt="o",
            color="#2e8b57",
            ms=6,
            label="GQE",
            capsize=2,
        )
    if np.isfinite(payload["GQKAE_mean"]).any():
        ax.errorbar(
            Rs,
            payload["GQKAE_mean"],
            yerr=payload["GQKAE_std"],
            fmt="s",
            color="#6b3fa0",
            ms=6,
            label="GQKAE",
            capsize=2,
        )
    ax.set_xlabel("Bond length R (Å)")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title(r"H$_4$ (8 qubits) — Fig. 4(a) style PES")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig4a_pes.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(CHEMICAL_ACCURACY_MHA, color="0.4", ls="--", label="chemical accuracy (1.6 mHa)")
    ax.semilogy(Rs, _err_mHa(payload["HF"], casci), ":", color="0.45", label="HF")
    ax.semilogy(Rs, _err_mHa(payload["CCSD"], casci), "--", color="0.45", label="CCSD")
    if np.isfinite(payload["VQE"]).any():
        ax.semilogy(Rs, np.clip(_err_mHa(payload["VQE"], casci), 1e-6, None), "^", color="#2a6fbb", label="VQE")
    if np.isfinite(payload["GQE_mean"]).any():
        ax.semilogy(
            Rs,
            np.clip(_err_mHa(payload["GQE_mean"], casci), 1e-6, None),
            "o",
            color="#2e8b57",
            label="GQE",
        )
    if np.isfinite(payload["GQKAE_mean"]).any():
        ax.semilogy(
            Rs,
            np.clip(_err_mHa(payload["GQKAE_mean"], casci), 1e-6, None),
            "s",
            color="#6b3fa0",
            label="GQKAE",
        )
    ax.set_xlabel("Bond length R (Å)")
    ax.set_ylabel(r"$|E - E_{\mathrm{CASCI}}|$ (mHa)")
    ax.set_title(r"H$_4$ — Fig. 5(a) style absolute error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig5a_error.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out / 'fig4a_pes.png'} and {out / 'fig5a_error.png'}")


def _write_summary(out: Path, payload: dict[str, Any]) -> None:
    casci = np.asarray(payload["CASCI"])
    Rs = np.asarray(payload["R"])
    lines = [
        f"# H₄ {payload['profile']} comparison",
        "",
        f"- Profile: `{payload['profile']}`",
        f"- Bond lengths R (Å): {', '.join(f'{r:.3f}' for r in Rs)}",
        "- Success criterion: |E − E_CASCI| ≤ 1.6 mHa (chemical accuracy).",
        "- CASCI is exact FCI in the (4e, 4o) active space.",
        "",
        "## Mean |error| vs CASCI (mHa)",
        "",
        "| R (Å) | HF | CCSD | VQE | GQE | GQKAE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(Rs):
        def cell(vals):
            v = vals[i]
            if v is None or not np.isfinite(v):
                return "—"
            return f"{abs(v - casci[i]) * 1e3:.3f}"

        lines.append(
            f"| {r:.3f} | {cell(payload['HF'])} | {cell(payload['CCSD'])} | "
            f"{cell(payload['VQE'])} | {cell(payload['GQE_mean'])} | {cell(payload['GQKAE_mean'])} |"
        )
    gqe_p = next((p for p in payload["GQE_params"] if p), 0)
    gqk_p = next((p for p in payload["GQKAE_params"] if p), 0)
    lines += [
        "",
        "## Nano model size (not paper Table II)",
        "",
        f"- GQE params: {gqe_p}",
        f"- GQKAE params: {gqk_p}",
        f"- GQE wall (sum over R, s): {sum(payload['GQE_wall_s']):.1f} s",
        f"- GQKAE wall (sum over R, s): {sum(payload['GQKAE_wall_s']):.1f} s",
        "",
        payload["notes"],
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines))
    print(f"wrote {out / 'summary.md'}")


def train_config_from_profile(profile: str, **overrides) -> TrainConfig:
    cfg = get_profile(profile)
    cfg.update(overrides)
    backbone = cfg.get("backbone", "gqkae")
    return _train_kwargs(
        cfg,
        bond_length=float(cfg.get("bond_length", 1.0)),
        seed=int(cfg.get("seed", 0)),
        out_dir=str(cfg.get("out_dir", "runs/h4")),
        backbone=backbone,
    )
