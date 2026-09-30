# GQE / GQKAE for Ising and QUBO

A small generative circuit search built on [Generative Quantum-inspired Kolmogorov–Arnold Eigensolver](https://arxiv.org/abs/2605.04604) (Lin et al.), pointed at combinatorial optimization instead of molecules.

The one-sentence version: **train a tiny GPT whose vocabulary is quantum gates, and reward it when the circuit it writes, once measured, puts its weight on low-energy bitstrings.** GQKAE is that, with the MLP inside each transformer block swapped for a slimmer KAN-style block (HQKAN with DARUAN activations). GQE is the plain-MLP version.

The H₄ molecular version of this code lives on the `qchem/h4-ground-state` branch.

## The problem

Any QUBO `min_x xᵀQx + c` over `x ∈ {0,1}ⁿ`, or equivalently an Ising model

```
E(s) = offset + Σ_i h_i s_i + Σ_{i<j} J_ij s_i s_j,     s_i = 1 − 2 x_i ∈ {±1}
```

A measured bit *is* the QUBO variable, and `H = offset + Σ h_i Z_i + Σ J_ij Z_i Z_j` is diagonal, so every shot is a candidate solution with an exact, cheap energy.

Bring your own instance as `.npy` (the `Q` matrix) or `.json` (`{"Q", "const"}` or `{"h", "J", "offset"}`):

```python
from gqkae import IsingProblem, TrainConfig, load_qubo, train

problem = IsingProblem(load_qubo("my_problem.json"), cvar_alpha=0.1)
res = train(problem, TrainConfig(n_iters=60, shots=128, backbone="gqkae"))
res.best_energy, res.best_index   # best bitstring ever measured
```

## How one training step works

1. The transformer samples a group of 20-token sequences. Each token is one fixed-angle gate from the pool (39 tokens at n = 14):
   - **cost layer** `exp(−iγ H/‖H‖)` for γ ∈ {0.1, 0.2, 0.4, 0.8}, with H scaled so its largest coefficient is 1
   - **mixer** `exp(−iβ Σ X_i)` for β ∈ {±0.1, ±0.2, ±0.4}
   - **local bias** `RY(±π/4)` on each qubit

   Any discrete-angle QAOA circuit up to depth 10 is one sequence, but the model is free to pick depths, angles and local tweaks per instance.
2. Each sequence is a circuit applied to `|+⟩ⁿ`.
3. Measure it `shots` times; each bitstring's energy comes straight from the diagonal H.
4. **Score** = CVaR_α of those energies (mean of the best α-fraction of shots). This is the diagonal-H version of the paper's QSCI reward: QSCI diagonalizes H on the sampled bitstrings, and for a diagonal H that is just the minimum. CVaR with α = 0.1 is smoother than the raw minimum and gives GRPO something to rank.
5. **GRPO**: rank circuits against their group mates, then take a PPO-style clipped step on the token log-probs.

The answer to the QUBO is the lowest-energy bitstring measured at any point during training, not the expected energy of a circuit.

## What's being compared

`gqkae compare` runs every method on random Sherrington–Kirkpatrick spin glasses (`J_ij ~ N(0, 1/n)`), scored the same way: best energy found minus the brute-force ground energy, and how often it hit the ground state exactly.

| Method | What it does |
|---|---|
| exact | Brute force over all 2ⁿ bitstrings. **This is the label.** |
| random | Uniform bitstrings, **same shot budget as one generator run**. |
| SA | Simulated annealing, 32 parallel chains. |
| QAOA | Depth-2 QAOA, COBYLA on exact ⟨H⟩, then sampled with the same budget. |
| GQE | Transformer (MLP blocks) + GRPO. |
| GQKAE | Transformer (HQKAN blocks) + GRPO. |

**The random baseline is the one to watch.** Starting from `|+⟩ⁿ`, an untrained generator is already a uniform sampler. If `n_iters × group_size × shots` approaches 2ⁿ, random sampling finds the ground state too and the comparison is meaningless (the same trap as H₄, where 36 valid determinants made every method look perfect). The profiles keep the budget well under 2ⁿ.

## Layout

```
src/gqkae/
  problem.py     the Problem protocol: pool, initial state, score(counts)
  gates.py       fixed-angle gate tokens
  circuit.py     one Catalyst-compiled lightning.qubit circuit per sequence length
  model.py       transformer with MLP (GQE) or HQKAN (GQKAE) blocks
  grpo.py        the RL loss
  train.py       generic GRPO loop over any Problem
  ising.py       Ising/QUBO model, conversions, instances, loader, pool, CVaR
  baselines.py   random, simulated annealing, QAOA
  profiles.py    smoke / laptop / nano budgets
  compare.py     the SK head-to-head
examples/
  run_compare.py
  sol_nano.slurm
tests/
```

Nothing outside `ising.py`, `baselines.py` and `compare.py` knows what an Ising model is. A new problem type only needs to implement `Problem`.

## Why it's fast

Circuits run on `lightning.qubit` and are compiled with [Catalyst](https://docs.pennylane.ai/projects/catalyst/en/stable/) `@qjit`. The token sequence is an *input* to one compiled program: each step looks up a local token's wires and angle in small tables and applies it on those (dynamic) wires with the other angles zeroed, and the handful of layer tokens run under a `cond` on the token id. It compiles once per sequence length, so the model can write a different circuit every time for free.

## Why the pool is layers, not gates

The first version used one fixed-angle `ZZ(θ)` token per coupling. That can't work: a single QAOA cost layer at n = 14 is 91 `ZZ` gates with J-weighted angles, so a 20-token circuit could barely move off the uniform distribution, the policy never saw a reward gradient, and GQE/GQKAE behaved exactly like random sampling. Putting the whole (normalized) cost Hamiltonian into one token fixed that immediately: on an n = 14 SK instance the best circuits reach CVaR₀.₁ ≈ −8.5 against −4.4 for uniform sampling (E₀ = −9.33), and the mean circuit improves steadily during training.

## First results (laptop profile, ~70 s)

n = 14 SK, 3 instances × 2 seeds, 6400 shots per generator run. Cell = mean gap to E₀ / fraction of runs that hit E₀ exactly.

| instance | E₀ | random | SA | QAOA p=2 | GQE | GQKAE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | −9.331 | 0.080 / 50% | 0 / 100% | 0.199 / 50% | 0 / 100% | 0.199 / 50% |
| 1 | −7.396 | 0.089 / 0% | 0 / 100% | 0.045 / 50% | 0 / 100% | 0 / 100% |
| 2 | −7.573 | 0 / 100% | 0 / 100% | 0.040 / 50% | 0 / 100% | 0.040 / 50% |

Six runs per method is anecdote, not evidence, and SA is very strong at this size. The interesting regime is larger n (and your own instances), where the shot budget covers a vanishing fraction of 2ⁿ.

## Run it

```bash
python3.13 -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
```

```bash
gqkae train --sk 14 --backbone gqkae              # one random SK instance
gqkae train --qubo my_problem.json --no-exact     # your instance (skip brute force past ~22 vars)

gqkae compare --profile smoke     # seconds, checks the plumbing
gqkae compare --profile laptop    # n=14, 3 instances × 2 seeds
gqkae compare --profile nano      # n=16, 5 instances × 3 seeds
```

On ASU Sol: `mkdir -p logs && sbatch examples/sol_nano.slurm`. Outputs land in `runs/<profile>/`: `results.json`, `summary.md`, and one folder per generator run.

## How we know it's right

Each of these is a test in `tests/test_ising.py`:

- QUBO → Ising → QUBO gives identical energies on every bitstring
- the precomputed energy vector, weighted by the circuit's probabilities, equals `qml.expval(H)` on `default.qubit` (so bit ordering is right)
- the Catalyst-compiled circuit matches a plain `default.qubit` circuit to 1e-10
- the empty circuit is exactly uniform; CVaR reduces to the mean at α = 1 and the minimum as α → 0
- max-cut on a 4-cycle has optimum −4 at `0101` / `1010`
- SA's final states (4 chains) find the n = 14 SK ground state; QAOA's ⟨H⟩ sits between E₀ and the uniform mean
- a 3-iteration training run on n = 6 finds the ground state
