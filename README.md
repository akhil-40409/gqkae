# gqe

The simplest, fastest repo I could write for training a tiny GPT to write quantum circuits. You give it an Ising model or a QUBO, it writes a circuit, the circuit gets measured, and the model gets rewarded when the bitstrings that come out have low energy. That's it. That's the whole thing.

It's built on [Generative Quantum-inspired Kolmogorov–Arnold Eigensolver](https://arxiv.org/abs/2605.04604) (Lin et al.), except pointed at combinatorial optimization instead of molecules. There are two backbones:

- **GQE**: a vanilla decoder-only transformer. Token embeddings, causal attention, MLP, the usual.
- **GQKAE**: the same thing, but the MLP inside each block is swapped for a slimmer KAN-style block (HQKAN with DARUAN activations). Fewer parameters, same job.

The code is small on purpose. ~1300 lines of Python, and every piece is something you can read in one sitting. If you're looking for the molecular (H₄) version, it lives on the `qchem/h4-ground-state` branch.

## the problem

Any QUBO `min_x xᵀQx + c` over `x ∈ {0,1}ⁿ`. Equivalently, an Ising model:

```
E(s) = offset + Σ_i h_i s_i + Σ_{i<j} J_ij s_i s_j,     s_i = 1 − 2 x_i ∈ {±1}
```

Here's the thing that makes this nice. A measured qubit *is* a QUBO variable, and `H = offset + Σ h_i Z_i + Σ J_ij Z_i Z_j` is diagonal. So every single shot is a candidate solution, and you get its exact energy for free. No estimation, no tomography, just a lookup.

## quick start

```bash
python3.13 -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
```

Train on one random Sherrington–Kirkpatrick instance:

```bash
gqe train --sk 14 --backbone gqkae
```

Or bring your own instance, as `.npy` (the `Q` matrix) or `.json` (`{"Q", "const"}` or `{"h", "J", "offset"}`). Past ~22 variables, skip the brute-force check:

```bash
gqe train --qubo my_problem.json --no-exact
```

Same thing from Python:

```python
from gqe import IsingProblem, TrainConfig, load_qubo, train

problem = IsingProblem(load_qubo("my_problem.json"), cvar_alpha=0.1)
res = train(problem, TrainConfig(n_iters=60, shots=128, backbone="gqkae"))
res.best_energy, res.best_index   # best bitstring ever measured
```

If you'd rather see every piece built from scratch first, start with the notebook: [`notebooks/gqe_nano_demo.ipynb`](notebooks/gqe_nano_demo.ipynb). It does an 8-qubit SK and a 6-qubit TFIM inline, and runs fine on an M4 MacBook CPU.

## one training step

Ok so here's what actually happens, step by step:

1. The transformer samples a group of 20-token sequences. Each token is one fixed-angle "gate" from a small pool (39 tokens at n = 14):
   - **cost layer** `exp(−iγ H/‖H‖)` for γ ∈ {0.1, 0.2, 0.4, 0.8}, where H is scaled so its largest coefficient is 1
   - **mixer** `exp(−iβ Σ X_i)` for β ∈ {±0.1, ±0.2, ±0.4}
   - **local bias** `RY(±π/4)` on each qubit
2. Each sequence becomes a circuit applied to `|+⟩ⁿ`.
3. We measure it `shots` times. Each bitstring's energy comes straight off the diagonal H.
4. **Score** = CVaR_α of those energies, i.e. the mean of the best α-fraction of shots. This is the diagonal-H version of the paper's QSCI reward. QSCI diagonalizes H on the sampled bitstrings, and for a diagonal H that's just the minimum. CVaR at α = 0.1 is a smoother version of the minimum, which gives GRPO something to actually rank.
5. **GRPO**: rank each circuit against its group mates, then take a PPO-style clipped step on the token log-probs. No critic, no value network.

Notice that any discrete-angle QAOA circuit up to depth 10 is just one particular sequence here. The model is free to pick depths, angles, and local tweaks per instance. So in a sense QAOA is a point in the search space, and we're learning to search around it.

One more subtle point: the answer to the QUBO is the lowest-energy bitstring measured *at any point during training*. It's not the expected energy of some final circuit. The circuits are a means to an end.

## why the pool is layers, not gates

I want to call this out because it's the thing that made it work. The first version used one fixed-angle `ZZ(θ)` token per coupling, which seems like the natural thing to do. It doesn't work. A single QAOA cost layer at n = 14 is 91 `ZZ` gates with J-weighted angles. A 20-token circuit could barely nudge the state off the uniform distribution, the policy never saw a reward gradient, and GQE/GQKAE behaved exactly like random sampling.

Putting the whole (normalized) cost Hamiltonian into one token fixed it immediately. On an n = 14 SK instance the best circuits reach CVaR₀.₁ ≈ −8.5 vs −4.4 for uniform sampling (E₀ = −9.33), and the mean circuit improves steadily over training. Lesson: give the model a vocabulary where a single token can actually move the needle.

## why it's fast

Circuits run on `lightning.qubit`, compiled with [Catalyst](https://docs.pennylane.ai/projects/catalyst/en/stable/) `@qjit`. The trick is that the token sequence is an *input* to one compiled program, not something baked into it. At each step, a local token's wires and angle get looked up in small tables and applied on those (dynamic) wires with the other angles zeroed, and the handful of layer tokens run under a `cond` on the token id. It compiles once per sequence length, and after that the model can write a different circuit every time basically for free.

## the comparison

`gqe compare` runs every method on random SK spin glasses (`J_ij ~ N(0, 1/n)`), all scored the same way: best energy found minus the brute-force ground energy, and how often each method hit the ground state exactly.

| method | what it does |
|---|---|
| exact | Brute force over all 2ⁿ bitstrings. **This is the label.** |
| random | Uniform bitstrings, **same shot budget as one generator run**. |
| SA | Simulated annealing, 32 parallel chains. |
| QAOA | Depth-2 QAOA, COBYLA on exact ⟨H⟩, then sampled with the same budget. |
| GQE | Transformer (MLP blocks) + GRPO. |
| GQKAE | Transformer (HQKAN blocks) + GRPO. |

**Keep your eye on the random baseline.** Starting from `|+⟩ⁿ`, an untrained generator is already a uniform sampler. If `n_iters × group_size × shots` gets anywhere near 2ⁿ, random sampling finds the ground state too and the whole comparison is meaningless. (This is exactly the trap H₄ fell into: 36 valid determinants and every method looks perfect.) The profiles keep the budget well under 2ⁿ.

```bash
gqe compare --profile smoke     # seconds, just checks the plumbing
gqe compare --profile laptop    # n=14, 3 instances × 2 seeds
gqe compare --profile nano      # n=16, 5 instances × 3 seeds
```

On ASU Sol: `mkdir -p logs && sbatch examples/sol_nano.slurm`. Outputs land in `runs/<profile>/`: `results.json`, `summary.md`, and one folder per generator run.

## first results

Laptop profile, ~70 s. n = 14 SK, 3 instances × 2 seeds, 6400 shots per generator run. Each cell is mean gap to E₀ / fraction of runs that hit E₀ exactly.

| instance | E₀ | random | SA | QAOA p=2 | GQE | GQKAE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | −9.331 | 0.080 / 50% | 0 / 100% | 0.199 / 50% | 0 / 100% | 0.199 / 50% |
| 1 | −7.396 | 0.089 / 0% | 0 / 100% | 0.045 / 50% | 0 / 100% | 0 / 100% |
| 2 | −7.573 | 0 / 100% | 0 / 100% | 0.040 / 50% | 0 / 100% | 0.040 / 50% |

Let's be honest about this table: six runs per method is an anecdote, not evidence, and SA is very strong at this size. I wouldn't conclude anything from it except "the plumbing works and the generators are in the right ballpark." The interesting regime is larger n (and your own instances), where the shot budget covers a vanishing fraction of 2ⁿ.

## files

```
src/gqe/
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
notebooks/
  gqe_nano_demo.ipynb
tests/
```

Nothing outside `ising.py`, `baselines.py`, and `compare.py` knows what an Ising model is. The trainer only sees a `Problem`. So if you want to point this at something new, you implement `Problem` and you're done.

## tests

I don't trust code I haven't checked against something exact, so each of these is a test in `tests/test_ising.py`:

- QUBO → Ising → QUBO gives identical energies on every bitstring
- the precomputed energy vector, weighted by the circuit's probabilities, equals `qml.expval(H)` on `default.qubit` (so the bit ordering is right, which is the classic place to get burned)
- the Catalyst-compiled circuit matches a plain `default.qubit` circuit to 1e-10
- the empty circuit is exactly uniform; CVaR reduces to the mean at α = 1 and the minimum as α → 0
- max-cut on a 4-cycle has optimum −4 at `0101` / `1010`
- SA's final states (4 chains) find the n = 14 SK ground state; QAOA's ⟨H⟩ sits between E₀ and the uniform mean
- a 3-iteration training run on n = 6 finds the ground state

## license

Apache-2.0
