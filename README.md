# gqe (H₄ branch)

A small, readable reimplementation of the H₄ experiment from [Generative Quantum-inspired Kolmogorov–Arnold Eigensolver](https://arxiv.org/abs/2605.04604) (Lin et al.).

Here's the whole idea in one sentence: **train a tiny GPT whose vocabulary is quantum gates, and reward it when the circuit it writes has low energy.** That's it. GQKAE is the same thing with the MLP inside each transformer block swapped for a slimmer KAN-style block. GQE is the plain-MLP version.

We only do H₄ (8 qubits), and that's on purpose. At 8 qubits the exact answer costs microseconds, so every number the model produces can be checked against ground truth. I'd much rather have one molecule I fully trust than six I don't. (The QUBO / Ising version of this code lives on the `qubo-ising` branch.)

## quick start

```bash
python3.13 -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
```

Train one model at one geometry:

```bash
gqe-h4 train --R 1.0 --backbone gqkae
gqe-h4 train --R 1.0 --backbone gqe
```

Run everything across bond lengths:

```bash
gqe-h4 compare --profile smoke     # ~1 min, just checks the plumbing
gqe-h4 compare --profile laptop    # a few minutes
gqe-h4 compare --profile nano      # ~15 min on a laptop, 7 R points × 3 seeds
```

On ASU Sol: `mkdir -p logs && sbatch examples/sol_h4_nano.slurm`. Outputs land in `runs/<profile>/`: `pes.json`, `fig4a_pes.png`, `fig5a_error.png`, `summary.md`.

If you've never touched chemistry, start with the notebook: [`notebooks/gqe_h4_demo.ipynb`](notebooks/gqe_h4_demo.ipynb). It walks through all of it assuming you know Hamiltonians but not molecules.

## what's being compared

| method | what it does |
|---|---|
| HF | Best single bitstring (mean-field). Breaks when bonds stretch. |
| CCSD | Classical correction on top of HF. Great near equilibrium. |
| CASCI | Diagonalize the 36×36 Hamiltonian exactly. **This is the label.** |
| VQE | Fixed UCCSD circuit, tune 26 angles with COBYLA. |
| GQE | Transformer picks a 20-gate sequence, trained with GRPO on a QSCI reward. |
| GQKAE | GQE, but the feed-forward MLP is replaced by HQKAN (DARUAN activations). |

"Chemical accuracy" means landing within 1.6 mHa of CASCI.

## one training step

1. The transformer samples a group of 20-token sequences. Each token is a gate from the UCCSD pool (27 choices, angles frozen at π/2).
2. Each sequence becomes a circuit, applied to the HF bitstring `11110000`.
3. Measure the circuit N times and keep the valid bitstrings (2 spin-up + 2 spin-down electrons).
4. **QSCI**: diagonalize H restricted to those bitstrings. The lowest eigenvalue is the score, and reward = −energy. So the circuit's job is just to *propose* the right bitstrings; linear algebra does the rest.
5. **GRPO**: rank circuits against their group mates, then take a PPO-style clipped step on the token log-probs. No critic.

## why it's fast

Circuits run on `lightning.qubit`, compiled with [Catalyst](https://docs.pennylane.ai/projects/catalyst/en/stable/) `@qjit`. The annoying part is that the model writes a different circuit every time, and you really don't want to recompile per circuit. The trick is to make the token sequence an *input* to one compiled program: at each step every gate gets applied, with its angle zeroed unless it's the chosen token. A rotation by zero does nothing, so the circuit is exact. Compile once, then a few ms per circuit. Transformer sampling runs inside a single `jax.jit`. Together these took one training run from ~35 s to ~3 s.

## files

```
src/gqe/
  molecule.py    H₄ geometry, PySCF integrals, HF/CCSD/CASCI, qubit Hamiltonian
  operators.py   the UCCSD gate pool (the "vocabulary")
  circuit.py     Catalyst-compiled lightning.qubit circuits
  qsci.py        bitstrings → subspace → eigenvalue
  model.py       transformer with MLP (GQE) or HQKAN (GQKAE) blocks
  grpo.py        the RL loss
  train.py       one geometry, one seed
  vqe.py         UCCSD + COBYLA, compiled
  profiles.py    smoke / laptop / nano / paperish budgets
  compare.py     the full Fig. 4(a)/5(a) comparison
examples/
  run_h4_compare.py
  sol_h4_nano.slurm
notebooks/
  gqe_h4_demo.ipynb
tests/test_h4.py
```

## tests

If you can't check it, you can't trust it. Each of these is a test in `tests/test_h4.py`:

- empty circuit → QSCI returns exactly HF
- hand QSCI all 36 valid bitstrings → exactly CASCI
- the Catalyst-compiled circuit matches a plain `default.qubit` circuit to 1e-10
- the qubit Hamiltonian's expectation on `11110000` equals the HF energy
- VQE at θ = 0 is HF, and it never goes below CASCI

## paper vs this repo

| | paper | laptop | nano |
|---|---|---|---|
| System | H₄ (4e,4o), 6-31G, 8 qubits | same | same |
| Sequence length | 20 | 20 | 20 |
| Shots / iters / seeds | 10⁵ / 100 / 5 | 512 / 20 / 1 | 8192 / 70 / 3 |
| Model | GPT-2 scale, ~14M (GQKAE) vs ~55M (GQE) | 16k vs 27k | 84k vs 203k |
| Simulator | CUDA-Q | Catalyst + lightning.qubit | same |

## results

Nano profile, 7 bond lengths × 3 seeds, ~13 min on a laptop. Each cell is |E − E_CASCI| in mHa; 0 means equal to machine precision.

| R (Å) | HF | CCSD | VQE | GQE | GQKAE |
|---:|---:|---:|---:|---:|---:|
| 0.80 | 16.2 | 0.04 | 0.05 | 0 | 0 |
| 1.27 | 54.7 | 0.03 | 0.07 | 0 | 0 |
| 1.73 | 119.8 | 1.24 | 1.03 | 0 | 0 |
| 2.20 | 203.8 | 9.50 | 2.49 | 0 | 0 |

HF falls apart as the chain stretches. CCSD and one-layer UCCSD-VQE leave chemical accuracy past ~1.8 Å. GQE and GQKAE nail it everywhere.

**Now, please don't get excited about that last column.** With 8192 shots, 30–40% of *random* 20-gate circuits already reach chemical accuracy on H₄. There are only 36 valid bitstrings and the shots find most of them. Training looks at ~560 circuits and keeps the best, so it basically can't miss. Every seed hits 1.6 mHa at iteration 0, which matches the paper's "nearly flat" H₄ curve. The policy does learn something (the average circuit goes from ~18 to ~12 mHa error), but H₄ is a correctness check, not a benchmark, and it can't tell GQE from GQKAE. The paper's interesting cases are N₂ and LiH, which are out of scope here.

Things we don't claim: the paper's wall-clock numbers, its exact parameter counts, or anything about the other five molecules.

## license

Apache-2.0
