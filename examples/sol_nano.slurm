#!/bin/bash
#SBATCH -J gqkae-sk-nano
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/sk_nano.%j.out
#SBATCH -e logs/sk_nano.%j.err
#
# ASU Sol nano comparison on random SK spin glasses (16 spins).
# Adjust partition/account/qos to your allocation, e.g.:
#   #SBATCH -p general
#   #SBATCH -q public
#   #SBATCH -A <your_account>
#
# Usage (from the repo root on a Sol login node):
#   mkdir -p logs
#   sbatch examples/sol_nano.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Load a Python 3.11+ env if the default is too old, e.g.:
#   module load mamba/latest
#   source .venv/bin/activate

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${PWD}/.mplconfig"
mkdir -p "$MPLCONFIGDIR" logs

python -m gqkae.cli compare --profile nano --out-dir runs/nano
