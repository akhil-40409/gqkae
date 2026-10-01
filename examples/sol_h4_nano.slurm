#!/bin/bash
#SBATCH -J gqe-h4-nano
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/h4_nano.%j.out
#SBATCH -e logs/h4_nano.%j.err
#
# ASU Sol nano comparison for linear H4 (8 qubits).
# Adjust partition/account/qos to your allocation, e.g.:
#   #SBATCH -p general
#   #SBATCH -q public
#   #SBATCH -A <your_account>
#
# Usage (from the repo root on a Sol login node):
#   mkdir -p logs
#   sbatch examples/sol_h4_nano.slurm

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

python -m gqe.cli compare --profile nano --out-dir runs/h4_nano
