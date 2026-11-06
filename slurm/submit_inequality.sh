#!/bin/bash
#SBATCH --job-name=dmurd-ineq
#SBATCH --partition=day
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --time=6:00:00
#SBATCH --output=logs/ineq-%j.out
#SBATCH --requeue

# Goal 3 / H4 supplement: three mean-preserving spreads of initial wealth crossed
# with the rank and median-ratio desperation rules (9 cells x 100 replications).
# Writes <results>/inequality_sweep/summary.csv.
#
# Submit with:  sbatch slurm/submit_inequality.sh

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"

export DMURD_RESULTS_DIR="$RESULTS_ROOT/final_batch_run_v2"
export DMURD_RAW=0

python batch_run_inequality.py
echo "finished    $(date)"
