#!/bin/bash
#SBATCH --job-name=dmurd-nbr
#SBATCH --partition=day
#SBATCH --array=1-75%12
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0:30:00
#SBATCH --output=logs/nbr-%A_%a.out
#SBATCH --requeue

# Comparison group size sensitivity for Supplementary Text 2.
# n in {5, 10, 20, 30, 48} x beta in {-1, 0, +1} x nu in {.05, .20, .35, .50, .65}
# alpha = 1, 15x15 grid, 20 replications, 25 steps.
#
# Submit with:  sbatch slurm/submit_neighbor_sensitivity.sh

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"

export DMURD_RESULTS_DIR="$RESULTS_ROOT/final_batch_run_lambda0.5"
export DMURD_RAW=0
export DMURD_LAMBDA=0.5
export DMURD_SENSITIVITY_SUFFIX="_lambda0.5"

python batch_run_neighbor_sensitivity.py
echo "finished    $(date)"
