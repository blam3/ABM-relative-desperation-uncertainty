#!/bin/bash
#SBATCH --job-name=dmurd-sens05
#SBATCH --partition=day
#SBATCH --array=1-18
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --time=6:00:00
#SBATCH --output=logs/sens-l05-%A_%a.out
#SBATCH --requeue

# Softmax and 50-step convergence sensitivity re-runs at lambda = 0.5.
# Tasks 1-9: tau x theta at beta = 0, alpha = 1, nu = 0.35, 25 steps
# Tasks 10-18: 50-step convergence across beta x nu at alpha = 1
#
# Submit with:  sbatch slurm/submit_sensitivity_lambda05.sh

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"

export DMURD_RESULTS_DIR="$RESULTS_ROOT/final_batch_run_lambda0.5"
export DMURD_RAW=0
export DMURD_LAMBDA=0.5
export DMURD_SENSITIVITY_SUFFIX="_lambda0.5"

python batch_run_sensitivity.py
echo "finished    $(date)"
