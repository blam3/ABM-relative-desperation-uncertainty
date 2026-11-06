#!/bin/bash
#SBATCH --job-name=dmurd-sens
#SBATCH --partition=day
#SBATCH --array=1-18
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --time=6:00:00
#SBATCH --output=logs/sens-%A_%a.out
#SBATCH --requeue

# Tasks 1-9: softmax robustness (tau x theta) at beta = 0, alpha = 1, threshold = .35.
# Tasks 10-18: 50-step convergence check across beta and threshold.
# Summaries only; the per-agent rows are not needed for these analyses.
#
# Submit with:  sbatch slurm/submit_sensitivity.sh

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"

export DMURD_RESULTS_DIR="$RESULTS_ROOT/final_batch_run_v2"
export DMURD_RAW=0

python batch_run_sensitivity.py
echo "finished    $(date)"
