#!/bin/bash
#SBATCH --job-name=dmurd-lambda
#SBATCH --partition=day
#SBATCH --array=1-25%12
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1:00:00
#SBATCH --output=logs/lambda-%A_%a.out
#SBATCH --requeue

# Lambda sweep: lambda in {0.05, 0.1, 0.2, 0.5, 1.0} x threshold in {0.05, 0.20, 0.35, 0.50, 0.65}
# Fixed at beta = 0, alpha = 1 (no distortion) so the effect isolates the
# reference-shift mechanism. Summaries only.
#
# Submit with:  sbatch slurm/submit_lambda_sweep.sh

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"

export DMURD_RESULTS_DIR="$RESULTS_ROOT/final_batch_run_v2"
export DMURD_RAW=0

python batch_run_lambda_sweep.py
echo "finished    $(date)"
