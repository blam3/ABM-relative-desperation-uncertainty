#!/bin/bash
#SBATCH --job-name=dmurd-l05
#SBATCH --partition=day
#SBATCH --array=1-45%12
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2:00:00
#SBATCH --output=logs/sweep-l05-%A_%a.out
#SBATCH --requeue

# Main parameter sweep re-run with lambda = 0.5 (default is 0.2). At lambda >= 0.5
# the desperate offending rate rises monotonically with the desperation threshold nu
# (see analyze_lambda_sweep.py output). Summaries only, no per-agent rows.
#
# Submit with:  sbatch slurm/submit_main_sweep_lambda05.sh

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"

export DMURD_RESULTS_DIR="$RESULTS_ROOT/final_batch_run_lambda0.5"
export DMURD_RAW=0
export DMURD_LAMBDA=0.5

python batch_run_hpc.py
echo "finished    $(date)"
