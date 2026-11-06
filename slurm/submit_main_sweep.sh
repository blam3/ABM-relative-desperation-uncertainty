#!/bin/bash
#SBATCH --job-name=dmurd-sweep
#SBATCH --partition=day
#SBATCH --array=1-45%12
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00
#SBATCH --output=logs/sweep-%A_%a.out
#SBATCH --requeue
## Uncomment and set your Yale address to get mail:
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=netid@yale.edu

# Main parameter sweep: 45 conditions x 100 replications, 10,000 agents, 25 steps.
# One array task per condition. Each task writes
#   summary_cond_<n>_*.csv   one row per replication (a few hundred KB)
#   cond_<n>_*.csv.gz        per-agent rows (about 570 MB gzipped per condition)
#
# Submit with:  sbatch slurm/submit_main_sweep.sh
# Summaries only (about 25 MB total, no per-agent rows):
#   sbatch --export=ALL,DMURD_RAW=0 slurm/submit_main_sweep.sh

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"

export DMURD_RESULTS_DIR="$RESULTS_ROOT/final_batch_run_v2"
export DMURD_GZIP="${DMURD_GZIP:-1}"
export DMURD_RAW="${DMURD_RAW:-1}"

python batch_run_hpc.py
echo "finished    $(date)"
