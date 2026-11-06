#!/bin/bash
#SBATCH --job-name=dmurd-smoke
#SBATCH --partition=devel
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0:30:00
#SBATCH --output=logs/smoke-%j.out

# Five-minute check that the environment, the paths, and the output format work
# before committing the real sweep: one condition, 30 x 30 grid, 4 replications.
#
# Submit with:  sbatch slurm/smoke_test.sh
# Then check:   ls $HOME/scratch/dmurd/smoke_test

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"

export DMURD_RESULTS_DIR="$RESULTS_ROOT/smoke_test"
export DMURD_GRID=30
export DMURD_REPLICATIONS=4
export DMURD_STEPS=10
export DMURD_GZIP=1
export SLURM_ARRAY_TASK_ID=21   # condition 21: beta = 0, alpha = 1, threshold = .05

python check_environment.py
python batch_run_hpc.py
python aggregate_results.py --results "$DMURD_RESULTS_DIR" --out "$DMURD_RESULTS_DIR/analysis"
echo "finished    $(date)"
