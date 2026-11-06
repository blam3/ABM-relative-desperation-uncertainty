# Shared setup for the DMURD batch jobs on Bouchet. Sourced by the submit scripts.
# Edit RESULTS_ROOT (or export DMURD_RESULTS_ROOT before submitting) if you keep
# output somewhere other than scratch.

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

module reset
module load miniconda
# `conda activate` needs the shell hook, which a non-interactive batch shell does
# not get from ~/.bashrc. Source it if the module did not already define it.
if ! declare -f conda >/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "${DMURD_CONDA_ENV:-dmurd}"

# One BLAS thread per process: the model parallelizes with multiprocessing, and
# threaded BLAS on top of that oversubscribes the cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Scratch is the right place for the per-agent output (10 TiB per group, purged
# after 30 days). Check your own path with: ls -l $HOME | grep -i scratch
RESULTS_ROOT="${DMURD_RESULTS_ROOT:-$HOME/scratch/dmurd}"
mkdir -p "$RESULTS_ROOT"

echo "host        $(hostname)"
echo "job         ${SLURM_JOB_ID:-none} array task ${SLURM_ARRAY_TASK_ID:-none}"
echo "cpus        ${SLURM_CPUS_PER_TASK:-?}"
echo "python      $(which python)"
echo "results     $RESULTS_ROOT"
echo "started     $(date)"
