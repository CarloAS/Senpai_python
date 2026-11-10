#!/bin/bash
# One array task = one (image, version) line of manifest.tsv: segmentation + separation (segment_neurons.py), then SWC skeletons.
# Submit with submit.sh (sets --account and --array). --partition/--time/--mem/--cpus-per-task can be overridden on the sbatch line.
#SBATCH --job-name=senpai
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
source "${CONFIG:?submit with submit.sh}"
cd "$WORK"

line=$(grep -v '^\s*#' manifest.tsv | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
read -r STEM VERSION <<< "$line"
case "$VERSION" in
    map2)      FLAG=--map2-only; TAG=_sigma2 ;;
    map2-tuj1) FLAG=;            TAG=_sigma2_map2-tuj1 ;;
    *) echo "unknown version '$VERSION' in manifest line $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac
echo "$(date '+%F %T') task $SLURM_ARRAY_TASK_ID: $STEM $VERSION on $(hostname), $SLURM_CPUS_PER_TASK CPUs"

export PATH="$ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
# k-means blocks run in $SLURM_CPUS_PER_TASK joblib workers (1 OpenMP thread each, set by joblib).
# Note: sklearn KMeans labels depend slightly on the thread count, so results differ from a Mac run at the ~0.3% level (Dice ~0.997).
python "$CODE/segment_neurons.py" "data/$STEM/" --sigma 2 $FLAG --n-jobs "$SLURM_CPUS_PER_TASK"
echo "$(date '+%F %T') segmentation done"
python "$CODE/NeuronSkeletonizer.py" "data/$STEM/" --tag "$TAG"
echo "$(date '+%F %T') skeletons done"
