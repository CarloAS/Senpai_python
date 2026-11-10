#!/bin/bash
# On LUMI: bash $CODE/hpc/lumi/submit.sh [extra sbatch options]
# Submits one array task per non-comment line of $WORK/manifest.tsv. Extra options override the #SBATCH defaults, e.g.
#   bash submit.sh --partition=debug --time=00:30:00 --mem=32G --cpus-per-task=16     (quick test)
#   bash submit.sh --array=1-4%2                                                       (only some lines / at most 2 at once)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/config.sh"
n=$(grep -v '^\s*#' "$WORK/manifest.tsv" | grep -vc '^\s*$')
[ "$n" -ge 1 ] || { echo "no tasks in $WORK/manifest.tsv"; exit 1; }
mkdir -p "$WORK/logs"
cd "$WORK"
sbatch --account="$PROJECT" --array="1-$n" --export=ALL,CONFIG="$here/config.sh" "$@" "$here/reconstruct.sbatch"
