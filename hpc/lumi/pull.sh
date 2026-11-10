#!/bin/bash
# From the Mac, in the repo root: bash hpc/lumi/pull.sh STEM [STEM ...]
# Copies the LUMI outputs into data_lumi/<stem>/ (NOT data/, so no local result is ever overwritten; --ignore-existing as well).
# Check them (e.g. against the Mac run), then move what you want into data/<stem>/ by hand. Also pulls the Slurm logs to data_lumi/logs/.
set -euo pipefail
source "$(dirname "$0")/config.sh"
[ $# -ge 1 ] || { echo "usage: $0 STEM [STEM ...]"; exit 1; }

for stem in "$@"; do
    stem=${stem%/}; stem=${stem#data/}
    mkdir -p "data_lumi/$stem"
    rsync -e "$RSYNC_RSH" -avP --ignore-existing \
        --include='neuron_segmentation*.tiff' --include='neuron_labels*.tiff' \
        --include='kmeans_*' --include='morphologies*/***' --exclude='*' \
        "$LUMI_USER@$LUMI_HOST:$WORK/data/$stem/" "data_lumi/$stem/"
done
mkdir -p data_lumi/logs
rsync -e "$RSYNC_RSH" -avP --ignore-existing "$LUMI_USER@$LUMI_HOST:$WORK/logs/" data_lumi/logs/
