#!/bin/bash
# From the Mac, in the repo root: bash hpc/lumi/push.sh STEM [STEM ...]
# Sends only the reconstruction inputs of each image to LUMI scratch, in the same data/ layout:
#   data/imgs/<image_path>  (original TIFF: Tuj1 is re-read from it for Mech images)
#   data/<stem>/processed_image.npy, processed_metadata.npy, soma_mask.tiff, somas_data.npy
# Old outputs are not sent, so segment_neurons.py's refuse-to-overwrite checks pass on LUMI.
set -euo pipefail
source "$(dirname "$0")/config.sh"
[ $# -ge 1 ] || { echo "usage: $0 STEM [STEM ...]"; exit 1; }
PY=${PY:-/Users/carlos/micromamba/envs/vadis/bin/python}   # any Python with numpy (reads image_path from the metadata)

list=$(mktemp)
for stem in "$@"; do
    stem=${stem%/}; stem=${stem#data/}
    for f in processed_image.npy processed_metadata.npy soma_mask.tiff somas_data.npy; do
        [ -f "data/$stem/$f" ] || { echo "missing data/$stem/$f (mark and save the somas in Image_Processor.py first)"; exit 1; }
        echo "$stem/$f" >> "$list"
    done
    img=$($PY -c "import sys, numpy as np; print(np.load(sys.argv[1], allow_pickle=True).item()['image_path'])" \
          "data/$stem/processed_metadata.npy")
    [ -n "$img" ] && [ -e "data/imgs/$img" ] || { echo "missing data/imgs/$img"; exit 1; }
    echo "imgs/$img" >> "$list"
done
echo "Sending:"; sed 's/^/  data\//' "$list"
rsync -avPL --files-from="$list" data/ "$LUMI_USER@$LUMI_HOST:$WORK/data/"
rsync -avP "$(dirname "$0")/manifest.tsv" "$LUMI_USER@$LUMI_HOST:$WORK/manifest.tsv"
rm -f "$list"
