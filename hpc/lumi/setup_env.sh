#!/bin/bash
# Run ONCE on LUMI (login node), from the repo clone: bash hpc/lumi/setup_env.sh
# Builds the headless Python env with the LUMI container wrapper (a plain conda env on Lustre would hit the file-count quota).
set -euo pipefail
source "$(dirname "$0")/config.sh"
REPO=$(cd "$(dirname "$0")/../.." && pwd)        # the clone this script is in
if [ "$REPO" != "$CODE" ]; then
    echo "WARNING: this clone is $REPO but config.sh has CODE=$CODE (the Slurm jobs run the scripts from \$CODE)."
    echo "         Move the clone there, or set CODE in hpc/lumi/config.sh to $REPO."
fi

module load LUMI lumi-container-wrapper
if [ -e "$ENV" ]; then
    echo "$ENV exists, not rebuilding (delete it by hand to rebuild)"; exit 1
fi
mkdir -p "$(dirname "$ENV")"
conda-containerize new --prefix "$ENV" "$REPO/envs/senpai_lumi.yml"
mkdir -p "$WORK/data/imgs" "$WORK/logs"

"$ENV/bin/python" -c "import numpy, scipy, skimage, sklearn, tifffile, joblib; print('env ok: numpy', numpy.__version__, 'sklearn', sklearn.__version__)"
"$ENV/bin/python" "$REPO/segment_neurons.py" --help > /dev/null && echo "segment_neurons.py imports ok"
