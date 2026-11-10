#!/bin/zsh
# Mac batch run, SEQUENTIAL (one segment_neurons run needs ~9 GB RAM: never run two at once on the 16 GB Mac).
# Usage, from the repo root:  zsh scripts/run_local.sh hpc/lumi/manifest.tsv [logdir]
# Same manifest format as LUMI (<stem> <version>, version map2 | map2-tuj1). Per line: segment -> skeletonize -> BSB/NEST network + HTML.
# Generalized from tmp/mech_runs/run_all.sh (2026-11-07).
PY=/Users/carlos/micromamba/envs/vadis/bin/python
NEST=/Users/carlos/micromamba/envs/vadis-nest/bin/python
MANIFEST=${1:?usage: run_local.sh MANIFEST [LOGDIR]}
L=${2:-tmp/runs}
mkdir -p $L
grep -v '^\s*#' $MANIFEST | grep -v '^\s*$' | while read -r stem v; do
  D=data/$stem/
  case $v in
    map2)      tag=_sigma2; flag=--map2-only ;;
    map2-tuj1) tag=_sigma2_map2-tuj1; flag= ;;
    *) echo "unknown version $v"; continue ;;
  esac
  echo "$(date '+%F %T') START $stem $v"
  $PY -u segment_neurons.py $D --sigma 2 ${=flag} > $L/seg_${stem}_$v.log 2>&1 \
    || { echo "$(date '+%F %T') FAILED segmentation $stem $v"; continue; }
  echo "$(date '+%F %T') segmented $stem $v: $(grep -E '^Separation' $L/seg_${stem}_$v.log)"
  $PY -u NeuronSkeletonizer.py $D --tag $tag > $L/skel_${stem}_$v.log 2>&1 \
    || { echo "$(date '+%F %T') FAILED skeleton $stem $v"; continue; }
  echo "$(date '+%F %T') skeletons $stem $v: $(tail -1 $L/skel_${stem}_$v.log)"
  BSB_IGNORE_PARALLEL=1 $NEST -u bsb_network.py $D --tag $tag > $L/net_${stem}_$v.log 2>&1 \
    || { echo "$(date '+%F %T') FAILED network $stem $v"; continue; }
  echo "$(date '+%F %T') network $stem $v done"
done
echo "$(date '+%F %T') ALL DONE"
