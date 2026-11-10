# Running the reconstruction on LUMI

Soma marking stays on the Mac (napari GUI). LUMI runs the heavy part, segmentation + separation (`segment_neurons.py`) and
SWC skeletons (`NeuronSkeletonizer.py`), as a Slurm job array: one task per (image, version) line of `manifest.tsv`.
Each task also runs its k-means blocks in parallel (`--n-jobs` = the CPUs of the task).
The network simulation (`bsb_network.py`, ~30 s per image) runs on the Mac with the pulled SWCs.

All paths and the project number live in `config.sh`.

## Once

1. **Mac**: fill in `PROJECT` and `LUMI_USER` in `hpc/lumi/config.sh`, commit, and push the branch to GitHub.
2. **LUMI** (`ssh <user>@lumi.csc.fi`):
   ```bash
   cd /project/<project>                          # code: shared project folder (CODE in config.sh); the env goes to /project/<project>/<user>/envs
   git clone -b bsb_implementation https://github.com/CarloAS/Senpai_python.git
   bash Senpai_python/hpc/lumi/setup_env.sh      # builds the container env (~10-20 min), creates $WORK/data and $WORK/logs
   ```
   The env (`envs/senpai_lumi.yml`) has the same package versions as the Mac env `vadis`, without napari/Qt/BSB.

## Each batch

```bash
# Mac, repo root: list the tasks in hpc/lumi/manifest.tsv, then send the inputs (+ the manifest)
bash hpc/lumi/push.sh IF_D14_Mech_40x_up_Ch1_003 IF_D7_Mech_40x_Stat_Down_Ch1_001

# LUMI
cd /project/<project>/Senpai_python        && git pull         # if the code changed
bash hpc/lumi/submit.sh                                         # one array task per manifest line
squeue --me ; tail -f $WORK/logs/senpai_<jobid>_1.out
seff <jobid>_1                                                  # peak memory / CPU efficiency after it ends

# Mac: outputs go to data_lumi/<stem>/ (never into data/), check them, then move by hand
bash hpc/lumi/pull.sh IF_D14_Mech_40x_up_Ch1_003
```

`push.sh` sends only `data/imgs/<image>.tif` (Tuj1 is re-read from it) and `processed_image.npy`, `processed_metadata.npy`,
`soma_mask.tiff`, `somas_data.npy`. It stops if a folder has no saved somas.
No old outputs are sent, so the scripts' refuse-to-overwrite checks pass on LUMI.

## Resources (defaults in `reconstruct.sbatch`, override on the submit line)

- `small` partition, 32 CPUs, 96 GB, 3 h per task. On the Mac a Mech `map2-tuj1` run took ~60 min (MAP2 23 + Tuj1 22 + separation 15)
  with 16 GB peak footprint. Only the k-means part runs in parallel. The refinement and the watershed stay serial, so expect
  ~25-35 min per task. Check `seff` on the first run and lower `--mem`/`--time`.
- Billing on `small`: memory above 2 GB/core is billed as extra cores (96 GB / 32 CPUs → ~48 core-h per hour).
- Quick test first with `test_data_pc` (37 MB). On the Mac (4 workers), segmentation took 2 min and separation 3 min (113 neurons),
  but skeletons took 49 min (dense, very large Purkinje cells; Mech images need ~1 min). That is too long for `debug` (30 min max):
  ```bash
  # manifest line: test_data_pc  map2
  bash hpc/lumi/submit.sh --time=01:30:00 --mem=32G --cpus-per-task=16
  ```

## Reproducibility

sklearn's KMeans labels depend slightly on the OpenMP thread count. This is not specific to LUMI: on the Mac, serial runs
with 10 vs 1-2 threads already differ. LUMI results therefore match a Mac run at Dice ~0.997, not bit-for-bit.
A fixed `--cpus-per-task` is reproducible. Compare the first LUMI run with the Mac run of the same image
(Dice, number of neurons, share of voxels with the same label) before using LUMI outputs.

## Scratch

`/scratch` is cleaned of old files. Keep the master copy of inputs and outputs on the Mac (`pull.sh` right after each batch).
