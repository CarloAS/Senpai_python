# CLAUDE.md (project: Senpai_python, SENPAI port + BSB network from confocal stacks)

General behavior rules are in `~/.claude/CLAUDE.md`. This file covers only this project.

## What this project does
Python port of the SENPAI MATLAB toolbox (3D neuron segmentation of confocal stacks), extended to
build a BSB network ("brain_on_chip") from the imaged tissue:
confocal stack (.tif / .nd2 / .czi) → soma marking → topology-informed k-means segmentation
→ watershed separation into single neurons (soma-seeded) → soma positions (+ later morphologies)
→ BSB scaffold saved to HDF5.

## MATLAB reference (`matlab_senpai/`, gitignored): read-only
The `.m` files in `matlab_senpai/` (also `crop_script.m`, `test_script.m`, `senpai_separator.mat`, `cicle_masks.txt`) are the original SENPAI code and the ground truth for what the Python is supposed to do.
Do not edit them. When changing a Python port, compare against its MATLAB source and state any deviation.

| MATLAB | Python port | Status |
|---|---|---|
| `senpai_seg_core_v4.m` | `NeuronSegmentation.py` (`SegmentationConfig`, `NeuronSegmenter`) | features/clustering/mask aligned to MATLAB (2026-10-17); refinement still has extras, see Known issues |
| `senpai_separator.m` | `NeuronSeparator.py` (`NeuronSeparator.separate_neurons`) | ported (imimposemin → watershed markers) |
| `senpai_somamark.m` | `NeuronalSomaMarker.py` (current), `legacy/somamark.py` (older copy) | ported, matplotlib GUI |
| `senpai_prune.m` | `legacy/senpai_prune.py` | draft: no imports, not runnable |
| `senpai_skeletonize.m` | `NeuronSkeletonizer.py` | replaced (TEASAR instead of bwskel thinning, see Pipeline 2c) |
| `senpai_estimateK.m`, `senpai_strahlerord.m`, `senpai_spinecatch.m` | none | not ported |

## Pipeline (summary; full step details, parameters and dated results: `docs/pipeline.md`)
Run everything from the repo root (all paths are relative to the working directory). Layout: see `README.md`.
0. Multi-channel ImageJ TIFFs / ND2 are read by `Image_Processor.py` (soma channel `-c`, channel names from the file name or `--channels`).
1. Soma marking (napari, Mac only): `python Image_Processor.py -i FILE -c CH -d data/` → `data/<stem>/processed_image.npy`,
   `processed_metadata.npy`, `somas_data.npy`, `soma_napari_ellipsoids.npy`, `soma_mask.tiff`; proposals cached in `auto_somas.npy` /
   `auto_cells.npy`. Older matplotlib marker: `NeuronalSomaMarker.py`.
2. Segmentation + separation: `python segment_neurons.py data/<stem>/ --sigma 2 [--map2-only] [--n-jobs N]` →
   `neuron_segmentation<tag>.tiff`, `neuron_labels<tag>.tiff` (tag `_sigma2` or `_sigma2_map2-tuj1`). Viewers: `neuron_viewer.py`, `kmeans_viewer.py`.
   `--n-jobs` (2026-11-10): k-means blocks in parallel (joblib). sklearn KMeans labels depend on the OpenMP thread count (pre-existing:
   serial 10 vs 1–2 threads already differ in 11/12 blocks of a Mech crop), so parallel/LUMI runs match a Mac run only to Dice ~0.997
   (crop: 554,589 vs 551,441 voxels), not bit-for-bit. A fixed thread count is reproducible. Crop speed-up k-means 89 → 51 s (4 workers).
2c. Morphologies: `python NeuronSkeletonizer.py data/<stem>/ --tag <tag>` → `morphologies<tag>/neuron_<label>.swc`.
3. Network: `BSB_Compile.py` (simple builder); `bsb_network.py` (env `vadis-nest`) + `network_html.py` → `results/network/<stem><tag>/`.
3c. Figures/statistics: `notebooks/presentation_stats.ipynb` → `results/presentation/`.
Batch runs: `scripts/run_local.sh` (Mac, sequential, ONE at a time: 9 GB RAM each) or LUMI (`hpc/lumi/README.md`, job array).

## Scripts that overwrite existing results (always ask before running on an existing image/folder)
- `Image_Processor.py` overwrites `data/<stem>/processed_*.npy` every time the viewer closes, and the soma files on save.
- `NeuronalSomaMarker` save overwrites `<image>_somas.json` and `<image>_somas_mask.tif`.
- `test.py` overwrites `results/test_output/*.pkl`.
- BSB builds: `Scaffold.compile()` raises `FileExistsError` if the HDF5 file exists (`new_network.hdf5` by default); it does not overwrite.

## Environment
micromamba env `vadis` (`/Users/carlos/micromamba/envs/vadis`, Python 3.12). Recreated on 2026-10-17 from the old in-repo venv
`vadis/` (now gitignored, broken symlink, can be deleted by the user): pip packages pinned to the old versions
(numpy 1.26.4, napari 0.6.5, bsb 6.0.7, ...), except `triangle==20250106` (20230923 is gone from PyPI).
Pinned list: `envs/vadis_requirements.txt`. Use `micromamba run -n vadis python ...`.
- matplotlib 3.10.9 (+ contourpy 1.3.3, cycler, fonttools, pyparsing) was installed on 2026-10-18 and added to `envs/vadis_requirements.txt` (167 pins; `pip install --dry-run -r` on the env installs nothing).
  Careful: a plain `pip install matplotlib` upgrades numpy to 2.x and breaks bsb/numba/neo. Use
  `pip install "matplotlib<3.11" "contourpy<1.4" "numpy==1.26.4"`. `pip check` is clean with this set.
- Env `vadis-nest` (2026-11-06, for NEST only; vadis untouched): conda-forge python 3.12, nest-simulator 3.9 (osx-arm64 builds all pull OpenMPI),
  numpy 1.26.4; pip bsb-core/hdf5/json/yaml/nest 6.0.7 with `-c envs/vadis_requirements.txt`. Plain `pip install bsb-nest` would install bsb-core 8 +
  numpy 2.5. Quirk 3: in vadis-nest `import h5py` MUST precede `import nest`, else NEST segfaults on import (exit 139). Needs `BSB_IGNORE_PARALLEL=1`.
- NOT installed: `aicspylibczi` (only `notebooks/extract_metadata.ipynb`), `bsb-plotting`, `bsb-native`, StarDist/Cellpose. Installing needs the user's OK.
- Quirk 1: `bsb` raises "MPI runtime detected" if ANY env var name contains "mpi". Inside VS Code,
  `VSCODE_NODE_COMPILE_CACHE_ROOT` ("co**mpi**le") triggers it: run with `env -u VSCODE_NODE_COMPILE_CACHE_ROOT` or set `BSB_IGNORE_PARALLEL=1`.
- Quirk 2: `import h5py` must come before `import bsb`, otherwise the hdf5 plugin fails with `H5Tget_class` (handled at the top of `BSB_Compile.py`).
- `senpai_prune.py` uses `ndarray.ptp`, removed in NumPy 2 (fine with the pinned NumPy 1.26.4).

## Contracts — breaking these gives silently wrong results, not errors
- **Channel order (default for all of Asia's images from now on, user decision 2026-11-05).** Fiji channel numbers (1-based, as in Fiji's
  Split Channels / Duplicate; Python/nd2 index = Fiji number − 1): **1 = DAPI** (blue, all nuclei), **2 = MAP2** (green, neurons; nuclei are
  dark voids inside MAP2 somas), **3 = F-actin** (red, neurons and astrocytes), **4 = Ki67** (grey, proliferating nuclei). The `IF_D14_*` stacks have
  a 5th channel that looks like transmitted light / brightfield (checked on `IF_D14_ID_40x_3petri_middle_Ch2_`, all 5 MIPs consistent). Not every stain follows it: `IF_D14_Mech_40x_up_Ch1_003.tif` (62×5×2048², 0.1766 µm/px, no z spacing → 1 µm) is 1 DAPI, 2 MAP2, 3 Piezo1, 4 Tuj1, 5 brightfield: chosen automatically (2026-11-06, `channel_order_from_name`): `_Mech_` in the file name → `dapi,map2,piezo1,tuj1`; `_ID_` or neither → `dapi,map2,actin,ki67` (case-insensitive); `--channels a,b,c,...` overrides it for other stains. Only channels named `actin`/`ki67` feed the astrocyte/Ki67+ classes; with the default names Tuj1 was read as Ki67 (73 "Ki67+", 62 of them MAP2 neurons) and Piezo1 as F-actin. With `--channels`: 259 nuclei = 130 neurons, 129 other; names are saved in `processed_metadata.npy` (read by `segment_neurons.py`); `auto_cells.npy` is cached: delete it after changing names. `test_asia` (from `40X_F50M50_50M_Static_down_C3_001.nd2`) does NOT follow this order (its stored colours
  suggest DAPI=1, grey punctate=2, green MAP2-like=3, red F-actin-like=4) and is set aside for multi-channel work.
- **Axis order.** TIFF on disk and napari/`NeuronalSomaMarker.image`: `(Z, Y, X)`.
  `NeuronSegmenter.image_data` and all segmentation/separation arrays: `transpose(1, 2, 0)` → `(Y, X, Z)`,
  but the code names the first axis "x" and `image_info['dimensions'] = [X, Y, Z]`. This only works for
  square XY images. `NeuronSeparator` applies the same transpose to the soma mask TIFF; any new mask
  writer must save in `(Z, Y, X)`.
- **Soma JSON** (`*_somas.json`, `somas_test`, `test_out.json`): `{"image_path", "somas", "num_somas", "image_dimensions"}`,
  each soma `[x, y, z, radius, z_factor]` in **pixels**; `image_dimensions` is `(Z, Y, X)`.
  The mask ellipsoid is `(x-xc)² + (y-yc)² + ((z-zc)·z_factor)² ≤ r²`, so the z semi-axis is `r / z_factor`.
- **Napari soma data** (`somas_data.npy`) are points in `(z, y, x)` pixel order; each soma has `2·z_radius+1` ellipse shapes
  in `soma_napari_ellipsoids.npy` (one per slice, 4 corners `[z, y±r, x±r]`). `data/test_asia/` also holds a stale
  `somas_napari_ellipsoids.npy` from before the filename fix (unused; the user can delete it).
- **Cluster ordering.** SENPAI's thresholding `KM >= mask` assumes k-means labels 1..K sorted by mean
  intensity ascending (`reorderKM` in the MATLAB) and background = 1. The Python port does not reorder and
  uses sklearn labels 0..K-1 with background also set to 1. Do not trust the threshold until this is fixed.
- **Thresholds depend on bit depth**: background `0.02·2^bit` (MATLAB v4 hard-codes `th_back=6000`),
  soma fill `0.55·2^bit`. Only uint8/uint16 inputs are accepted.
- **BSB units are µm**; soma positions from marking are pixels in `(z, y, x)` (napari) or `[x, y, z]` (`NeuronalSomaMarker`) order.
  `BSB_compile.load_somas` does the napari conversion (reverse axes, scale by `xy_res, xy_res, z_res`); positions from
  `NeuronalSomaMarker` would need their own conversion. Network size = `width_px·xy_res`, `height_px·xy_res`, `z_px·z_res`.

## Known issues (verified by reading code, not yet fixed; ask before fixing the ones that change results)
- `NeuronSegmentation.py` (aligned to MATLAB on 2026-10-17: `len(sig_G)>1`/`sig_G[1]` for the 2nd scale and `TOT_KM2` is now stored;
  k-means labels 0..K-1 sorted by mean intensity with background = 0 (MATLAB `reorderKM`, background=1); second derivatives = Sobel-of-Sobel
  like `imgradientxyz` (`second_derivatives`); `image_info['dtype']` set; `_mask_threshold` keeps all clusters instead of crashing when every
  cluster has all-negative curvature (MATLAB errors there); median filter via `ndimage.median_filter`; 2D min-area filter `config.min_area=7` applied after the
  per-slice opening, before the 3D closing). Effect on `results/test_output/test_neuron_1.tif` (size_lim [1024,1024,5], sig_G=[0]):
  `refined_seg_1` 276,613 → 225,734 voxels, Dice 0.65 old vs new; 1 label after separation in both (that mask has one soma).
  Not verified against MATLAB output (none available): the claim is "closer to the MATLAB code", not "correct". Remaining deviations:
  - `th_back` = 0.02·R instead of hard-coded 6000. The fill threshold 0.55·R matches MATLAB with the default `intensity_reference='max'`
    (MATLAB computes it on the recomposed `cIM`, Python on the whole loaded image: same max). See "Bit-depth thresholds" below.
  - For `sig_G>0` the Gaussian sigma ratio uses real µm/px (`res[0]/res[2]`); MATLAB forces `xres=yres=1`. Median filter (`sig_G[0]==-1`) is applied only to the derivatives, not to the k-means intensity.
  - Two-scale case: MATLAB ORs both levels into a single segmentation; Python refines each level separately (`refined_seg_1`, `refined_seg_2`).
  - Final refinement (2D opening, 3D closing) is kept on purpose (not in MATLAB); the single-scale MATLAB path has no min-area filter (only the two-scale path does).
  - Refinement loops (`binary_mask_1` voxel loops) are pure Python and slow (~2 min for 1024×1024×19).
- `NeuronSeparator.separate_neurons` (changed 2026-11-05): watershed now on the whole image (outside seg = max) and multiplied by the
  segmentation, as MATLAB `senpai_separator.m:32-39` (was `mask=senpai_seg`, which silently dropped seeds outside the segmentation:
  crop test 15/20 neurons → 19/20). New `fill_soma_holes=True` (not in MATLAB): per z-slice, fill segmentation holes containing a seed,
  then OR the seeds in (20/20). BUT on MAP2 (crop of IF_D14) it is not enough: the segmentation barely covers soma cytoplasm (fill
  threshold 0.55·R = 2253 with R=4096, MAP2 soma cytoplasm ~600–1500; 'percentile' R=1881 → 1035 helps little), rings are open, and seeds in
  the dark nucleus lose their own MAP2 ring to neighbours flooding through bright tissue: 13/20 "neurons" are ~only the seed.
  Prototype (scratch, not in the code): grow the labelled seeds with `expand_labels` (6 µm, anisotropic spacing) restricted to the cell body
  = per-slice grayscale-hole-filled smoothed MAP2 > Otsu, use them as markers and OR them into the segmentation: somas follow the cell
  outline and keep their dendrites. Unrestricted growth (3–5 µm) gave artificial discs. 6 pairs of accepted somas lie < 7 µm apart in XY
  (different z): probably one nucleus accepted twice; they split one neuron in two labels.
- MAP2 labels soma + dendrites, not axons: morphologies from it have no axons (matters for VoxelIntersection connectivity).
- Reconstruction on IF_D14 (viewer, max projection): somas + proximal dendrites are labelled; most long thin neurites stay unsegmented
  (segmentation = 0.8% of the stack with test.py settings, sigma_gaussian=0, 6 clusters): morphologies are truncated. Unaccepted MAP2 somas
  leave their MAP2 unassigned or attached to a neighbour. Many neurons span z 0–36 of 38 (axial blur + dendrites at different depths).
  sigma_gaussian crop test (2026-11-05, IF_D14 dense crop (640,128) + sparse crop (1150,1100), 512² × 38, scratch only): neurites ARE
  detected with [0] but beaded/fragmented (sparse: 84 connected pieces), and the separator's pruning drops pieces not connected to the
  soma. [2] / [3] px (≈0.35 / 0.5 µm) make them continuous (48 / 28 pieces; seg 0.9% → 1.5% / 1.7% sparse, 4.5% → 7.2% / 9.1% dense)
  with no more background voxels (smoothed MAP2 < median+3·MAD: ≤0.10 in all); [3] and [0,3] (OR, as MATLAB) merge neighbouring cells in
  the dense cluster; [-1] (median) ≈ [0]. Candidate for 0.18 µm/px MAP2: [2]. Not adopted (default still [0], as test.py): ask first.
  Full IF_D14 runs (2026-11-05, 61 somas, `--sigma 2` / `--sigma 0 3`, outputs tagged `_sigma2` / `_sigma0-3`; the untagged [0] outputs
  were deleted at the user's request): [2] 15 min, [0,3] 28 min. Segmented 1.8 / 2.2% (was 0.8% with [0]); pieces 1713 / 1563 (2300);
  background 4.0 / 4.1%; assigned to a neuron 85 / 87% (75%); median neuron 1805 / 1852 µm³; neurite volume >10 µm from soma 388 / 479 µm³;
  reach 23.0 / 24.2 µm; skeleton 92.9 / 83.4 µm; Dice [2] vs [0,3] 0.84. Visually almost identical; [0,3] gives fuller, blobbier somas.
  Recommended: `--sigma 2` (same result at half the time).
- IF_D14 z calibration doubt: MAP2 intensity through somas stays above half-max over a median 28 of 38 µm (profile nearly flat), so neurons
  are ~30 µm columns in 3D. Either strong axial blur or the z step is not 1 µm (ImageJ voxel depth 1 = also "uncalibrated"); to check in
  the acquisition metadata before using z for morphology / BSB.
- Duplicate accepted somas split one neuron into several labels (e.g. one isolated IF_D14 soma at ~(1500, 1250) px → 3 labels).
- Purkinje demo (`data/test_data_pc/`, 2026-11-05; `data/imgs/test_data_pc.tif` is a symlink to `reference_data/test_data_pc.tif`): 113 detect_somas
  proposals accepted as-is (102/113 inside a `somas_pc.mat` soma, 102/108 reference somas covered) → segmentation 20% of the stack,
  113 neurons, median 27,500 µm³, 97% of the segmentation assigned. In this dense tissue the separation is a tessellation: dendrite
  territories are split between neighbouring somas (molecular layer in stripes; one neuron gets a large dendritic region), and the 11 false
  proposals also claim territory. No MATLAB parcellation of this demo exists to compare with.
- `BSB_Compile.py`: `plot_network` calls `Scaffold.build`/`plot_3d`, which don't exist in bsb 6 (plotting needs the separate `bsb-plotting` package).
  A single `cell_type` and a single morphology are used for all somas; `cell_densities` is only used for random placement.
  Morphologies must be `.swc` in µm; the project has no SWC producer yet (`senpai_skeletonize.m` not ported).
  `TouchDetection`/`TouchDetector` does not exist in bsb 6.0.7 (the docs list AllToAll, FixedIndegree, FixedOutdegree, VoxelIntersection and
  `SegmentIntersection`; the latter needs `bsb-native`, not installed). `VoxelIntersection` params: `contacts` (default 1), `voxels_pre/post` (50), `affinity`, `favor_cache`.
- `Image_Processor.py`: `folder_save` uses `image_path.split('.')[0]` (breaks on names/paths with extra dots);
  `check_already_processed` returns False when only the image was processed, so the image is re-read (harmless).
  `_on_point_added(last_point=True)` is dead code. The napari GUI part of the mask/reload fixes was tested only on saved data, not in the viewer.
- Repo tidy 2026-11-10: stale/draft scripts in `legacy/`, test fixtures in `reference_data/`, notebooks in `notebooks/` (first cell
  chdirs to the repo root), `__pycache__`/`.DS_Store` untracked. Remaining cleanup candidates (ask first): `reference_data/somas.mat`
  (byte-identical to `somas_pc.mat`), `reference_data/processed_metadata.npy` (stale), `data/vadis_requirements.txt` (copied to `envs/`),
  stale `results/presentation/summary_by_group.csv` and `summary_tuj1.csv` (no longer written by the notebook).

## Soma detection evaluations (2026-10-17/18, 2026-11-06): moved verbatim to `docs/results_log.md`

## Project rules
- Ask before changing segmentation parameters/defaults (clusters, `sig_G`, thresholds, margins),
  the k-means feature space, or soma mask geometry: these change every downstream parcellation and network.
- When proposing a change to a port, say whether it moves the Python closer to or further from the MATLAB, and why.
- Keep `legacy/somamark.py` untouched unless asked; `NeuronalSomaMarker.py` is the maintained marker.
- GUI code (napari, matplotlib widgets) cannot be run here: change it only with a description of how to test it by hand.

## Verification (no test suite)
- Segmentation changes: run `test.py` on one fixed stack before and after; report voxel counts of
  `refined_seg_1`, number of labels after separation, and the Dice between old and new segmentation.
- Port fidelity: when MATLAB output exists (`senpai_final.mat`, `parcel_final`), compare against it
  after transposing to the same axis order.
- Soma I/O: round-trip JSON → mask → `_extract_somas_from_mask` and check centers match within 1 px.
- BSB: check placed positions lie inside `network.x/y/z` and count equals the number of somas.
