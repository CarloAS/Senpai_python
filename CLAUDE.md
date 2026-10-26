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
| `senpai_somamark.m` | `NeuronalSomaMarker.py` (current), `somamark.py` (older copy) | ported, matplotlib GUI |
| `senpai_prune.m` | `senpai_prune.py` | draft: no imports, not runnable |
| `senpai_estimateK.m`, `senpai_skeletonize.m`, `senpai_strahlerord.m`, `senpai_spinecatch.m` | none | not ported |

## Pipeline (run from the folder containing the data, all paths are relative to the working directory)
1. Soma marking, one of:
   - `python Image_Processor.py -i FILE.{tif,nd2} -c CHANNEL -d data/` (napari, newest). Reads
     `data/imgs/FILE`, writes to `data/<FILE stem>/`: `processed_image.npy`, `processed_metadata.npy`
     (on window close), `somas_data.npy`, `soma_napari_ellipsoids.npy`, `soma_mask.tiff` (on "Save Somas").
     If `processed_image.npy` + `processed_metadata.npy` already exist in that folder they are loaded instead of
     re-reading the image; if `somas_data.npy` also exists a "Load Somas" button appears. The soma mask is one disc
     per z-slice shape (a true ellipsoid, z anisotropy as drawn), checked on `data/test_asia` (15 somas → 15 components).
     Automatic soma proposals (2026-10-19): on init, `load_or_detect_somas` loads `data/<stem>/auto_somas.npy` or, if missing, runs
     `detect_somas` once and saves it (rows `[z, y, x, r_um]`, pixel coords of the processed image, r = distance-transform value ≈ soma radius).
     Shown in napari as the yellow `Auto Somas (proposed)` points layer (added below `Soma Centroids`, so clicks still go to the manual layer);
     they are NOT copied into `somas_data.npy` or the soma mask. Cached: delete `auto_somas.npy` to re-run after changing parameters.
     `detect_somas(image, xy_res, z_res, radius_um=8, voxel_um=1, background_um=20)`: block-average to ~1 µm voxels → Gaussian 1 µm minus
     Gaussian 20 µm (background removal) → Otsu → opening with a 0.6·r ball (done with distance transforms) → EDT peaks (min distance 1.2·r).
     Results: `test_data_pc` vs `somas_pc.mat` 108 proposals, P 0.90 R 0.90 (0.92/0.92 without background removal) in ~3 s;
     `test_asia` ch2 20 proposals, visually mostly on somas (without background removal: 50 blobs on the illumination gradient, Otsu fg 46%).
     Background σ 30 µm lets neurite beads in, 50 µm fails again. Napari display not tested here (no OpenGL); the layer object builds headless.
   - `python NeuronalSomaMarker.py IMAGE.tif [-i init.json|mask.tif] [-r 10] [-z 3] [--batch]`
     (matplotlib) → `IMAGE_somas.json` + `IMAGE_somas_mask.tif` next to the image.
2. Segmentation + separation: `test.py` is the working driver (`load_image` → `k_means_voxels`
   → `process_segmentation` → `NeuronSeparator.separate_neurons`). Writes `results/test_output/{seg,refined_seg}_{1,2}.pkl`.
   `run_segmentation.py` is stale (imports `neuron_segmentation`/`NeuronSegmentation` that don't exist; calls removed methods).
3. Network: `BSB_Compile.py`: `BSB_compile(somas_file='data/<stem>/somas_data.npy', output_file=..., data_path='data/<stem>/')`
   → `initialize_network(conn_strategy=None)` → `config_network` → `build_and_save` (`Scaffold.compile()`).
   `somas_file` is the napari `(z, y, x)` pixel array; it is converted to `(x, y, z)` µm using the metadata in `data_path`.
   No soma file → random placement. No morphology file → FixedIndegree. Valid strategies: `AllToAll`, `FixedIndegree`,
   `FixedOutdegree`, `VoxelIntersection` (needs `morph_file`).
   `morph_file` = a single `.swc` (µm) used for every cell: it is registered as a BSB morphology, set on the cell type and
   switches the default connectivity to `VoxelIntersection` (tested with a synthetic SWC: 27 contacts for 15 cells).
   Smoke-tested on `data/test_asia` (15 cells placed, inside the network box, FixedIndegree 105 / AllToAll 225 connections).
   `bsb_test.py` / `bsb_test.ipynb` are scratch versions.
4. Metadata exploration only: `extract_metadata.ipynb` (CZI/ImageJ/OME), `extract_channels.ipynb` (OME/ND2 channels).

## Scripts that overwrite existing results (always ask before running on an existing image/folder)
- `Image_Processor.py` overwrites `data/<stem>/processed_*.npy` every time the viewer closes, and the soma files on save.
- `NeuronalSomaMarker` save overwrites `<image>_somas.json` and `<image>_somas_mask.tif`.
- `test.py` overwrites `results/test_output/*.pkl`.
- BSB builds: `Scaffold.compile()` raises `FileExistsError` if the HDF5 file exists (`new_network.hdf5` by default); it does not overwrite.

## Environment
micromamba env `vadis` (`/Users/carlos/micromamba/envs/vadis`, Python 3.12). Recreated on 2026-10-17 from the old in-repo venv
`vadis/` (now gitignored, broken symlink, can be deleted by the user): pip packages pinned to the old versions
(numpy 1.26.4, napari 0.6.5, bsb 6.0.7, ...), except `triangle==20250106` (20230923 is gone from PyPI).
Pinned list: `data/vadis_requirements.txt`. Use `micromamba run -n vadis python ...`.
- matplotlib 3.10.9 (+ contourpy 1.3.3, cycler, fonttools, pyparsing) was installed on 2026-10-18 and added to `data/vadis_requirements.txt` (167 pins; `pip install --dry-run -r` on the env installs nothing).
  Careful: a plain `pip install matplotlib` upgrades numpy to 2.x and breaks bsb/numba/neo. Use
  `pip install "matplotlib<3.11" "contourpy<1.4" "numpy==1.26.4"`. `pip check` is clean with this set.
- NOT installed: `aicspylibczi` (only `extract_metadata.ipynb`), `bsb-plotting`, `bsb-native`, StarDist/Cellpose. Installing needs the user's OK.
- Quirk 1: `bsb` raises "MPI runtime detected" if ANY env var name contains "mpi". Inside VS Code,
  `VSCODE_NODE_COMPILE_CACHE_ROOT` ("co**mpi**le") triggers it: run with `env -u VSCODE_NODE_COMPILE_CACHE_ROOT` or set `BSB_IGNORE_PARALLEL=1`.
- Quirk 2: `import h5py` must come before `import bsb`, otherwise the hdf5 plugin fails with `H5Tget_class` (handled at the top of `BSB_Compile.py`).
- `senpai_prune.py` uses `ndarray.ptp`, removed in NumPy 2 (fine with the pinned NumPy 1.26.4).

## Contracts — breaking these gives silently wrong results, not errors
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
- `BSB_Compile.py`: `plot_network` calls `Scaffold.build`/`plot_3d`, which don't exist in bsb 6 (plotting needs the separate `bsb-plotting` package).
  A single `cell_type` and a single morphology are used for all somas; `cell_densities` is only used for random placement.
  Morphologies must be `.swc` in µm; the project has no SWC producer yet (`senpai_skeletonize.m` not ported).
  `TouchDetection`/`TouchDetector` does not exist in bsb 6.0.7 (the docs list AllToAll, FixedIndegree, FixedOutdegree, VoxelIntersection and
  `SegmentIntersection`; the latter needs `bsb-native`, not installed). `VoxelIntersection` params: `contacts` (default 1), `voxels_pre/post` (50), `affinity`, `favor_cache`.
- `Image_Processor.py`: `folder_save` uses `image_path.split('.')[0]` (breaks on names/paths with extra dots);
  `check_already_processed` returns False when only the image was processed, so the image is re-read (harmless).
  `_on_point_added(last_point=True)` is dead code. The napari GUI part of the mask/reload fixes was tested only on saved data, not in the viewer.
- Cleanup candidates (ask first): `run_segmentation.py` (stale), tracked `__pycache__/*.pyc` (not in `.gitignore`, includes
  `SENPAI.cpython-312.pyc` and `neuron_segmentation.cpython-312.pyc` with no source), root-level `processed_metadata.npy`, `somas_coords*.npy` (deleted in the working tree).
- `CLAUDE.md` and `.claude/` are untracked.

## Soma detection and napari notes (evaluated 2026-10-17/18; implemented 2026-10-19: see Pipeline step 1)
- `Image_Processor.py` opens napari by hand: `napari.Viewer()` + `add_image` (no `scale`, so z shown isotropic), a 3D `Points` layer and a `Shapes` layer
  with one 2D ellipse per z-slice per soma (11 shapes/soma at `z_radius=5`), magicgui dock widgets (radius sliders, undo, save, load), and it replaces
  `viewer.window._qt_window.closeEvent` to save metadata (this bypasses napari's own close handling and rewrites the full `processed_image.npy` on every close).
  Napari cannot open headless here (no OpenGL), so none of this was run; test by hand.
- The hand-marked somas (`data/test_asia/somas_data.npy`, `results/test_output/*_somas.json`) were a quick test and cover only a small part of the
  real somas: do NOT use them as ground truth (recall against them is only a weak sanity check; precision cannot be measured).
- Soma localisation tests (2026-10-18, scratch scripts, `test_asia` 2× XY downsampled, judged on max-projection overlays):
  - ch2 (SD 477, 517 nm, neuronal marker with neurites; the channel used so far): DoG / LoG / smoothed-intensity maxima / ball-minus-ring template
    mostly fire on varicosities along neurites and on background; methods agree only 40–70% on their top 150. Not usable as soma detectors on this channel.
  - Otsu + opening + distance transform: fails on `test_asia` (bad threshold); on `test_neuron_1`'s `refined_seg_1` it gives 2–3 peaks per soma.
  - ch0 (SD 405, 438 nm, looks nuclear: clean oval nuclei): DoG r=3 µm with a robust threshold (response > median + 20·MAD of the DoG) gives
    149 nuclei, visually almost all on real nuclei (a few double hits on elongated nuclei, a few dim spots). Thresholds relative to the max response or Otsu
    on the image fail (hundreds–thousands of peaks in empty background).
  - Only ~10–20% of ch0 nuclei have clearly raised ch2 around them (>1.3× median: 28, >1.5×: 14), and some bright ch2 somas have no visible ch0 nucleus:
    nuclei ≠ neurons; a nuclear channel needs a neuronal-marker test (or a NeuN-like channel) to keep only neurons.
  - Recommended: nuclei from ch0 (classical DoG now; Cellpose/StarDist if needed) → neuron/non-neuron decision with ch2 (or another marker) →
    shown as a candidates layer in napari to accept/reject. Not implemented.
- Reference data for soma tests: `test_data_pc.tif` (repo root; byte-identical to `results/test_output/test_data.tif` and `../SENPAI/test_data_pc.tif`;
  SENPAI Purkinje-cell demo, uint8, 143×512×512, 0.617 µm/px XY, 1.2 µm Z) with the hand-made SENPAI soma reference `../SENPAI/somas_pc.mat`
  (`somas`, logical (Y,X,Z) → transpose(2,0,1); 1815 components, 108 somas ≥1000 voxels, equiv. radius ~11 µm; the rest are specks).
  `matlab_senpai/res_test_full/` (copied from `../SENPAI/` on 2026-10-19; `senpai_final.mat` + `sl*_1_1360_1_1360.mat` crops, i.e. the MATLAB segmentation of a 1360×1360 stack, probably `test_asia`; not inspected yet) and `matlab_senpai/neurons_morph/*.swc`
  (real morphologies, candidates for a `VoxelIntersection` test). `somas_pc.mat` is still only in `../SENPAI/`. All read-only.
- Soma localisation on `test_data_pc.tif` vs `somas_pc.mat` (2026-10-18; hit = candidate inside a reference soma, each soma counted once):
  | method | best F1 (P/R) | no-tuning threshold result | time | mask Dice vs reference |
  |---|---|---|---|---|
  | Otsu + opening (ball 0.6·r) + EDT peaks, r=8 µm | 0.93 (0.93/0.93) | same: every EDT peak, no threshold | ~1 min | 0.77 (opening mask) |
  | ball-minus-ring template, r=10 µm | 0.92 (0.95/0.90) | median+10·MAD: P 0.96 R 0.86 | 4 s | – |
  | DoG, r=8 µm | 0.87 (0.90/0.84) | median+10·MAD: P 0.87 R 0.86 | 8 s | – |
  | LoG, r=10 µm | 0.86 | median+10·MAD: P 0.87 R 0.83 | 9 s | – |
  | smoothed-intensity maxima, r=10 µm | 0.84 | median+5·MAD: P 0.88 R 0.77 | 5 s | – |
  | Cellpose-SAM 3D (cellpose 4.2.1, `cpsam`, anisotropy 1.945, MPS), objects with equiv. radius ≥10 µm | 0.94 (0.97/0.91) | needs the size filter: unfiltered 1277 objects, P 0.08 | 69 min | 0.70 (masks ~2× reference volume) |
  Radius matters: r ≤ 6 µm (too small for ~11 µm Purkinje somas) drops F1 to 0.5–0.8. Cellpose finds every soma as an object (108/108 ≥30% covered) but
  also segments dendrite pieces; it only matches the classical methods after a size filter, at ~1000× the run time.
  Contrast with `test_asia` (thin sparse neuronal marker, ch2): there the same classical methods fail; nuclear ch0 + DoG works (see above).
- Cellpose lives in its own micromamba env `cellpose` (python 3.11, cellpose 4.2.1.1, torch 2.14.1, numpy 2.4.6; model weights ~1.15 GB in
  `~/.cellpose/models`). Do not install it into `vadis` (numpy 2 breaks bsb/numba/neo). Exchange data via .npy/.tif files.
- Bit-depth thresholds, measured (`bg = 0.02·2^bit`, `fill = 0.55·2^bit`; MATLAB: bg=6000, fill=`0.55·2^ceil(log2(max))`):
  `test_asia` uint16 but max 4095: bg=1311 is 32% of max (only 0.5% of voxels pass), fill=36045 is never reached; MATLAB fill would be 2253.
  `test_data` uint8 max 255: bg=5, fill=141, fine. `test_neuron_1` uint16 max 65535: fine.
  Implemented 2026-10-18: `SegmentationConfig.intensity_reference` sets R for both thresholds (background `0.02·R`, fill `0.55·R`):
  `'max'` (default, MATLAB `2^ceil(log2(max))`), `'percentile'` (99.9th pct), int bit depth (e.g. `12`; ND2 gives it as
  `ND2File.attributes.bitsPerComponentSignificant`, = 12 for `test_asia.nd2`), `'dtype'` (old behaviour, `2^8`/`2^16`). `image_info['intensity_ref']` stores R.
  `test_asia`: 'max'/12 → R=4096 (fill 2253, bg 82), 'percentile' → R=2053 (fill 1129, bg 41), 'dtype' → R=65536.
  Caveat: the background threshold is only a memory-saving mask; with a camera offset (~700 in `test_asia`) `0.02·R` keeps every voxel in the
  k-means (MATLAB's fixed 6000 would keep none, i.e. MATLAB fails on this image). Set `background_threshold` explicitly if needed.

## Project rules
- Ask before changing segmentation parameters/defaults (clusters, `sig_G`, thresholds, margins),
  the k-means feature space, or soma mask geometry: these change every downstream parcellation and network.
- When proposing a change to a port, say whether it moves the Python closer to or further from the MATLAB, and why.
- Keep `somamark.py` untouched unless asked; `NeuronalSomaMarker.py` is the maintained marker.
- GUI code (napari, matplotlib widgets) cannot be run here: change it only with a description of how to test it by hand.

## Verification (no test suite)
- Segmentation changes: run `test.py` on one fixed stack before and after; report voxel counts of
  `refined_seg_1`, number of labels after separation, and the Dice between old and new segmentation.
- Port fidelity: when MATLAB output exists (`senpai_final.mat`, `parcel_final`), compare against it
  after transposing to the same axis order.
- Soma I/O: round-trip JSON → mask → `_extract_somas_from_mask` and check centers match within 1 px.
- BSB: check placed positions lie inside `network.x/y/z` and count equals the number of somas.
