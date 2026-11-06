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
| `senpai_skeletonize.m` | `NeuronSkeletonizer.py` | replaced (TEASAR instead of bwskel thinning, see Pipeline 2c) |
| `senpai_estimateK.m`, `senpai_strahlerord.m`, `senpai_spinecatch.m` | none | not ported |

## Pipeline (run from the folder containing the data, all paths are relative to the working directory)
0. Multi-channel ImageJ TIFFs (Z, C, Y, X, e.g. `data/imgs/IF_D14_ID_40x_3petri_middle_Ch2_.tif`, 5 channels) are read directly by
   `Image_Processor.py` (no split files, user decision 2026-11-05): `tiff_preprocess` keeps all channels in `self.channels`
   (names from `CHANNEL_NAMES`, Fiji order, extra channels `ch<N>`), `self.image_data` = the soma channel (`-c`, 0-based; default `SOMA_CHANNEL`
   = map2 = index 1), napari shows every channel as an additive layer (dapi + soma channel visible). `processed_image.npy` caches only the soma
   channel; multi-channel TIFFs are always re-read (metadata stores `channel_names`). `.nd2` now needs `-c` (CLI default changed 0 → None).
   Verified headless 2026-11-05: soma channel pixel-identical to `data/imgs/map2_asia_test.tif` (the earlier Fiji MAP2 export, formerly
   `new_asia_test.tif`), same 52 proposals (written to `data/IF_D14_ID_40x_3petri_middle_Ch2_/auto_somas.npy`); single-channel TIFFs still load
   (`test_asia.tif` 0.3532/0.5 µm). Napari layers not tested (no OpenGL).
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
     Gaussian 20 µm (background removal) → triangle threshold → per-slice fill of holes < π·r² (`remove_small_holes`) → opening with a 0.6·r ball
     (done with distance transforms) → EDT peaks (min distance 1.2·r).
     Threshold changed Otsu → triangle + small-hole fill on 2026-11-05 for MAP2 images (`new_asia_test`: nuclei are dim voids inside the somas;
     Otsu kept 3% of the volume and only bright somas). Results (triangle / old Otsu): `test_data_pc` vs `somas_pc.mat` 113 proposals P 0.89 R 0.94
     / 108, P 0.89 R 0.89 (CLAUDE.md earlier said 0.90/0.90), ~3 s; `test_asia` ch2 20 / 20 (without background removal: 50 blobs on the illumination
     gradient, Otsu fg 46%); `new_asia_test` (z = 1 µm) 52 / 19, visually mostly on soma centres, a few double hits and neurite-junction hits.
     Unbounded 2D `binary_fill_holes` gave the same counts but could fill neurite loops in dense cultures; hence the size limit.
     `data/imgs/map2_asia_test.tif` (called `new_asia_test.tif` in the results above): ImageJ export of Fiji channel 2 of 5 (MAP2) of `IF_D14_ID_40x_3petri_middle_Ch2_`, 38×2048×2048 uint16,
     0.1766 µm/px, voxel depth 1 µm (no .nd2 available). ImageJ omits `spacing=` when voxel depth is 1, so `tiff_preprocess` defaults to 1.0.
     Background σ 30 µm lets neurite beads in, 50 µm fails again. Napari display not tested here (no OpenGL); the layer object builds headless.
     Multi-channel cell proposals (2026-11-05, only for multi-channel TIFFs): `load_or_detect_cells` caches `data/<stem>/auto_cells.npy` from
     `detect_cells(channels, xy_res, z_res)`: DAPI nuclei (~0.5 µm XY grid, Otsu, seeds = maxima of DAPI smoothed 1.5 µm, watershed,
     equiv. radius ≥ 2.5 µm) → per nucleus robust z-scores of MAP2 and F-actin in a 0.5–3 µm shell outside the nucleus and Ki67 inside →
     neuron (MAP2 shell > 5) / astrocyte (else F-actin shell > 5) / other; Ki67+ if > 10. Rows `[z, y, x, r_um, class, ki67_pos, map2_shell,
     actin_shell, ki67_in]`. Napari layers (2026-11-05, one colour each): `Neuron somas (proposed)` (yellow, neuron & Ki67−),
     `Astrocyte somas (proposed)` (magenta, astrocyte & Ki67−), `Other nuclei (proposed)` (white, hidden, other & Ki67−),
     `Proliferating (Ki67+) cells (proposed)` (cyan, every Ki67+ cell; hover `class` shows its MAP2/F-actin class). IF_D14: 61 / 25 / 84 / 37.
     `self.cell_layers` keys 0 neuron, 1 astrocyte, 2 other, 3 proliferating; no accept button for 3 (Ki67+ cells are not accepted
     as neurons or astrocytes; add one by hand in the neuron/astrocyte layer if it is one). The MAP2-only `Auto Somas` layer is then hidden. Like `auto_somas`, NOT saved as somas.
     MAP2 shell/inside ratio was tried and dropped: in 3D the axial blur fills the nuclear void (inside ≈ shell). Thresholds 5/5/10 chosen by eye
     on `IF_D14_ID_40x_3petri_middle_Ch2_` (MAP2 shell histogram: peak at 0 + long tail; Ki67 clearly bimodal, gap ~12). Result there:
     207 nuclei = 80 neurons, 40 astrocytes, 87 other (mostly isolated nuclei with no stain around, plus small bright debris); 37 Ki67+,
     of which 19 are "neurons" (suspicious, post-mitotic: progenitors or a neighbour's MAP2 in the shell). Judged visually only, no reference.
     Known limits: dense DAPI clumps are partly under-segmented; a glial nucleus touching a neuron can pick up the neuron's MAP2 in its shell.
     Reviewing proposals (2026-11-05): the class layers are plain editable Points layers (`self.cell_layers`, `out_of_slice_display=True`;
     select/move/delete/add with napari's own tools in 2D; hand-added points get the median nucleus size). Dock "Cell Proposals":
     `Accept neuron somas` copies the neuron layer into `Soma Centroids` with ellipsoids of `NEURON_SEED_FRACTION`=0.5 × the nucleus radius
     (z radius = same µm / z_res), skips points inside existing somas, removes the layer; `Save Somas` then writes the usual files.
     0.5 because `NeuronSeparator` uses soma-mask components as watershed seeds: full-size ellipsoids merged 80 neurons into 51 components.
     `Accept astrocytes` writes `data/<stem>/astrocytes_data.npy` (`[z, y, x, r_um]`, pixel coords; re-press to re-save; reloaded instead of the
     proposals on the next opening); nothing downstream reads it yet. `Discard other nuclei` removes that layer (auto_cells.npy kept).
     `undo_soma` now removes the ellipses centred on the last soma's (y, x) (works for any size and after Load Somas).
     Single-channel images (no `auto_cells`): the yellow `Auto Somas (proposed)` layer is `cell_layers[0]` (editable, slice-limited) and the
     dock has only `Accept neuron somas` (seeds = 0.5 × the detect_somas radius). `open_viewer` tested with a stub viewer (Qt offscreen) on
     `test_data_pc` and IF_D14 (multi-channel unchanged).
     Slice display: napari's `out_of_slice_display` takes the point size in pixels (~75) as z extent, so every proposal showed on all
     38 slices (seen in the user's screen recording). `_show_cells_near_slice` (on `dims.current_step`/`ndisplay` events) sets each layer's
     `shown` mask to |z − slice| ≤ radius_um / z_res (median ~7 slices; IF_D14 mid-stack: 36/80 neurons shown, ends 4–8); all shown in 3D.
     Tested headless with real layers and a stub viewer (edit → accept → duplicate guard → undo → save: 81 somas → 81 mask components);
     the buttons/dock/editing in the real viewer were NOT run (no OpenGL).
   - `python NeuronalSomaMarker.py IMAGE.tif [-i init.json|mask.tif] [-r 10] [-z 3] [--batch]`
     (matplotlib) → `IMAGE_somas.json` + `IMAGE_somas_mask.tif` next to the image.
2. Segmentation + separation: `test.py` is the working driver (`load_image` → `k_means_voxels`
   → `process_segmentation` → `NeuronSeparator.separate_neurons`). Writes `results/test_output/{seg,refined_seg}_{1,2}.pkl`.
   `run_segmentation.py` is stale (imports `neuron_segmentation`/`NeuronSegmentation` that don't exist; calls removed methods).
2b. Multi-channel images (2026-11-05): `python segment_neurons.py data/<stem>/` reads `processed_image.npy` (soma channel = MAP2),
   `processed_metadata.npy`, `soma_mask.tiff`; if the metadata `channel_names` contain `tuj1` (Mech stain) Tuj1 is read from the original TIFF
   (`data_path/imgs/image_path`), segmented with the same settings and OR-ed with MAP2 (tag `_map2-tuj1`, e.g. `--tag _sigma2_map2-tuj1` for the viewer);
   the separation still runs on MAP2. ID images: MAP2 only (2026-11-06). First run on `IF_D14_Mech_40x_up_Ch1_003` (`--sigma 2`, 130 accepted somas
   → 128 seeds: 2 pairs touch): MAP2 1.6% of the stack (20 min), Tuj1 2.3% (17 min), union 2.8% (42% of it Tuj1-only), separation 12.5 min;
   86% of the segmentation assigned; median neuron 2493 µm³ (5–95%: 681–6051); median z extent 37.5 of 62 slices. Viewer
   `neuron_viewer_sigma2_map2-tuj1.html` (24 MB, rendered in headless Chrome): long Tuj1 neurites now attached to their somas; some bright
   isolated neurites stay unassigned. Merging the channels before one segmentation (half the time) was discussed, not tried. segmentation as `test.py` (size_lim [1024,1024,5], sigma 0) via `NeuronSegmenter.load_array`
   (in-memory (Z,Y,X) stack; `load_image` now calls it); writes `neuron_segmentation.tiff` + `neuron_labels.tiff` ((Z,Y,X)), refuses to overwrite.
   Seeds are grown inside the cell body (`NeuronSeparator`, `resolution=` given, `--soma-grow-um` 6) and, after the watershed, holes
   enclosed by one neuron are filled per slice (≤ π(2·6 µm)², nuclei). Full run on IF_D14 (2026-11-05, before the hole fill was added):
   segmentation 572 s, separation 265 s, 60 neurons from 60 seeds, median 1461 µm³ (5–95%: 209–3142); 75% of the MAP2 segmentation
   assigned to a neuron (the rest is not connected to any accepted soma); 20/60 neurons still had nuclear holes in some slices (now fixed
   in the code, outputs NOT regenerated). Crop vs full run over the same 512² region: 4.6% vs 4.5% segmented, Dice 0.85 (size_lim 512 vs 1024).
   `python neuron_viewer.py data/<stem>/ [--dapi-tif MULTICHANNEL.tif]` → self-contained `neuron_viewer.html` (~8 MB): every slice at 2×
   downsampling (labels by block max), max projection (`#projection` in the URL opens it), overlay opacity, DAPI toggle, soma centres,
   hover id, click / table row isolates a neuron. Refuses to overwrite. Checked by rendering in headless Chrome (screenshots, DOM).
   3D panel (WebGL, no library): labelled voxels on a ~0.7 µm XY grid (block max) as a point cloud, coloured per neuron, unassigned
   segmentation in grey (toggle), z stretch / point size sliders, drag rotate, wheel zoom, shift-drag pan; selection synced with 2D.
   Shading (2026-11-06): only surface cells of each label are drawn, lit with normals of the label's smoothed mask (σ 1 µm) + ambient
   occlusion (labelled fraction within 3 µm) + mild depth cue; point size covers max(XY cell, z cell × z stretch) (else stripes when tilted);
   'shading' checkbox = old flat colours. Mech: 1.05 M → 0.48 M points (24 → 20.5 MB with DAPI). Checked by headless-Chrome screenshots.
   Grid coarsened ×2 (XY and z) until ≤ `max_points` (1.5 M) surface points: Purkinje demo 7.5 M → 0.37 M points (87 → 11 MB). `--tag` reads tagged outputs.
   K-means viewer (2026-11-06, `kmeans_viewer.py`): raw | k-means clusters | kept clusters (≥ block curvature threshold) | final segmentation,
   per slice and channel, synced zoom, per-block cluster table (mean I, Gxx/Gyy/Gzz). `segment_neurons.py --save_kmeans` (new runs only) writes
   `kmeans_<channel><tag>.tiff` ((Z,Y,X) uint8, level 1 only), `kmeans_blocks<tag>.json`, `kmeans_viewer<tag>.html` (2× downsampled).
   `python kmeans_viewer.py data/<stem>/ --sigma 2 --tile Y0 X0 --z Z0 Z1` recomputes only some blocks (each block clustered independently,
   random_state 42). Mech, tile (1024, 0), z 25–39: 6 blocks in ~2 min; kept clusters (MAP2 ∪ Tuj1) 98% inside the saved final segmentation, Dice 0.93.
   Note: `segment_neurons` shares one `config` across channels, so Tuj1 uses MAP2's background threshold (0.02·R_map2); the viewer reproduces this.
   z-consistent hole fill (2026-11-06, NOT in MATLAB, off by default): `SegmentationConfig.fill_z_support` / `segment_neurons.py --fill-z-support 0.5`
   (tag `_zfill0.5`): in both per-slice XY fills of the refinement a hole is filled only if ≥ f of it is filled in z−1 or z+1. Motivation: Mech Tuj1
   z 36 (y 1245, x 876): kept clusters closed a ring around a gap between cells (280 µm², Tuj1 median 1197 vs ring 1558, bg 924) and the fill
   took it, one slice only. Default off = identical to HEAD (checked on a crop, both channels). Full Mech run `_sigma2_zfill0.5_map2-tuj1`
   (with `--save_kmeans`, MAP2 20 min, Tuj1 23 min, separation 14 min): segmentation −1.8% (Dice 0.991 vs `_sigma2_map2-tuj1`, 133k voxels removed,
   1 added; most at z 36), that box at z 36 23,370 → 14,945 px (neighbours ~15k), 128 neurons, labels unchanged where both labelled.
   Soma centres were not filled by the segmentation in either version (soma bodies come from the separator's seed growth).
2c. Morphologies (2026-11-06): `python NeuronSkeletonizer.py data/<stem>/ --tag <tag>` → `morphologies<tag>/neuron_<label>.swc` (µm, x = column,
   y = row, z = slice; type 1 soma, 3 neurite; radius = distance to boundary) + `summary.csv`; refuses to overwrite; ~1 min for 128 neurons.
   Same pre-steps as `senpai_skeletonize.m` (soma ∪ neuron, component of the soma, 3D fill, EDT, soma = one node at the seed centroid,
   spur pruning), but the skeleton is TEASAR (penalised shortest paths from the deepest seed voxel, each new branch joins the tree at its
   cheapest point, rolling ball 3·DBF + 1 µm, branches < 1.5 µm dropped, paths cut at the cell body = seed DBF radius), on an XY-binned grid
   (~0.53 µm). Reason: skimage 0.25.2 3D `skeletonize` (Lee = bwskel) returns NOTHING for any even-square cross-section (2×2, 4×4, ... a
   6×6×50 box → 0 voxels), so neurites break/vanish. Synthetic neuron test (scratch): soma, 40 µm neurite, 30 µm + Y fork, ring → stems 3,
   fork found at the right place, length 165 µm. Mech `_sigma2_map2-tuj1` (128): length median 172 µm (5–95%: 33–511), 2–5 stems typical;
   scale 1.5 gave 3× the branches (spurs in blobby cell bodies). Checked by overlays on label MIPs; bushy where neurites are thick/tufted;
   no reference morphologies to compare with. BSB `parse_morphology_file` reads them (1 root, bounds = SWC in µm). `BSB_Compile` still takes
   ONE `morph_file` for all cells (per-cell morphologies not wired in).
3. Network: `BSB_Compile.py`: `BSB_compile(somas_file='data/<stem>/somas_data.npy', output_file=..., data_path='data/<stem>/')`
   → `initialize_network(conn_strategy=None)` → `config_network` → `build_and_save` (`Scaffold.compile()`).
   `somas_file` is the napari `(z, y, x)` pixel array; it is converted to `(x, y, z)` µm using the metadata in `data_path`.
   No soma file → random placement. No morphology file → FixedIndegree. Valid strategies: `AllToAll`, `FixedIndegree`,
   `FixedOutdegree`, `VoxelIntersection` (needs `morph_file`).
   `morph_file` = a single `.swc` (µm) used for every cell: it is registered as a BSB morphology, set on the cell type and
   switches the default connectivity to `VoxelIntersection` (tested with a synthetic SWC: 27 contacts for 15 cells).
   Smoke-tested on `data/test_asia` (15 cells placed, inside the network box, FixedIndegree 105 / AllToAll 225 connections).
   `bsb_test.py` / `bsb_test.ipynb` are scratch versions.
3b. Simulation (2026-11-06): `BSB_IGNORE_PARALLEL=1 micromamba run -n vadis-nest python bsb_network.py data/<stem>/ --tag <tag>` →
   `results/network/<stem><tag>/` (refuses to overwrite; ~30 s): two BSB networks of the same cells (SWC somas, µm):
   *morph* = each cell its own soma-centred SWC (custom distributor `ByNearestSoma`, checked: all cells carry their own SWC) +
   `VoxelIntersection` (voxels 500; autapses removed by `VoxelIntersectionNoAutapse`); *soma* = custom `DistanceDependent`, symmetric
   Gaussian p(d) = p0·exp(−d²/2σ²), σ fitted (ML) to the morph network pairs, p0 set to match its connection count.
   Mech `_sigma2_map2-tuj1`: 470 vs 468 connections, σ 28.1 µm, p0 0.71 (morph p ≈ 0.7 < 20 µm, 0.2 at 30–50, 0.05 at 50–80; exponential fit badly);
   92% of morph connections reciprocal; 229/253 BSB pairs have skeletons within 3 µm (+radii) of each other (direct KD-tree check).
   NEST (`bsb-nest`): `aeif_cond_alpha` (τ_syn_ex 2 ms, adaptation a 2 nS, b 80 pA, τw 500 ms), w_rec 8 nS, delay 1.5 ms, Poisson 2000 Hz × 1 nS
   per cell (~2.6 Hz alone), 10 s, 5 repeats (input seeds; soma network re-drawn). iaf_cond_alpha without adaptation was bistable
   (3.5 Hz at 1 nS, 100–300 Hz runaway from 1.5 nS). Result (Mech): network bursts ~1/s in both; morph: synchrony χ 0.51 ± 0.02 vs 0.47 ± 0.04,
   cells per burst 0.76 ± 0.03 vs 0.62 ± 0.06, burst 111 vs 90 ms, spikes in bursts 0.79 vs 0.66, clustering 0.38 vs 0.31; bursts travel as
   waves across x in the morph raster. Model choices (weights, adaptation, excitatory only) are not fitted to data.
   `network_html.py` writes `network_activity.html` (Plotly 2.35.2 inlined from a cached copy `results/network/plotly-2.35.2.min.js`, works
   offline; jsDelivr fallback): synced 3D replay (somas glow on spikes), rasters, population rate, bursts, comparison table, histograms.
   Checked by headless-Chrome screenshot (no JS errors); the replay animation itself was not watched.
3c. `presentation_stats.ipynb` (2026-11-06): ID vs Mech per image (auto-discovered in `data/`, group from `_ID_`/`_Mech_`): soma composition and
   densities, segmentation %, unassigned %, neuron/neurite volumes, SWC length/branch points/tips/stems/reach/section length, Sholl, CDF, summary
   table (Mann–Whitney on per-image values when ≥ 3 images per group). Cache `results/presentation/cache/`, figures PNG+PDF and CSVs in
   `results/presentation/`. Astrocytes/Ki67 = not measured for Mech (no channel). Written as JSON (no nbformat in vadis); tested by running all
   code cells as a script (8 s), not in Jupyter.
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
- Env `vadis-nest` (2026-11-06, for NEST only; vadis untouched): conda-forge python 3.12, nest-simulator 3.9 (osx-arm64 builds all pull OpenMPI),
  numpy 1.26.4; pip bsb-core/hdf5/json/yaml/nest 6.0.7 with `-c data/vadis_requirements.txt`. Plain `pip install bsb-nest` would install bsb-core 8 +
  numpy 2.5. Quirk 3: in vadis-nest `import h5py` MUST precede `import nest`, else NEST segfaults on import (exit 139). Needs `BSB_IGNORE_PARALLEL=1`.
- NOT installed: `aicspylibczi` (only `extract_metadata.ipynb`), `bsb-plotting`, `bsb-native`, StarDist/Cellpose. Installing needs the user's OK.
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
- Purkinje demo (`data/test_data_pc/`, 2026-11-05; `data/imgs/test_data_pc.tif` is a symlink to the repo-root file): 113 detect_somas
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
- Label-free soma benchmark (2026-11-06): `soma_benchmark/` = Zenodo 20797635 (Pal et al., oblique/Dodt, rat/mouse/human, CC-BY; md5-checked),
  half kept by `prepare_dataset.py` (20/40 stacks per subset: authors' val+test = held-out, 10 seeded-random train = tuning; `kept_stacks.json`; 4.9 GB, zips deleted).
  `soma_detection_benchmark.ipynb` (no outputs saved: `vadis` has no nbformat/nbconvert; figures/CSVs in `soma_benchmark/results/`, ~13 min):
  point-in-mask matching, pixel units (no µm calibration in the data), radius/anisotropy from tuning GT, raw vs gradient input, stack bootstrap CIs,
  permutation test vs random points. Result: `detect_somas` fails on relief contrast (held-out P 0.48, R 0.05, F1 0.09; 10× chance, p ≤ 0.001).
  Says nothing about MAP2/DAPI accuracy; a fluorescence benchmark / hand-annotated IF_D14 subset is still needed.
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
