<!-- Moved verbatim from CLAUDE.md on 2026-11-10 (repo tidy); file paths updated to the new layout. -->
# Pipeline (run from the folder containing the data, all paths are relative to the working directory)
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
   `legacy/run_segmentation.py` is stale (imports `neuron_segmentation`/`NeuronSegmentation` that don't exist; calls removed methods).
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
   NaN statistics (e.g. `ibi_cv` with < 2 bursts) are written as null (`_finite`; crashed the HTML for Ch1_004 MAP2 before).
   `python network_html.py results/network/<stem><tag>/ data/<stem>/` rebuilds the page from the saved outputs (no new simulation).
3c. `notebooks/presentation_stats.ipynb` (restructured 2026-11-07, user request): section 1 (somas) ID vs Mech; sections 2–4 Mech MAP2 (`_sigma2`) vs
   MAP2 + Tuj1 (`_sigma2_map2-tuj1`), paired per image (lines, paired Wilcoxon: with 3 images min p = 0.25, + per-image ratio), outputs
   `summary_soma_by_stain.csv`, `summary_by_version.csv`, `per_image_reconstruction.csv` (old `summary_by_group.csv` is stale); section 5 = paired
   neurons + stain-matched ID; section 6 (fig11/fig12, `summary_network.csv`, `network_fits.csv`, `network_runs.csv`): soma-only vs morphology
   network of the `_sigma2_map2-tuj1` runs (3 images). 3 Mech images: Tuj1 ratio segmented ×1.69, skeleton length ×1.51, neurite volume ×2.05;
   network: rate/connections equal, cells per burst ×1.09/1.23/1.34 (morph > soma), χ and bursts/s not consistent. Ch1_004: 9 SWC soma nodes
   differ between versions (labels without their seed, as Ch1_003 26/32).
   Original description (2026-11-06): ID vs Mech per image (auto-discovered in `data/`, group from `_ID_`/`_Mech_`): soma composition and
   densities, segmentation %, unassigned %, neuron/neurite volumes, SWC length/branch points/tips/stems/reach/section length, Sholl, CDF, summary
   table (Mann–Whitney on per-image values when ≥ 3 images per group).
   Section 5 (2026-11-06): Tuj1 impact on Mech: `segment_neurons.py --map2-only` (new flag, tag `_sigma2`) + `NeuronSkeletonizer.py --tag _sigma2`
   (MAP2 18 min, separation 11 min, 128 neurons). MAP2-only segmentation ⊂ union exactly (deterministic k-means); 99.1% of voxels labelled in
   both keep the same label (label k = same soma). Tuj1 = 42% of the union segmentation (75% of it assigned); per neuron median 16% Tuj1-only,
   volume ×1.11, skeleton length ×1.30. Stain-matched ID vs Mech (MAP2): segmented 1.77 / 1.64%, skeleton median 148 / 102 µm.
   Ch1_002 / Ch1_004 (2026-11-07, 88 slices, 125 / 170 somas → 122 / 167 neurons), both versions, run one after the other
   (`tmp/mech_runs/run_all.sh`, logs there): MAP2 ~21–27 min, Tuj1 ~22–27 min, separation 14–22 min per version. PEAK RAM 8.3–8.9 GB per
   segment_neurons run: never run more than one at a time on this 16 GB Mac (4 in parallel → watchdog kernel panic + reboot on 2026-11-06).
   k-means is not bit-reproducible between runs (Ch1_002 MAP2: 6,558,113 vs 6,558,153 voxels).
   Labels 26 and 32 do not contain their soma seed (in one or both runs): SWC rooted at the thickest point (pre-existing, not fixed). Cache `results/presentation/cache/`, figures PNG+PDF and CSVs in
   `results/presentation/`. Astrocytes/Ki67 = not measured for Mech (no channel). Written as JSON (no nbformat in vadis); tested by running all
   code cells as a script (8 s), not in Jupyter.
4. Metadata exploration only: `notebooks/extract_metadata.ipynb` (CZI/ImageJ/OME), `notebooks/extract_channels.ipynb` (OME/ND2 channels).

