<!-- Moved verbatim from CLAUDE.md on 2026-11-10 (repo tidy); file paths updated to the new layout. -->
# Soma detection and napari notes (evaluated 2026-10-17/18; implemented 2026-10-19: see Pipeline step 1)
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
- Reference data for soma tests: `reference_data/test_data_pc.tif` (moved from the repo root 2026-11-10; byte-identical to `results/test_output/test_data.tif` and `../SENPAI/test_data_pc.tif`;
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

