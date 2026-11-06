# From somas to morphology: automated reconstruction pipeline

Input: a 3D confocal stack of a neuronal culture (multi-channel immunofluorescence:
1 DAPI = all nuclei, 2 MAP2 = neurons (soma + dendrites), 3 F-actin = neurons and astrocytes, 4 Ki67 = proliferating cells).

Output: one 3D label per neuron (volumetric reconstruction). Later: one skeleton (`.swc`) per neuron and a network model in the Brain Scaffold Builder (BSB).

The segmentation and separation steps are a Python port of **SENPAI** (Cauzzo et al., 2024).
The automatic soma detection and the cell classification (step 1) are our own additions. They are not a published method; they chain standard image-processing steps (references below).

---

## Overview

```
confocal stack (DAPI, MAP2, F-actin, Ki67)
   │
   ├─ 1. Automatic soma detection ──────── nuclei from DAPI → classified with MAP2 / F-actin / Ki67
   │
   ├─ 2. Review by the user (napari) ───── accept / move / delete / add somas
   │
   ├─ 3. Neuron segmentation (SENPAI) ──── k-means on intensity + curvature → binary neuron mask
   │
   ├─ 4. Separation into single neurons ── watershed seeded by the accepted somas (SENPAI separator)
   │
   ├─ 5. Morphology (skeleton → SWC) ───── SENPAI skeletonization  [MATLAB only, not yet ported]
   │
   └─ 6. Network model ─────────────────── soma positions (+ morphologies) → BSB scaffold
```

---

## 1. Automatic soma detection

**Goal:** propose the position and size of every cell body, and say which ones are neurons.

**a. Find all nuclei (DAPI channel)**
1. Resample to ~0.5 µm voxels.
2. Remove the uneven background: Gaussian blur at 0.7 µm minus Gaussian blur at 20 µm (difference of Gaussians).
3. Threshold with Otsu's method → nuclear foreground.
4. One seed per nucleus: local maxima of DAPI blurred at 1.5 µm (at least 2.5 µm apart).
5. Split touching nuclei with a marker-controlled **watershed** grown from those seeds.
6. Drop objects smaller than 2.5 µm equivalent radius (debris).

**b. Classify each nucleus with the other markers**
For each nucleus, measure the stain in a thin **shell around it** (0.5–3 µm outside the nucleus), as a robust z-score (median and MAD of the background-subtracted image):
- MAP2 in the shell > 5 → **neuron** (MAP2 fills the cytoplasm around the neuronal nucleus)
- otherwise F-actin in the shell > 5 → **astrocyte**
- otherwise → **other** (isolated nuclei, debris)
- Ki67 inside the nucleus > 10 → **proliferating** (Ki67+)

Why a shell: in MAP2 images the nucleus is a dark hole inside a bright soma, so a nuclear marker gives the position and the MAP2 ring around it says "neuron".

**Single-channel images (MAP2 only):** a simpler detector on the neuronal marker: background removal (difference of Gaussians, 1 µm − 20 µm) → triangle threshold → fill small holes per slice (nuclei) → morphological opening with a ball of 0.6 × soma radius (removes neurites) → peaks of the **distance transform** (one peak per soma, radius ≈ peak value).
On the SENPAI Purkinje-cell demo it gets precision 0.89 / recall 0.94 against the hand-made SENPAI soma masks.

## 2. Review by the user (napari)

The proposals are shown as coloured point layers on the image in napari (yellow neurons, magenta astrocytes, cyan Ki67+, white others).
The user moves, deletes or adds points, then presses **Accept neuron somas**.
Each accepted neuron becomes a small ellipsoid (0.5 × nuclear radius) in a **soma mask**. These ellipsoids are the seeds for step 4.

## 3. Neuron segmentation (SENPAI core)

**Goal:** a binary mask of all neuronal tissue (somas + dendrites) in the MAP2 channel.

1. The stack is processed in tiles (1024 × 1024 × 5 voxels).
2. For each voxel, compute features: intensity and the **second derivatives** (curvature) along x, y and z, optionally after Gaussian smoothing (we use σ = 2 px ≈ 0.35 µm, which keeps thin neurites continuous).
3. **k-means clustering** (6 clusters) on those features; clusters are sorted by mean intensity.
4. **Topology rule:** a bright tubular structure (neurite) has negative curvature across it. The clusters whose curvature is negative are kept as neuron; the rest is background.
5. Soma interiors that fail the curvature rule are filled back with an intensity threshold (0.55 × image range).
6. Clean-up: per-slice opening, removal of tiny 2D objects, 3D closing.

## 4. Separation into single neurons (SENPAI separator)

**Goal:** assign every segmented voxel to one neuron.

1. Grow each accepted soma seed by up to 6 µm, only inside the cell body (smoothed MAP2 above Otsu), so seeds follow the real soma outline.
2. Invert the (median-filtered) image: bright neurites become valleys.
3. **Watershed** from the soma seeds on this inverted image: each neuron floods outward along its own bright processes until it meets a neighbour (topographic distance, Meyer 1994).
4. Keep only voxels inside the segmentation from step 3 → one label per neuron.
5. Fill holes enclosed by one neuron (the dark nucleus) per slice.

Result on IF_D14 (one test image, 61 accepted somas): 61 neurons, 85% of the MAP2 segmentation assigned to a neuron, median neuron volume ~1800 µm³.

## 5. Morphology: skeleton and SWC (SENPAI, MATLAB, not ported yet)

**Goal:** turn each neuron's voxels into a tree (soma → branches → tips) in the standard SWC format.

1. Keep the largest connected piece of the neuron (+ its soma), fill holes.
2. 3D **skeletonization** (thinning to a 1-voxel-wide centre line, MATLAB `bwskel`).
3. Convert skeleton voxels into a **graph** (neighbouring voxels = edges).
4. Break loops: remove the weakest edge of each cycle (thinnest and dimmest).
5. Connect branches that touch the soma to a soma node; compute a **minimum spanning tree** rooted at the soma.
6. Write the **SWC** file: each node with position, radius (from the distance transform) and parent.
7. Optional: **Strahler order** of the branches (branching statistics).

Limits to state: MAP2 labels dendrites, not axons, so morphologies have no axon; thin distal neurites are often only partly segmented.

## 6. Network model (BSB)

Soma positions (pixels → µm with the image resolution) are placed as cells in a **Brain Scaffold Builder** network.
Connectivity: fixed in-degree for now; with SWC morphologies, connections can be made where the neurites of two neurons overlap (`VoxelIntersection`).

---

## References

### Core methods
- **SENPAI** (segmentation, separation, skeletonization):
  Cauzzo S., Bruno E., Boulet D., Nazac P., Basile M., Callara A.L., Tozzi F., Ahluwalia A., Magliaro C., Danglot L., Vanello N.
  *A modular framework for multi-scale tissue imaging and neuronal segmentation.* Nature Communications 15, 4102 (2024).
  doi:10.1038/s41467-024-48146-y. Code: github.com/cauzzo-s5/SENPAI
- **Brain Scaffold Builder** (network model):
  De Schepper R., Geminiani A., Masoli S., Rizza M.F., Antonietti A., Casellato C., D'Angelo E.
  *Model simulations unveil the structure-function-dynamics relationship of the cerebellar cortical microcircuit.*
  Communications Biology 5, 1240 (2022). doi:10.1038/s42003-022-04213-y

### Building blocks used in the steps above
| Step | Method | Reference |
|---|---|---|
| 1 | Otsu threshold | Otsu N. *A threshold selection method from gray-level histograms.* IEEE Trans. Syst. Man Cybern. 9(1):62–66 (1979) |
| 1 | Triangle threshold | Zack G.W., Rogers W.E., Latt S.A. *Automatic measurement of sister chromatid exchange frequency.* J. Histochem. Cytochem. 25(7):741–753 (1977) |
| 1 | Marker-controlled watershed for touching nuclei | Malpica N. et al. *Applying watershed algorithms to the segmentation of clustered nuclei.* Cytometry 28(4):289–297 (1997) |
| 1 | Robust z-score (median / MAD) | Leys C. et al. *Detecting outliers: do not use standard deviation around the mean, use absolute deviation around the median.* J. Exp. Soc. Psychol. 49(4):764–766 (2013) |
| 3 | k-means | Lloyd S. *Least squares quantization in PCM.* IEEE Trans. Inf. Theory 28(2):129–137 (1982); k-means++ init: Arthur D., Vassilvitskii S., SODA 2007 |
| 4 | Watershed | Beucher S., Meyer F. *The morphological approach to segmentation: the watershed transformation.* In: Mathematical Morphology in Image Processing, 433–481 (1993); Meyer F. *Topographic distance and watershed lines.* Signal Processing 38(1):113–125 (1994) |
| 5 | 3D skeletonization (thinning) | Lee T.C., Kashyap R.L., Chu C.N. *Building skeleton models via 3-D medial surface/axis thinning algorithms.* CVGIP: Graph. Models Image Process. 56(6):462–478 (1994) |
| 5 | SWC format | Cannon R.C. et al. *An on-line archive of reconstructed hippocampal neurons.* J. Neurosci. Methods 84(1–2):49–54 (1998) |
| 5 | Strahler order | Strahler A.N. *Quantitative analysis of watershed geomorphology.* Trans. Am. Geophys. Union 38(6):913–920 (1957) |

### Software
- napari: Sofroniew N. et al. *napari: a multi-dimensional image viewer for Python.* Zenodo, doi:10.5281/zenodo.3555620
- scikit-image: van der Walt S. et al. PeerJ 2:e453 (2014)
- SciPy: Virtanen P. et al. Nature Methods 17:261–272 (2020)
- scikit-learn: Pedregosa F. et al. JMLR 12:2825–2830 (2011)

### Tested for comparison, not used in the pipeline
- Cellpose: Stringer C. et al. Nature Methods 18:100–106 (2021). On the Purkinje demo it found every soma but also many dendrite pieces, and was ~1000× slower than the classical detector.
