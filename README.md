# Senpai_python

A Python port of the SENPAI MATLAB toolbox (3D neuron segmentation of confocal stacks). It is extended to build a
BSB network of the imaged culture:

```
confocal stack (.tif / .nd2)
  → soma marking (napari)                    Image_Processor.py
  → k-means segmentation + soma-seeded split  segment_neurons.py  (NeuronSegmentation.py, NeuronSeparator.py)
  → SWC morphologies                          NeuronSkeletonizer.py
  → BSB network + NEST simulation             bsb_network.py, network_html.py
  → figures / statistics                      notebooks/presentation_stats.ipynb
```

Run every command from the repo root. Paths are relative to the working directory.

## Layout

| Path | What |
|---|---|
| `*.py` (root) | pipeline modules and command-line scripts (see below) |
| `notebooks/` | `presentation_stats.ipynb` (figures/stats), metadata exploration notebooks |
| `hpc/lumi/` | running segmentation + skeletons on LUMI as a Slurm job array (see its README) |
| `scripts/run_local.sh` | sequential batch run on the Mac (same manifest format as LUMI) |
| `envs/` | environment specs: `vadis_requirements.txt` (Mac, pinned), `vadis-nest.yml` (NEST), `senpai_lumi.yml` (LUMI) |
| `docs/` | `pipeline.md` (detailed step notes and dated results), `results_log.md` (soma-detection evaluations) |
| `reference_data/` | SENPAI Purkinje demo stack + hand-made soma reference, old soma test files |
| `legacy/` | stale or draft scripts kept for reference (not maintained) |
| `data/`, `results/`, `tmp/`, `matlab_senpai/` | images and outputs, MATLAB original (gitignored) |

Root scripts:

| Script | Step |
|---|---|
| `Image_Processor.py` | read the stack, propose somas/cells, mark/accept somas in napari → `data/<stem>/` |
| `NeuronalSomaMarker.py` | older matplotlib soma marker |
| `segment_neurons.py` | segmentation (MAP2, + Tuj1 for Mech images) and separation into neurons |
| `NeuronSegmentation.py`, `NeuronSeparator.py` | the segmentation / separation code (ports of `senpai_seg_core_v4.m`, `senpai_separator.m`) |
| `NeuronSkeletonizer.py` | SWC skeletons per neuron (TEASAR) |
| `neuron_viewer.py`, `kmeans_viewer.py` | self-contained HTML viewers of the results |
| `bsb_network.py`, `network_html.py` | morphology vs soma-distance BSB networks, NEST simulation, HTML report |
| `BSB_Compile.py` | simple BSB builder from soma positions |
| `test.py` | fixed-stack segmentation check (writes `results/test_output/`) |

## Typical run

```bash
micromamba run -n vadis python Image_Processor.py -i IMAGE.tif -d data/                 # mark + save somas (GUI)
micromamba run -n vadis python segment_neurons.py data/IMAGE/ --sigma 2                  # ~1 h, 9 GB RAM
micromamba run -n vadis python NeuronSkeletonizer.py data/IMAGE/ --tag _sigma2_map2-tuj1
BSB_IGNORE_PARALLEL=1 micromamba run -n vadis-nest python bsb_network.py data/IMAGE/ --tag _sigma2_map2-tuj1
```

Many images: `zsh scripts/run_local.sh MANIFEST` (one at a time on the Mac) or LUMI (`hpc/lumi/README.md`).

## Environments

- `vadis` (Mac, Python 3.12): `pip install -r envs/vadis_requirements.txt`. Keep numpy 1.26.4 (numpy 2 breaks bsb/numba/neo).
- `vadis-nest`: NEST 3.9 + bsb 6.0.7 (`envs/vadis-nest.yml`). `import h5py` must come before `import nest`.
- LUMI: `envs/senpai_lumi.yml` via the container wrapper (`hpc/lumi/setup_env.sh`).

Conventions, known issues and environment quirks: `CLAUDE.md`.
