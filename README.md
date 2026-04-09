# Sparse Approximation and Compressed Sensing

This repository focuses on sparse recovery and compressed reconstruction based on **Orthogonal Matching Pursuit (OMP)** and least-squares decoding.

The work is organized around three main directions:

- noiseless phase-transition experiments
- noisy sparse recovery under different stopping rules
- compressed image and compressed audio reconstruction

## Project Highlights

- configurable OMP implementation for several recovery settings
- Monte Carlo experiments across varying measurement dimensions and sparsity levels
- comparison between OMP and least-squares reconstruction
- reconstruction of compressed images from linear measurements
- sparse decoding of compressed audio signals

## Included Materials

- `pr3_pr4.py`
  Sparse-recovery experiments in noiseless and noisy settings.
- `pr5_pr6.py`
  Reconstruction pipeline for compressed image and audio examples.
- `plot_results.py`
  Plotting utilities for heatmaps and summary figures.
- `load_data.py`
  Helper loader for the main scripts.
- `pr6/load_data.py`
- `pr6/loading_data.m`
  Additional loading utilities used by the audio workflow.
- `results/`
  Selected plots generated from the experiments.
- project-description PDF

## Run Examples

Run commands from the repository root.

### Noiseless Phase Transition

```bash
python pr3_pr4.py noiseless --N 20
python pr3_pr4.py noiseless --N 50
python pr3_pr4.py noiseless --N 100
```

### Noisy Recovery with Known Sparsity

```bash
python pr3_pr4.py noisy_known_s --N 20 --sigma 0.05
python pr3_pr4.py noisy_known_s --N 50 --sigma 0.05
python pr3_pr4.py noisy_known_s --N 100 --sigma 0.05
```

### Noisy Recovery with Known Noise Norm

```bash
python pr3_pr4.py noisy_known_normn --N 20 --sigma 0.05
python pr3_pr4.py noisy_known_normn --N 50 --sigma 0.05
python pr3_pr4.py noisy_known_normn --N 100 --sigma 0.05
```

### Plot Summary Heatmaps

```bash
python plot_results.py results/noiseless
python plot_results.py results/noisy_known_s
python plot_results.py results/noisy_known_normn
```

### Compressed Image Reconstruction

```bash
python pr5_pr6.py pr5
```

### Compressed Audio Reconstruction

```bash
python pr5_pr6.py pr6
```

## Public Scope

This public version keeps the implementation, the command structure, selected output figures, and the main documentation PDF.

External raw input assets and bulky generated arrays are intentionally excluded to keep the repository lightweight and easier to browse.
