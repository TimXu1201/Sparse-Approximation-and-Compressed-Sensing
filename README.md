# ECE 269 Mini Project 1: Sparse Approximation and Compressed Sensing

This project implements several classical sparse-recovery experiments built around **Orthogonal Matching Pursuit (OMP)** and least-squares reconstruction.

The work is organized into three parts:

- **Phase-transition experiments** in the noiseless setting
- **Noisy sparse recovery** with different stopping rules
- **Compressed image and audio decoding** using sparse representations

## Project Highlights

- clean OMP implementation with configurable stopping rules
- Monte Carlo phase-transition experiments over varying measurement dimensions and sparsity levels
- comparison between OMP and least-squares recovery
- compressed image reconstruction from linear measurements
- compressed audio reconstruction with dictionary-based sparse coding

## Main Files

- `pr3_pr4.py`
  Phase-transition experiments for noiseless and noisy sparse recovery.
- `pr5_pr6.py`
  Reconstruction pipeline for compressed image and audio examples.
- `plot_results.py`
  Plotting utilities for heatmaps and summary figures.
- `load_data.py`
  Helper loader used by the PR6 audio task.
- `results/`
  Selected plots generated from the experiments.

## What Is Published Here

This public version focuses on:

- the source code
- the command structure
- selected result plots that demonstrate the outcomes

To keep the repository clean, course-distributed raw input assets and bulky generated arrays are excluded by `.gitignore`.

## Run Commands

Run all commands from the project root.

### PR3: Noiseless Phase Transition

```bash
python pr3_pr4.py noiseless --N 20
python pr3_pr4.py noiseless --N 50
python pr3_pr4.py noiseless --N 100
```

### PR4(a): Noisy Recovery with Known Sparsity

```bash
python pr3_pr4.py noisy_known_s --N 20 --sigma 0.05
python pr3_pr4.py noisy_known_s --N 50 --sigma 0.05
python pr3_pr4.py noisy_known_s --N 100 --sigma 0.05
```

### PR4(b): Noisy Recovery with Known Noise Norm

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

### PR5: Compressed Image Recovery

```bash
python pr5_pr6.py pr5
```

### PR6: Compressed Audio Recovery

```bash
python pr5_pr6.py pr6
```

## Outputs

Typical outputs include:

- success-rate heatmaps
- normalized-error heatmaps
- reconstructed image comparisons
- reconstructed audio waveforms

## Notes

- The original project was developed for a course setting, so some inputs were distributed separately and are not included in this public repository.
- The existing `read me.md` file is kept as a historical note; this `README.md` is the polished public-facing version.
