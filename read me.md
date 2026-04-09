# Sparse Approximation and Compressed Sensing: Command Reference

This file keeps a compact command reference for the main scripts. For the full repository overview, project summary, and public-scope notes, see `README.md`.

## Directory Layout

```text
Project_Root/
|-- pr3_pr4.py
|-- pr5_pr6.py
|-- plot_results.py
|-- load_data.py
|-- pr5/
`-- pr6/
```

Generated plots, audio `.wav` files, and `.npy` arrays are written to `results/` when the required local input assets are available.

## Commands

### Noiseless Sparse Recovery

```bash
python pr3_pr4.py noiseless --N 20
python pr3_pr4.py noiseless --N 50
python pr3_pr4.py noiseless --N 100
```

### Noisy Sparse Recovery with Known Sparsity

```bash
python pr3_pr4.py noisy_known_s --N 20 --sigma 0.05
python pr3_pr4.py noisy_known_s --N 50 --sigma 0.05
python pr3_pr4.py noisy_known_s --N 100 --sigma 0.05
```

### Noisy Sparse Recovery with Known Noise Norm

```bash
python pr3_pr4.py noisy_known_normn --N 20 --sigma 0.05
python pr3_pr4.py noisy_known_normn --N 50 --sigma 0.05
python pr3_pr4.py noisy_known_normn --N 100 --sigma 0.05
```

### Plotting

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
