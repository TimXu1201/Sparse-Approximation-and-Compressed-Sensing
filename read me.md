# ECE 269 Mini Project 1: Execution Commands

This file contains the exact commands required to execute each part of the mini project. Please run these commands from the root directory of the project.

## 0. Directory Structure & Data Setup

```text
Project_Root/
├── pr3_pr4.py
├── pr5_pr6.py
├── plot_results.py
├── load_data.py                  <-- Provided script for PR6
├── pr5/
│   └── Y1 Y2 Y3 and A1 A2 A3.mat <-- Data for PR5
└── pr6/
    ├── compressedSignal.mat      <-- Data for PR6
    ├── compressionMatrix.mat     <-- Data for PR6
    └── CompressedBasis.tiff      <-- Data for PR6
```
**Output Destination:** All generated plots, audio `.wav` files, and `.npy` arrays will be automatically saved into a `results/` folder created in this root directory.

---

## 1. Phase Transitions (`pr3_pr4.py`)

**Part 3: Noiseless Case**
```bash
python pr3_pr4.py noiseless --N 20
python pr3_pr4.py noiseless --N 50
python pr3_pr4.py noiseless --N 100
```

**Part 4(a): Noisy Case (Known Sparsity)**
*(Example using sigma = 0.05, can be changed as needed)*
```bash
python pr3_pr4.py noisy_known_s --N 20 --sigma 0.05
python pr3_pr4.py noisy_known_s --N 50 --sigma 0.05
python pr3_pr4.py noisy_known_s --N 100 --sigma 0.05
```

**Part 4(b): Noisy Case (Unknown Sparsity, Known Noise Norm)**
```bash
python pr3_pr4.py noisy_known_normn --N 20 --sigma 0.05
python pr3_pr4.py noisy_known_normn --N 50 --sigma 0.05
python pr3_pr4.py noisy_known_normn --N 100 --sigma 0.05
```

---

## 2. Visualization (`plot_results.py`)

Run these commands to generate heatmaps from the `.npy` files created in the previous step. Plots will be saved in a `plots/` subfolder within each target directory.

```bash
python plot_results.py results/noiseless
python plot_results.py results/noisy_known_s
python plot_results.py results/noisy_known_normn
```

---

## 3. Decoding Messages (`pr5_pr6.py`)

Parameters such as tolerance, maximum sparsity, and sampling frequency are configured at the top of the `pr5_pr6.py` script. 

**Part 5: Decode Compressed Image**
```bash
python pr5_pr6.py pr5
```

**Part 6: Decode Compressed Audio**
```bash
python pr5_pr6.py pr6
```