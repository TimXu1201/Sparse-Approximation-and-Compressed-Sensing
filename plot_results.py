import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_heatmap_xs_yM(Z, title, out_path, vmin=None, vmax=None, cmap="viridis"):
    # Z shape: (M, s).
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=[1, Z.shape[1], 1, Z.shape[0]],
    )
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel("s")
    plt.ylabel("M")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] {out_path}")


def parse_tag(name: str):
    # Get clean tag from filename
    base = os.path.splitext(name)[0]
    base = re.sub(r"^(avg_ne_|success_rate_)", "", base)
    return base


def process_and_plot(in_dir, out_dir, files, prefix, title_prefix, vmin=None, vmax=None):
    # Filter files and plot
    target_files = sorted([f for f in files if f.startswith(prefix)])
    for f in target_files:
        path = os.path.join(in_dir, f)
        Z = np.load(path)

        # Transpose if shape is (s, M)
        if Z.shape[0] <= Z.shape[1]:
            m = re.search(r"_smax(\d+)", f)
            if m:
                smax = int(m.group(1))
                if Z.shape[0] == smax:
                    Z = Z.T

        tag = parse_tag(f)
        title = f"{title_prefix} ({tag})"
        out_png = os.path.join(out_dir, f"{prefix}{tag}.png")
        plot_heatmap_xs_yM(Z, title, out_png, vmin=vmin, vmax=vmax)


def main():
    # Require 1 argument: the input directory
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <target_folder>")
        print("Example: python plot_results.py results/noiseless")
        sys.exit(1)

    in_dir = sys.argv[1]
    out_dir = os.path.join(in_dir, "plots")

    if not os.path.exists(in_dir):
        print(f"[error] Directory not found: {in_dir}")
        return

    ensure_dir(out_dir)
    files = [f for f in os.listdir(in_dir) if f.endswith(".npy")]

    if not files:
        print(f"[warn] No .npy files found in {in_dir}")
        return

    # Plot both automatically
    process_and_plot(in_dir, out_dir, files, "avg_ne_", "Avg Normalized Error")
    process_and_plot(in_dir, out_dir, files, "success_rate_", "Success Rate", vmin=0.0, vmax=1.0)

    print("[done] All plots generated.")


if __name__ == "__main__":
    main()