import os
import sys
import numpy as np
from scipy.io import loadmat, wavfile

# OMP core
from pr3_pr4 import OMPConfig, omp

# PR5 Parameters
PR5_TOL = 1e-2
PR5_SMAX = 600
SAVE_PNG = True

# PR6 Parameters
PR6_SPARSITY = 100
PR6_TOL = 1e-2
PR6_FS = 7350
PR6_KS = [10, 50, 100, 200, 300, 1000, 2000, 3000]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def col_normalize(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.maximum(np.linalg.norm(A, axis=0), eps)
    return A / nrm


def ls_solve(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(A, y, rcond=None)[0]



# PR5
def load_pr5(mat_path: str):
    d = loadmat(mat_path)
    y1, y2, y3 = [np.asarray(d[k]).squeeze().astype(float) for k in ["y1", "y2", "y3"]]
    A1, A2, A3 = [np.asarray(d[k]).astype(float) for k in ["A1", "A2", "A3"]]
    return [y1, y2, y3], [A1, A2, A3]


def run_pr5(mat_path: str, out_dir: str, tol: float, smax: int, H: int=90, W: int=160, save_png: bool=False):
    ensure_dir(out_dir)
    print(f"[PR5] Loading: {mat_path}")
    Y_list, A_list = load_pr5(mat_path)

    # Show original compressed images (Y)
    if save_png:
        import matplotlib.pyplot as plt
        import math
        fig_comp = plt.figure(figsize=(12, 4))
        for idx, y_curr in enumerate(Y_list):
            side = math.ceil(math.sqrt(y_curr.size))
            y_pad = np.pad(y_curr, (0, side * side - y_curr.size), 'constant')
            ax = fig_comp.add_subplot(1, 3, idx + 1)
            ax.imshow(y_pad.reshape((side, side)), cmap="gray")
            ax.set_title(f"Compressed Y{idx+1}")
            ax.axis("off")
        fig_comp.tight_layout()
        fig_comp.savefig(os.path.join(out_dir, "pr5_5a_compressed.png"), dpi=200)
        plt.close(fig_comp)

    # Recover X
    for i, (y, A) in enumerate(zip(Y_list, A_list)):
        A = col_normalize(A)
        
        cfg = OMPConfig(max_iters=smax, known_sparsity=None, tol=tol)
        x_omp, supp = omp(A, y, cfg)
        x_ls = ls_solve(A, y)

        np.save(os.path.join(out_dir, f"pr5_case{i+1}_x_omp.npy"), x_omp)
        np.save(os.path.join(out_dir, f"pr5_case{i+1}_x_ls.npy"), x_ls)

        img_omp = x_omp.reshape((H, W), order="F")
        img_ls = x_ls.reshape((H, W), order="F")
        np.save(os.path.join(out_dir, f"pr5_case{i+1}_img_omp.npy"), img_omp)
        np.save(os.path.join(out_dir, f"pr5_case{i+1}_img_ls.npy"), img_ls)

        err_omp = np.linalg.norm(A@x_omp-y) / (np.linalg.norm(y)+1e-12)
        err_ls = np.linalg.norm(A@x_ls-y) / (np.linalg.norm(y)+1e-12)
        print(f"[PR5] Case {i+1} | OMP err: {err_omp:.6g} | LS err: {err_ls:.6g} | Supp: {len(supp)}")

        if save_png:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(12, 4))
            for j, (img, name) in enumerate(zip([img_omp, img_ls], ["OMP", "LS"])):
                ax = fig.add_subplot(1, 2, j+1)
                ax.imshow(img, cmap="gray")
                ax.set_title(f"PR5 Case {i+1}: {name}")
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"pr5_case{i+1}_cmp.png"), dpi=200)
            plt.close(fig)


# PR6
def run_pr6(folder: str, out_dir: str, sparsity: int, fs: int, ks: list, tol: float):
    ensure_dir(out_dir)
    print(f"[PR6] Folder: {folder}")

    if folder not in sys.path:
        sys.path.insert(0, folder)
    from load_data import loading_data

    address = folder + os.sep
    
    y, D, A = loading_data(address)

    y = np.asarray(y).reshape(-1).astype(np.float64)
    A = np.asarray(A).astype(np.float64)
    D = np.asarray(D).astype(np.float32)

    M_total = A.shape[0]

    # Save original compressed audio
    y_wav = y / (np.max(np.abs(y)) + 1e-12)
    wavfile.write(os.path.join(out_dir, "pr6_y.wav"), fs, (y_wav * 32767).astype(np.int16))
    np.save(os.path.join(out_dir, "pr6_y.npy"), y)

    for k in ks:
        if k > M_total:
            continue
            
        yk = y[:k]
        Ak = col_normalize(A[:k, :])
        Phi = (Ak.astype(np.float32) @ D).astype(np.float32)

        cfg = OMPConfig(max_iters=sparsity, known_sparsity=sparsity, tol=tol)
        s_hat, supp = omp(Phi, yk.astype(np.float32), cfg)
        x_hat = (D @ s_hat.astype(np.float32)).astype(np.float32)

        s_ls = ls_solve(Phi.astype(np.float64), yk.astype(np.float64)).astype(np.float32)
        x_ls = (D @ s_ls).astype(np.float32)

        for arr, name in [(s_hat, "s_omp"), (x_hat, "x_omp"), (s_ls, "s_ls"), (x_ls, "x_ls")]:
            np.save(os.path.join(out_dir, f"pr6_k{k}_{name}.npy"), arr)

        for arr, name in [(x_hat, "omp"), (x_ls, "ls")]:
            arr_norm = arr / (np.max(np.abs(arr)) + 1e-12)
            wavfile.write(os.path.join(out_dir, f"pr6_k{k}_{name}.wav"), fs, (arr_norm * 32767).astype(np.int16))

        err_omp = np.linalg.norm(Phi @ s_hat - yk.astype(np.float32)) / (np.linalg.norm(yk)+1e-12)
        err_ls = np.linalg.norm(Phi.astype(np.float64) @ s_ls.astype(np.float64) - yk) / (np.linalg.norm(yk)+1e-12)
        print(f"[PR6] k={k} | Supp: {len(supp)} | OMP err: {err_omp:.6g} | LS err: {err_ls:.6g}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python pr5_pr6.py [pr5 | pr6]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    base = os.path.dirname(os.path.abspath(__file__))

    if cmd == "pr5":
        mat_path = os.path.join(base, "pr5", "Y1 Y2 Y3 and A1 A2 A3.mat")
        out_dir = os.path.join(base, "results", "pr5")
        run_pr5(mat_path, out_dir, PR5_TOL, PR5_SMAX, save_png=SAVE_PNG)

    elif cmd == "pr6":
        folder = os.path.join(base, "pr6")
        out_dir = os.path.join(base, "results", "pr6")
        run_pr6(folder, out_dir, PR6_SPARSITY, PR6_FS, PR6_KS, PR6_TOL)
        

if __name__ == "__main__":
    main()