import argparse
import os
from dataclasses import dataclass
import numpy as np

# Definition
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed):
    return np.random.default_rng(seed)


def col_normalize(A, eps=1e-12):
    norms = np.linalg.norm(A, axis=0)
    norms = np.maximum(norms, eps)
    return A / norms


def generate_A(rng, M, N):
    A = rng.standard_normal(size=(M, N))
    return col_normalize(A)


def generate_sparse_x(rng, N, s):
    # Support: uniform random
    support = rng.choice(N, size=s, replace=False)
    support.sort()

    signs = rng.choice([-1.0, 1.0], size=s)
    mags = rng.uniform(1.0, 10.0, size=s)
    vals = signs * mags

    x = np.zeros(N, dtype=float)
    x[support] = vals
    return x, support


def generate_noise(rng, M, sigma):
    return rng.normal(loc=0.0, scale=sigma, size=M)


def normalized_error(x_hat, x, eps=1e-12):
    denom = np.linalg.norm(x)
    denom = max(denom, eps)
    return float(np.linalg.norm(x_hat - x) / denom)



# OMP core
@dataclass
class OMPConfig:
    max_iters: int
    known_sparsity: object = None
    tol: object = None
    eps: float = 1e-12


def omp(A, y, cfg):
    # OMP
    M, N = A.shape
    r = y.copy()
    support = []
    x_hat = np.zeros(N, dtype=float)
    z = np.zeros(0, dtype=float)

    for _ in range(cfg.max_iters):
        corr = A.T @ r

        # Avoid reselecting
        if len(support) > 0:
            corr[support] = 0.0

        j = int(np.argmax(np.abs(corr)))
        support.append(j)

        As = A[:, support]

        # Least squares on selected atoms
        z, *_ = np.linalg.lstsq(As, y, rcond=None)
        r = y - As @ z

        # Stopping rules
        if cfg.known_sparsity is not None and len(support) >= int(cfg.known_sparsity):
            break
        if cfg.tol is not None and np.linalg.norm(r) <= float(cfg.tol) + cfg.eps:
            break

    x_hat[support] = z
    return x_hat, support



# Experiments
def _smax_sparse(N):
    # Hard-code s_max = floor(0.3*N)
    return max(1, int(np.floor(0.3 * N)))


def run_noiseless(N, num_mc, seed, res_tol, out_dir):
    # Noiseless phase transition
    ensure_dir(out_dir)
    rng = set_seed(seed)

    s_max = _smax_sparse(N)
    success_rate = np.full((s_max, N), np.nan, dtype=float)
    avg_ne = np.full((s_max, N), np.nan, dtype=float)

    for M in range(1, N + 1):
        s_up = min(M, s_max)
        for s in range(1, s_up + 1):
            succ = 0
            ne_sum = 0.0

            for _ in range(num_mc):
                A = generate_A(rng, M, N)
                x, supp = generate_sparse_x(rng, N, s)
                y = A @ x

                cfg = OMPConfig(max_iters=s, known_sparsity=s, tol=res_tol)
                x_hat, supp_hat = omp(A, y, cfg)

                supp_hat_sorted = np.array(sorted(supp_hat), dtype=int)
                if supp_hat_sorted.shape[0] == supp.shape[0] and np.array_equal(supp_hat_sorted, supp):
                    succ += 1

                ne_sum += normalized_error(x_hat, x)

            success_rate[s - 1, M - 1] = succ / num_mc
            avg_ne[s - 1, M - 1] = ne_sum / num_mc

        print(f"[noiseless] done M={M}/{N} (s<=min(M,{s_max}))")

    tag = f"N{N}_smax{s_max}"
    np.save(os.path.join(out_dir, f"success_rate_{tag}.npy"), success_rate)
    np.save(os.path.join(out_dir, f"avg_ne_{tag}.npy"), avg_ne)
    print(f"[saved npy] {out_dir}  tag={tag}")


def run_noisy_known_s(N, sigma, num_mc, seed, success_ne, out_dir):
    # known sparsity
    ensure_dir(out_dir)
    rng = set_seed(seed)

    s_max = _smax_sparse(N)
    success_rate = np.full((s_max, N), np.nan, dtype=float)
    avg_ne = np.full((s_max, N), np.nan, dtype=float)

    for M in range(1, N + 1):
        s_up = min(M, s_max)
        for s in range(1, s_up + 1):
            succ = 0
            ne_sum = 0.0

            for _ in range(num_mc):
                A = generate_A(rng, M, N)
                x, _ = generate_sparse_x(rng, N, s)
                n = generate_noise(rng, M, sigma)
                y = A @ x + n

                cfg = OMPConfig(max_iters=s, known_sparsity=s, tol=None)
                x_hat, _ = omp(A, y, cfg)

                ne = normalized_error(x_hat, x)
                ne_sum += ne
                if ne < success_ne:
                    succ += 1

            success_rate[s - 1, M - 1] = succ / num_mc
            avg_ne[s - 1, M - 1] = ne_sum / num_mc

        print(f"[noisy_known_s sigma={sigma}] done M={M}/{N} (s<=min(M,{s_max}))")

    tag = f"N{N}_sigma{sigma}_smax{s_max}"
    np.save(os.path.join(out_dir, f"success_rate_{tag}.npy"), success_rate)
    np.save(os.path.join(out_dir, f"avg_ne_{tag}.npy"), avg_ne)
    print(f"[saved npy] {out_dir}  tag={tag}")


def run_noisy_known_normn(N, sigma, num_mc, seed, success_ne, out_dir):
    # unknown sparsity, known noise norm
    ensure_dir(out_dir)
    rng = set_seed(seed)

    s_max = _smax_sparse(N)
    success_rate = np.full((s_max, N), np.nan, dtype=float)
    avg_ne = np.full((s_max, N), np.nan, dtype=float)

    for M in range(1, N + 1):
        s_up = min(M, s_max)
        for s in range(1, s_up + 1):
            succ = 0
            ne_sum = 0.0

            for _ in range(num_mc):
                A = generate_A(rng, M, N)
                x, _ = generate_sparse_x(rng, N, s)

                n = generate_noise(rng, M, sigma)
                y = A @ x + n
                normn = float(np.linalg.norm(n))

                # Allow up to min(M, N) iterations
                cfg = OMPConfig(max_iters=min(M, N), known_sparsity=None, tol=normn)
                x_hat, _ = omp(A, y, cfg)

                ne = normalized_error(x_hat, x)
                ne_sum += ne
                if ne < success_ne:
                    succ += 1

            success_rate[s - 1, M - 1] = succ / num_mc
            avg_ne[s - 1, M - 1] = ne_sum / num_mc

        print(f"[noisy_known_normn sigma={sigma}] done M={M}/{N} (s<=min(M,{s_max}))")

    tag = f"N{N}_sigma{sigma}_smax{s_max}"
    np.save(os.path.join(out_dir, f"success_rate_{tag}.npy"), success_rate)
    np.save(os.path.join(out_dir, f"avg_ne_{tag}.npy"), avg_ne)
    print(f"[saved npy] {out_dir}  tag={tag}")



# CLI
def main():
    parser = argparse.ArgumentParser(
        description="ECE269 Mini Project: OMP (Parts 3 & 4; save .npy only)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("noiseless", help="Part 3: noiseless phase transition")
    p1.add_argument("--N", type=int, required=True, choices=[20, 50, 100])
    p1.add_argument("--num_mc", type=int, default=2000)
    p1.add_argument("--seed", type=int, default=0)
    p1.add_argument("--res_tol", type=float, default=1e-8)
    p1.add_argument("--out_dir", type=str, default="results/noiseless")

    p2 = sub.add_parser("noisy_known_s", help="Part 4(a): noisy, s is known")
    p2.add_argument("--N", type=int, required=True, choices=[20, 50, 100])
    p2.add_argument("--sigma", type=float, required=True)
    p2.add_argument("--num_mc", type=int, default=2000)
    p2.add_argument("--seed", type=int, default=0)
    p2.add_argument("--success_ne", type=float, default=1e-3)
    p2.add_argument("--out_dir", type=str, default="results/noisy_known_s")

    p3 = sub.add_parser("noisy_known_normn", help="Part 4(b): noisy, ||n||_2 known, s unknown")
    p3.add_argument("--N", type=int, required=True, choices=[20, 50, 100])
    p3.add_argument("--sigma", type=float, required=True)
    p3.add_argument("--num_mc", type=int, default=2000)
    p3.add_argument("--seed", type=int, default=0)
    p3.add_argument("--success_ne", type=float, default=1e-3)
    p3.add_argument("--out_dir", type=str, default="results/noisy_known_normn")

    args = parser.parse_args()

    if args.cmd == "noiseless":
        run_noiseless(args.N, args.num_mc, args.seed, args.res_tol, args.out_dir)
    elif args.cmd == "noisy_known_s":
        run_noisy_known_s(args.N, args.sigma, args.num_mc, args.seed, args.success_ne, args.out_dir)
    elif args.cmd == "noisy_known_normn":
        run_noisy_known_normn(args.N, args.sigma, args.num_mc, args.seed, args.success_ne, args.out_dir)
    else:
        raise ValueError("Unknown cmd")


if __name__ == "__main__":
    main()