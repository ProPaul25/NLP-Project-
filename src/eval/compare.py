"""Paired comparison of two models on the same test sentences.

Problem: two models can have overlapping confidence intervals and still differ reliably (or the reverse),
because both were scored on the same sentences. The honest test compares them sentence by sentence.

Input : config `compare:` section - pairs of model names [A, B] and the metrics to compare
        <results>/eval/<model>__<set>_per_sentence.tsv  (written by evaluate.py, one row per test sentence)
Output: <results>/eval/compare.csv   one row per (set, A, B, metric)

For each pair and metric (a per-sentence 1 = right / 0 = wrong column):
  a, b           accuracy of A and of B (%)
  diff           b - a in percentage points
  diff_lo/_hi    95% paired bootstrap interval of the difference: the same resampled sentences are used for
                 both models; whole groups (IN22-Conv conversations) are resampled together
  p_value        two-sided bootstrap p-value for "no difference"
  only_a/only_b  sentences only A got right / only B got right (the sentences the difference comes from)
  verdict        "B better" / "A better" if the interval excludes 0, otherwise "no clear difference"
Translation metrics are stored as 1 = correct retrieval, so for every metric higher is better.
No GPU computation happens here; it runs under Slurm with the GPU header by project policy.
"""
import os
# Must be set BEFORE importing torch / numpy. Inside a Slurm job the allocated GPU is index 0.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# him2 has more CPU cores than OpenBLAS supports (max 64): cap math-library threads before numpy loads.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("SLURM_CPUS_PER_TASK", "16"))

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available - refusing to run on CPU. "
                       "Did you submit via sbatch/srun with --gres=gpu:1?")
print("Using GPU:", torch.cuda.get_device_name(0))

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import load_config, resolve, set_seed  # noqa: E402


def paired_bootstrap(a, b, groups, n_boot, seed):
    """Interval and p-value for mean(b) - mean(a), resampling whole groups for both models together."""
    uniq, inv = np.unique(groups, return_inverse=True)
    sa, sb = np.bincount(inv, weights=a), np.bincount(inv, weights=b)
    cnt = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    n = cnt[idx].sum(axis=1)
    d = (sb[idx].sum(axis=1) - sa[idx].sum(axis=1)) / n
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return 100 * lo, 100 * hi, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)
    C, E = cfg["compare"], cfg["evaluate"]
    ev_dir = resolve(cfg["paths"]["results"]) / "eval"
    sets = [f"{s['name']}_{s['split']}" for s in E["sets"]]

    rows = []
    for set_name in sets:
        print(f"\n=== {set_name}")
        for model_a, model_b in C["pairs"]:
            fa, fb = ev_dir / f"{model_a}__{set_name}_per_sentence.tsv", ev_dir / f"{model_b}__{set_name}_per_sentence.tsv"
            for f in (fa, fb):
                if not f.exists():
                    raise SystemExit(f"missing {f} - run evaluate.py for that model first")
            A = pd.read_csv(fa, sep="\t", dtype={"id": str, "group_id": str})
            B = pd.read_csv(fb, sep="\t", dtype={"id": str, "group_id": str})
            m = A.merge(B, on=["id", "group_id"], suffixes=("_a", "_b"))
            if len(m) != len(A) or len(m) != len(B):
                raise SystemExit(f"{set_name}: {model_a} and {model_b} were scored on different sentences")
            for metric in C["metrics"]:
                a, b = m[f"{metric}_a"].to_numpy(float), m[f"{metric}_b"].to_numpy(float)
                lo, hi, p = paired_bootstrap(a, b, m["group_id"].to_numpy(), int(C["n_boot"]), seed)
                diff = 100 * (b.mean() - a.mean())
                verdict = "B better" if lo > 0 else "A better" if hi < 0 else "no clear difference"
                rows.append({"set": set_name, "A": model_a, "B": model_b, "metric": metric,
                             "a": round(100 * a.mean(), 2), "b": round(100 * b.mean(), 2),
                             "diff": round(diff, 2), "diff_lo": round(lo, 2), "diff_hi": round(hi, 2),
                             "p_value": round(p, 4), "only_a": int(((a == 1) & (b == 0)).sum()),
                             "only_b": int(((a == 0) & (b == 1)).sum()), "n": len(m), "verdict": verdict})
                print(f"{model_b:12s} vs {model_a:12s} {metric:16s} {100 * a.mean():6.2f} -> {100 * b.mean():6.2f}  "
                      f"diff {diff:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]  p={p:.3f}  {verdict}")

    out = ev_dir / "compare.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()