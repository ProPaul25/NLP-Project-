"""Combine repeat training runs (seeds) into one honest summary.

Problem: one training run can be lucky or unlucky (random batches, a checkpoint picked on only 99 dev
sentences). A claim about a training setting only holds if it shows up for every seed, not just one.

Input : config `aggregate:` (seeds, runs, metrics) and `compare:` (pairs, metrics, n_boot)
        <results>/eval/results.csv                              (evaluate.py, one row per model and set)
        <results>/eval/<model>__<set>_per_sentence.tsv          (evaluate.py, right/wrong per test sentence)
        Run names: the config seed has no suffix (e5_both); other seeds add _s<seed> (e5_both_s43).
        Base models (e5, labse) are not trained, so they have no seeds and are used as they are.
Output: <results>/eval/seeds_summary.csv   per set, run, metric: one column per seed, mean, sd, min, max
        <results>/eval/seeds_compare.csv   per set, pair, metric:
            diff_s<seed>, lo_s<seed>, hi_s<seed>  paired bootstrap difference B - A for that seed (as compare.py)
            mean_diff                             mean of the per-seed differences
            pooled_lo / pooled_hi / pooled_p      paired bootstrap on per-sentence scores averaged over seeds
            seeds_b_better / seeds_a_better       number of seeds whose own interval excludes 0 in that direction
            verdict                               "B better (all seeds)" / "A better (all seeds)": every seed's interval
                                                  excludes 0 in that direction AND the pooled interval does too;
                                                  "no clear difference": no seed's interval and not the pooled one excludes 0;
                                                  "mixed across seeds": anything in between (some seeds show it, some not)
Caveat: the pooled interval only reflects test-sentence uncertainty. With 3 seeds, the spread between
runs is shown (sd, min, max, per-seed results) but is too few to give a confidence interval of its own.
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
    """Same test as compare.py: interval and p-value for mean(b) - mean(a), whole groups resampled together."""
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


def run_name(run, seed, base_seed):
    return run if seed == base_seed else f"{run}_s{seed}"


def load_per_sentence(ev_dir, model, set_stem):
    f = ev_dir / f"{model}__{set_stem}_per_sentence.tsv"
    if not f.exists():
        raise SystemExit(f"missing {f} - run evaluate.py for {model} first")
    return pd.read_csv(f, sep="\t", dtype={"id": str, "group_id": str}).sort_values("id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    base_seed = cfg["seed"]
    set_seed(base_seed)
    A, C, E = cfg["aggregate"], cfg["compare"], cfg["evaluate"]
    seeds, runs = [int(s) for s in A["seeds"]], list(A["runs"])
    ev_dir = resolve(cfg["paths"]["results"]) / "eval"
    res = pd.read_csv(ev_dir / "results.csv")

    # ---- 1. spread of every metric across seeds
    rows = []
    for set_cfg in E["sets"]:
        set_key = f"{set_cfg['name']}:{set_cfg['split']}"
        print(f"\n=== {set_key}   (mean ± sd over seeds {seeds}; min-max)")
        for run in runs:
            for metric in A["metrics"]:
                vals = {}
                for s in seeds:
                    r = res[(res["model"] == run_name(run, s, base_seed)) & (res["set"] == set_key)]
                    if len(r) != 1:
                        raise SystemExit(f"{run_name(run, s, base_seed)} on {set_key}: {len(r)} rows in results.csv")
                    vals[s] = float(r[metric].iloc[0])
                v = np.array(list(vals.values()))
                row = {"set": set_key, "run": run, "metric": metric,
                       **{f"s{s}": round(x, 2) for s, x in vals.items()},
                       "mean": round(v.mean(), 2), "sd": round(v.std(ddof=1), 2),
                       "min": round(v.min(), 2), "max": round(v.max(), 2)}
                rows.append(row)
                print(f"{run:12s} {metric:16s} {v.mean():6.2f} ± {v.std(ddof=1):5.2f}   "
                      f"[{v.min():6.2f} - {v.max():6.2f}]   " + "  ".join(f"s{s}={x:.2f}" for s, x in vals.items()))
    pd.DataFrame(rows).to_csv(ev_dir / "seeds_summary.csv", index=False)

    # ---- 2. paired comparisons, per seed and pooled over seeds
    n_boot = int(C["n_boot"])
    out = []
    for set_cfg in E["sets"]:
        set_key = f"{set_cfg['name']}:{set_cfg['split']}"
        stem = set_key.replace(":", "_")
        print(f"\n=== {set_key}   paired B - A per seed, then pooled")
        for model_a, model_b in C["pairs"]:
            names = lambda m: [run_name(m, s, base_seed) if m in runs else m for s in seeds]
            dfs_a = [load_per_sentence(ev_dir, m, stem) for m in names(model_a)]
            dfs_b = [load_per_sentence(ev_dir, m, stem) for m in names(model_b)]
            ref = dfs_a[0][["id", "group_id"]]
            for d in dfs_a + dfs_b:
                if not d[["id", "group_id"]].equals(ref):
                    raise SystemExit(f"{set_key}: {model_a}/{model_b} files were scored on different sentences")
            groups = ref["group_id"].to_numpy()
            for metric in C["metrics"]:
                row = {"set": set_key, "A": model_a, "B": model_b, "metric": metric, "n": len(ref)}
                diffs, n_b, n_a = [], 0, 0
                for s, da, db in zip(seeds, dfs_a, dfs_b):
                    a, b = da[metric].to_numpy(float), db[metric].to_numpy(float)
                    lo, hi, _ = paired_bootstrap(a, b, groups, n_boot, base_seed)
                    d = 100 * (b.mean() - a.mean())
                    diffs.append(d)
                    n_b += lo > 0
                    n_a += hi < 0
                    row.update({f"diff_s{s}": round(d, 2), f"lo_s{s}": round(lo, 2), f"hi_s{s}": round(hi, 2)})
                a_avg = np.mean([d[metric].to_numpy(float) for d in dfs_a], axis=0)
                b_avg = np.mean([d[metric].to_numpy(float) for d in dfs_b], axis=0)
                plo, phi, pp = paired_bootstrap(a_avg, b_avg, groups, n_boot, base_seed)
                k = len(seeds)
                verdict = ("B better (all seeds)" if n_b == k and plo > 0 else
                           "A better (all seeds)" if n_a == k and phi < 0 else
                           "no clear difference" if n_b == 0 and n_a == 0 and plo <= 0 <= phi else
                           "mixed across seeds")
                row.update({"mean_diff": round(float(np.mean(diffs)), 2), "pooled_lo": round(plo, 2),
                            "pooled_hi": round(phi, 2), "pooled_p": round(pp, 4),
                            "seeds_b_better": int(n_b), "seeds_a_better": int(n_a), "verdict": verdict})
                out.append(row)
                print(f"{model_b:12s} vs {model_a:12s} {metric:16s} "
                      + " ".join(f"{d:+6.2f}" for d in diffs)
                      + f"  mean {np.mean(diffs):+6.2f} [{plo:+6.2f}, {phi:+6.2f}]  "
                        f"seeds B>{n_b} A>{n_a}  {verdict}")
    pd.DataFrame(out).to_csv(ev_dir / "seeds_compare.csv", index=False)
    print(f"\nWrote {ev_dir / 'seeds_summary.csv'} and {ev_dir / 'seeds_compare.csv'}")


if __name__ == "__main__":
    main()