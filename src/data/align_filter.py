"""Step 2 — drop misaligned Hindi–Kangri TRAINING pairs using LaBSE cross-lingual similarity.

Input : data/interim/pairs_rule_clean.tsv   (training pairs only — the test set never reaches this step)
Output: data/interim/pairs_scored.tsv       all training pairs + labse_sim + keep flag
        data/processed/pairs.tsv            kept pairs only (the file every later step uses)
        <results>/align_report.json         thresholds, counts, baselines, threshold table
        <results>/sims.csv                  per-pair similarity (+ both baselines) for the dashboard

Two baselines, to choose the threshold from data rather than by guess:
  random pair  — each Hindi line vs a *shuffled* Kangri line: unrelated sentences.
  shifted pair — each Hindi line vs the Kangri line of its *neighbour* in the same file
                 (line i vs line i+1). This is exactly what an off-by-one misalignment looks
                 like, and neighbours usually share a topic, so it is the harder baseline.
The share of shifted pairs above the threshold estimates how many misaligned pairs the
threshold would still let through.

Each source has its own cut-off (config align.thresholds; null = no LaBSE filter). Chosen
from a blind human check of 140 pairs (data/review/): every dropped Mendeley pair was a
correct translation, and in Chauhan the 0.30-0.45 band held no wrong pairs while the band
below 0.30 was about half wrong. So: Mendeley unfiltered, Chauhan cut at 0.30.
"""
import os
# Must be set BEFORE importing torch / transformers.
# Inside a Slurm job, Slurm already restricts which GPU is visible, and the
# allocated GPU appears as index 0. So use "0" -- never a physical GPU number.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available - refusing to run on CPU. "
                       "Did you submit via sbatch/srun with --gres=gpu:1?")
device = torch.device("cuda")
print("Using GPU:", torch.cuda.get_device_name(0))

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import load_config, resolve, save_json, set_seed  # noqa: E402


def encode(model, texts, batch_size):
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    set_seed(cfg["seed"])
    P, A = cfg["paths"], cfg["align"]

    from sentence_transformers import SentenceTransformer
    df = pd.read_csv(resolve(P["interim"]) / "pairs_rule_clean.tsv", sep="\t", dtype=str, keep_default_na=False)
    df["orig_line"] = df["orig_line"].astype(int)
    print(f"Scoring {len(df)} pairs with {A['model']}")

    model = SentenceTransformer(A["model"], device="cuda")
    h = encode(model, df["hindi"].tolist(), A["batch_size"])
    k = encode(model, df["kangri"].tolist(), A["batch_size"])
    df["labse_sim"] = (h * k).sum(axis=1).round(4)

    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(len(df))
    random_sim = (h * k[perm]).sum(axis=1)

    # Shifted baseline: Hindi line i vs the Kangri line of the next line in the same source file.
    order = df.sort_values(["source", "orig_line"]).index.to_numpy()
    src_sorted = df.loc[order, "source"].to_numpy()
    same_src = src_sorted[:-1] == src_sorted[1:]
    a, b = order[:-1][same_src], order[1:][same_src]
    shifted_sim = (h[a] * k[b]).sum(axis=1)
    shifted_src = df.loc[a, "source"].to_numpy()

    thrs = A["thresholds"]
    missing = set(df["source"]) - set(thrs)
    if missing:
        raise SystemExit(f"align.thresholds has no entry for source(s) {sorted(missing)}")
    df["keep"] = True
    for src, t in thrs.items():
        if t is not None:
            m = df["source"] == src
            df.loc[m, "keep"] = df.loc[m, "labse_sim"] >= float(t)
    numeric = sorted({float(t) for t in thrs.values() if t is not None})
    # One cut-off value for the baselines and the dashboard charts: the (lowest) numeric threshold in use.
    thr = numeric[0] if numeric else 0.0

    interim = resolve(P["interim"]) / "pairs_scored.tsv"
    df.to_csv(interim, sep="\t", index=False)
    kept = df[df["keep"]].drop(columns=["keep"])
    out = resolve(P["processed"]) / "pairs.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out, sep="\t", index=False)
    print(f"Wrote {interim}\nWrote {out}  ({len(kept)} of {len(df)} pairs kept; thresholds {thrs})")
    print(df.groupby("source")["keep"].agg(total="size", kept="sum"))

    res = resolve(P["results"])
    res.mkdir(parents=True, exist_ok=True)
    sims = pd.concat([
        df[["id", "source", "split", "orig_line", "labse_sim"]].assign(kind="real pair"),
        pd.DataFrame({"id": "", "source": df["source"].values, "split": df["split"].values,
                      "orig_line": -1, "labse_sim": random_sim.round(4), "kind": "random pair"}),
        pd.DataFrame({"id": "", "source": shifted_src, "split": "pool",
                      "orig_line": -1, "labse_sim": shifted_sim.round(4), "kind": "shifted pair"}),
    ])
    sims.to_csv(res / "sims.csv", index=False)

    by_group = (df.groupby(["source", "split"])
                  .agg(total=("keep", "size"), kept=("keep", "sum"), median_sim=("labse_sim", "median"))
                  .reset_index())
    by_group["kept_pct"] = (100 * by_group["kept"] / by_group["total"]).round(1)
    by_group["median_sim"] = by_group["median_sim"].astype(float).round(3)
    filtered = df["source"].map(lambda x: thrs[x] is not None)   # sources the LaBSE filter applies to
    lowest = df[filtered].nsmallest(25, "labse_sim")[["id", "source", "labse_sim", "hindi", "kangri"]]
    near = filtered & (df["labse_sim"] >= thr - 0.05) & (df["labse_sim"] < thr + 0.05)
    border = df[near].sample(n=min(25, int(near.sum())), random_state=cfg["seed"])[
        ["id", "source", "labse_sim", "hindi", "kangri"]]

    save_json({
        "model": A["model"], "thresholds": thrs, "threshold": thr,
        "pairs_scored": int(len(df)), "pairs_kept": int(df["keep"].sum()),
        "by_source": by_group.to_dict(orient="records"),
        "real_sim_percentiles": {p: round(float(np.percentile(df["labse_sim"], p)), 3) for p in (1, 5, 10, 25, 50)},
        "random_sim_percentiles": {p: round(float(np.percentile(random_sim, p)), 3) for p in (50, 90, 99, 99.9)},
        "random_pairs_above_threshold_pct": round(float(100 * (random_sim >= thr).mean()), 3),
        "shifted_sim_percentiles": {p: round(float(np.percentile(shifted_sim, p)), 3) for p in (50, 90, 95, 99)},
        # Mendeley neighbours only (guaranteed wrong pairs), as in threshold_table below.
        "shifted_pairs_above_threshold_pct": round(float(100 * (shifted_sim[shifted_src == "mendeley"] >= thr).mean()), 2),
        # How each candidate threshold trades kept pairs against shifted pairs let through.
        "threshold_table": [
            {"threshold": t,
             "real_kept_pct": round(float(100 * (df["labse_sim"] >= t).mean()), 1),
             "mendeley_kept_pct": round(float(100 * (df.loc[df["source"] == "mendeley", "labse_sim"] >= t).mean()), 1),
             "chauhan_kept_pct": round(float(100 * (df.loc[df["source"] == "chauhan", "labse_sim"] >= t).mean()), 1),
             # Mendeley neighbours only: that file is correctly aligned, so its shifted pairs are
             # guaranteed wrong. (In Chauhan's drifted stretches a "shifted" pair can be the right one.)
             "shifted_pass_pct": round(float(100 * (shifted_sim[shifted_src == "mendeley"] >= t).mean()), 2),
             "random_pass_pct": round(float(100 * (random_sim >= t).mean()), 3)}
            for t in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)],
        "lowest_sim_examples": lowest.to_dict(orient="records"),
        "near_threshold_examples": border.sort_values("labse_sim").to_dict(orient="records"),
    }, res / "align_report.json")


if __name__ == "__main__":
    main()