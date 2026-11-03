"""Check the LaBSE filter by eye: a blind sample of dropped and kept pairs for a Kangri reader.

Input : data/interim/pairs_scored.tsv        every training pair with its LaBSE score and keep flag
Output: data/review/review_sheet.csv         shuffled pairs to judge; score and band are HIDDEN
                                             (UTF-8 with BOM so Excel shows Devanagari)
        data/review/review_key.csv           id -> source, band, score (do not open while judging)

Per source (Mendeley, Chauhan) it samples:
  low   : dropped, score below 0.30
  near  : dropped, score 0.30 to just below the threshold
  kept  : kept, score from the threshold up to 0.60 (a control group)
The reader marks each pair y (correct translation), n (wrong), or ? (unsure).
Mixing kept pairs in, and hiding scores, keeps the judgement honest.

The script refuses to overwrite an existing sheet, so judgements can never be wiped.
No GPU computation happens here; it runs under Slurm with the GPU header by project policy.
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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import load_config, resolve, set_seed  # noqa: E402

PER_BAND = {"low": 25, "near": 25, "kept": 20}   # per source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed = cfg["seed"]
    set_seed(seed)
    thr = float(cfg["align"]["threshold"])

    out_dir = resolve("data/review")
    sheet_path, key_path = out_dir / "review_sheet.csv", out_dir / "review_key.csv"
    if sheet_path.exists():
        raise SystemExit(f"{sheet_path} already exists - refusing to overwrite possible judgements. "
                         "Move or rename it first if you really want a new sample.")

    df = pd.read_csv(resolve(cfg["paths"]["interim"]) / "pairs_scored.tsv", sep="\t",
                     dtype=str, keep_default_na=False)
    df["labse_sim"] = df["labse_sim"].astype(float)
    s = df["labse_sim"]
    df["band"] = None
    df.loc[s < 0.30, "band"] = "low"
    df.loc[(s >= 0.30) & (s < thr), "band"] = "near"
    df.loc[(s >= thr) & (s < 0.60), "band"] = "kept"

    parts = []
    for (src, band), g in df.dropna(subset=["band"]).groupby(["source", "band"]):
        n = min(PER_BAND[band], len(g))
        parts.append(g.sample(n=n, random_state=seed))
        print(f"{src:9s} {band:5s}: {n} sampled of {len(g)}")
    sample = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = sample[["id", "hindi", "kangri"]].copy()
    sheet["correct"] = ""       # y / n / ?
    sheet["notes"] = ""
    sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")
    sample[["id", "source", "band", "labse_sim"]].to_csv(key_path, index=False)
    print(f"Wrote {sheet_path} ({len(sheet)} pairs)\nWrote {key_path}")


if __name__ == "__main__":
    main()