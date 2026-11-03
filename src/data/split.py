"""Week 2, step 1 — set aside the hand-labelled development (gold) set BEFORE any topic clustering.

Problem: we need 1,000 pairs that a person labels by topic, and that clustering and training
never see. They must be chosen before clustering, so no cluster label can bias the annotator.

Input : split.input   data/processed/pairs.tsv  (42,336 clean pairs from week 1; read only)
Output: split.output  data/processed/pairs_split.tsv  all pairs + split column:
            train         used for clustering and training
            gold_dev      the 1,000 development pairs
            excluded_dup  training pairs that are near-copies of a gold pair; used nowhere
        <annotation_dir>/gold_dev_ids.txt          gold ids, one per line (cluster.py checks against it)
        <annotation_dir>/gold_dev_to_annotate.csv  gold pairs, shuffled, columns to fill by hand:
            pair_ok  y = correct translation, p = partial, n = wrong pair (leave topic empty)
            topic    one name from the fixed topic list (fill AFTER the topic list is fixed)
            notes    free text
            (UTF-8 with BOM so Excel shows Devanagari)
        <results>/split_report.json                counts per source and split, removed near-copies

The sample is stratified by source, so Mendeley and Chauhan keep their shares; it is fixed by the seed.
Near-copies: exact duplicates were removed in week 1, but two pairs that differ only in punctuation
or spacing would otherwise sit on both sides of the split. Such training pairs are set aside.
The script refuses to overwrite the annotation sheet or the id list, so finished labels are never wiped.
Use --dry_run first: it prints the counts and writes nothing.
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
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import load_config, resolve, save_json, set_seed  # noqa: E402

# Devanagari letters, vowel signs and digits (no danda U+0964/5), plus Latin letters and digits.
# Python's \w would drop Devanagari vowel signs, so the range is spelled out.
WORD = re.compile(r"[\u0900-\u0963\u0966-\u097FA-Za-z0-9]+")


def norm(text):
    """Text reduced to its words: NFC, no punctuation, lower-case Latin, single spaces."""
    return " ".join(WORD.findall(unicodedata.normalize("NFC", text).lower()))


def stratified_sample(df, n_total, seed):
    """Pick n_total row positions, split across sources in proportion to their size."""
    rng = np.random.default_rng(seed)
    sizes = df["source"].value_counts().sort_index()
    quota = (sizes / sizes.sum() * n_total).round().astype(int)
    quota.iloc[-1] = n_total - quota.iloc[:-1].sum()   # make the quotas add up exactly
    picked = []
    for src, q in quota.items():
        pos = np.flatnonzero(df["source"].to_numpy() == src)
        if q > len(pos):
            raise SystemExit(f"quota {q} for {src} is larger than its {len(pos)} pairs")
        picked.extend(rng.choice(pos, size=int(q), replace=False).tolist())
    return sorted(picked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true", help="print counts, write nothing")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)
    P, S = cfg["paths"], cfg["split"]

    src_path, out_pairs = resolve(S["input"]), resolve(S["output"])
    ann_dir = resolve(S["annotation_dir"])
    ann_path, ids_path = ann_dir / "gold_dev_to_annotate.csv", ann_dir / "gold_dev_ids.txt"
    if not args.dry_run:
        for p in (ann_path, ids_path):
            if p.exists():
                raise SystemExit(f"{p} already exists - refusing to overwrite possible annotations. "
                                 "Move or rename it first if you really want a new split.")

    df = pd.read_csv(src_path, sep="\t", dtype=str, keep_default_na=False)
    print(f"Read {len(df)} pairs from {src_path}")
    if not df["id"].is_unique:
        raise SystemExit("ids in the input are not unique - the split would be ambiguous")
    if "split" in df.columns:
        # Week 1 carried a split column through cleaning; it must be a single training pool.
        old = sorted(df["split"].unique())
        if len(old) != 1:
            raise SystemExit(f"input already has several split values {old}; expected one training pool")
        print(f"Replacing the week-1 split column (value {old[0]!r}) with the new split")
        df = df.drop(columns=["split"])

    n_gold = int(S["dev_size"])
    if n_gold >= len(df):
        raise SystemExit(f"dev_size {n_gold} is not smaller than the number of pairs {len(df)}")

    gold_pos = stratified_sample(df, n_gold, seed)
    df["split"] = "train"
    df.loc[df.index[gold_pos], "split"] = "gold_dev"
    gold = df["split"] == "gold_dev"

    dup = pd.Series(False, index=df.index)
    if S.get("near_dup_check", True):
        hn, kn = df["hindi"].map(norm), df["kangri"].map(norm)
        dup = ~gold & (hn.isin(set(hn[gold])) | kn.isin(set(kn[gold])))
        df.loc[dup, "split"] = "excluded_dup"

    counts = df.groupby(["source", "split"]).size().unstack(fill_value=0)
    print(counts)
    print(f"near-copies of gold pairs set aside from training: {int(dup.sum())}")
    if args.dry_run:
        print("Dry run: nothing written.")
        return

    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_pairs, sep="\t", index=False)
    print(f"Wrote {out_pairs}")

    ann_dir.mkdir(parents=True, exist_ok=True)
    ids_path.write_text("\n".join(sorted(df.loc[gold, "id"])) + "\n", encoding="utf-8")
    sheet = df[gold].sample(frac=1.0, random_state=seed)[["id", "hindi", "kangri"]].copy()   # shuffled order
    sheet["pair_ok"] = ""
    sheet["topic"] = ""
    sheet["notes"] = ""
    sheet.to_csv(ann_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {ids_path}\nWrote {ann_path} ({len(sheet)} pairs to label by hand)")

    report = {
        "seed": seed,
        "input": str(S["input"]),
        "n_pairs": int(len(df)),
        "n_train": int((df["split"] == "train").sum()),
        "n_gold_dev": int(gold.sum()),
        "n_excluded_dup": int(dup.sum()),
        "per_source": {s: {k: int(v) for k, v in row.items()} for s, row in counts.iterrows()},
        "excluded_dup_examples": df[dup].head(20)[["id", "source", "hindi", "kangri"]].to_dict(orient="records"),
    }
    res = resolve(P["results"])
    res.mkdir(parents=True, exist_ok=True)
    save_json(report, res / "split_report.json")
    print(f"Wrote {res / 'split_report.json'}")


if __name__ == "__main__":
    main()