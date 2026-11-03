"""Bengali–Hindi pilot — check that no test sentence also appears, exactly or nearly, in the data used to
train or choose the model, and that topic voters in a test set are not near-copies of the sentence they score.

Problem: the evaluation is only fair if the labelled sentences that train the encoder (MASSIVE train,
SIB-200 train), the sentences that choose the checkpoint (SIB-200 dev) and the sentences that vote on topics
share no sentences or near-duplicates with the sentences being scored. The earlier check covered only exact
text against MASSIVE train. This script covers SIB-200 train/dev too, near-duplicates, and near-duplicates
inside each test set across different documents/conversations (those are allowed to vote for each other).

Input : data/processed/<pilot>/{massive,sib200,in22_gen,in22_conv}.tsv   (from prepare_pilot.py)
Output: <results>/overlap/overlap_report.json   counts per test set, per check, per threshold
        <results>/overlap/near_duplicates.tsv    every flagged pair with its similarity and both texts,
                                                 highest similarity first, for reading by hand
Nothing is removed here: the counts decide whether anything has to be dropped before re-scoring.

Two tests per sentence pair, run separately on the Bengali (src) and Hindi (tgt) side:
  exact      identical after normalising (lower-case, punctuation removed, spaces collapsed)
  near       cosine similarity of untrained LaBSE vectors >= threshold (0.85, 0.90, 0.95 reported)
A pair is flagged if either side passes. High similarity can also mean a close paraphrase on the same
subject rather than a copy, so flagged pairs must be read before anything is dropped.
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
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import load_config, resolve, save_json, set_seed  # noqa: E402

THRESHOLDS = [0.85, 0.90, 0.95]
LOWEST = min(THRESHOLDS)
DETECTOR = "sentence-transformers/LaBSE"    # untrained model, used only as a similarity detector


def norm(text):
    """Lower-case, drop punctuation and symbols, collapse spaces (Bengali/Devanagari letters are kept)."""
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return re.sub(r"\s+", " ", t).strip()


def encode(model, texts):
    return model.encode(texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True,
                        show_progress_bar=False).astype(np.float32)


def best_match(q, d, block=1024):
    """For every row of q: index and cosine of its most similar row in d."""
    idx, sim = np.empty(len(q), dtype=int), np.empty(len(q), dtype=np.float32)
    dt = torch.from_numpy(d).cuda()
    for s in range(0, len(q), block):
        sims = torch.from_numpy(q[s:s + block]).cuda() @ dt.T
        v, i = sims.max(dim=1)
        idx[s:s + block], sim[s:s + block] = i.cpu().numpy(), v.cpu().numpy()
    return idx, sim


def best_match_other_group(e, groups, block=1024):
    """Within one set: most similar sentence from a DIFFERENT group (same-group sentences never vote)."""
    idx, sim = np.empty(len(e), dtype=int), np.empty(len(e), dtype=np.float32)
    et = torch.from_numpy(e).cuda()
    codes = torch.from_numpy(pd.factorize(groups)[0]).cuda()
    for s in range(0, len(e), block):
        sims = torch.from_numpy(e[s:s + block]).cuda() @ et.T
        same = codes[s:s + block, None] == codes[None, :]          # also covers the sentence itself
        sims[same] = -2.0
        v, i = sims.max(dim=1)
        idx[s:s + block], sim[s:s + block] = i.cpu().numpy(), v.cpu().numpy()
    return idx, sim


def compare(test, ref, emb_test, emb_ref, check, test_name, ref_name, within=False):
    """Flag test sentences whose closest reference sentence is an exact or near copy, on either side."""
    flags = {}
    for side in ("src", "tgt"):
        if within:
            j, s = best_match_other_group(emb_test[side], test["group_id"].to_numpy())
        else:
            j, s = best_match(emb_test[side], emb_ref[side])
        exact = test[side].map(norm).to_numpy() == ref[side].map(norm).to_numpy()[j]
        flags[side] = (j, s, exact)
    rows = []
    for i in range(len(test)):
        best_side = max(("src", "tgt"), key=lambda sd: flags[sd][1][i])
        j, s, exact = flags[best_side][0][i], float(flags[best_side][1][i]), bool(flags[best_side][2][i])
        any_exact = bool(flags["src"][2][i] or flags["tgt"][2][i])
        if any_exact or s >= LOWEST:
            rows.append({"check": check, "test_set": test_name, "test_id": test["id"].iloc[i],
                         "ref_set": ref_name, "ref_id": ref["id"].iloc[j], "side": best_side,
                         "similarity": round(s, 4), "exact_after_normalising": any_exact,
                         "same_topic": test["topic"].iloc[i] == ref["topic"].iloc[j],
                         "test_hindi": test["tgt"].iloc[i], "ref_hindi": ref["tgt"].iloc[j],
                         "test_bengali": test["src"].iloc[i], "ref_bengali": ref["src"].iloc[j]})
    counts = {"sentences": int(len(test)),
              "exact": int(sum(flags["src"][2] | flags["tgt"][2]))}
    best = np.maximum(flags["src"][1], flags["tgt"][1])
    for t in THRESHOLDS:
        counts[f"near_{t:.2f}"] = int((best >= t).sum())
    return counts, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    proc = resolve(cfg["paths"]["processed"])
    out_dir = resolve(cfg["paths"]["results"]) / "overlap"
    out_dir.mkdir(parents=True, exist_ok=True)

    read = lambda n: pd.read_csv(proc / f"{n}.tsv", sep="\t", dtype=str, keep_default_na=False)
    massive, sib, gen, conv = read("massive"), read("sib200"), read("in22_gen"), read("in22_conv")
    refs = {"training (MASSIVE train + SIB-200 train)":
                pd.concat([massive[massive["split"] == "train"], sib[sib["split"] == "train"]], ignore_index=True),
            "checkpoint choice (SIB-200 dev)": sib[sib["split"] == "dev"].reset_index(drop=True)}
    tests = {"SIB-200 test": sib[sib["split"] == "test"].reset_index(drop=True),
             "IN22-Gen": gen[gen["split"] == "test"].reset_index(drop=True),
             "IN22-Conv": conv[conv["split"] == "test"].reset_index(drop=True)}

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(DETECTOR, device="cuda")
    emb = lambda df: {side: encode(model, df[side].tolist()) for side in ("src", "tgt")}
    emb_refs = {n: emb(df) for n, df in refs.items()}
    emb_tests = {n: emb(df) for n, df in tests.items()}

    report, all_rows = {"detector": DETECTOR, "thresholds": THRESHOLDS, "checks": {}}, []
    for t_name, t_df in tests.items():
        for r_name, r_df in refs.items():
            c, rows = compare(t_df, r_df, emb_tests[t_name], emb_refs[r_name], "test vs " + r_name, t_name, r_name)
            report["checks"].setdefault(t_name, {})["vs " + r_name] = c
            all_rows += rows
        c, rows = compare(t_df, t_df, emb_tests[t_name], emb_tests[t_name], "voters inside the test set",
                          t_name, t_name, within=True)
        report["checks"][t_name]["vs other groups in the same test set (topic voters)"] = c
        all_rows += rows

    for t_name, checks in report["checks"].items():
        print(f"\n=== {t_name}")
        for name, c in checks.items():
            print(f"  {name:58s} sentences {c['sentences']:5d}  exact {c['exact']:4d}  "
                  + "  ".join(f"near>={t:.2f} {c[f'near_{t:.2f}']:4d}" for t in THRESHOLDS))

    save_json(report, out_dir / "overlap_report.json")
    pairs = pd.DataFrame(all_rows)
    if len(pairs):
        pairs = pairs.sort_values("similarity", ascending=False)
    pairs.to_csv(out_dir / "near_duplicates.tsv", sep="\t", index=False)
    print(f"\nWrote {out_dir / 'overlap_report.json'} and {out_dir / 'near_duplicates.tsv'} ({len(pairs)} flagged pairs)")


if __name__ == "__main__":
    main()