"""Bengali–Hindi pilot, step 1 — put the four downloaded datasets into one common table format.

Problem: the pilot must exercise the same pipeline the Kangri data will use, so every dataset is
converted to one layout that training and evaluation code can read without knowing where it came from.

Input  (under paths.raw):
  massive/bn-BD.jsonl, hi-IN.jsonl         MASSIVE 1.1, paired on `id`            -> training (+ its own dev/test)
  sib200/{ben_Beng,hin_Deva}/{train,dev,test}.tsv, labels.txt
                                           SIB-200, paired on `index_id`          -> evaluation only
  in22_conv.parquet                        IN22-Conv, one row = one sentence in all languages -> evaluation only
  in22_gen.parquet                         IN22-Gen, same layout                  -> evaluation only

Output (under paths.processed), one TSV per dataset, same columns everywhere:
  id        unique id, prefixed with the dataset name
  dataset   massive | sib200 | in22_conv | in22_gen
  split     train | dev | test (IN22 sets: test only)
  src       Bengali sentence        (Kangri in the main project)
  tgt       Hindi sentence
  topic     topic label (MASSIVE scenario, SIB-200 category, IN22 domain)
  group_id  sentences that must stay on the same side of any split (IN22-Conv conversation, IN22-Gen
            source document); for MASSIVE and SIB-200 each sentence is its own group
  extra columns: Dogri (doi_Deva) for the IN22 sets, IN22-Conv fine-grained `topic` as `subtopic`
Plus <results>/prepare_report.json: counts per dataset/split/topic, pairing checks, train/eval overlap.

Checks that stop the script: unequal pairing (ids present on one side only), labels that differ between the
two languages of a pair, empty sentences. Train/eval text overlap is only reported, not removed.
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
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import load_config, resolve, save_json, set_seed  # noqa: E402


def check_pairs(df, name):
    """Stop if a pair has an empty side or ids repeat."""
    for c in ("src", "tgt"):
        empty = df[c].isna() | (df[c].astype(str).str.strip() == "")
        if empty.any():
            raise SystemExit(f"{name}: {int(empty.sum())} empty {c} sentences, e.g. ids {df.loc[empty, 'id'].head(5).tolist()}")
    if not df["id"].is_unique:
        raise SystemExit(f"{name}: ids are not unique")


def load_massive(raw, P):
    d = raw / P["massive"]["dir"]
    read = lambda f: pd.DataFrame([json.loads(l) for l in open(d / f, encoding="utf-8")])
    s, t = read(P["massive"]["src_file"]), read(P["massive"]["tgt_file"])
    only_s, only_t = set(s["id"]) - set(t["id"]), set(t["id"]) - set(s["id"])
    if only_s or only_t:
        raise SystemExit(f"MASSIVE: ids on one side only (src {len(only_s)}, tgt {len(only_t)})")
    m = s.merge(t, on="id", suffixes=("_s", "_t"))
    bad = (m["scenario_s"] != m["scenario_t"]) | (m["partition_s"] != m["partition_t"])
    if bad.any():
        raise SystemExit(f"MASSIVE: {int(bad.sum())} pairs disagree on scenario or split")
    out = pd.DataFrame({
        "id": "massive-" + m["id"].astype(str),
        "dataset": "massive",
        "split": m["partition_s"],
        "src": m["utt_s"], "tgt": m["utt_t"],
        "topic": m["scenario_s"],
        "group_id": "massive-" + m["id"].astype(str),
        "intent": m["intent_s"],
    })
    return out, {"pairs": len(out)}


def load_sib(raw, P):
    d = raw / P["sib200"]["dir"]
    sd, td = d / P["sib200"]["src_dir"], d / P["sib200"]["tgt_dir"]
    labels_s = (sd / "labels.txt").read_text(encoding="utf-8").split()
    labels_t = (td / "labels.txt").read_text(encoding="utf-8").split()
    if labels_s != labels_t:
        raise SystemExit(f"SIB-200: label lists differ between languages: {labels_s} vs {labels_t}")
    parts = []
    for split in ("train", "dev", "test"):
        s = pd.read_csv(sd / f"{split}.tsv", sep="\t", dtype=str, keep_default_na=False)
        t = pd.read_csv(td / f"{split}.tsv", sep="\t", dtype=str, keep_default_na=False)
        if set(s["index_id"]) != set(t["index_id"]):
            raise SystemExit(f"SIB-200 {split}: index_id sets differ between languages")
        m = s.merge(t, on="index_id", suffixes=("_s", "_t"))
        if (m["category_s"] != m["category_t"]).any():
            raise SystemExit(f"SIB-200 {split}: categories differ between languages")
        unknown = set(m["category_s"]) - set(labels_s)
        if unknown:
            raise SystemExit(f"SIB-200 {split}: categories not in labels.txt: {unknown}")
        parts.append(pd.DataFrame({
            "id": "sib200-" + m["index_id"], "dataset": "sib200", "split": split,
            "src": m["text_s"], "tgt": m["text_t"], "topic": m["category_s"],
            "group_id": "sib200-" + m["index_id"],
        }))
    out = pd.concat(parts, ignore_index=True)
    return out, {"pairs": len(out), "labels": labels_s}


def load_in22(raw, P, file_key, name):
    path = raw / P[file_key]
    try:
        df = pd.read_parquet(path)
    except ImportError as e:
        raise SystemExit(f"Reading parquet needs pyarrow in env shiv ({e}). "
                         "Install it via slurm/install.sbatch (python -m pip install pyarrow).")
    sc, tc = P["src_col"], P["tgt_col"]
    for c in (sc, tc, "domain"):
        if c not in df.columns:
            raise SystemExit(f"{name}: column {c} missing; columns are {list(df.columns)}")
    if name == "in22_conv":
        rid = df["doc_id"].astype(str) + "-" + df["sent_id"].astype(str)
        group = name + "-" + df["doc_id"].astype(str)
    else:   # IN22-Gen: one row per sentence; sentences from the same source page share a url/context
        rid = pd.Series(range(len(df)), index=df.index).astype(str)
        key = df["url"].fillna("").astype(str)
        key = key.where(key != "", df["context"].fillna("").astype(str).str[:80])
        group = name + "-" + pd.Series(pd.factorize(key)[0], index=df.index).astype(str)
    out = pd.DataFrame({
        "id": name + "-" + rid, "dataset": name, "split": "test",
        "src": df[sc], "tgt": df[tc], "topic": df["domain"], "group_id": group,
    })
    if "topic" in df.columns:
        out["subtopic"] = df["topic"].values
    for c in P.get("extra_cols", []):
        if c in df.columns:
            out[c] = df[c].values
    return out, {"pairs": len(out), "groups": int(out["group_id"].nunique())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    P = cfg["prepare"]
    raw, proc, res = resolve(cfg["paths"]["raw"]), resolve(cfg["paths"]["processed"]), resolve(cfg["paths"]["results"])
    proc.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)

    report = {"src_lang": P["src_lang"], "tgt_lang": P["tgt_lang"], "datasets": {}}
    tables = {}
    for name, fn in (("massive", lambda: load_massive(raw, P)),
                     ("sib200", lambda: load_sib(raw, P)),
                     ("in22_conv", lambda: load_in22(raw, P, "in22_conv", "in22_conv")),
                     ("in22_gen", lambda: load_in22(raw, P, "in22_gen", "in22_gen"))):
        df, info = fn()
        check_pairs(df, name)
        tables[name] = df
        info["per_split"] = df["split"].value_counts().to_dict()
        info["topics"] = {s: g["topic"].value_counts().to_dict() for s, g in df.groupby("split")}
        report["datasets"][name] = info
        out = proc / f"{name}.tsv"
        df.to_csv(out, sep="\t", index=False)
        print(f"{name:10s} {len(df):6d} pairs  splits={info['per_split']}  topics={df['topic'].nunique()}  -> {out}")

    # Exact-text overlap between the training data and every evaluation set (reported, not removed).
    train = tables["massive"][tables["massive"]["split"] == "train"]
    tr_src, tr_tgt = set(train["src"].str.strip()), set(train["tgt"].str.strip())
    overlap = {}
    for name, df in tables.items():
        ev = df if name != "massive" else df[df["split"] != "train"]
        overlap[name] = {"src_in_train": int(ev["src"].str.strip().isin(tr_src).sum()),
                         "tgt_in_train": int(ev["tgt"].str.strip().isin(tr_tgt).sum()),
                         "eval_rows": int(len(ev))}
    report["overlap_with_massive_train"] = overlap
    print("overlap with MASSIVE train (exact text):", overlap)

    save_json(report, res / "prepare_report.json")
    print(f"Wrote {res / 'prepare_report.json'}")


if __name__ == "__main__":
    main()