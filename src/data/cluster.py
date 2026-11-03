"""Week 2, step 2 — exploratory topic clustering of the Hindi side of the TRAINING pairs.

Problem: find out which topics the Kangri data actually contains, before fixing a topic set.
This is exploration, not evaluation: clusters must be read and named by a person.

Input : cluster.input    data/processed/pairs_split.tsv  (only rows with split == cluster.split are used)
        cluster.gold_ids data/annotation/gold_dev_ids.txt (the script stops if any of these ids reach it)
Output (all in <results>/, or <results>/smoke/ with --smoke):
  clusters.tsv               id + one column per (model, k) with the cluster number
  summary_<model>_k<k>.json  per cluster: size, source mix, keywords, example pairs
  cluster_metrics.json       silhouette per (model, k), LaBSE-vs-E5 agreement (ARI), block contiguity
  <interim>/emb_<model>_<side>.npy + _ids.txt   embeddings and the ids they belong to (full run only)

Block contiguity: both corpora were written topic by topic, so neighbouring lines tend to share a
topic. For each source we measure how often two truly adjacent lines (original line numbers n and
n+1, both in the training pool) fall in the same cluster, against chance for that cluster
distribution. A ratio well above 1 means the clusters track the real topic blocks, not noise.

Smoke test: --smoke 2000 clusters 2,000 random training pairs with k <= 8 and writes to <results>/smoke/.
"""
import os
# Must be set BEFORE importing torch / transformers.
# Inside a Slurm job, Slurm already restricts which GPU is visible, and the
# allocated GPU appears as index 0. So use "0" -- never a physical GPU number.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# him2 has more CPU cores than OpenBLAS supports (max 64). Cap math-library threads
# BEFORE numpy/torch load, otherwise KMeans crashes with "double free or corruption".
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("SLURM_CPUS_PER_TASK", "16"))

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available - refusing to run on CPU. "
                       "Did you submit via sbatch/srun with --gres=gpu:1?")
device = torch.device("cuda")
print("Using GPU:", torch.cuda.get_device_name(0))

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import load_config, resolve, save_json, set_seed  # noqa: E402

# Devanagari letters and marks only: excludes danda (U+0964/5) and Devanagari digits (U+0966–096F).
TOKEN = re.compile(r"[ऀ-ॣ॰-ॿ]+")
HINDI_STOP = set("""
और का की के को में है हैं था थी थे से पर भी तो ही यह वह ये वो इस उस इन उन एक कि जो
ना न नहीं हो होता होती होते होना हुआ हुई हुए कर करना करते करती करता किया कुछ सब
अपने अपनी अपना मैं हम तुम आप वे उसे उसका उसकी उनके उनकी मेरा मेरी मेरे हमारे हमारी
तक लिए साथ बाद फिर अब जब तब कब क्या क्यों कैसे कहाँ यहाँ वहाँ बहुत सकता सकती सकते
रहा रही रहे गया गई गए जाता जाती जाते जा दिया दी दे ले लेकिन या तरह ऐसे वाले वाली
ने जी मुझे मुझ हूं हूँ कोई किसी कभी बारे हाँ हां मैंने उसने उसको उन्हें उन्होंने इसे इसका
इसकी इसके उसके तुम्हें तुम्हारा तुम्हारी आपका आपकी आपके कहा कहते कहती कहता बोला बोली
बात चाहिए जैसे वैसे इतना उतना जितना सबसे कौन किस किसे यदि अगर तथा एवं द्वारा रूप
""".split())


def tokens(text):
    return [t for t in TOKEN.findall(text) if len(t) > 1 and t not in HINDI_STOP]


def ctfidf_keywords(texts, labels, k, n):
    """Class-based TF-IDF (as in BERTopic): words frequent in one cluster but not everywhere."""
    counts = [Counter() for _ in range(k)]
    for t, c in zip(texts, labels):
        counts[c].update(tokens(t))
    total_words = sum(sum(c.values()) for c in counts)
    avg = total_words / k
    freq_all = Counter()
    for c in counts:
        freq_all.update(c)
    out = []
    for c in counts:
        size = sum(c.values()) or 1
        scored = {w: (f / size) * math.log(1 + avg / freq_all[w]) for w, f in c.items() if f >= 3}
        out.append([w for w, _ in sorted(scored.items(), key=lambda x: -x[1])[:n]])
    return out


def contiguity(df, labels):
    """Same-cluster rate for truly adjacent lines (n, n+1) vs chance, per source."""
    res = {}
    for src, g in df.assign(_c=labels).groupby("source"):
        g = g.sort_values("orig_line")
        c, line = g["_c"].to_numpy(), g["orig_line"].to_numpy()
        adj = np.diff(line) == 1          # gaps = lines removed in cleaning or held out as gold
        p = np.bincount(c, minlength=labels.max() + 1) / len(c)
        chance = float((p ** 2).sum())
        if adj.sum() == 0:
            res[src] = {"same_cluster_rate": None, "chance_rate": round(chance, 3), "ratio": None,
                        "n_adjacent": 0}
            continue
        same = float((c[1:] == c[:-1])[adj].mean())
        res[src] = {"same_cluster_rate": round(same, 3), "chance_rate": round(chance, 3),
                    "ratio": round(same / chance, 2) if chance else None, "n_adjacent": int(adj.sum())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", type=int, default=0,
                    help="cluster only N random training pairs with k <= 8; results go to <results>/smoke")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)
    P, C = cfg["paths"], cfg["cluster"]
    res = resolve(P["results"]) / ("smoke" if args.smoke else "")
    res.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(resolve(C["input"]), sep="\t", dtype=str, keep_default_na=False)
    if "split" not in df.columns:
        raise SystemExit(f"{C['input']} has no split column - run split.py and point cluster.input at its output")
    df["orig_line"] = df["orig_line"].astype(int)
    df = df[df["split"] == C["split"]].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"no pairs with split == {C['split']!r} in {C['input']}")

    gold_path = resolve(C["gold_ids"])
    if not gold_path.exists():
        raise SystemExit(f"{gold_path} not found - the gold set must be split off before clustering")
    gold_ids = set(gold_path.read_text(encoding="utf-8").split())
    leak = gold_ids & set(df["id"])
    if leak:
        raise SystemExit(f"{len(leak)} gold ids are in the clustering input - stopping (e.g. {sorted(leak)[:5]})")
    print(f"Gold-id check passed: none of the {len(gold_ids)} gold ids are in the input")

    k_values = list(C["k_values"])
    if args.smoke:
        df = df.sample(n=min(args.smoke, len(df)), random_state=seed).reset_index(drop=True)
        k_values = [k for k in k_values if k <= 8] or k_values[:1]
        print(f"SMOKE TEST: {len(df)} pairs, k = {k_values}")
    texts = df[C["text_side"]].tolist()
    print(f"Clustering {len(df)} '{C['split']}' pairs on the {C['text_side']} side")

    from sentence_transformers import SentenceTransformer
    assignments = df[["id"]].copy()
    metrics = {"n_pairs": int(len(df)), "silhouette": {}, "contiguity": {}, "agreement_ari": {}}
    rng = np.random.default_rng(seed)
    sil_idx = rng.choice(len(df), size=min(C["silhouette_sample"], len(df)), replace=False)

    for m in C["models"]:
        print(f"\n=== {m['name']} ({m['hf_id']})")
        model = SentenceTransformer(m["hf_id"], device="cuda")
        emb = model.encode([m["prefix"] + t for t in texts], batch_size=C["batch_size"],
                           convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
        if not args.smoke:
            stem = resolve(P["interim"]) / f"emb_{m['name']}_{C['text_side']}"
            np.save(f"{stem}.npy", emb)
            Path(f"{stem}_ids.txt").write_text("\n".join(df["id"]) + "\n", encoding="utf-8")
        del model
        torch.cuda.empty_cache()

        metrics["silhouette"][m["name"]] = {}
        metrics["contiguity"][m["name"]] = {}
        for k in k_values:
            labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(emb)
            col = f"{m['name']}_k{k}"
            assignments[col] = labels
            sil = float(silhouette_score(emb[sil_idx], labels[sil_idx], metric="cosine"))
            metrics["silhouette"][m["name"]][str(k)] = round(sil, 4)
            metrics["contiguity"][m["name"]][str(k)] = contiguity(df, labels)
            print(f"k={k:3d}  silhouette={sil:.4f}")

            kw = ctfidf_keywords(texts, labels, k, C["n_keywords"])
            clusters = []
            for c in range(k):
                idx = np.flatnonzero(labels == c)
                ex = rng.choice(idx, size=min(C["n_examples"], len(idx)), replace=False)
                sub = df.iloc[idx]
                clusters.append({
                    "cluster": c, "size": int(len(idx)),
                    "share_pct": round(100 * len(idx) / len(df), 2),
                    "sources": {s: int(n) for s, n in sub["source"].value_counts().items()},
                    "keywords": kw[c],
                    "examples": df.iloc[ex][["id", "source", "orig_line", "hindi", "kangri"]].to_dict(orient="records"),
                })
            clusters.sort(key=lambda x: -x["size"])
            save_json({"model": m["name"], "k": k, "silhouette": round(sil, 4), "clusters": clusters},
                      res / f"summary_{m['name']}_k{k}.json")

    names = [m["name"] for m in C["models"]]
    if len(names) >= 2:
        a, b = names[0], names[1]
        for k in k_values:
            metrics["agreement_ari"][str(k)] = round(float(adjusted_rand_score(
                assignments[f"{a}_k{k}"], assignments[f"{b}_k{k}"])), 4)
        metrics["agreement_models"] = [a, b]

    assignments.to_csv(res / "clusters.tsv", sep="\t", index=False)
    save_json(metrics, res / "cluster_metrics.json")
    print(f"Done. Results in {res}")


if __name__ == "__main__":
    main()