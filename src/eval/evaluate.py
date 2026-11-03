"""Score sentence encoders on held-out sets: translation retrieval and topic structure.

Problem: every model in the results table (untrained baselines now, fine-tuned models later) must be
measured the same way, on data it never trained on.

Input : config `evaluate:` section  - sets (processed TSV + split) and models (HF id or local checkpoint)
        data/processed/<pilot>/<set>.tsv  columns id, src, tgt, topic, group_id (from prepare_pilot.py)
Output: <results>/eval/<model>__<set>.json   all metrics + 95% bootstrap intervals + per-topic accuracy
        <results>/eval/<model>__<set>_per_sentence.tsv   right/wrong per sentence (for paired model comparisons)
        <results>/eval/results.csv            one row per (model, set); re-running a model replaces its rows

Metrics (all on L2-normalised embeddings, cosine similarity):
  xsim_src2tgt / xsim_tgt2src  translation retrieval error (%): for each source sentence, is the nearest of
                               ALL target sentences its own translation? Lower is better. Same idea as the
                               xsim used in the LASER3 papers (plain cosine here, not margin-based).
  knn_acc_src / _tgt           topic accuracy: majority topic of the k nearest OTHER sentences, same language.
  knn_acc_cross                query in source language, neighbours from the target language (its own
                               translation excluded): does topic structure survive across languages?
  knn_f1_*                     macro-F1 of the same predictions (topics are unbalanced in IN22-Conv)
  nmi_src / ari_src            KMeans with k = number of topics on source embeddings vs gold topics
Uncertainty: *_lo / *_hi = 95% bootstrap interval (n_boot resamples). Whole groups are resampled together,
so the interval for IN22-Conv reflects its 44 conversations, not 1,503 independent sentences.
With exclude_same_group, neighbours from the same group (IN22-Conv conversation) are ignored, otherwise
turns of one conversation would vote for each other and inflate topic accuracy.
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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import load_config, resolve, save_json, set_seed  # noqa: E402


def encode(model, texts, prefix, batch_size):
    emb = model.encode([prefix + t for t in texts], batch_size=batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
    return emb.astype(np.float32)


def xsim_wrong(q, d, q_text_gold, d_text):
    """Per query: True if its nearest document is not its translation (identical texts count as correct)."""
    nn = (q @ d.T).argmax(axis=1)
    return np.array([d_text[j] != g for j, g in zip(nn, q_text_gold)])


def bootstrap_ci(values, groups, n_boot, seed):
    """95% percentile interval of the mean (in %), resampling whole groups so that sentences of one
    conversation/document are drawn together (for MASSIVE/SIB-200 every sentence is its own group)."""
    values = np.asarray(values, dtype=float)
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(100 * lo, 2), round(100 * hi, 2)


def knn_predict(q, d, labels, groups, k, exclude_same_group, exclude_self):
    """Majority topic of the k most similar items in d for each item in q (ties: higher summed similarity)."""
    sims = q @ d.T
    n = len(q)
    if exclude_self:                         # q and d are the same items (or translations of each other)
        sims[np.arange(n), np.arange(n)] = -np.inf
    if exclude_same_group:
        same = groups[:, None] == groups[None, :]
        if exclude_self:
            sims[same] = -np.inf
    top = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    preds = []
    for i in range(n):
        score = {}
        for j in top[i]:
            if np.isfinite(sims[i, j]):
                score[labels[j]] = score.get(labels[j], 0.0) + 1.0 + 1e-3 * sims[i, j]
        preds.append(max(score, key=score.get) if score else None)
    return np.array(preds, dtype=object)


def topic_scores(gold, pred, tag):
    ok = pred != None  # noqa: E711
    return {f"knn_acc_{tag}": round(100 * float((pred[ok] == gold[ok]).mean()), 2),
            f"knn_f1_{tag}": round(100 * f1_score(gold[ok], pred[ok].astype(str), average="macro"), 2)}


def score_set(src, tgt, df, k, excl_group, seed, n_boot):
    gold = df["topic"].astype(str).to_numpy()
    groups = df["group_id"].astype(str).to_numpy()
    s_text, t_text = df["src"].tolist(), df["tgt"].tolist()
    m = {"n": int(len(df)), "n_groups": int(len(set(groups))), "n_topics": int(len(set(gold)))}
    w_st = xsim_wrong(src, tgt, t_text, t_text)
    w_ts = xsim_wrong(tgt, src, s_text, s_text)
    m["xsim_src2tgt"] = round(100 * w_st.mean(), 2)
    m["xsim_tgt2src"] = round(100 * w_ts.mean(), 2)
    m["xsim_src2tgt_lo"], m["xsim_src2tgt_hi"] = bootstrap_ci(w_st, groups, n_boot, seed)
    p_src = knn_predict(src, src, gold, groups, k, excl_group, exclude_self=True)
    p_tgt = knn_predict(tgt, tgt, gold, groups, k, excl_group, exclude_self=True)
    p_x = knn_predict(src, tgt, gold, groups, k, excl_group, exclude_self=True)
    for tag, p in (("src", p_src), ("tgt", p_tgt), ("cross", p_x)):
        m.update(topic_scores(gold, p, tag))
    ok_src, ok_x = (p_src == gold), (p_x == gold)
    m["knn_acc_src_lo"], m["knn_acc_src_hi"] = bootstrap_ci(ok_src, groups, n_boot, seed)
    m["knn_acc_cross_lo"], m["knn_acc_cross_hi"] = bootstrap_ci(ok_x, groups, n_boot, seed)
    km = KMeans(n_clusters=m["n_topics"], n_init=10, random_state=seed).fit_predict(src)
    m["nmi_src"] = round(100 * normalized_mutual_info_score(gold, km), 2)
    m["ari_src"] = round(100 * adjusted_rand_score(gold, km), 2)
    per_topic = pd.DataFrame({"gold": gold, "pred": p_src}).assign(ok=lambda x: x.gold == x.pred)
    m["per_topic_acc_src"] = {t: round(100 * float(g.ok.mean()), 1) for t, g in per_topic.groupby("gold")}
    m["majority_baseline_acc"] = round(100 * pd.Series(gold).value_counts(normalize=True).iloc[0], 2)
    per_sentence = pd.DataFrame({
        "id": df["id"], "group_id": groups, "topic": gold,
        "pred_src": p_src, "ok_src": ok_src.astype(int),
        "pred_cross": p_x, "ok_cross": ok_x.astype(int),
        "xsim_ok_src2tgt": (~w_st).astype(int), "xsim_ok_tgt2src": (~w_ts).astype(int),
    })
    return m, per_sentence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", help="only these model names from the config (default: all)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)
    E = cfg["evaluate"]
    proc = resolve(cfg["paths"]["processed"])
    out_dir = resolve(cfg["paths"]["results"]) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    sets = {}
    for s in E["sets"]:
        df = pd.read_csv(proc / f"{s['name']}.tsv", sep="\t", dtype=str, keep_default_na=False)
        df = df[df["split"] == s["split"]].reset_index(drop=True)
        if len(df) == 0:
            raise SystemExit(f"{s['name']}: no rows with split == {s['split']}")
        sets[f"{s['name']}:{s['split']}"] = df
        print(f"set {s['name']}:{s['split']}  {len(df)} pairs, {df['topic'].nunique()} topics")

    from sentence_transformers import SentenceTransformer
    models = [m for m in E["models"] if not args.models or m["name"] in args.models]
    if not models:
        raise SystemExit(f"no models selected; config has {[m['name'] for m in E['models']]}")
    rows = []
    for mcfg in models:
        print(f"\n=== {mcfg['name']}  ({mcfg['path']})")
        model = SentenceTransformer(str(resolve(mcfg["path"])) if mcfg["path"].startswith(("outputs/", "data/"))
                                    else mcfg["path"], device="cuda")
        for set_name, df in sets.items():
            src = encode(model, df["src"].tolist(), mcfg["prefix"], E["batch_size"])
            tgt = encode(model, df["tgt"].tolist(), mcfg["prefix"], E["batch_size"])
            m, per_sentence = score_set(src, tgt, df, E["knn_k"], E["exclude_same_group"], seed,
                                        int(E.get("n_boot", 1000)))
            m.update({"model": mcfg["name"], "model_path": mcfg["path"], "set": set_name})
            stem = f"{mcfg['name']}__{set_name.replace(':', '_')}"
            save_json(m, out_dir / f"{stem}.json")
            per_sentence.to_csv(out_dir / f"{stem}_per_sentence.tsv", sep="\t", index=False)
            rows.append({k: v for k, v in m.items() if not isinstance(v, dict)})
            print(f"{set_name:16s} xsim s>t {m['xsim_src2tgt']:6.2f}  t>s {m['xsim_tgt2src']:6.2f}  "
                  f"knn src {m['knn_acc_src']:6.2f} [{m['knn_acc_src_lo']:.1f}-{m['knn_acc_src_hi']:.1f}]  "
                  f"tgt {m['knn_acc_tgt']:6.2f}  cross {m['knn_acc_cross']:6.2f}  "
                  f"nmi {m['nmi_src']:6.2f}  (majority {m['majority_baseline_acc']:.2f})")
        del model
        torch.cuda.empty_cache()

    table = out_dir / "results.csv"
    new = pd.DataFrame(rows)
    if table.exists():
        old = pd.read_csv(table)
        old = old[~old["model"].isin(new["model"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(table, index=False)
    print(f"\nWrote {table}")


if __name__ == "__main__":
    main()