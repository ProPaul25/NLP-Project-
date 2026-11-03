"""Fine-tune a sentence encoder with a translation loss, a topic loss, or both.

Problem: make sentences on the same topic lie close together (topic loss) while a sentence stays closest
to its own translation (translation loss). Pilot: Bengali–Hindi; the same script later trains on Kangri–Hindi.

Input : config `train:` section; data/processed/<pilot>/<set>.tsv (id, src, tgt, topic, split, ...)
        --base   a key of train.base_models   (e.g. e5, labse)
        --variant a key of train.variants     (trans | topic | both)
Output: <output_dir>/<base>_<variant>/best/      the checkpoint with the best dev score (SentenceTransformer format)
        <output_dir>/<base>_<variant>/train_log.json   loss and dev scores over training (for the dashboard)

Losses (embeddings L2-normalised, cosine similarity / temperature):
  translation  in-batch InfoNCE in both directions: src_i must be closer to tgt_i than to every other tgt_j
               in the batch, and vice versa. Pairs whose target texts are identical are not used as negatives.
  topic        supervised contrastive loss (SupCon) over all 2B embeddings of the batch (src and tgt):
               every sentence is pulled toward all others with the same topic and pushed from the rest.
Batches come from one dataset at a time (batch_from_one_dataset), so topics of different label lists never
act as negatives for each other.
Dev score (picks the checkpoint): topic accuracy of dev sentences against the training-set sentences of the
same dataset (5 nearest neighbours, source side); translation error on dev is logged as a guard rail.
The test sets are never touched here.
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
DEVICE = "cuda"
print("Using GPU:", torch.cuda.get_device_name(0))

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import load_config, resolve, save_json, set_seed  # noqa: E402


# ---------------------------------------------------------------- data
def load_split(proc, name, split):
    df = pd.read_csv(proc / f"{name}.tsv", sep="\t", dtype=str, keep_default_na=False)
    df = df[df["split"] == split].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"{name}: no rows with split == {split}")
    return df


def make_batches(sets, batch_size, one_dataset, rng):
    """List of batches; each batch is a list of rows (dicts). Sets are repeated as configured."""
    pools = {}
    for name, df, repeat in sets:
        rows = df[["src", "tgt", "topic"]].to_dict("records") * repeat
        for r in rows:
            r["topic"] = f"{name}:{r['topic']}"          # keep label lists of different datasets apart
        pools[name] = rows
    batches = []
    if one_dataset:
        for rows in pools.values():
            rng.shuffle(rows)
            batches += [rows[i:i + batch_size] for i in range(0, len(rows) - batch_size + 1, batch_size)]
    else:
        rows = [r for rs in pools.values() for r in rs]
        rng.shuffle(rows)
        batches = [rows[i:i + batch_size] for i in range(0, len(rows) - batch_size + 1, batch_size)]
    rng.shuffle(batches)
    return batches


# ---------------------------------------------------------------- losses
def translation_loss(s, t, tgt_texts, temp):
    logits = s @ t.T / temp
    n = len(tgt_texts)
    same = torch.tensor([[tgt_texts[i] == tgt_texts[j] and i != j for j in range(n)] for i in range(n)],
                        device=s.device)
    logits = logits.masked_fill(same, float("-inf"))           # identical targets are not negatives
    labels = torch.arange(n, device=s.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def supcon_loss(z, labels, temp):
    """Supervised contrastive loss (Khosla et al., 2020), averaged over anchors that have a positive."""
    n = len(labels)
    lab = torch.tensor(pd.factorize(pd.Series(labels))[0], device=z.device)
    sim = z @ z.T / temp
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (lab[:, None] == lab[None, :]) & ~self_mask
    has_pos = pos.any(dim=1)
    if not has_pos.any():
        return z.sum() * 0.0
    mean_log_pos = (log_prob.masked_fill(~pos, 0.0).sum(dim=1) / pos.sum(dim=1).clamp(min=1))[has_pos]
    return -mean_log_pos.mean()


# ---------------------------------------------------------------- dev score
@torch.no_grad()
def embed(model, texts, prefix, bs=256):
    model.eval()
    out = model.encode([prefix + t for t in texts], batch_size=bs, convert_to_tensor=True,
                       normalize_embeddings=True, show_progress_bar=False)
    model.train()
    return out.float()


def dev_scores(model, prefix, dev, pool, k=5):
    q = embed(model, dev["src"].tolist(), prefix)
    qt = embed(model, dev["tgt"].tolist(), prefix)
    p = embed(model, pool["src"].tolist(), prefix)
    nn = (q @ p.T).topk(k, dim=1).indices.cpu().numpy()
    pool_lab, gold = pool["topic"].to_numpy(), dev["topic"].to_numpy()
    pred = [pd.Series(pool_lab[row]).value_counts().index[0] for row in nn]
    acc = float((np.array(pred) == gold).mean() * 100)
    best = (q @ qt.T).argmax(dim=1).cpu().numpy()
    tgt = dev["tgt"].to_numpy()
    xsim = float((tgt[best] != tgt).mean() * 100)
    return round(acc, 2), round(xsim, 2)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--seed", type=int, default=None, help="override the config seed (repeat runs)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    T = cfg["train"]
    base, var = T["base_models"][args.base], T["variants"][args.variant]
    w_tr, w_tp = float(var["translation"]), float(var["topic"])
    proc = resolve(cfg["paths"]["processed"])
    run = (f"{args.base}_{args.variant}" + (f"_s{seed}" if seed != cfg["seed"] else "")
           + ("_smoke" if args.max_steps else ""))
    out_dir = resolve(T["output_dir"]) / run
    out_dir.mkdir(parents=True, exist_ok=True)

    train_sets = [(d["name"], load_split(proc, d["name"], d["split"]), int(d.get("repeat", 1))) for d in T["data"]]
    dev = load_split(proc, T["dev"]["name"], T["dev"]["split"])
    pool = next(df for name, df, _ in train_sets if name == T["dev"]["name"])   # same-dataset training pool
    print(f"run {run}: weights translation={w_tr} topic={w_tp}; "
          + ", ".join(f"{n}={len(df)}x{r}" for n, df, r in train_sets) + f"; dev {len(dev)}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(base["path"], device=DEVICE)
    model.max_seq_length = int(T["max_len"])
    prefix, temp = base["prefix"], float(T["temperature"])

    rng = random.Random(seed)
    steps_per_epoch = len(make_batches(train_sets, T["batch_size"], T["batch_from_one_dataset"], random.Random(0)))
    total = steps_per_epoch * int(T["epochs"])
    if args.max_steps:
        total = min(total, args.max_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=float(T["lr"]), weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(float(T["warmup_ratio"]) * total), total)
    use_bf16 = torch.cuda.is_bf16_supported()

    log = {"run": run, "base": base["path"], "weights": var, "steps_total": total, "dev": [], "loss": []}
    acc0, x0 = dev_scores(model, prefix, dev, pool)
    log["dev"].append({"step": 0, "topic_acc": acc0, "xsim": x0})
    print(f"step 0  dev topic acc {acc0:.2f}  dev xsim {x0:.2f}  (before training)")
    best = acc0
    model.save(str(out_dir / "best"))
    log["best"] = {"step": 0, "topic_acc": acc0, "xsim": x0}

    step, t0 = 0, time.time()
    model.train()
    for epoch in range(int(T["epochs"])):
        for batch in make_batches(train_sets, T["batch_size"], T["batch_from_one_dataset"], rng):
            if step >= total:
                break
            src = [prefix + r["src"] for r in batch]
            tgt = [prefix + r["tgt"] for r in batch]
            feats = model.tokenize(src + tgt)
            feats = {k: v.to(DEVICE) for k, v in feats.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                emb = model(feats)["sentence_embedding"]
            emb = F.normalize(emb.float(), dim=-1)
            s, t = emb[:len(batch)], emb[len(batch):]
            loss = emb.sum() * 0.0
            parts = {}
            if w_tr > 0:
                lt = translation_loss(s, t, [r["tgt"] for r in batch], temp)
                loss = loss + w_tr * lt
                parts["translation"] = round(lt.item(), 4)
            if w_tp > 0:
                labels = [r["topic"] for r in batch]
                lc = supcon_loss(emb, labels + labels, temp)
                loss = loss + w_tp * lc
                parts["topic"] = round(lc.item(), 4)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 10 == 0:
                log["loss"].append({"step": step, "loss": round(loss.item(), 4), **parts})
            if step % int(T["eval_every"]) == 0 or step == total:
                acc, x = dev_scores(model, prefix, dev, pool)
                log["dev"].append({"step": step, "topic_acc": acc, "xsim": x})
                mark = ""
                if acc > best:
                    best = acc
                    model.save(str(out_dir / "best"))
                    log["best"] = {"step": step, "topic_acc": acc, "xsim": x}
                    mark = "  <- best, saved"
                print(f"step {step:4d}/{total}  epoch {epoch + 1}  loss {loss.item():.4f} {parts}  "
                      f"dev topic acc {acc:.2f}  dev xsim {x:.2f}  ({time.time() - t0:.0f}s){mark}")
        if step >= total:
            break

    log["train_seconds"] = round(time.time() - t0, 1)
    save_json(log, out_dir / "train_log.json")
    print(f"\nBest dev topic accuracy {log['best']['topic_acc']} at step {log['best']['step']} "
          f"(before training: {acc0}). Checkpoint: {out_dir / 'best'}")
    print("Add to the evaluate: models list ->  "
          + json.dumps({"name": run, "path": str((out_dir / 'best').relative_to(resolve('.'))), "prefix": prefix},
                       ensure_ascii=False))


if __name__ == "__main__":
    main()