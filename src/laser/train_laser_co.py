"""LASER3-CO on the Bengali–Hindi pilot (Tan et al., EACL 2023), optionally plus our topic loss.

Problem: fine-tune the Bengali LASER encoder (student) so each Bengali sentence lands next to the frozen Hindi
encoder's (teacher) embedding of its translation, using InfoNCE with a queue of earlier teacher embeddings.
Input : config `laser:` (languages, model_dir), `laser_co:` (hyper-parameters, variants), `train:` (data, dev)
        --variant co | co_filter | co_topic  --seed N  --max_steps N (smoke test)
Output: <laser_co.output_dir>/laser_<variant>[_s<seed>]/best_student.pt   student weights at the best dev score
        .../train_log.json                                                losses and dev scores (dashboard)
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("SLURM_CPUS_PER_TASK", "16"))

import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available - refusing to run on CPU. Submit via sbatch with --gres=gpu:1")
DEVICE = "cuda"
print("Using GPU:", torch.cuda.get_device_name(0))

# PyTorch 2.6+ compatibility: weights_only defaults to True, which blocks
# loading laser_encoders checkpoints containing argparse.Namespace and fairseq objects.
try:
    import argparse
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([argparse.Namespace])
except Exception:
    pass

_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import load_config, resolve, save_json, set_seed  # noqa: E402


# ---------------------------------------------------------------- LASER encoders as torch modules
def torch_modules(obj, depth=0, seen=None):
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 4:
        return []
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        return [obj]
    found = []
    for val in (vars(obj).values() if hasattr(obj, "__dict__") else []):
        found += torch_modules(val, depth + 1, seen)
    return found


class LaserSide:
    """One LASER encoder (one language) as a torch module: trainable (student) or frozen (teacher)."""
    def __init__(self, lang, model_dir, trainable):
        from laser_encoders import LaserEncoderPipeline
        self.pipe = LaserEncoderPipeline(lang=lang, model_dir=str(model_dir))
        mods = torch_modules(self.pipe)
        if not mods:
            raise SystemExit(f"{lang}: no torch module found inside the LASER pipeline")
        self.module = max(mods, key=lambda m: sum(p.numel() for p in m.parameters())).to(DEVICE)
        self.trainable = trainable
        for p in self.module.parameters():
            p.requires_grad_(trainable)
        self.module.train(trainable)

    def forward(self, texts):
        """Embeddings (batch x dim) WITH gradients for the student.
        Follows the code path of laser_encoders' encode_sentences without torch.no_grad().
        """
        if not texts:
            return torch.empty((0, 1024), device=DEVICE)
        enc = self.pipe.encoder
        tok = self.pipe.tokenizer
        tokens = [enc._tokenize(tok.tokenize(t)) for t in texts]
        lengths = torch.tensor([t.numel() for t in tokens], dtype=torch.long)

        # Sort descending by length in case an LSTM encoder is used
        sort_order = torch.argsort(-lengths)
        rev_order = torch.argsort(sort_order)

        sorted_tokens = [tokens[i] for i in sort_order]
        sorted_lengths = lengths[sort_order].to(DEVICE)

        max_len = max(len(t) for t in tokens)
        pad_idx = enc.pad_index
        left_pad = enc.left_padding

        batch_toks = torch.full((len(texts), max_len), pad_idx, dtype=torch.long)
        for i, t in enumerate(sorted_tokens):
            if not left_pad:
                batch_toks[i, :len(t)] = t
            else:
                batch_toks[i, -len(t):] = t
        batch_toks = batch_toks.to(DEVICE)

        out = self.module(batch_toks, sorted_lengths)
        sentemb = out["sentemb"] if isinstance(out, dict) else out[0]
        return sentemb[rev_order]

    @torch.no_grad()
    def encode(self, texts, bs=128):
        was = self.module.training
        self.module.eval()
        out = [F.normalize(self.forward(texts[i:i + bs]).float(), dim=-1) for i in range(0, len(texts), bs)]
        self.module.train(was)
        return torch.cat(out)


# ---------------------------------------------------------------- data (same sets as the pilot)
def load_split(proc, name, split):
    df = pd.read_csv(proc / f"{name}.tsv", sep="\t", dtype=str, keep_default_na=False)
    df = df[df["split"] == split].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"{name}: no rows with split == {split}")
    return df


def make_batches(sets, batch_size, shuffle, rng):
    """Batches from one dataset at a time. shuffle=False: sort by Bengali length, keep that order (-Filter)."""
    batches = []
    for name, df, repeat in sets:
        rows = df[["src", "tgt", "topic"]].to_dict("records") * repeat
        for r in rows:
            r["topic"] = f"{name}:{r['topic']}"  # label lists of different datasets never mix
        if shuffle:
            rng.shuffle(rows)
        else:
            rows.sort(key=lambda r: len(r["src"].split()))
        batches += [rows[i:i + batch_size] for i in range(0, len(rows) - batch_size + 1, batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


# ---------------------------------------------------------------- losses
class Queue:
    """FIFO queue of teacher embeddings used as extra negatives (no momentum encoder)."""
    def __init__(self, size):
        self.size, self.buf = size, None

    def get(self):
        return self.buf

    @torch.no_grad()
    def push(self, k):
        self.buf = k.detach() if self.buf is None else torch.cat([k.detach(), self.buf])[: self.size]


def queue_infonce(q, k, queue, temp, sigma, in_batch, tgt_texts):
    """InfoNCE: q_i must be closer to k_i than to the negatives (other batch targets and/or the queue).
    sigma: drop negatives whose cosine with the true target k_i is >= sigma (pre-filter of -Filter)."""
    parts = [(q * k).sum(-1, keepdim=True)]  # positive similarity, column 0
    if in_batch:
        n = len(tgt_texts)
        drop = torch.tensor([[i == j or tgt_texts[i] == tgt_texts[j] for j in range(n)] for i in range(n)],
                            device=q.device)
        if sigma is not None:
            drop = drop | ((k @ k.T) >= sigma)
        parts.append((q @ k.T).masked_fill(drop, float("-inf")))
    if queue is not None and len(queue):
        nq = q @ queue.T
        if sigma is not None:
            nq = nq.masked_fill((k @ queue.T) >= sigma, float("-inf"))
        parts.append(nq)
    logits = torch.cat(parts, dim=1) / temp
    return F.cross_entropy(logits, torch.zeros(len(q), dtype=torch.long, device=q.device))


def supcon_loss(z, labels, temp):
    """Supervised contrastive loss (Khosla et al., 2020), as in our pilot train.py."""
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
def dev_scores(student, teacher, dev, pool, k=5):
    q = student.encode(dev["src"].tolist())
    qt = teacher.encode(dev["tgt"].tolist())
    p = student.encode(pool["src"].tolist())
    nn = (q @ p.T).topk(k, dim=1).indices.cpu().numpy()
    pool_lab, gold = pool["topic"].to_numpy(), dev["topic"].to_numpy()
    pred = [pd.Series(pool_lab[row]).value_counts().index[0] for row in nn]
    acc = float((np.array(pred) == gold).mean() * 100)
    best = (q @ qt.T).argmax(dim=1).cpu().numpy()
    tgt = dev["tgt"].to_numpy()
    xsim = float((tgt[best] != tgt).mean() * 100)
    return round(acc, 2), round(xsim, 2)


def better(acc, x, best, select_by):
    if select_by == "xsim":
        return (x, -acc) < (best["xsim"], -best["topic_acc"])
    return (acc, -x) > (best["topic_acc"], -best["xsim"])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (smoke test)")
    # Allow arguments passed as single space-separated strings from bash (e.g. sbatch ... "--max_steps 20")
    raw_args = []
    for a in sys.argv[1:]:
        raw_args.extend(a.split())
    args = ap.parse_args(raw_args)
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    L, C, T = cfg["laser"], cfg["laser_co"], cfg["train"]
    var = C["variants"][args.variant]
    w_topic, sigma, select_by = float(var["topic"]), var["filter_sigma"], var["select_by"]
    proc = resolve(cfg["paths"]["processed"])
    run = (f"laser_{args.variant}" + (f"_s{seed}" if seed != cfg["seed"] else "")
           + ("_smoke" if args.max_steps else ""))
    out_dir = resolve(C["output_dir"]) / run
    out_dir.mkdir(parents=True, exist_ok=True)

    train_sets = [(d["name"], load_split(proc, d["name"], d["split"]), int(d.get("repeat", 1))) for d in T["data"]]
    dev = load_split(proc, T["dev"]["name"], T["dev"]["split"])
    pool = next(df for name, df, _ in train_sets if name == T["dev"]["name"])
    model_dir = resolve(L["model_dir"])
    student = LaserSide(L["src_lang"], model_dir, trainable=True)  # Bengali, trained
    teacher = LaserSide(L["tgt_lang"], model_dir, trainable=False)  # Hindi, frozen
    temp, bs = float(C["temperature"]), int(C["batch_size"])

    steps_per_epoch = len(make_batches(train_sets, bs, var["shuffle"], random.Random(0)))
    total = steps_per_epoch * int(C["epochs"])
    if args.max_steps:
        total = min(total, args.max_steps)
    warm = max(1, int(float(C["warmup_ratio"]) * total))
    params = [p for p in student.module.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(C["lr"]), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    queue = Queue(int(C["queue_size"]))
    print(f"run {run}: variant {var}; {steps_per_epoch} steps/epoch, {total} steps; "
          + ", ".join(f"{n}={len(df)}x{r}" for n, df, r in train_sets) + f"; dev {len(dev)}")

    log = {"run": run, "variant": var, "steps_total": total, "dev": [], "loss": []}
    acc0, x0 = dev_scores(student, teacher, dev, pool)
    log["dev"].append({"step": 0, "topic_acc": acc0, "xsim": x0})
    best = {"step": 0, "topic_acc": acc0, "xsim": x0}
    torch.save(student.module.state_dict(), out_dir / "best_student.pt")
    print(f"step 0 dev topic acc {acc0:.2f} dev xsim {x0:.2f} (before training)")

    rng, step, t0 = random.Random(seed), 0, time.time()
    for epoch in range(int(C["epochs"])):
        for batch in make_batches(train_sets, bs, var["shuffle"], rng):
            if step >= total:
                break
            q = F.normalize(student.forward([r["src"] for r in batch]).float(), dim=-1)
            with torch.no_grad():
                k = F.normalize(teacher.forward([r["tgt"] for r in batch]).float(), dim=-1)
            l_co = queue_infonce(q, k, queue.get(), temp, sigma, C["in_batch_negatives"], [r["tgt"] for r in batch])
            loss, parts = l_co, {"contrastive": round(l_co.item(), 4)}
            if w_topic > 0:
                labels = [r["topic"] for r in batch]
                l_tp = supcon_loss(torch.cat([q, k]), labels + labels, temp)  # gradient reaches q only
                loss = loss + w_topic * l_tp
                parts["topic"] = round(l_tp.item(), 4)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            queue.push(k)
            step += 1
            if step % 10 == 0:
                log["loss"].append({"step": step, "loss": round(loss.item(), 4), **parts})
            if step % int(C["eval_every"]) == 0 or step == total:
                acc, x = dev_scores(student, teacher, dev, pool)
                log["dev"].append({"step": step, "topic_acc": acc, "xsim": x})
                mark = ""
                if better(acc, x, best, select_by):
                    best = {"step": step, "topic_acc": acc, "xsim": x}
                    torch.save(student.module.state_dict(), out_dir / "best_student.pt")
                    mark = " <- best, saved"
                print(f"step {step:4d}/{total} loss {loss.item():.4f} {parts} dev topic acc {acc:.2f} "
                      f"dev xsim {x:.2f} ({time.time() - t0:.0f}s){mark}")
            if step >= total:
                break

    log["best"], log["train_seconds"] = best, round(time.time() - t0, 1)
    save_json(log, out_dir / "train_log.json")
    print(f"\nBest (by {select_by}) at step {best['step']}: topic acc {best['topic_acc']}, xsim {best['xsim']}. "
          f"Before training: {acc0} / {x0}. Weights: {out_dir / 'best_student.pt'}")


if __name__ == "__main__":
    main()
