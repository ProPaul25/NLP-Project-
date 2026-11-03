"""Score LASER3 (untrained baseline or fine-tuned student) exactly like evaluate.py scores the other encoders.

Problem: LASER3 is the encoder family of the bitext-mining papers this project builds on, so it is the
natural untrained baseline. It cannot be loaded by sentence-transformers and needs fairseq, which does not
install in env `shiv` (Python 3.11). It therefore runs in its own env `laser` (Python 3.8, torch 2.1.2),
and only the encoding differs; all metrics come from evaluate.score_set, so the numbers are comparable.

Input : config `laser:` section (name, src_lang, tgt_lang as FLORES codes, model_dir)
        config `evaluate:` section (the same sets, knn_k, exclude_same_group, n_boot)
        data/processed/<pilot>/<set>.tsv
        optional: --name <name> and --student_ckpt <path_to_best_student.pt>
Output: the same files evaluate.py writes, under the model name from `laser:` (or --name):
        <results>/eval/<name>__<set>.json, <name>__<set>_per_sentence.tsv, and its rows in results.csv
How LASER3 is used: each language has its own encoder (laser_encoders picks the LASER3 encoder for the
code if one exists, else LASER2); both map into the same LASER2 space, so source sentences are encoded
with the source-language encoder and target sentences with the target-language encoder.
Embeddings are L2-normalised, so similarities are cosine, as for the other models.
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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import load_config, resolve, save_json, set_seed  # noqa: E402
from src.eval.evaluate import score_set  # noqa: E402


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


def laser_encoder(lang, model_dir):
    from laser_encoders import LaserEncoderPipeline
    model_dir.mkdir(parents=True, exist_ok=True)
    return LaserEncoderPipeline(lang=lang, model_dir=str(model_dir))


def encode(enc, texts):
    emb = enc.encode_sentences(list(texts), normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", type=str, default=None, help="override model run name (e.g. laser_co)")
    ap.add_argument("--student_ckpt", type=str, default=None, help="path to best_student.pt checkpoint")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)
    E, L = cfg["evaluate"], cfg["laser"]
    name = args.name if args.name else L["name"]
    proc = resolve(cfg["paths"]["processed"])
    out_dir = resolve(cfg["paths"]["results"]) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = resolve(L["model_dir"])

    enc_src = laser_encoder(L["src_lang"], model_dir)
    enc_tgt = laser_encoder(L["tgt_lang"], model_dir)

    if args.student_ckpt:
        ckpt_path = resolve(args.student_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        mods = torch_modules(enc_src)
        if not mods:
            raise SystemExit(f"{L['src_lang']}: no torch module found inside the LASER pipeline")
        student_mod = max(mods, key=lambda m: sum(p.numel() for p in m.parameters()))
        device = next(student_mod.parameters()).device
        student_mod.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded student weights from {ckpt_path}")

    files = sorted(p.name for p in model_dir.iterdir())
    print(f"=== {name}: src {L['src_lang']}, tgt {L['tgt_lang']}; model files in {model_dir}: {files}")

    rows = []
    for s in E["sets"]:
        set_name = f"{s['name']}:{s['split']}"
        df = pd.read_csv(proc / f"{s['name']}.tsv", sep="\t", dtype=str, keep_default_na=False)
        df = df[df["split"] == s["split"]].reset_index(drop=True)
        if len(df) == 0:
            raise SystemExit(f"{set_name}: no rows")
        src = encode(enc_src, df["src"])
        tgt = encode(enc_tgt, df["tgt"])
        if src.shape[0] != len(df) or tgt.shape[0] != len(df):
            raise SystemExit(f"{set_name}: got {src.shape[0]}/{tgt.shape[0]} embeddings for {len(df)} pairs")
        m, per_sentence = score_set(src, tgt, df, E["knn_k"], E["exclude_same_group"], seed,
                                    int(E.get("n_boot", 1000)))
        m.update({"model": name,
                  "model_path": f"laser_encoders {L['src_lang']}/{L['tgt_lang']}" + (f" (student: {args.student_ckpt})" if args.student_ckpt else ""),
                  "set": set_name, "laser_files": files})
        stem = f"{name}__{set_name.replace(':', '_')}"
        save_json(m, out_dir / f"{stem}.json")
        per_sentence.to_csv(out_dir / f"{stem}_per_sentence.tsv", sep="\t", index=False)
        rows.append({k: v for k, v in m.items() if not isinstance(v, (dict, list))})
        print(f"{set_name:16s} xsim s>t {m['xsim_src2tgt']:6.2f}  t>s {m['xsim_tgt2src']:6.2f}  "
              f"knn src {m['knn_acc_src']:6.2f} [{m['knn_acc_src_lo']:.1f}-{m['knn_acc_src_hi']:.1f}]  "
              f"tgt {m['knn_acc_tgt']:6.2f}  cross {m['knn_acc_cross']:6.2f}  "
              f"nmi {m['nmi_src']:6.2f}  (majority {m['majority_baseline_acc']:.2f})")

    table = out_dir / "results.csv"
    new = pd.DataFrame(rows)
    if table.exists():
        old = pd.read_csv(table)
        old = old[old["model"] != name]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(table, index=False)
    print(f"\nWrote {table}")


if __name__ == "__main__":
    main()