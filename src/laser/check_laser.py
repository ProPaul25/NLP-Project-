"""Inspect the LASER encoders for Bengali, Hindi (and Kangri if any): which model, which torch module,
trainable?"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("SLURM_CPUS_PER_TASK", "16"))

import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available - submit through sbatch with --gres=gpu:1")

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

import inspect
from pathlib import Path

MODEL_DIR = Path("outputs/laser_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def torch_modules(obj, name="pipe", depth=0, seen=None):
    """All torch.nn.Module objects reachable from obj's attributes (not inside other modules)."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 4:
        return []
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        return [(name, obj)]
    found = []
    for key, val in (vars(obj).items() if hasattr(obj, "__dict__") else []):
        found += torch_modules(val, f"{name}.{key}", depth + 1, seen)
    return found


from laser_encoders import LaserEncoderPipeline  # noqa: E402

for lang in ["ben_Beng", "hin_Deva"]:
    pipe = LaserEncoderPipeline(lang=lang, model_dir=str(MODEL_DIR))
    print(f"\n=== {lang}: pipeline {type(pipe).__module__}.{type(pipe).__name__}")
    print("pipeline attributes:", sorted(vars(pipe)))
    for name, mod in torch_modules(pipe):
        n = sum(p.numel() for p in mod.parameters())
        p0 = next(mod.parameters())
        print(f"module {name}: {type(mod).__module__}.{type(mod).__name__}, {n/1e6:.1f}M params, "
              f"device {p0.device}, requires_grad {p0.requires_grad}")
        print(" forward signature:", inspect.signature(mod.forward))
        print(" first children:", [c for c, _ in list(mod.named_children())[:8]])
    emb = pipe.encode_sentences(["এটি একটি পরীক্ষা।" if lang == "ben_Beng" else "यह एक परीक्षा है।"],
                                normalize_embeddings=True)
    print("embedding shape:", getattr(emb, "shape", None))

print("\nfiles in", MODEL_DIR, ":", sorted(p.name for p in MODEL_DIR.iterdir()))
import laser_encoders  # noqa: E402
print("laser_encoders source folder:", Path(laser_encoders.__file__).parent)

for lang in ["kangri", "xnr_Deva"]:  # expected to fail: Kangri is not an NLLB-200 language
    try:
        LaserEncoderPipeline(lang=lang, model_dir=str(MODEL_DIR))
        print(f"{lang}: an encoder EXISTS")
    except Exception as e:
        print(f"{lang}: no encoder ({type(e).__name__}: {str(e)[:120]})")
