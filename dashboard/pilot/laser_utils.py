"""Utilities for loading and encoding with LASER / LASER3-CO in the dashboard.

Handles:
  - PyTorch 2.6+ unpickling compatibility (weights_only=False)
  - Loading LaserEncoderPipeline for Bengali (student) and Hindi (frozen teacher)
  - Injecting fine-tuned student checkpoints (best_student.pt)
  - Tokenizing with the pipeline's sentencepiece tokenizer
  - Caching pipelines on GPU
"""
import os
import re
from pathlib import Path

import numpy as np
import streamlit as st
import torch

# PyTorch 2.6+ compatibility for fairseq / laser_encoders checkpoints
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


def torch_modules(obj, depth=0, seen=None):
    """Find all torch.nn.Module objects in obj."""
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


@st.cache_resource(show_spinner="Loading LASER encoder on GPU…")
def load_laser_pipeline(lang, model_dir, ckpt_path=None):
    """Load LaserEncoderPipeline for lang and optionally load student weights."""
    from laser_encoders import LaserEncoderPipeline
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    pipe = LaserEncoderPipeline(lang=lang, model_dir=str(model_dir))
    if ckpt_path and Path(ckpt_path).exists():
        mods = torch_modules(pipe)
        if mods:
            student_mod = max(mods, key=lambda m: sum(p.numel() for p in m.parameters()))
            device = next(student_mod.parameters()).device
            student_mod.load_state_dict(torch.load(ckpt_path, map_location=device))
    return pipe


def detect_laser_lang(text):
    """Detect if text contains Bengali or Devanagari script."""
    for ch in text:
        cp = ord(ch)
        if 0x0980 <= cp <= 0x09FF:
            return "ben_Beng"
        if 0x0900 <= cp <= 0x097F:
            return "hin_Deva"
    return "ben_Beng"


def _laser_root(project_root=None):
    """Ensure LASER models and checkpoints are read from Project_NLP_2."""
    if project_root is not None:
        p = Path(project_root)
        if (p / "outputs" / "laser_models").exists() or (p / "outputs" / "checkpoints" / "pilot_bn_laser").exists():
            return p
    default_laser = Path("/raid/home/loitongbam/Project_NLP_2")
    if default_laser.exists():
        return default_laser
    if project_root is not None:
        p2 = Path(project_root).parent / "Project_NLP_2"
        if p2.exists():
            return p2
        return Path(project_root)
    return Path.cwd()


def encode_laser_texts(model_name_or_path, texts, side="src", project_root=None):
    """Encode texts with LASER or fine-tuned LASER3-CO student."""
    root = _laser_root(project_root)
    model_dir = root / "outputs" / "laser_models"

    ckpt_path = None
    if isinstance(model_name_or_path, str) and ":" in model_name_or_path:
        ckpt_path = model_name_or_path.split(":", 1)[1]
    elif model_name_or_path in ["laser_co", "laser_co_filter", "laser_co_topic"]:
        base_ck = root / "outputs" / "checkpoints" / "pilot_bn_laser"
        p = base_ck / model_name_or_path / "best_student.pt"
        p_s42 = base_ck / f"{model_name_or_path}_s42" / "best_student.pt"
        if p.exists():
            ckpt_path = str(p)
        elif p_s42.exists():
            ckpt_path = str(p_s42)
        else:
            ckpt_path = str(p)

    # Target language (Hindi) is always the frozen teacher
    if side in ("tgt", "hin_Deva"):
        pipe = load_laser_pipeline("hin_Deva", model_dir, ckpt_path=None)
    else:
        pipe = load_laser_pipeline("ben_Beng", model_dir, ckpt_path=ckpt_path)

    emb = pipe.encode_sentences(list(texts), normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def tokenize_laser_text(text, side="src", project_root=None):
    """Return subword tokens for a sentence using LASER's tokenizer."""
    root = _laser_root(project_root)
    model_dir = root / "outputs" / "laser_models"
    lang = "hin_Deva" if side in ("tgt", "hin_Deva") else "ben_Beng"
    pipe = load_laser_pipeline(lang, model_dir)
    try:
        if hasattr(pipe, "tokenizer") and hasattr(pipe.tokenizer, "tokenize"):
            return pipe.tokenizer.tokenize(text)
        if hasattr(pipe, "spm") and hasattr(pipe.spm, "encode_as_pieces"):
            return pipe.spm.encode_as_pieces(text)
        if hasattr(pipe, "encoder") and hasattr(pipe.encoder, "tokenizer"):
            return pipe.encoder.tokenizer.tokenize(text)
    except Exception:
        pass
    return text.split()
