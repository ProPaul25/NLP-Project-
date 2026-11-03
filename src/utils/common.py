"""Shared helpers: project paths, config loading, seeding, text I/O and normalisation.

This module never imports torch at import time, so GPU scripts can import it
*after* their mandatory CUDA header without changing the import order.
"""
import json
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]   # .../Project_NLP


def resolve(rel_path):
    """Path relative to the project root."""
    return ROOT / rel_path


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config {path}")
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    print(f"Seed: {seed}")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"Wrote {path}")


def read_lines(path):
    """Read a text file line by line, keeping line positions exactly as `wc -l` sees them.

    Uses split('\\n') instead of splitlines(): splitlines() also breaks on rare
    Unicode separators, which would silently shift a parallel file out of alignment.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = [l.rstrip("\r") for l in text.split("\n")]
    if lines and lines[-1] == "":
        lines.pop()          # trailing newline at end of file
    return lines


_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\ufeff"), None)
_LEADING_NUMBER = re.compile(r"^\s*[0-9\u0966-\u096F]+\s*[.)]\s*")


def normalize(s):
    """NFC, drop BOM / zero-width space, strip leading enumeration like '767.', collapse whitespace."""
    s = unicodedata.normalize("NFC", s).translate(_ZERO_WIDTH)
    s = _LEADING_NUMBER.sub("", s)
    return " ".join(s.split())


def devanagari_words(s):
    """Number of whitespace tokens that contain at least one Devanagari letter."""
    return sum(1 for w in s.split()
               if any("\u0900" <= ch <= "\u0963" or "\u0970" <= ch <= "\u097F" for ch in w))


def devanagari_ratio(s):
    """Share of letters/marks in the string that are Devanagari."""
    total = dev = 0
    for ch in s:
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "M"):
            total += 1
            if "\u0900" <= ch <= "\u097F":
                dev += 1
    return dev / total if total else 0.0