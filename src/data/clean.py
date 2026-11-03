"""Step 1 — rule-based cleaning of the Hindi–Kangri parallel corpora (+ Kangri monolingual).

Input : data/raw/chauhan_kangri/{Train,Test}/Kr_*.txt, data/raw/mendeley_hi_kangri/*.txt
Output: data/interim/pairs_rule_clean.tsv        TRAINING pairs only (id, hindi, kangri, source, orig_line, split)
        data/interim/chauhan_test_untouched.tsv  official Chauhan test pairs, NOT filtered or changed
        data/interim/mono_kangri.tsv             TRAINING monolingual Kangri (id, text, source, orig_line, split)
        <results>/cleaning_report.json           per-filter counts + removed TRAINING examples (dashboard)

Test-set policy: the official Chauhan test files are never filtered, inspected or shown.
They are used for exactly one thing — removing exact copies of test sentences from the
TRAINING data (decontamination). The report records only how many were removed.
No GPU computation happens here; it runs under Slurm with the GPU header by project policy.
"""
import os
# Must be set BEFORE importing torch / transformers.
# Inside a Slurm job, Slurm already restricts which GPU is visible, and the
# allocated GPU appears as index 0. So use "0" -- never a physical GPU number.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available - refusing to run on CPU. "
                       "Did you submit via sbatch/srun with --gres=gpu:1?")
device = torch.device("cuda")
print("Using GPU:", torch.cuda.get_device_name(0))

import argparse
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import (devanagari_ratio, devanagari_words, load_config,  # noqa: E402
                              normalize, read_lines, resolve, save_json, set_seed)

N_EXAMPLES = 8  # removed TRAINING examples kept per filter, for the dashboard
SOURCES = ["mendeley", "chauhan"]


class Funnel:
    """Counts and examples of what each filter removed from the TRAINING data, per source."""

    def __init__(self):
        self.counts = defaultdict(OrderedDict)
        self.examples = defaultdict(list)

    def add(self, source, step, n):
        self.counts[source][step] = self.counts[source].get(step, 0) + n

    def example(self, step, source, hi, kr):
        if len(self.examples[step]) < N_EXAMPLES:
            self.examples[step].append({"source": source, "hindi": hi, "kangri": kr})

    def table(self, c):
        """Funnel as an ordered list of rows (step, mendeley, chauhan)."""
        order = ["raw lines (Hindi file)",
                 "lines with no partner (file length mismatch)",
                 "removed: blank on either side",
                 "pairs after dropping blank lines",
                 f"removed: fewer than {c['min_words']} Devanagari words",
                 "removed: not mainly Devanagari",
                 "removed: Kangri side identical to Hindi (copied, not translated)",
                 f"removed: length ratio > {c['max_len_ratio']}",
                 "removed: duplicate pair",
                 "removed: exact copy of a test sentence (decontamination)",
                 "kept after rule-based cleaning"]
        seen = {s for v in self.counts.values() for s in v}
        order += sorted(seen - set(order))
        return [{"step": s, **{src: int(self.counts[src].get(s, 0)) for src in SOURCES}}
                for s in order if s in seen]


# ---------------------------------------------------------------- loading
def load_mendeley(raw_dir, funnel):
    """Mendeley: drop blank lines on EACH side independently.

    A stray blank at Hindi line 14,215 and Kangri line 14,446 shifts ~230 pairs by one;
    both files hold exactly 25,000 non-blank lines, so dropping blanks re-aligns them.
    """
    hi_raw = read_lines(Path(raw_dir) / "Hindi sentences.txt")
    kr_raw = read_lines(Path(raw_dir) / "Kangri sentences.txt")
    hi = [(i + 1, l) for i, l in enumerate(hi_raw) if l.strip()]
    kr = [(i + 1, l) for i, l in enumerate(kr_raw) if l.strip()]
    if len(hi) != len(kr):
        raise ValueError(f"Mendeley non-blank line counts differ: Hindi {len(hi)} vs Kangri {len(kr)}. "
                         "The raw files are not the expected version — stop and check them.")
    funnel.add("mendeley", "raw lines (Hindi file)", len(hi_raw))
    funnel.add("mendeley", "pairs after dropping blank lines", len(hi))
    return [{"hindi": h, "kangri": k, "source": "mendeley", "orig_line": hn, "split": "pool"}
            for (hn, h), (_, k) in zip(hi, kr)]


def load_chauhan_train(raw_dir, funnel):
    """Chauhan Kr_4 train: pair by line position. Misaligned stretches are removed later by LaBSE."""
    hi = read_lines(Path(raw_dir) / "Train" / "Kr_4_Hindi.txt")
    kr = read_lines(Path(raw_dir) / "Train" / "Kr_4_kangri.txt")
    funnel.add("chauhan", "raw lines (Hindi file)", len(hi))
    funnel.add("chauhan", "lines with no partner (file length mismatch)", abs(len(hi) - len(kr)))
    rows, blank = [], 0
    for i in range(min(len(hi), len(kr))):
        if not hi[i].strip() or not kr[i].strip():
            blank += 1
            continue
        rows.append({"hindi": hi[i], "kangri": kr[i], "source": "chauhan", "orig_line": i + 1, "split": "pool"})
    funnel.add("chauhan", "removed: blank on either side", blank)
    return rows


def load_chauhan_test(raw_dir):
    """Official Chauhan Kr_4 test pairs: normalised only, never filtered.

    Returns all pairs by line position (blank lines kept as empty strings), so the
    saved file is the official test set exactly, just in one TSV.
    """
    hi = read_lines(Path(raw_dir) / "Test" / "Kr_4_Hindi.txt")
    kr = read_lines(Path(raw_dir) / "Test" / "Kr_4_Kangri.txt")
    if len(hi) != len(kr):
        raise ValueError(f"Chauhan test files differ in length: Hindi {len(hi)} vs Kangri {len(kr)}.")
    return [{"id": f"chauhan_test-{i + 1}", "hindi": normalize(h), "kangri": normalize(k),
             "source": "chauhan", "orig_line": i + 1, "split": "chauhan_test"}
            for i, (h, k) in enumerate(zip(hi, kr))]


# ---------------------------------------------------------------- filters (TRAINING data only)
def apply_filters(rows, c, funnel):
    """Pair-level filters, in order. Each removal is counted against its source."""
    out = []
    for r in rows:
        src = r["source"]
        hi, kr = normalize(r["hindi"]), normalize(r["kangri"])
        r["hindi"], r["kangri"] = hi, kr
        nh, nk = devanagari_words(hi), devanagari_words(kr)
        if nh < c["min_words"] or nk < c["min_words"]:
            step = f"removed: fewer than {c['min_words']} Devanagari words"
        elif devanagari_ratio(hi) < c["min_devanagari_ratio"] or devanagari_ratio(kr) < c["min_devanagari_ratio"]:
            step = "removed: not mainly Devanagari"
        elif hi == kr:
            step = "removed: Kangri side identical to Hindi (copied, not translated)"
        elif max(nh, nk) / min(nh, nk) > c["max_len_ratio"]:
            step = f"removed: length ratio > {c['max_len_ratio']}"
        else:
            out.append(r)
            continue
        funnel.add(src, step, 1)
        funnel.example(step, src, hi, kr)
    return out


def dedupe(rows, funnel):
    seen, kept = set(), []
    for r in rows:
        key = (r["hindi"], r["kangri"])
        if key in seen:
            funnel.add(r["source"], "removed: duplicate pair", 1)
            continue
        seen.add(key)
        kept.append(r)
    return kept


def decontaminate(rows, test_sentences, funnel):
    """Remove TRAINING pairs whose Hindi or Kangri side is an exact copy of a test sentence.
    Only counts and line numbers are recorded — no test content goes into any report."""
    kept, removed_lines = [], defaultdict(list)
    for r in rows:
        if r["hindi"] in test_sentences or r["kangri"] in test_sentences:
            funnel.add(r["source"], "removed: exact copy of a test sentence (decontamination)", 1)
            removed_lines[r["source"]].append(r["orig_line"])
            continue
        kept.append(r)
    return kept, dict(removed_lines)


# ---------------------------------------------------------------- monolingual (TRAINING only)
def clean_mono(raw_dir, c, parallel_kr, parallel_hi, test_sentences):
    """Kangri monolingual TRAINING text: Kr_1 (stories/social media) and Kr_3 (poems).
    Kr_2 is a dictionary (excluded). The Kr_1 test file is used only for decontamination."""
    stats, rows = {}, []
    for name, sub in [("kr1", "Train/Kr_1.txt"), ("kr3", "Train/Kr_3.txt")]:
        lines = read_lines(Path(raw_dir) / sub)
        s = OrderedDict(raw_lines=len(lines))
        seen, kept = set(), 0
        for i, line in enumerate(lines):
            t = normalize(line)
            if not t:
                reason = "blank"
            elif devanagari_words(t) < c["min_words"]:
                reason = "too_short"
            elif devanagari_ratio(t) < c["min_devanagari_ratio"]:
                reason = "not_devanagari"
            elif t in parallel_hi:
                reason = "is_a_hindi_line_from_parallel_data"
            elif t in parallel_kr:
                reason = "repeats_parallel_kangri"
            elif t in test_sentences:
                reason = "exact_copy_of_test_sentence"
            elif t in seen:
                reason = "duplicate"
            else:
                seen.add(t)
                kept += 1
                rows.append({"id": f"{name}-{i + 1}", "text": t, "source": name,
                             "orig_line": i + 1, "split": "pool"})
                continue
            s[reason] = s.get(reason, 0) + 1
        s["kept"] = kept
        stats[name] = s
    return rows, stats


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    set_seed(cfg["seed"])
    P, C = cfg["paths"], cfg["clean"]
    raw_c, raw_m, interim = resolve(P["raw_chauhan"]), resolve(P["raw_mendeley"]), resolve(P["interim"])
    interim.mkdir(parents=True, exist_ok=True)

    # --- official test sets: saved unchanged, used only for decontamination
    test_pairs = load_chauhan_test(raw_c)
    test_sentences = {p["hindi"] for p in test_pairs} | {p["kangri"] for p in test_pairs}
    # Kr_1 test (monolingual) overlaps heavily with the parallel train data. Decontaminating
    # against it costs ~700 Chauhan train pairs but keeps that test file clean for later use.
    if C.get("decontaminate_mono_test", True):
        test_sentences |= {normalize(l) for l in read_lines(raw_c / "Test" / "Kr_1.txt")}
    test_sentences -= {""}
    test_out = interim / "chauhan_test_untouched.tsv"
    pd.DataFrame(test_pairs)[["id", "hindi", "kangri", "source", "orig_line", "split"]].to_csv(
        test_out, sep="\t", index=False)
    print(f"Wrote {test_out}  ({len(test_pairs)} official test pairs, unchanged)")

    # --- training parallel data
    funnel = Funnel()
    rows = load_mendeley(raw_m, funnel) + load_chauhan_train(raw_c, funnel)
    rows = apply_filters(rows, C, funnel)
    rows = dedupe(rows, funnel)
    rows, decontam_lines = decontaminate(rows, test_sentences, funnel)
    for r in rows:
        r["id"] = f"{r['source']}-{r['orig_line']}"
    for src in SOURCES:
        funnel.add(src, "kept after rule-based cleaning", sum(1 for r in rows if r["source"] == src))

    df = pd.DataFrame(rows)[["id", "hindi", "kangri", "source", "orig_line", "split"]]
    out = interim / "pairs_rule_clean.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"Wrote {out}  ({len(df)} training pairs)")

    # --- training monolingual data
    mono, mono_stats = clean_mono(raw_c, C, parallel_kr=set(df["kangri"]),
                                  parallel_hi=set(df["hindi"]), test_sentences=test_sentences)
    mono_out = interim / "mono_kangri.tsv"
    pd.DataFrame(mono, columns=["id", "text", "source", "orig_line", "split"]).to_csv(
        mono_out, sep="\t", index=False)
    print(f"Wrote {mono_out}  ({len(mono)} training lines)")

    save_json({"config": C,
               "parallel_funnel": funnel.table(C),
               "removed_examples": dict(funnel.examples),
               "monolingual": mono_stats,
               "decontaminated_train_lines": decontam_lines,
               "test_set": {"chauhan_kr4_test_pairs_saved_unchanged": len(test_pairs),
                            "note": "Official test sets are not filtered or shown. They are used only to "
                                    "remove exact copies of test sentences from the training data."},
               "notes": ["Kr_2 (Hindi–Kangri dictionary word pairs) is excluded: not sentences.",
                         "Chauhan Kr_4 train is paired by line position; misaligned stretches are removed "
                         "in step 2 (LaBSE similarity), not here.",
                         "Monolingual lines that repeat a parallel-corpus sentence are removed, so the "
                         "same sentence never appears both as parallel and as monolingual data."]},
              resolve(P["results"]) / "cleaning_report.json")


if __name__ == "__main__":
    main()