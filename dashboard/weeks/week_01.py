"""Week 1 — data cleaning and alignment filtering. (Topic clustering is on the week-2 page.)"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

WEEK_DIR = "week_01_data_and_clusters"   # folder name on the server; kept as is


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def _missing(step):
    st.info(f"No results yet — run `{step}` first.")


def render(results_root):
    d = Path(results_root) / WEEK_DIR
    st.title("Week 1 — Data cleaning")
    st.caption("Kangri: Mendeley Hindi–Kangri corpus + Chauhan et al. Kr_4 parallel corpus. "
               "All numbers on this page are about TRAINING data; the official test set is kept "
               "untouched and is never shown.")

    notes = d / "notes.md"
    if notes.exists():
        with st.expander("Notes for this week", expanded=True):
            st.markdown(notes.read_text(encoding="utf-8"))

    _summary(d)
    t1, t2 = st.tabs(["1 · Cleaning", "2 · Alignment filter"])
    with t1:
        _cleaning(d)
    with t2:
        _alignment(d)


def _cutoffs(rep):
    """'mendeley: none · chauhan: 0.3' — per-corpus LaBSE cut-offs (older reports had one threshold)."""
    t = rep.get("thresholds")
    if not t:
        return f"all: {rep['threshold']}"
    return " · ".join(f"{s}: {'none' if v is None else v}" for s, v in t.items())


def _cutoff_short(rep):
    """Short card value, e.g. '0.3 · Chauhan only' (full per-corpus detail goes in the tooltip)."""
    t = rep.get("thresholds")
    if not t:
        return str(rep["threshold"])
    used = {s: v for s, v in t.items() if v is not None}
    if len(used) == 1:
        (s, v), = used.items()
        return f"{v} · {s.capitalize()} only"
    return " / ".join(str(v) for v in used.values()) or "none"


# ---------------------------------------------------------------- headline numbers
def _summary(d):
    clean, align = _json(d / "cleaning_report.json"), _json(d / "align_report.json")
    if not clean:
        return
    funnel = pd.DataFrame(clean["parallel_funnel"]).set_index("step")
    raw = int(funnel.loc["raw lines (Hindi file)"].sum())
    rule = int(funnel.loc["kept after rule-based cleaning"].sum())
    mono = sum(v.get("kept", 0) for v in clean["monolingual"].values())
    c = st.columns(4)
    c[0].metric("Raw parallel lines", f"{raw:,}")
    c[1].metric("After rule-based cleaning", f"{rule:,}")
    if align:
        c[2].metric("Final clean pairs", f"{align['pairs_kept']:,}", help=_cutoffs(align))
    c[3].metric("Monolingual Kangri lines", f"{mono:,}")

def _funnel(rows):
    """Cleaning funnel with every row filled in for both corpora.

    The loaders record blank-line removal differently per corpus (Mendeley drops blanks on
    each side to re-align; Chauhan pairs by position), so fill the missing cells by arithmetic.
    """
    f = pd.DataFrame(rows).set_index("step")
    raw, nop = "raw lines (Hindi file)", "lines with no partner (file length mismatch)"
    blank, usable = "removed: blank on either side", "pairs after dropping blank lines"
    for c in f.columns:
        if f.at[usable, c] == 0:                       # usable count not recorded: derive it
            f.at[usable, c] = f.at[raw, c] - f.at[nop, c] - f.at[blank, c]
        else:                                          # blank count not recorded: derive it
            f.at[blank, c] = f.at[raw, c] - f.at[nop, c] - f.at[usable, c]
    return f.rename(index={usable: "usable pairs"})

# ---------------------------------------------------------------- tab 1
def _cleaning(d):
    rep = _json(d / "cleaning_report.json")
    if not rep:
        return _missing("sbatch slurm/clean.sbatch configs/week01.yaml")

    st.subheader("Parallel training data: what each filter removed")
    st.dataframe(_funnel(rep["parallel_funnel"]), width="stretch")

    st.subheader("Removed examples (training data only)")
    for step, ex in rep["removed_examples"].items():
        with st.expander(f"{step} ({len(ex)} shown)"):
            st.dataframe(pd.DataFrame(ex), width="stretch", hide_index=True)

    lines = rep.get("decontaminated_train_lines", {})
    if lines:
        st.subheader("Decontamination: where the removed training lines sit")
        st.caption("Training pairs removed because they are exact copies of a test sentence. "
                   "Only line numbers are shown — no test content. A narrow band means one topic "
                   "block lost more data than others.")
        dl = pd.DataFrame([{"source": s, "orig_line": n} for s, ns in lines.items() for n in ns])
        st.altair_chart(alt.Chart(dl).mark_bar().encode(
            x=alt.X("orig_line:Q", bin=alt.Bin(step=500), title="line in original training file"),
            y=alt.Y("count():Q", title="pairs removed"), color="source:N"), width="stretch")

    st.subheader("Monolingual Kangri (Kr_1 stories/social media, Kr_3 poems)")
    st.dataframe(pd.DataFrame(rep["monolingual"]).fillna(0).astype(int), width="stretch")

    if rep.get("test_set"):
        st.caption(f"Official test set: {rep['test_set']['chauhan_kr4_test_pairs_saved_unchanged']} pairs "
                   f"saved unchanged. {rep['test_set']['note']}")
    for n in rep.get("notes", []):
        st.caption(n)


# ---------------------------------------------------------------- tab 2
def _alignment(d):
    rep = _json(d / "align_report.json")
    if not rep:
        return _missing("sbatch slurm/align.sbatch configs/week01.yaml")
    thr = rep["threshold"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Pairs kept", f"{rep['pairs_kept']:,} / {rep['pairs_scored']:,}")
    c2.metric("LaBSE cut-off", _cutoff_short(rep), help=_cutoffs(rep))
    if "shifted_pairs_above_threshold_pct" in rep:
        c3.metric("Shifted (off-by-one) pairs that pass", f"{rep['shifted_pairs_above_threshold_pct']}%",
                  help="Share of deliberately misaligned neighbour pairs that the threshold still accepts.")

    st.dataframe(pd.DataFrame(rep["by_source"]), width="stretch", hide_index=True)

    if rep.get("threshold_table"):
        st.subheader("Threshold trade-off")
        st.caption("kept % = real pairs kept · shifted pass % = misaligned (Mendeley neighbour) pairs that "
                   "would get through · random pass % = unrelated pairs that would get through.")
        tt = pd.DataFrame(rep["threshold_table"])
        st.dataframe(tt.style.apply(lambda r: ["font-weight: bold" if r["threshold"] == thr else ""] * len(r),
                                    axis=1), width="stretch", hide_index=True)
        long = tt.melt("threshold", value_vars=["mendeley_kept_pct", "chauhan_kept_pct", "shifted_pass_pct"],
                       var_name="series", value_name="percent")
        long["series"] = long["series"].map({"mendeley_kept_pct": "Mendeley pairs kept",
                                             "chauhan_kept_pct": "Chauhan pairs kept",
                                             "shifted_pass_pct": "Shifted (wrong) pairs let through"})
        st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("threshold:Q", title="LaBSE cut-off"), y=alt.Y("percent:Q", title="% of pairs"),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom", labelLimit=0)))
            + alt.Chart(pd.DataFrame({"x": [thr]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q"),
            width="stretch")

    sims_path = d / "sims.csv"
    if sims_path.exists():
        sims = pd.read_csv(sims_path, dtype={"id": str})
        st.subheader("Similarity: real pairs vs shifted and random pairs")
        hist = alt.Chart(sims).mark_bar(opacity=0.55).encode(
            x=alt.X("labse_sim:Q", bin=alt.Bin(maxbins=60), title="LaBSE cosine similarity"),
            y=alt.Y("count():Q", stack=None, title="pairs"),
            color=alt.Color("kind:N", title=None))
        rule = alt.Chart(pd.DataFrame({"x": [thr]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart(hist + rule, width="stretch")
        st.caption("Real pairs should sit to the right of both baselines. Dashed line = threshold. "
                   "LaBSE was not trained on Kangri, so correct pairs score lower than for languages it knows.")

        st.subheader("Similarity along each file (rolling mean of 200 lines)")
        st.caption("Dips show stretches where the Hindi and Kangri files drift out of alignment.")
        real = sims[sims["kind"] == "real pair"].sort_values(["source", "orig_line"]).copy()
        real["rolling_sim"] = real.groupby("source")["labse_sim"].transform(
            lambda s: s.rolling(200, min_periods=20).mean())
        st.altair_chart(alt.Chart(real).mark_line().encode(
            x=alt.X("orig_line:Q", title="line in original file"),
            y=alt.Y("rolling_sim:Q", title="mean similarity", scale=alt.Scale(zero=False)),
            color="source:N"), width="stretch")

    with st.expander("Pairs just around the threshold — read these to judge the threshold"):
        st.dataframe(pd.DataFrame(rep["near_threshold_examples"]), width="stretch", hide_index=True)
    with st.expander("Lowest-similarity pairs (clearly misaligned)"):
        st.dataframe(pd.DataFrame(rep["lowest_sim_examples"]), width="stretch", hide_index=True)