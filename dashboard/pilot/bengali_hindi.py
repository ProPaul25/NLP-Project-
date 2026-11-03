"""Pilot — Bengali–Hindi: contrastive fine-tuning of LaBSE and E5, tested on held-out sets.

Tabs 1–5 read only files written by the pipeline (no model is run there):
  outputs/results/pilot_bn/prepare_report.json            data counts and overlap checks
  outputs/results/pilot_bn/eval/results.csv               every model on every test set
  outputs/results/pilot_bn/eval/seeds_compare.csv         paired comparisons, per seed and pooled
  outputs/results/pilot_bn/eval/<model>__<set>_per_sentence.tsv   right/wrong per test sentence
  outputs/checkpoints/pilot_bn/<run>/train_log.json       dev scores and loss during training
  data/processed/pilot_bn/<set>.tsv                       sentence texts (for the error examples)
Tabs 6 (the vector space) and 7 (live demo) load the models and therefore only work when the dashboard
runs inside a GPU job.
"""
import json
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PILOT = "pilot_bn"
BASES = {"labse": "LaBSE", "e5": "E5", "laser": "LASER"}
VARIANTS = {
    "trans": "+ translation loss",
    "topic": "+ topic loss",
    "both": "+ both losses",
    "co": "3-CO",
    "co_filter": "3-CO-Filter",
    "co_topic": "3-CO + topic loss",
}
SETS = {"sib200:test": "SIB-200 test", "in22_gen:test": "IN22-Gen", "in22_conv:test": "IN22-Conv"}
SET_NOTE = {
    "sib200:test": "Same 7 topic labels as the SIB-200 training part — measures learning of seen labels.",
    "in22_gen:test": "13 domains never used in training — the main test of transfer to new topics.",
    "in22_conv:test": "Turns from 44 conversations; 16 domains, never used in training. Every turn carries the "
                      "domain of its whole conversation.",
}
MAIN_CLAIMS = [  # fixed BEFORE the repeat seeds were run
    ("in22_gen:test", "e5", "e5_both", "ok_src"),
    ("in22_gen:test", "labse", "labse_both", "ok_src"),
    ("sib200:test", "e5_topic", "e5_both", "xsim_ok_src2tgt"),
    ("sib200:test", "labse_topic", "labse_both", "xsim_ok_src2tgt"),
    ("sib200:test", "laser", "laser_co", "xsim_ok_src2tgt"),
    ("in22_gen:test", "laser", "laser_co", "xsim_ok_src2tgt"),
    ("in22_gen:test", "laser", "laser_co_topic", "ok_src"),
    ("sib200:test", "laser_co", "laser_co_topic", "ok_src"),
]
METRICS = {"knn_acc_src": "Topic accuracy (%)", "knn_acc_cross": "Topic accuracy across languages (%)",
           "xsim_src2tgt": "Translation error, Bengali→Hindi (%)", "xsim_tgt2src": "Translation error, Hindi→Bengali (%)",
           "nmi_src": "Cluster agreement (NMI)"}
PAIR_METRICS = {"ok_src": "topic accuracy", "ok_cross": "topic accuracy across languages",
                "xsim_ok_src2tgt": "translation accuracy"}
VERDICT_LABEL = {  # wording shown on the page for the verdicts written by aggregate.py
    "B better (all seeds)": "Higher on all 3 runs",
    "mixed across seeds": "Differs between runs",
    "no clear difference": "No measurable difference",
    "A better (all seeds)": "Lower on all 3 runs",
}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def _missing(step):
    st.info(f"No results yet — run `{step}` first.")


def split_name(model):
    """'labse_both_s43' -> ('labse', 'both', 43); base models -> ('labse', None, None)."""
    m = re.fullmatch(r"(?P<b>[a-z0-9]+?)(?:_(?P<v>trans|topic|both|co_filter|co_topic|co))?(?:_s(?P<s>\d+))?", model)
    if not m:
        return model, None, None
    return m["b"], m["v"], int(m["s"]) if m["s"] else None


def nice(model):
    b, v, s = split_name(model)
    name = run_label(model)
    if v and not s:
        s = 42          # trained runs without a suffix used the config seed (42)
    return name + (f" · seed {s}" if s else "")


def run_label(run):
    b, v, _ = split_name(run)
    base = BASES.get(b, b)
    if not v:
        return base + " (untrained)"
    if b == "laser":
        lbl = VARIANTS.get(v, v)
        if lbl.startswith("3-"):
            return f"LASER{lbl}"
        return f"LASER3-CO {lbl}"
    return base + f" {VARIANTS.get(v, v)}"


@st.cache_data
def _csv(path, mtime, **kw):
    return pd.read_csv(path, **kw)


def _read(path, **kw):
    return _csv(str(path), Path(path).stat().st_mtime, **kw) if Path(path).exists() else None


def _find_roots(results_root):
    r = Path(results_root)
    proj = r.parents[1] if len(r.parents) > 1 else r
    nlp_root = Path("/raid/home/loitongbam/Project_NLP")
    laser_root = Path("/raid/home/loitongbam/Project_NLP_2")
    if not nlp_root.exists():
        nlp_root = proj.parent / "Project_NLP" if (proj.parent / "Project_NLP").exists() else proj
    if not laser_root.exists():
        laser_root = proj.parent / "Project_NLP_2" if (proj.parent / "Project_NLP_2").exists() else proj
    return nlp_root, laser_root


def render(results_root):
    results_root = Path(results_root)
    nlp_root, laser_root = _find_roots(results_root)
    project_root = nlp_root
    d = nlp_root / "outputs" / "results" / PILOT
    ev = d / "eval"
    st.title("Pilot — Bengali–Hindi")
    st.caption("The Kangri topic labels are still being annotated. To build and test the method in the "
               "meantime, the full pipeline was run on Bengali–Hindi, where topic-labelled parallel data "
               "already exists. The same code will be run on Kangri–Hindi once the labels are complete.")

    notes = d / "notes.md"
    if not notes.exists():
        notes = laser_root / "outputs" / "results" / PILOT / "notes.md"
    if notes.exists():
        with st.expander("Notes", expanded=True):
            st.markdown(notes.read_text(encoding="utf-8"))

    res_nlp = _read(ev / "results.csv")
    res_laser = _read(laser_root / "outputs" / "results" / PILOT / "eval" / "results.csv")
    if res_nlp is not None and res_laser is not None:
        res = pd.concat([res_nlp, res_laser], ignore_index=True).drop_duplicates(subset=["model", "set"], keep="last")
    else:
        res = res_nlp if res_nlp is not None else res_laser

    if res is None:
        return _missing("sbatch slurm/evaluate.sbatch configs/pilot_bn.yaml")

    tabs = st.tabs(["1 · Setup", "2 · Results", "3 · Is the difference real?", "4 · Error analysis",
                    "5 · Training curves", "6 · The vector space", "7 · Try it (live)"])
    with tabs[0]:
        _setup(d, res)
        _data_plots(project_root)
    with tabs[1]:
        _results(res)
    with tabs[2]:
        _compare(ev)
    with tabs[3]:
        _errors(ev, project_root, res)
    with tabs[4]:
        _curves(project_root)
    with tabs[5]:
        _space(project_root, res)
    with tabs[6]:
        _live(project_root, res)


# ---------------------------------------------------------------- tab 1
def _setup(d, res):
    st.subheader("What was done")
    st.markdown(
        "- **Goal:** sentence vectors where sentences on the same topic lie close together, while every "
        "sentence still lies closest to its own translation.\n"
        "- **Starting models:** **LaBSE** (trained on translation pairs), **E5** (multilingual-e5-base, "
        "trained for search), and **LASER3** (language-specific sentence encoders; LASER3-CO fine-tunes Bengali student against frozen Hindi teacher). Each is also tested untouched, as a baseline.\n"
        "- **Method — contrastive fine-tuning**, with two losses:\n"
        "  - *translation loss*: a Bengali sentence is pulled toward its Hindi translation and pushed away "
        "from the other sentences in the batch;\n"
        "  - *topic loss* (supervised contrastive): sentences with the same topic label are pulled together, "
        "different topics pushed apart.\n"
        "- **Three settings per model:** translation loss only, topic loss only, both losses.\n"
        "- **Repeats:** every setting trained 3 times (seeds 42, 43, 44) to see how much results move "
        "by chance.\n"
        "- **Checkpoint choice:** the model is scored every 100 training steps (and at the start and end) on the "
        "SIB-200 development part (99 sentences); the version with the best topic accuracy there is kept. "
        "Test sets are used only for the final scores.")

    rep = _json(d / "prepare_report.json")
    if rep:
        st.subheader("Data")
        role = {"massive": "training (+ its own dev/test, not used)", "sib200": "training (train part), "
                "checkpoint choice (dev), test (test)", "in22_conv": "test only", "in22_gen": "test only"}
        rows = []
        for name, info in rep["datasets"].items():
            ps = info.get("per_split", {})
            n_topics = len({t for split in info.get("topics", {}).values() for t in split})
            rows.append({"dataset": name, "used for": role.get(name, ""), "pairs": info["pairs"],
                         "train": ps.get("train", 0), "dev": ps.get("dev", 0), "test": ps.get("test", 0),
                         "topic labels": n_topics})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        ov = rep.get("overlap_with_massive_train", {})
        if ov:
            o = pd.DataFrame(ov).T.rename(columns={"src_in_train": "Bengali in training", "tgt_in_train":
                                                   "Hindi in training", "eval_rows": "rows checked"})
            with st.expander("Leakage check: test sentences that also appear in the MASSIVE training data"):
                st.dataframe(o, width="stretch")
                st.caption("Exact-text matches. The three test sets have none; MASSIVE's own dev/test "
                           "repeat some training utterances, which is why MASSIVE is not used for testing.")

    st.subheader("How it is scored")
    st.markdown(
        "- **Topic accuracy:** for each test sentence, the 5 most similar *other* test sentences vote on its "
        "topic. For IN22-Conv, turns from the same conversation are not allowed to vote.\n"
        "- **Translation error:** share of sentences whose nearest sentence in the other language is *not* "
        "its translation (lower is better).\n"
        "- **Majority baseline:** the accuracy of always guessing the most common topic.\n"
        "- **Uncertainty:** 95% confidence intervals (bootstrap) over test sentences (whole conversations for IN22-Conv), "
        "and comparisons made sentence by sentence (paired) — see tab 3.")

DATA_NAMES = {"massive": "MASSIVE", "sib200": "SIB-200", "in22_gen": "IN22-Gen", "in22_conv": "IN22-Conv"}


@st.cache_data(show_spinner=False)
def _all_data(project_root):
    parts = []
    for name in DATA_NAMES:
        f = Path(project_root) / "data" / "processed" / PILOT / f"{name}.tsv"
        if f.exists():
            df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, usecols=["dataset", "split", "src", "tgt",
                                                                                      "topic"])
            parts.append(df)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df["dataset"] = df["dataset"].map(DATA_NAMES)
    df["Bengali"] = df["src"].str.split().str.len()
    df["Hindi"] = df["tgt"].str.split().str.len()
    return df


def _data_plots(project_root):
    df = _all_data(str(project_root))
    if df is None:
        return st.info("Processed data not found; dataset plots cannot be shown.")
    st.subheader("Topic distribution")
    c1, _ = st.columns([1, 2])
    name = c1.selectbox("Dataset", list(DATA_NAMES.values()), index=1, key="w3_data_set")
    sub = df[df["dataset"] == name]
    counts = sub.groupby(["topic", "split"]).size().rename("sentences").reset_index()
    order = sub["topic"].value_counts().index.tolist()
    st.altair_chart(alt.Chart(counts).mark_bar().encode(
        y=alt.Y("topic:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220)),
        x=alt.X("sentences:Q", stack="zero", title="sentence pairs"),
        color=alt.Color("split:N", sort=["train", "dev", "test"],
                        scale=alt.Scale(domain=["train", "dev", "test"], range=["#222222", "#9B9B9B", "#9B1B30"])),
        tooltip=["topic", "split", "sentences"]).properties(height=max(220, 30 * len(order))), width="stretch")
    top = sub["topic"].value_counts(normalize=True)
    st.caption(f"{name}: {sub['topic'].nunique()} labels, {len(sub):,} pairs. Largest label "
               f"{top.index[0]} ({100 * top.iloc[0]:.1f}%), smallest {top.index[-1]} ({100 * top.iloc[-1]:.1f}%).")

    st.subheader("Sentence length")
    long = df.melt(id_vars=["dataset"], value_vars=["Bengali", "Hindi"], var_name="language", value_name="words")
    st.altair_chart(alt.Chart(long).mark_boxplot(extent="min-max", size=22).encode(
        x=alt.X("language:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("words:Q", title="words per sentence", scale=alt.Scale(type="log")),
        color=alt.Color("language:N", scale=alt.Scale(range=["#9B1B30", "#222222"])),
        column=alt.Column("dataset:N", sort=list(DATA_NAMES.values()), title=None)), width=120)
    med = long.groupby(["dataset", "language"])["words"].median().unstack().reindex(list(DATA_NAMES.values()))
    st.dataframe(med.rename(columns=lambda c: f"median words, {c}"), width="content")
    st.caption("Words = space-separated tokens. Box = middle half of the sentences, line = median, "
               "whiskers = shortest and longest (log scale).")
# ---------------------------------------------------------------- tab 2
def _summary_table(res, set_key, metric):
    sub = res[res["set"] == set_key].copy()
    sub["run"] = sub["model"].map(lambda m: re.sub(r"_s\d+$", "", m))
    g = sub.groupby("run")[metric]
    out = pd.DataFrame({"mean": g.mean(), "min": g.min(), "max": g.max(), "runs": g.size()}).reset_index()
    out["model"] = out["run"].map(run_label)
    out["base"] = out["run"].map(lambda r: BASES.get(split_name(r)[0], r))
    order = {v: i for i, v in enumerate([None, "trans", "topic", "both", "co", "co_filter", "co_topic"])}
    out["order"] = out["run"].map(lambda r: order.get(split_name(r)[1], 9))
    return out.sort_values(["base", "order"]), sub


def _results(res):
    c1, c2, c3 = st.columns(3)
    set_key = c1.selectbox("Test set", list(SETS), format_func=SETS.get, key="w3_set")
    metric = c2.selectbox("Measure", list(METRICS), format_func=METRICS.get, key="w3_metric")
    family = c3.selectbox("Model family", ["All models", "LaBSE", "E5", "LASER3 / LASER3-CO"], key="w3_family")
    st.caption(SET_NOTE[set_key])
    table, sub = _summary_table(res, set_key, metric)
    if family == "LaBSE":
        table = table[table["base"] == "LaBSE"]
    elif family == "E5":
        table = table[table["base"] == "E5"]
    elif family.startswith("LASER"):
        table = table[table["base"] == "LASER"]
    if table.empty:
        return st.info("No results for this set / model selection.")

    bars = alt.Chart(table).mark_bar().encode(
        y=alt.Y("model:N", sort=table["model"].tolist(), title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("mean:Q", title=METRICS[metric]),
        color=alt.Color("base:N", title="starting model"),
        tooltip=["model", alt.Tooltip("mean:Q", format=".2f"), alt.Tooltip("min:Q", format=".2f"),
                 alt.Tooltip("max:Q", format=".2f"), "runs"])
    spread = alt.Chart(table[table["runs"] > 1]).mark_rule(strokeWidth=2, color="#222222").encode(
        y=alt.Y("model:N", sort=table["model"].tolist()), x="min:Q", x2="max:Q")
    layers = bars + spread
    if metric.startswith("knn_acc"):
        maj = float(sub["majority_baseline_acc"].iloc[0])
        layers += alt.Chart(pd.DataFrame({"x": [maj]})).mark_rule(strokeDash=[4, 4], color="#9B1B30").encode(x="x:Q")
        st.caption(f"Red dashed line = majority baseline ({maj:.1f}%).")
    st.altair_chart(layers, width="stretch")
    st.caption("Bars = mean over the 3 training runs (seeds); black line = lowest to highest run. "
               "Untrained models have one run. "
               + ("Lower is better for translation error." if metric.startswith("xsim") else ""))

    show = table[["model", "mean", "min", "max", "runs"]].round(2)
    st.dataframe(show, width="stretch", hide_index=True)

    with st.expander("All runs, all measures, with 95% confidence intervals"):
        names = {"knn_acc_src": "topic acc %", "knn_acc_src_lo": "95% CI low", "knn_acc_src_hi": "95% CI high",
                 "knn_acc_cross": "topic acc across languages %", "knn_f1_src": "macro-F1 %",
                 "xsim_src2tgt": "translation error bn→hi %", "xsim_tgt2src": "translation error hi→bn %",
                 "nmi_src": "NMI", "ari_src": "ARI", "majority_baseline_acc": "majority baseline %",
                 "n": "sentences", "n_groups": "groups"}
        full = sub[["model"] + [c for c in names if c in sub.columns]].copy()
        full.insert(0, "run", full["model"].map(nice))
        full = full.drop(columns="model").rename(columns=names).sort_values("run")
        st.dataframe(full, width="stretch", hide_index=True)
        st.caption("One row per training run. The 95% confidence interval covers test-sentence sampling for that "
                   "run only; run-to-run variation is the min–max above. Groups = units resampled together "
                   "(source documents for IN22-Gen, conversations for IN22-Conv).")


# ---------------------------------------------------------------- tab 3
def _compare(ev):
    nlp_root, laser_root = _find_roots(ev.parents[1])
    cmp_nlp = _read(nlp_root / "outputs" / "results" / PILOT / "eval" / "seeds_compare.csv") or _read(nlp_root / "outputs" / "results" / PILOT / "eval" / "compare.csv")
    p_laser = laser_root / "outputs" / "results" / PILOT / "eval" / "seeds_compare.csv"
    p_laser_cmp = laser_root / "outputs" / "results" / PILOT / "eval" / "compare.csv"
    cmp_laser = _read(p_laser) if p_laser.exists() else (_read(p_laser_cmp) if p_laser_cmp.exists() else None)
    if cmp_nlp is not None and cmp_laser is not None:
        cmp_ = pd.concat([cmp_nlp, cmp_laser], ignore_index=True).drop_duplicates(subset=["set", "A", "B", "metric"], keep="last")
    else:
        cmp_ = cmp_nlp if cmp_nlp is not None else cmp_laser

    if cmp_ is None:
        return _missing("sbatch slurm/aggregate.sbatch configs/pilot_bn.yaml")
    cmp_ = cmp_.copy()
    cmp_["result"] = cmp_["verdict"].map(VERDICT_LABEL)
    cmp_["model"] = cmp_["B"].map(run_label)
    cmp_["compared with"] = cmp_["A"].map(run_label)
    st.markdown(
        "Two models are compared **on the same test sentences**: the difference is the model's score minus "
        "the score of the model it is compared with, in percentage points (positive = the model scores higher). "
        "Because each trained setting was run 3 times, the comparison is made for each run separately:\n"
        "- **Higher / lower on all 3 runs:** for every run the 95% confidence interval of the difference lies "
        "entirely above (or below) zero, and so does the interval for the 3 runs taken together.\n"
        "- **Differs between runs:** the interval excludes zero for some runs but not for others.\n"
        "- **No measurable difference:** for no run does the interval exclude zero.")

    st.subheader("Pre-specified comparisons")
    st.caption("These four comparisons were chosen as the main questions before the two extra runs were trained.")
    rows = []
    for set_key, a, b, metric in MAIN_CLAIMS:
        r = cmp_[(cmp_["set"] == set_key) & (cmp_["A"] == a) & (cmp_["B"] == b) & (cmp_["metric"] == metric)]
        if len(r) == 1:
            r = r.iloc[0]
            rows.append({"model": r["model"], "compared with": r["compared with"],
                         "measure": PAIR_METRICS[metric], "test set": SETS[set_key],
                         "mean difference": r["mean_diff"],
                         "95% CI (3 runs together)": f"[{r['pooled_lo']:+.2f}, {r['pooled_hi']:+.2f}]",
                         "result": r["result"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("All comparisons")
    c1, c2 = st.columns(2)
    set_key = c1.selectbox("Test set", list(SETS), format_func=SETS.get, key="w3_cmp_set")
    metric = c2.selectbox("Measure", list(PAIR_METRICS), format_func=PAIR_METRICS.get, key="w3_cmp_metric")
    sub = cmp_[(cmp_["set"] == set_key) & (cmp_["metric"] == metric)].copy()
    if sub.empty:
        return st.info("No comparisons for this choice.")
    sub["pair"] = sub["model"] + "  vs  " + sub["compared with"]
    forest = alt.Chart(sub).mark_rule(strokeWidth=3).encode(
        y=alt.Y("pair:N", sort=sub["pair"].tolist(), title=None, axis=alt.Axis(labelLimit=420)),
        x=alt.X("pooled_lo:Q", title=f"difference in {PAIR_METRICS[metric]} (percentage points)"),
        x2="pooled_hi:Q",
        color=alt.Color("result:N", title="result", scale=alt.Scale(
            domain=list(VERDICT_LABEL.values()), range=["#2E7D32", "#B08900", "#888888", "#9B1B30"])))
    dots = alt.Chart(sub).mark_point(filled=True, size=60, color="#222222").encode(
        y=alt.Y("pair:N", sort=sub["pair"].tolist()), x="mean_diff:Q",
        tooltip=["pair", alt.Tooltip("mean_diff:Q", title="mean difference"),
                 alt.Tooltip("pooled_lo:Q", title="95% CI low"), alt.Tooltip("pooled_hi:Q", title="95% CI high"),
                 "result"])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart(forest + dots + zero, width="stretch")
    st.caption("Dot = mean difference over the 3 runs; line = 95% confidence interval for the 3 runs together; "
               "dashed line = no difference. The first model in each row is the one whose score is shown "
               "relative to the second.")

    seed_cols = sorted(c for c in sub.columns if re.fullmatch(r"diff_s\d+", c))
    show = sub[["model", "compared with"] + seed_cols + ["mean_diff", "pooled_lo", "pooled_hi", "pooled_p",
                                                         "result"]]
    show = show.rename(columns={**{c: f"difference, run seed {c[6:]}" for c in seed_cols},
                                "mean_diff": "mean difference", "pooled_lo": "95% CI low",
                                "pooled_hi": "95% CI high", "pooled_p": "p-value"})
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption("About 90 comparisons are made in total (3 test sets × 3 measures × 10 model pairs). With that "
               "many, a few results that differ between runs are expected by chance; the pre-specified "
               "comparisons above are the main questions.")


# ---------------------------------------------------------------- tab 4
def _errors(ev, project_root, res):
    c1, c2, c3 = st.columns(3)
    set_key = c1.selectbox("Test set", list(SETS), format_func=SETS.get, key="w3_err_set")
    models = sorted(res.loc[res["set"] == set_key, "model"].unique(), key=lambda m: (split_name(m)[0], nice(m)))
    trained = [m for m in models if split_name(m)[1]]
    if not trained:
        return st.info("No trained models evaluated yet.")
    model = c2.selectbox("Model", trained, format_func=nice, key="w3_err_model",
                         index=trained.index("labse_both") if "labse_both" in trained else 0)
    base = split_name(model)[0]
    ref = c3.selectbox("Compare with", [m for m in models if m != model], format_func=nice, key="w3_err_ref",
                       index=[m for m in models if m != model].index(base) if base in models else 0)
    stem = set_key.replace(":", "_")
    nlp_root, laser_root = _find_roots(project_root / "outputs" / "results")
    def _read_per(m):
        p1 = laser_root / "outputs" / "results" / PILOT / "eval" / f"{m}__{stem}_per_sentence.tsv"
        p2 = nlp_root / "outputs" / "results" / PILOT / "eval" / f"{m}__{stem}_per_sentence.tsv"
        return _read(p1, sep="\t", dtype=str) if m.startswith("laser") else (_read(p2, sep="\t", dtype=str) or _read(p1, sep="\t", dtype=str))
    pm, pr = _read_per(model), _read_per(ref)
    if pm is None or pr is None:
        return _missing("sbatch slurm/evaluate.sbatch configs/pilot_bn.yaml")
    for p in (pm, pr):
        for c in ("ok_src", "ok_cross", "xsim_ok_src2tgt"):
            p[c] = p[c].astype(int)

    st.subheader("Accuracy per topic")
    per = pd.concat([pm.groupby("topic")["ok_src"].agg(["mean", "size"]).assign(model=nice(model)),
                     pr.groupby("topic")["ok_src"].agg(["mean", "size"]).assign(model=nice(ref))]).reset_index()
    per["accuracy %"] = (100 * per["mean"]).round(1)
    order = per[per["model"] == nice(model)].sort_values("accuracy %", ascending=False)["topic"].tolist()
    st.altair_chart(alt.Chart(per).mark_bar().encode(
        y=alt.Y("topic:N", sort=order, title=None), x=alt.X("accuracy %:Q", scale=alt.Scale(domain=[0, 100])),
        yOffset=alt.YOffset("model:N", sort=[nice(model), nice(ref)]),
        color=alt.Color("model:N", scale=alt.Scale(domain=[nice(model), nice(ref)], range=["#9B1B30", "#888888"])),
        tooltip=["topic", "model", "accuracy %", alt.Tooltip("size:Q", title="sentences")]), width="stretch")
    st.caption("Topics with few sentences give unstable accuracies; hover to see the count.")

    st.subheader(f"Which topics get confused — {nice(model)}")
    cm = pd.crosstab(pm["topic"], pm["pred_src"].fillna("(none)"), normalize="index") * 100
    long = cm.reset_index().melt(id_vars="topic", var_name="predicted", value_name="pct")
    st.altair_chart(alt.Chart(long).mark_rect().encode(
        x=alt.X("predicted:N", title="predicted topic"), y=alt.Y("topic:N", title="true topic"),
        color=alt.Color("pct:Q", title="% of true topic", scale=alt.Scale(scheme="reds")),
        tooltip=["topic", "predicted", alt.Tooltip("pct:Q", format=".1f")]), width="stretch")
    off = long[long["topic"] != long["predicted"]].sort_values("pct", ascending=False).head(5)
    if len(off):
        st.caption("Most frequent confusions: " + "; ".join(
            f"{r.topic} → {r.predicted} ({r.pct:.0f}%)" for r in off.itertuples()))

    st.subheader("Example sentences")
    name = set_key.split(":")[0]
    texts = _read(project_root / "data" / "processed" / PILOT / f"{name}.tsv", sep="\t", dtype=str,
                  keep_default_na=False)
    if texts is None:
        return st.info("Processed data not found; examples cannot be shown.")
    m = pm.merge(pr[["id", "ok_src", "pred_src", "xsim_ok_src2tgt"]], on="id", suffixes=("", "_ref"))
    m = m.merge(texts[["id", "src", "tgt"]], on="id", how="left")
    kinds = {"fixed": ("Topic right now, wrong before", (m.ok_src == 1) & (m.ok_src_ref == 0)),
             "broken": ("Topic wrong now, right before", (m.ok_src == 0) & (m.ok_src_ref == 1)),
             "both_wrong": ("Topic wrong for both", (m.ok_src == 0) & (m.ok_src_ref == 0)),
             "xsim_lost": ("Translation found before, lost now", (m.xsim_ok_src2tgt == 0) & (m.xsim_ok_src2tgt_ref == 1))}
    counts = {k: int(mask.sum()) for k, (_, mask) in kinds.items()}
    kind = st.radio("Show", list(kinds), horizontal=True, key="w3_err_kind",
                    format_func=lambda k: f"{kinds[k][0]} ({counts[k]})")
    ex = m[kinds[kind][1]]
    cols = ["src", "tgt", "topic", "pred_src", "pred_src_ref"]
    st.dataframe(ex[cols].rename(columns={"src": "Bengali", "tgt": "Hindi", "topic": "true topic",
                                          "pred_src": f"predicted ({nice(model)})",
                                          "pred_src_ref": f"predicted ({nice(ref)})"}).head(200),
                 width="stretch", hide_index=True)
    st.caption(f"{counts['fixed']} sentences changed from wrong to right and {counts['broken']} from right to "
               "wrong; the difference between the two counts is the change in accuracy. "
               "Up to 200 rows are shown.")


# ---------------------------------------------------------------- tab 5
def _curves(project_root):
    nlp_root, laser_root = _find_roots(project_root / "outputs" / "results")
    ck_dirs = [nlp_root / "outputs" / "checkpoints" / PILOT,
               laser_root / "outputs" / "checkpoints" / f"{PILOT}_laser",
               laser_root / "outputs" / "checkpoints" / PILOT]
    logs = {}
    for d in ck_dirs:
        if d.exists():
            for p in sorted(d.glob("*/train_log.json")):
                if "smoke" not in p.parent.name:
                    logs[p.parent.name] = _json(p)
    if not logs:
        return _missing("sbatch slurm/train.sbatch configs/pilot_bn.yaml e5 both or sbatch slurm/train_laser_co.sbatch configs/pilot_bn.yaml co")
    runs = sorted({re.sub(r"_s\d+$", "", r) for r in logs})
    run = st.selectbox("Setting", runs, format_func=run_label, key="w3_curve_run",
                       index=runs.index("labse_both") if "labse_both" in runs else 0)
    rows, best = [], []
    for name, log in logs.items():
        if re.sub(r"_s\d+$", "", name) != run:
            continue
        seed = split_name(name)[2] or 42
        for r in log["dev"]:
            rows.append({"seed": str(seed), "step": r["step"], "dev topic accuracy %": r["topic_acc"],
                         "dev translation error %": r["xsim"]})
        best.append({"seed": str(seed), "step": log["best"]["step"], "dev topic accuracy %": log["best"]["topic_acc"]})
    df, bdf = pd.DataFrame(rows), pd.DataFrame(best)
    c1, c2 = st.columns(2)
    acc = alt.Chart(df).mark_line(point=True).encode(x="step:Q", y=alt.Y("dev topic accuracy %:Q",
                                                                            scale=alt.Scale(zero=False)),
                                                     color="seed:N")
    star = alt.Chart(bdf).mark_point(shape="diamond", size=150, filled=True, color="#9B1B30").encode(
        x="step:Q", y="dev topic accuracy %:Q", tooltip=["seed", "step", "dev topic accuracy %"])
    c1.altair_chart(acc + star, width="stretch")
    c2.altair_chart(alt.Chart(df).mark_line(point=True).encode(x="step:Q", y="dev translation error %:Q",
                                                               color="seed:N"), width="stretch")
    st.caption("Measured on the 99-sentence SIB-200 development part; one sentence ≈ 1 point. "
               "Red diamond = the checkpoint kept for testing. The kept step differs a lot between runs, "
               "which is one source of the run-to-run spread in tab 2. Step 0 = before training.")


# ---------------------------------------------------------------- tabs 6 and 7: models loaded on the GPU
LIVE_POOLS = {  # label lists for the topic vote: (shown name, dataset, split) of the labelled reference sentences
    # test sentences vote, as in the results table (a sentence never votes for itself)
    "sib200": ("SIB-200 topics (7 labels, seen in training)", "sib200", "test"),
    "in22_gen": ("IN22-Gen domains (13 labels, never seen in training)", "in22_gen", "test"),
}
SEARCH_SETS = [("sib200", "test"), ("in22_gen", "test"), ("in22_conv", "test")]   # sentences searched
LANG = {"src": "Bengali", "tgt": "Hindi"}
SPACE_SETS = {"sib200": "SIB-200 test (7 topics, seen in training)",
              "in22_gen": "IN22-Gen (13 domains, never seen in training)"}


def _gpu_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _live_models(project_root):
    """Untrained models and the seed-42 runs from the config's evaluate list: {name: (path, prefix)}."""
    import yaml
    nlp_root, laser_root = _find_roots(project_root / "outputs" / "results")
    cfg_path = laser_root / "configs" / "pilot_bn.yaml"
    if not cfg_path.exists():
        cfg_path = nlp_root / "configs" / "pilot_bn.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out = {}
    for m in cfg["evaluate"]["models"]:
        if split_name(m["name"])[2]:          # skip the extra seeds; one run per setting is enough for a demo
            continue
        path = m["path"]
        if path.startswith(("outputs/", "data/")):
            p_nlp = nlp_root / path
            p_laser = laser_root / path
            path = str(p_nlp if p_nlp.exists() else p_laser)
        out[m["name"]] = (path, m.get("prefix", ""))

    # Include LASER models from Project_NLP_2
    laser_ck = laser_root / "outputs" / "checkpoints" / "pilot_bn_laser"
    out["laser"] = ("laser", "")
    for variant in ["laser_co", "laser_co_filter", "laser_co_topic"]:
        ckpt = laser_ck / variant / "best_student.pt"
        ckpt_s42 = laser_ck / f"{variant}_s42" / "best_student.pt"
        chosen_ckpt = str(ckpt) if ckpt.exists() else (str(ckpt_s42) if ckpt_s42.exists() else str(ckpt))
        out[variant] = (f"laser:{chosen_ckpt}", "")
    return out


@st.cache_data(show_spinner=False)
def _sentences(project_root, name, split):
    nlp_root, laser_root = _find_roots(Path(project_root) / "outputs" / "results")
    p1 = laser_root / "data" / "processed" / PILOT / f"{name}.tsv"
    p2 = nlp_root / "data" / "processed" / PILOT / f"{name}.tsv"
    path = p2 if p2.exists() else p1
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df[df["split"] == split].reset_index(drop=True)


@st.cache_resource(show_spinner="Loading model on the GPU…")
def _model(path):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(path, device="cuda")


def _encode(path, prefix, texts, side="src", project_root=None):
    if path.startswith("laser"):
        from pilot.laser_utils import encode_laser_texts
        return encode_laser_texts(path, texts, side=side, project_root=project_root)
    import numpy as np
    emb = _model(path).encode([prefix + t for t in texts], batch_size=128, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype="float32")


@st.cache_resource(show_spinner="Encoding the reference sentences (once per model)…")
def _reference(path, prefix, project_root, key, side):
    """Embeddings + table of one reference collection on one language side."""
    if key == "search":
        df = pd.concat([_sentences(project_root, n, sp).assign(source=n) for n, sp in SEARCH_SETS],
                       ignore_index=True)
    else:
        _, n, sp = LIVE_POOLS[key]
        df = _sentences(project_root, n, sp).assign(source=n)
    return _encode(path, prefix, df[side].tolist(), side=side, project_root=project_root), df


def _vote(path, prefix, project_root, q, text, side, pool_key, k=5, group=None):
    """Topic = majority label of the k most similar labelled sentences (ties broken by similarity).

    group: for a test sentence filled in from this same pool, its source document; sentences from the same
    document may not vote, exactly as in the evaluation."""
    import numpy as np
    pool_emb, pool = _reference(path, prefix, project_root, pool_key, side)
    sims = pool_emb @ q
    sims[(pool[side] == text).to_numpy()] = -np.inf     # a sentence may not vote for itself
    if group is not None and "group_id" in pool.columns:
        sims[(pool["group_id"] == group).to_numpy()] = -np.inf   # nor may sentences from its own document
    top = np.argsort(-sims)[:k]
    score = {}
    for j in top:                                     # same vote as in the evaluation: count, ties by similarity
        score[pool["topic"][j]] = score.get(pool["topic"][j], 0) + 1 + 1e-3 * float(sims[j])
    topic = max(score, key=score.get)
    neighbours = pd.DataFrame({"sentence": pool[side].iloc[top].values, "topic": pool["topic"].iloc[top].values,
                                "similarity": np.round(sims[top], 3)})
    return topic, int(round(score[topic])), neighbours


def _search(path, prefix, project_root, q, text, k=5):
    """The k most similar test sentences in each language ."""
    import numpy as np
    per_lang = {}
    for side in ("src", "tgt"):
        emb, df = _reference(path, prefix, project_root, "search", side)
        sims = emb @ q
        sims[(df[side] == text).to_numpy()] = -np.inf      # leave out the input itself
        top = np.argsort(-sims)[:k]
        per_lang[side] = pd.DataFrame({"sentence": df[side].iloc[top].values, "language": LANG[side],
                                       "from": df["source"].iloc[top].map(DATA_NAMES).values,
                                       "similarity": np.round(sims[top], 3)})
    both = pd.concat(per_lang.values()).sort_values("similarity", ascending=False).head(k)
    return both, per_lang


def _steps(path, prefix, text, side="src", project_root=None):
    """What the model does with the sentence: its tokens and the vector it produces."""
    if path.startswith("laser"):
        from pilot.laser_utils import tokenize_laser_text, encode_laser_texts
        tok = tokenize_laser_text(text, side=side, project_root=project_root)
        v = encode_laser_texts(path, [text], side=side, project_root=project_root)[0]
        st.markdown(f"**1. Tokens** — the sentence is cut into {len(tok)} sub-word pieces (SentencePiece):")
        st.code(" · ".join(tok), language=None, wrap_lines=True)
        st.markdown(f"**2. Vector** — LASER turns the tokens into {len(v)} numbers (scaled to length 1). First 8:")
        st.code(", ".join(f"{x:+.3f}" for x in v[:8]) + ", …", language=None)
        st.markdown("Everything else on this page is found by comparing this vector with the vectors of other sentences.")
        return
    tok = _model(path).tokenizer.tokenize(prefix + text)
    v = _encode(path, prefix, [text], side=side, project_root=project_root)[0]
    st.markdown(f"**1. Tokens** — the sentence is cut into {len(tok)} sub-word pieces"
                + (f" (the first ones are the prefix `{prefix.strip()}` this model expects)" if prefix else "")
                + ":")
    st.code(" · ".join(tok), language=None, wrap_lines=True)
    st.markdown(f"**2. Vector** — the encoder turns the tokens into {len(v)} numbers "
                "(scaled to length 1). First 8:")
    st.code(", ".join(f"{x:+.3f}" for x in v[:8]) + ", …", language=None)
    st.markdown("Everything else on this page is found by comparing this vector with the vectors of other "
                "sentences.")


def _pick_example(project_root, name):
    """Fill the text box with a random test sentence of one set and remember its true label and translation."""
    df = _sentences(str(project_root), name, "test")
    row = df.sample(1).iloc[0]
    side = st.session_state.get("live_side", "src")
    st.session_state["live_text"] = row[side]
    st.session_state["live_gold"] = {"text": row[side], "topic": row["topic"], "set": name,
                                     "group": row["group_id"] if "group_id" in row.index else None,
                                     "translation": row["tgt" if side == "src" else "src"]}


def _best_setting(res, names):
    """Trained setting with the highest mean topic accuracy on IN22-Gen over its runs (the main test)."""
    sub = res[res["set"] == "in22_gen:test"].copy()
    sub["run"] = sub["model"].map(lambda m: re.sub(r"_s\d+$", "", m))
    sub = sub[sub["run"].map(lambda r: split_name(r)[1] is not None) & sub["run"].isin(names)]
    if sub.empty:
        return None, None
    means = sub.groupby("run")["knn_acc_src"].mean()
    return means.idxmax(), float(means.max())


def _load_models(project_root):
    if not _gpu_ok():
        st.info("This tab needs a GPU. It works when the dashboard is started through "
                "`slurm/dashboard.sbatch` on a GPU node; the other tabs work without one.")
        return None
    try:
        return _live_models(project_root)
    except Exception as e:  # config missing or unreadable
        st.error(f"Could not read the model list from configs/pilot_bn.yaml: {e}")
        return None


# Tabs 6 and 7 run as fragments: a click inside one of them re-runs only that tab, not the whole page,
# so the heavy maps of tab 6 are not rebuilt and redrawn on every click in tab 7 (and vice versa).
_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None) or (lambda f: f)


# ---------------------------------------------------------------- tab 6: the vector space
@st.cache_data(show_spinner="Drawing the 2D map of the vectors (t-SNE, first time only)…", max_entries=8)
def _tsne(path, prefix, project_root, key):
    """2D positions of all test sentences of one set (both languages). Computed once per model and set,
    then kept for the whole dashboard job; a typed sentence does NOT trigger a new map (see _star)."""
    import time
    import numpy as np
    from sklearn.manifold import TSNE
    t0 = time.time()
    e_src, df = _reference(path, prefix, project_root, key, "src")
    e_tgt, _ = _reference(path, prefix, project_root, key, "tgt")
    xy = TSNE(n_components=2, init="pca", perplexity=30, random_state=42, n_jobs=-1).fit_transform(
        np.vstack([e_src, e_tgt]))
    n = len(df)
    pts = pd.DataFrame({"x": xy[:2 * n, 0], "y": xy[:2 * n, 1],
                        "language": ["Bengali"] * n + ["Hindi"] * n,
                        "topic": list(df["topic"]) * 2,
                        "sentence": list(df["src"]) + list(df["tgt"])})
    links = pd.DataFrame({"x": xy[:n, 0], "y": xy[:n, 1], "x2": xy[n:2 * n, 0], "y2": xy[n:2 * n, 1],
                          "topic": df["topic"].values})
    return pts, links, round(time.time() - t0, 1)


def _star(path, prefix, project_root, key, query, pts, k=5):
    """Place a typed sentence on an existing map: at the mean 2D position of its k most similar test
    sentences (similarity of the real vectors). Fast, and the map itself does not change."""
    import numpy as np
    e_src, _ = _reference(path, prefix, project_root, key, "src")
    e_tgt, _ = _reference(path, prefix, project_root, key, "tgt")
    side = "src"
    if path.startswith("laser"):
        from pilot.laser_utils import detect_laser_lang
        side = "tgt" if detect_laser_lang(query) == "hin_Deva" else "src"
    sims = np.vstack([e_src, e_tgt]) @ _encode(path, prefix, [query], side=side, project_root=project_root)[0]
    top = np.argsort(-sims)[:k]
    return pd.DataFrame({"x": [float(pts["x"].iloc[top].mean())], "y": [float(pts["y"].iloc[top].mean())],
                         "label": ["your sentence"]})


def _space_chart(pts, links, star, title, show_links, highlight):
    """Scatter of the 2D map: colour = true topic, shape = language; optional translation lines and star."""
    topics = sorted(pts["topic"].unique())
    scale = alt.Scale(domain=topics, scheme="tableau20")
    pts = pts.assign(fade=1.0 if highlight is None else (pts["topic"] == highlight).map({True: 1.0, False: 0.08}))
    links = links.assign(fade=0.45 if highlight is None else (links["topic"] == highlight).map({True: 0.6, False: 0.03}))
    noaxis_x, noaxis_y = alt.X("x:Q", axis=None), alt.Y("y:Q", axis=None)
    layers = []
    if show_links:
        layers.append(alt.Chart(links).mark_rule(strokeWidth=0.6, color="#888888").encode(
            x=noaxis_x, y=noaxis_y, x2="x2:Q", y2="y2:Q", opacity=alt.Opacity("fade:Q", scale=None)))
    layers.append(alt.Chart(pts).mark_point(filled=True, size=38).encode(
        x=noaxis_x, y=noaxis_y,
        color=alt.Color("topic:N", scale=scale, title="true topic", legend=alt.Legend(orient="right")),
        shape=alt.Shape("language:N", scale=alt.Scale(domain=["Bengali", "Hindi"], range=["circle", "triangle"]),
                        legend=None),
        opacity=alt.Opacity("fade:Q", scale=None),
        tooltip=["sentence", "topic", "language"]))
    if star is not None:
        layers.append(alt.Chart(star).mark_point(shape="diamond", size=320, filled=True, color="#222222").encode(
            x=noaxis_x, y=noaxis_y))
        layers.append(alt.Chart(star).mark_text(dx=12, align="left", fontWeight="bold").encode(
            x=noaxis_x, y=noaxis_y, text="label:N"))
    return alt.layer(*layers).properties(title=title, height=470)


@_fragment
def _space(project_root, res):
    st.markdown(
        "Each dot is one **test sentence**, placed by its vector: circle = Bengali, triangle = Hindi, colour = "
        "its true topic. A thin grey line joins each sentence to its translation; when a sentence and its "
        "translation land on almost the same spot, the line is too short to see. **Left:** the model before "
        "training. **Right:** after training.\n\n"
        "This is the encoder's vector space itself. Training should move "
        "same-colour dots together while every line stays short (each sentence stays next to its translation).")
    models = _load_models(project_root)
    if models is None:
        return
    names = list(models)
    trained = [n for n in names if split_name(n)[1]]
    if not trained:
        return st.info("No trained models found in the config.")
    best, _ = _best_setting(res, names)
    c1, c2, c3 = st.columns([2, 2, 1])
    key = c1.radio("Test sentences", list(SPACE_SETS), format_func=SPACE_SETS.get, key="space_set")
    model = c2.selectbox("Trained model", trained, format_func=run_label, key="space_model",
                         index=trained.index(best) if best in trained else 0)
    show_links = c3.checkbox("Translation lines", value=True, key="space_links")
    base = split_name(model)[0]
    if base not in models:
        return st.info(f"The untrained {BASES.get(base, base)} model is not in the config.")
    topics = sorted(_sentences(str(project_root), LIVE_POOLS[key][1], LIVE_POOLS[key][2])["topic"].unique())
    c4, c5 = st.columns([1, 2])
    pick = c4.selectbox("Highlight one topic", ["(all topics)"] + topics, key=f"space_hl_{key}")
    highlight = None if pick == "(all topics)" else pick
    query = c5.text_input("Place your own sentence on both maps (optional, Bengali or Hindi)",
                          key="space_text").strip()
    # Streamlit runs every tab on every page load: without this switch the two models and the t-SNE maps
    # would be computed before the page finishes loading, even when nobody opens this tab.
    if not st.checkbox("Draw the maps", key="space_draw"):
        return st.info("Tick 'Draw the maps' to compute them. The first time takes up to a minute (both models "
                       "are loaded and every test sentence is encoded); after that it is fast.")

    cols = st.columns(2)
    for col, name in zip(cols, (base, model)):
        path, prefix = models[name]
        try:
            pts, links, secs = _tsne(path, prefix, str(project_root), key)
            star = _star(path, prefix, str(project_root), key, query, pts) if query else None
        except Exception as e:
            col.error(f"Could not draw the map for {run_label(name)}: {e}")
            continue
        col.altair_chart(_space_chart(pts, links, star, run_label(name), show_links, highlight), width="stretch")
        col.caption(f"Map computed once in {secs} s; reused until the dashboard job restarts.")
    st.caption("The 768 numbers of each vector are squeezed into 2 dimensions with t-SNE, which keeps "
               "near neighbours together but distorts larger distances, and each map is drawn separately. "
               "So the picture is an illustration; the measured numbers are in tabs 2 and 3. "
               "A typed sentence is placed at the average position of its 5 most similar test sentences "
               "(similarity of the real vectors), so the map does not have to be recomputed. "
               "Hover over a dot to read the sentence; use 'Highlight one topic' to fade the others.")


# ---------------------------------------------------------------- tab 7: try it (live)
@_fragment
def _live(project_root, res):
    st.markdown(
        "Type or paste a sentence. Each model turns it into **one vector**, and everything below is read from "
        "that vector by comparing it with the vectors of other sentences.\n\n"
       
        "The same vector can be compared with any set of labelled sentences — below, two different label "
        "lists are used without any retraining.")
    models = _load_models(project_root)
    if models is None:
        return
    names = list(models)
    c1, c2, c3 = st.columns(3)
    side = c1.radio("Input language", list(LANG), format_func=LANG.get, horizontal=True, key="live_side")
    best, best_acc = _best_setting(res, names)
    model = c2.selectbox("Model", names, format_func=run_label, key="live_model",
                         index=names.index(best) if best in names else 0)
    ref_opts = [n for n in names if n != model]
    ref = c3.selectbox("Compare with", ref_opts, format_func=run_label, key="live_ref",
                       index=ref_opts.index(split_name(model)[0]) if split_name(model)[0] in ref_opts else 0)
    b1, b2, _ = st.columns([1, 1, 1])
    b1.button("Fill in a random SIB-200 test sentence", on_click=_pick_example, args=(project_root, "sib200"))
    b2.button("Fill in a random IN22-Gen sentence", on_click=_pick_example, args=(project_root, "in22_gen"))
    text = st.text_area(f"Sentence ({LANG[side]})", key="live_text", height=90)
    if not text.strip():
        return
    text = text.strip()
    gold = st.session_state.get("live_gold")
    if not (gold and gold["text"].strip() == text):
        gold = None

    out = {}
    for name in (model, ref):
        path, prefix = models[name]
        try:
            q = _encode(path, prefix, [text], side=side, project_root=project_root)[0]
            out[name] = {"search": _search(path, prefix, str(project_root), q, text),
                         "votes": {pk: _vote(path, prefix, str(project_root), q, text, side, pk,
                                             group=gold["group"] if gold and gold["set"] == LIVE_POOLS[pk][1]
                                             else None)
                                   for pk in LIVE_POOLS}}
        except Exception as e:
            st.error(f"Could not run {run_label(name)}: {e}")
    if not out:
        return
    shown = [n for n in (model, ref) if n in out]
    other = "tgt" if side == "src" else "src"

    # 1. the encoder on its own: nearest sentences, no labels
    st.markdown("---")
    st.subheader("1 · Most similar sentences")
    st.caption("The 5 sentences closest to the input among the 2,731 test sentences, in Bengali and Hindi. "
               "If the input is one of the test sentences, its own translation should be the closest sentence "
               "in the other language. For a new sentence there is no translation to find: these are simply the "
               "sentences closest in meaning.")
    cols = st.columns(len(shown))
    for col, name in zip(cols, shown):
        path, prefix = models[name]
        both, per_lang = out[name]["search"]
        with col:
            st.markdown(f"**{run_label(name)}**")
            if gold:
                hit = per_lang[other]["sentence"].iloc[0] == gold["translation"]
                st.caption(f"Its own {LANG[other]} translation is ranked first ✓" if hit else
                           f"Its own {LANG[other]} translation is not ranked first ✗")
            st.dataframe(both, width="stretch", hide_index=True)
            with st.expander("Step by step: sentence → tokens → vector"):
                _steps(path, prefix, text, side=side, project_root=project_root)

    # 2. topics read from neighbours, for two label lists
    st.markdown("---")
    st.subheader("2 · Topic read from the 5 nearest labelled sentences")
    st.caption("For each label list, the 5 most similar labelled sentences vote and the majority label wins "
               "(the same vote as in the evaluation; a sentence never votes for itself). Changing the label "
               "list only changes which labelled sentences are compared — the model is not retrained.")
    if gold:
        st.markdown(f"True label of this test sentence ({DATA_NAMES[gold['set']]}): **{gold['topic']}**")
    summary = []
    for pk, (label, _, _) in LIVE_POOLS.items():
        st.markdown(f"**{label}**")
        cols = st.columns(len(shown))
        row = {"label list": label}
        for col, name in zip(cols, shown):
            topic, votes, neighbours = out[name]["votes"][pk]
            mark = (" ✓" if topic == gold["topic"] else " ✗") if (gold and pk == gold["set"]) else ""
            with col:
                st.markdown(f"{run_label(name)}: **{topic}**{mark} — {votes} of the 5 nearest sentences")
                st.dataframe(neighbours, width="stretch", hide_index=True)
            row[run_label(name)] = f"{topic}{mark} ({votes}/5)"
        summary.append(row)

    st.markdown("**Summary**")
    st.dataframe(pd.DataFrame(summary), width="stretch", hide_index=True)
    st.caption("(x/5) = how many of the 5 nearest labelled sentences carry that label; 1/5 means no majority. "
               "Typed sentences have no known true topic; it is shown only for sentences filled in with a "
               "button, and right/wrong is marked only for the label list the true topic comes from. For an "
               "IN22-Gen sentence, sentences from its own source document are not allowed to vote, as in the "
               "evaluation.")
    if best:
        st.caption(f"Default model: {run_label(best)}, the setting with the highest mean topic accuracy on "
                   f"IN22-Gen (domains not seen in training) over its 3 runs ({best_acc:.1f}%). "
                   "Fine-tuned models shown here are the seed-42 runs. Similarity = cosine similarity of the vectors.")