"""Pilot — LASER3-CO on Bengali–Hindi (Tan et al., EACL 2023).

Displays results, comparisons, error analysis, training curves, 2D vector space,
and interactive live testing for:
  - LASER (untrained baseline)
  - LASER3-CO (queue contrastive learning against frozen Hindi teacher)
  - LASER3-CO-Filter (no shuffling + removal of near-duplicate negatives)
  - LASER3-CO + topic loss (optional SupCon topic loss)
With LaBSE and multilingual-E5 rows included for reference.
"""
import json
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from pilot.laser_utils import (
    detect_laser_lang,
    encode_laser_texts,
    load_laser_pipeline,
    tokenize_laser_text,
)

PILOT = "pilot_bn"
BASES = {"laser": "LASER", "labse": "LaBSE", "e5": "E5"}
VARIANTS = {
    "co": "3-CO",
    "co_filter": "3-CO-Filter",
    "co_topic": "3-CO + topic loss",
    "trans": "+ translation loss",
    "topic": "+ topic loss",
    "both": "+ both losses",
}
SETS = {"sib200:test": "SIB-200 test", "in22_gen:test": "IN22-Gen", "in22_conv:test": "IN22-Conv"}
SET_NOTE = {
    "sib200:test": "Same 7 topic labels as the SIB-200 training part — measures learning of seen labels.",
    "in22_gen:test": "13 domains never used in training — the main test of transfer to new topics.",
    "in22_conv:test": "Turns from 44 conversations; 16 domains, never used in training.",
}
METRICS = {
    "xsim_src2tgt": "Translation error, Bengali→Hindi (%)",
    "knn_acc_src": "Topic accuracy (%)",
    "knn_acc_cross": "Topic accuracy across languages (%)",
    "xsim_tgt2src": "Translation error, Hindi→Bengali (%)",
    "nmi_src": "Cluster agreement (NMI)",
}
PAIR_METRICS = {
    "xsim_ok_src2tgt": "translation accuracy",
    "ok_src": "topic accuracy",
    "ok_cross": "topic accuracy across languages",
}
VERDICT_LABEL = {
    "B better (all seeds)": "Higher on all 3 runs",
    "mixed across seeds": "Differs between runs",
    "no clear difference": "No measurable difference",
    "A better (all seeds)": "Lower on all 3 runs",
}
LASER_CLAIMS = [
    ("sib200:test", "laser", "laser_co", "xsim_ok_src2tgt"),
    ("in22_gen:test", "laser", "laser_co", "xsim_ok_src2tgt"),
    ("sib200:test", "laser", "laser_co_filter", "xsim_ok_src2tgt"),
    ("in22_gen:test", "laser", "laser_co_filter", "xsim_ok_src2tgt"),
    ("in22_gen:test", "laser", "laser_co_topic", "ok_src"),
    ("sib200:test", "laser_co", "laser_co_topic", "ok_src"),
]

LIVE_POOLS = {
    "sib200": ("SIB-200 topics (7 labels, seen in training)", "sib200", "test"),
    "in22_gen": ("IN22-Gen domains (13 labels, never seen in training)", "in22_gen", "test"),
}
SEARCH_SETS = [("sib200", "test"), ("in22_gen", "test"), ("in22_conv", "test")]
LANG = {"src": "Bengali", "tgt": "Hindi"}
SPACE_SETS = {
    "sib200": "SIB-200 test (7 topics, seen in training)",
    "in22_gen": "IN22-Gen (13 domains, never seen in training)",
}
DATA_NAMES = {"sib200": "SIB-200", "in22_gen": "IN22-Gen", "in22_conv": "IN22-Conv"}

_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None) or (lambda f: f)


def _gpu_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def _missing(step):
    st.info(f"No results yet — run `{step}` first.")


def split_name(model):
    """'laser_co_s43' -> ('laser', 'co', 43); 'laser' -> ('laser', None, None)."""
    m = re.fullmatch(r"(?P<b>[a-z0-9]+?)(?:_(?P<v>co_filter|co_topic|co|trans|topic|both))?(?:_s(?P<s>\d+))?", model)
    if not m:
        return model, None, None
    return m["b"], m["v"], int(m["s"]) if m["s"] else None


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


def nice(model):
    b, v, s = split_name(model)
    name = run_label(model)
    if v and not s:
        s = 42
    return name + (f" · seed {s}" if s else "")


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
    d = laser_root / "outputs" / "results" / PILOT
    ev = d / "eval"

    st.title("LASER3-CO — Bengali–Hindi Pilot")
    st.caption("Multilingual Representation Distillation with Contrastive Learning (Tan et al., EACL 2023). "
               "Fine-tuning the Bengali LASER encoder (student) against a frozen Hindi encoder (teacher) "
               "using InfoNCE with a queue of earlier teacher embeddings.")

    res_laser = _read(ev / "results.csv")
    res_nlp = _read(nlp_root / "outputs" / "results" / PILOT / "eval" / "results.csv")
    if res_laser is not None and res_nlp is not None:
        res = pd.concat([res_nlp, res_laser], ignore_index=True).drop_duplicates(subset=["model", "set"], keep="last")
    else:
        res = res_laser if res_laser is not None else res_nlp

    if res is None:
        return _missing("sbatch slurm/evaluate_laser.sbatch configs/pilot_bn.yaml")

    tabs = st.tabs(["1 · Setup", "2 · Results (LASER & Benchmarks)", "3 · Statistical Comparison",
                    "4 · Error Analysis", "5 · Training Curves", "6 · The vector space", "7 · Try it (live)"])
    with tabs[0]:
        _setup()
    with tabs[1]:
        _results(res)
    with tabs[2]:
        _compare(ev)
    with tabs[3]:
        _errors(ev, laser_root, res)
    with tabs[4]:
        _curves(laser_root)
    with tabs[5]:
        _space(laser_root, res)
    with tabs[6]:
        _live(laser_root, res)


# ---------------------------------------------------------------- tab 1: setup
def _setup():
    st.subheader("LASER3-CO Pipeline Setup")
    st.markdown(
        "- **Reference:** Tan, Heffernan, Schwenk, Koehn (EACL 2023, arXiv 2210.05033).\n"
        "- **Architecture:**\n"
        "  - **Teacher:** Hindi encoder (`hin_Deva`), frozen throughout training.\n"
        "  - **Student:** Bengali encoder (`ben_Beng`), trained via contrastive distillation.\n"
        "- **Loss Function (InfoNCE):**\n"
        r"  $$\mathcal{L} = -\log \frac{\exp(q \cdot k_+ / \tau)}{\sum_i \exp(q \cdot k_i / \tau)}$$"
        "  where $q$ is the student embedding of the Bengali sentence, $k_+$ is the teacher embedding of its Hindi translation, "
        "and negatives are a FIFO queue of earlier teacher embeddings.\n"
        "- **Hyperparameters:** Queue size 4096, batch size 32, $\\tau = 0.05$, learning rate $1\\times 10^{-5}$ (AdamW, 10% warmup), 3 epochs.\n"
        "- **Four Evaluated Settings:**\n"
        "  1. **`laser`**: LASER encoders as-is (untrained baseline).\n"
        "  2. **`laser_co`**: LASER3-CO queue contrastive learning against frozen teacher.\n"
        "  3. **`laser_co_filter`**: LASER3-CO-Filter (no shuffling + filter negatives with cosine $\\ge 0.9$).\n"
        "  4. **`laser_co_topic`**: LASER3-CO plus supervised contrastive topic loss (SupCon).\n"
        "- **Checkpoints:** Selected by best dev translation error (`laser_co`, `laser_co_filter`) or best dev topic accuracy (`laser_co_topic`) on SIB-200 dev."
    )


# ---------------------------------------------------------------- tab 2: results
def _summary_table(res, set_key, metric):
    sub = res[res["set"] == set_key].copy()
    if sub.empty or metric not in sub.columns:
        return pd.DataFrame()
    sub["run"] = sub["model"].map(lambda m: re.sub(r"_s\d+$", "", m))
    sub["seed"] = sub["model"].map(lambda m: split_name(m)[2] or 42)
    sub["family"] = sub["run"].map(lambda r: split_name(r)[0])

    fam_order = {"laser": 0, "labse": 1, "e5": 2}
    sub["fam_rank"] = sub["family"].map(lambda f: fam_order.get(f, 9))

    piv = sub.pivot_table(index=["fam_rank", "family", "run"], columns="seed", values=metric).reset_index()
    seed_cols = [c for c in piv.columns if isinstance(c, (int, np.integer))]
    piv["mean"] = piv[seed_cols].mean(axis=1).round(2)
    piv["std"] = piv[seed_cols].std(axis=1).round(2)
    piv["model_label"] = piv["run"].map(run_label)
    piv = piv.sort_values(by=["fam_rank", "run"])

    disp = pd.DataFrame({"Model": piv["model_label"]})
    for s in sorted(seed_cols):
        disp[f"Seed {s}"] = piv[s].map(lambda v: f"{v:.2f}" if pd.notnull(v) else "—")
    disp["Mean"] = piv["mean"].map(lambda v: f"{v:.2f}" if pd.notnull(v) else "—")
    disp["Std"] = piv["std"].map(lambda v: f"±{v:.2f}" if pd.notnull(v) and v > 0 else "")
    return disp


def _results(res):
    st.subheader("LASER3-CO & Benchmark Models Results")
    st.caption("Benchmark comparison across all 3 random seeds (42, 43, 44).")

    set_key = st.selectbox("Evaluation set", list(SETS), format_func=SETS.get, key="laser_res_set")
    st.markdown(f"*{SET_NOTE.get(set_key, '')}*")

    st.markdown("#### Primary Metric: Topic Accuracy (%)")
    t_acc = _summary_table(res, set_key, "knn_acc_src")
    if not t_acc.empty:
        st.dataframe(t_acc, width="stretch", hide_index=True)
    else:
        st.info("No topic accuracy records found.")

    st.markdown("#### Translation Retrieval Error: Bengali→Hindi (%)")
    t_err = _summary_table(res, set_key, "xsim_src2tgt")
    if not t_err.empty:
        st.dataframe(t_err, width="stretch", hide_index=True)
    else:
        st.info("No translation retrieval records found.")


# ---------------------------------------------------------------- tab 3: comparison
def _compare(ev):
    nlp_root, laser_root = _find_roots(ev.parents[1])
    path_laser = laser_root / "outputs" / "results" / PILOT / "eval" / "compare.csv"
    path_nlp = nlp_root / "outputs" / "results" / PILOT / "eval" / "compare.csv"
    cmp_l = pd.read_csv(path_laser) if path_laser.exists() else None
    cmp_n = pd.read_csv(path_nlp) if path_nlp.exists() else None
    if cmp_l is not None and cmp_n is not None:
        cmp_df = pd.concat([cmp_n, cmp_l], ignore_index=True).drop_duplicates(subset=["set", "model", "compared with", "metric"], keep="last")
    else:
        cmp_df = cmp_l if cmp_l is not None else cmp_n

    if cmp_df is None:
        return _missing("sbatch slurm/evaluate_laser_all.sbatch configs/pilot_bn.yaml")

    laser_cmp = cmp_df[cmp_df["model"].str.startswith("laser") | cmp_df["compared with"].str.startswith("laser")]
    if laser_cmp.empty:
        return st.info("No LASER comparisons found in compare.csv.")

    st.subheader("LASER3-CO Statistical Significance (Paired Bootstrap)")
    st.caption("Testing whether differences over 3 random seeds (10,000 bootstrap resamples) are statistically significant.")

    metric = st.selectbox("Metric to compare", list(PAIR_METRICS), format_func=PAIR_METRICS.get, key="laser_cmp_metric")
    sub = laser_cmp[laser_cmp["metric"] == metric].copy()
    if sub.empty:
        return st.info(f"No comparisons available for {PAIR_METRICS.get(metric, metric)}.")

    sub["pair"] = sub["set"].map(SETS.get) + ": " + sub["model"].map(nice) + " vs " + sub["compared with"].map(nice)
    sub["result"] = sub["verdict"].map(lambda v: VERDICT_LABEL.get(v, v))

    forest = alt.Chart(sub).mark_rule(strokeWidth=2).encode(
        y=alt.Y("pair:N", sort=sub["pair"].tolist(), title=None),
        x=alt.X("pooled_lo:Q", title=f"Difference in {PAIR_METRICS.get(metric, metric)} (95% CI)"),
        x2="pooled_hi:Q",
        color=alt.Color("result:N", scale=alt.Scale(
            domain=list(VERDICT_LABEL.values()), range=["#2E7D32", "#B08900", "#888888", "#9B1B30"])))
    dots = alt.Chart(sub).mark_point(filled=True, size=60, color="#222222").encode(
        y=alt.Y("pair:N", sort=sub["pair"].tolist()), x="mean_diff:Q")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart(forest + dots + zero, width="stretch")

    seed_cols = sorted(c for c in sub.columns if re.fullmatch(r"diff_s\d+", c))
    show = sub[["model", "compared with"] + seed_cols + ["mean_diff", "pooled_lo", "pooled_hi", "pooled_p", "result"]]
    st.dataframe(show.rename(columns={"mean_diff": "mean diff", "pooled_lo": "95% CI low", "pooled_hi": "95% CI high"}),
                 width="stretch", hide_index=True)


# ---------------------------------------------------------------- tab 4: error analysis
def _errors(ev, project_root, res):
    nlp_root, laser_root = _find_roots(project_root / "outputs" / "results")
    c1, c2, c3 = st.columns(3)
    set_key = c1.selectbox("Test set", list(SETS), format_func=SETS.get, key="laser_err_set")
    models = sorted(res.loc[res["set"] == set_key, "model"].unique(), key=lambda m: (split_name(m)[0], nice(m)))
    laser_models = [m for m in models if m.startswith("laser")]
    trained_laser = [m for m in laser_models if split_name(m)[1]]

    if not trained_laser:
        return st.info("No trained LASER checkpoints evaluated yet.")

    model = c2.selectbox("Trained model", trained_laser, format_func=nice, key="laser_err_model")
    ref_opts = [m for m in laser_models if m != model]
    ref = c3.selectbox("Compare with", ref_opts, format_func=nice, key="laser_err_ref",
                       index=ref_opts.index("laser") if "laser" in ref_opts else 0)

    stem = set_key.replace(":", "_")
    def _read_per(m):
        p1 = laser_root / "outputs" / "results" / PILOT / "eval" / f"{m}__{stem}_per_sentence.tsv"
        p2 = nlp_root / "outputs" / "results" / PILOT / "eval" / f"{m}__{stem}_per_sentence.tsv"
        return _read(p1, sep="\t", dtype=str) if m.startswith("laser") else (_read(p2, sep="\t", dtype=str) or _read(p1, sep="\t", dtype=str))

    pm, pr = _read_per(model), _read_per(ref)
    if pm is None or pr is None:
        return _missing("sbatch slurm/evaluate_laser.sbatch configs/pilot_bn.yaml")

    for p in (pm, pr):
        for c in ("ok_src", "ok_cross", "xsim_ok_src2tgt"):
            if c in p.columns:
                p[c] = p[c].astype(int)

    st.subheader(f"Per-Topic Accuracy: {nice(model)} vs {nice(ref)}")
    per = pd.concat([pm.groupby("topic")["ok_src"].agg(["mean", "size"]).assign(model=nice(model)),
                     pr.groupby("topic")["ok_src"].agg(["mean", "size"]).assign(model=nice(ref))]).reset_index()
    per["accuracy %"] = (100 * per["mean"]).round(1)
    st.altair_chart(alt.Chart(per).mark_bar().encode(
        y=alt.Y("topic:N", sort="-x", title=None), x=alt.X("accuracy %:Q", scale=alt.Scale(domain=[0, 100])),
        yOffset="model:N", color=alt.Color("model:N", scale=alt.Scale(range=["#1F77B4", "#888888"])),
        tooltip=["topic", "model", "accuracy %", "size"]), width="stretch")


# ---------------------------------------------------------------- tab 5: curves
def _curves(project_root):
    nlp_root, laser_root = _find_roots(project_root / "outputs" / "results")
    ck_dirs = [laser_root / "outputs" / "checkpoints" / "pilot_bn_laser",
               laser_root / "outputs" / "checkpoints" / "pilot_bn",
               nlp_root / "outputs" / "checkpoints" / "pilot_bn"]
    logs = {}
    for d in ck_dirs:
        if d.exists():
            for p in sorted(d.glob("*/train_log.json")):
                if "smoke" not in p.parent.name and p.parent.name.startswith("laser"):
                    logs[p.parent.name] = _json(p)

    if not logs:
        return _missing("sbatch slurm/train_laser_co.sbatch configs/pilot_bn.yaml co 42")

    runs = sorted({re.sub(r"_s\d+$", "", r) for r in logs})
    run = st.selectbox("Setting", runs, format_func=run_label, key="laser_curve_run")
    rows, best = [], []
    for name, log in logs.items():
        if re.sub(r"_s\d+$", "", name) != run:
            continue
        seed = split_name(name)[2] or 42
        for r in log.get("dev", []):
            rows.append({"seed": str(seed), "step": r["step"], "dev topic accuracy %": r["topic_acc"],
                         "dev translation error %": r["xsim"]})
        if "best" in log:
            best.append({"seed": str(seed), "step": log["best"]["step"], "dev topic accuracy %": log["best"].get("topic_acc", 0)})

    df, bdf = pd.DataFrame(rows), pd.DataFrame(best)
    if df.empty:
        return st.info("No training log points to plot.")

    c1, c2 = st.columns(2)
    acc = alt.Chart(df).mark_line(point=True).encode(x="step:Q", y=alt.Y("dev topic accuracy %:Q", scale=alt.Scale(zero=False)), color="seed:N")
    if not bdf.empty:
        acc += alt.Chart(bdf).mark_point(shape="diamond", size=150, filled=True, color="#9B1B30").encode(x="step:Q", y="dev topic accuracy %:Q")
    c1.altair_chart(acc, width="stretch")

    xsim = alt.Chart(df).mark_line(point=True).encode(x="step:Q", y=alt.Y("dev translation error %:Q", scale=alt.Scale(zero=False)), color="seed:N")
    c2.altair_chart(xsim, width="stretch")
    st.caption("Measured on SIB-200 dev. Red diamond = checkpoint kept. Step 0 = before training.")


# ---------------------------------------------------------------- tab 6 & 7 helpers
@st.cache_data(show_spinner=False)
def _sentences(project_root, name, split):
    nlp_root, laser_root = _find_roots(Path(project_root) / "outputs" / "results")
    p1 = laser_root / "data" / "processed" / PILOT / f"{name}.tsv"
    p2 = nlp_root / "data" / "processed" / PILOT / f"{name}.tsv"
    p = p1 if p1.exists() else p2
    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    return df[df["split"] == split].reset_index(drop=True)


@st.cache_resource(show_spinner="Encoding LASER reference sentences (once per model)…")
def _laser_reference(model_name, project_root, key, side):
    """Embeddings + table of one reference collection on one language side."""
    if key == "search":
        df = pd.concat([_sentences(project_root, n, sp).assign(source=n) for n, sp in SEARCH_SETS],
                       ignore_index=True)
    else:
        _, n, sp = LIVE_POOLS[key]
        df = _sentences(project_root, n, sp).assign(source=n)
    emb = encode_laser_texts(model_name, df[side].tolist(), side=side, project_root=project_root)
    return emb, df


@st.cache_data(show_spinner="Drawing the 2D map of LASER vectors (t-SNE, first time only)…", max_entries=8)
def _laser_tsne(model_name, project_root, key):
    """2D positions of all test sentences of one set (Bengali + Hindi) in LASER's space."""
    import time
    from sklearn.manifold import TSNE
    t0 = time.time()
    e_src, df = _laser_reference(model_name, project_root, key, "src")
    e_tgt, _ = _laser_reference(model_name, project_root, key, "tgt")
    xy = TSNE(n_components=2, init="pca", perplexity=30, random_state=42, n_jobs=-1).fit_transform(
        np.vstack([e_src, e_tgt]))
    n = len(df)
    pts = pd.DataFrame({
        "x": xy[:2 * n, 0],
        "y": xy[:2 * n, 1],
        "language": ["Bengali"] * n + ["Hindi"] * n,
        "topic": list(df["topic"]) * 2,
        "sentence": list(df["src"]) + list(df["tgt"]),
    })
    links = pd.DataFrame({
        "x": xy[:n, 0],
        "y": xy[:n, 1],
        "x2": xy[n:2 * n, 0],
        "y2": xy[n:2 * n, 1],
        "topic": df["topic"].values,
    })
    return pts, links, round(time.time() - t0, 1)


def _laser_star(model_name, project_root, key, query, pts, k=5):
    """Place user typed sentence on map at mean position of its 5 most similar test sentences."""
    e_src, _ = _laser_reference(model_name, project_root, key, "src")
    e_tgt, _ = _laser_reference(model_name, project_root, key, "tgt")
    side = "tgt" if detect_laser_lang(query) == "hin_Deva" else "src"
    q = encode_laser_texts(model_name, [query], side=side, project_root=project_root)[0]
    sims = np.vstack([e_src, e_tgt]) @ q
    top = np.argsort(-sims)[:k]
    return pd.DataFrame({
        "x": [float(pts["x"].iloc[top].mean())],
        "y": [float(pts["y"].iloc[top].mean())],
        "label": ["your sentence"],
    })


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


def _pick_example(project_root, name):
    df = _sentences(str(project_root), name, "test")
    row = df.sample(1).iloc[0]
    side = st.session_state.get("laser_live_side", "src")
    st.session_state["laser_live_text"] = row[side]
    st.session_state["laser_live_gold"] = {
        "text": row[side],
        "topic": row["topic"],
        "set": name,
        "group": row["group_id"] if "group_id" in row.index else None,
        "translation": row["tgt" if side == "src" else "src"],
    }


def _laser_search(model_name, project_root, q, text, k=5):
    """The k most similar test sentences in each language for LASER."""
    per_lang = {}
    for side in ("src", "tgt"):
        emb, df = _laser_reference(model_name, project_root, "search", side)
        sims = emb @ q
        sims[(df[side] == text).to_numpy()] = -np.inf
        top = np.argsort(-sims)[:k]
        per_lang[side] = pd.DataFrame({
            "sentence": df[side].iloc[top].values,
            "language": LANG[side],
            "from": df["source"].iloc[top].map(DATA_NAMES).values,
            "similarity": np.round(sims[top], 3),
        })
    both = pd.concat(per_lang.values()).sort_values("similarity", ascending=False).head(k)
    return both, per_lang


def _laser_vote(model_name, project_root, q, text, side, pool_key, k=5, group=None):
    """Topic = majority label of the k most similar labelled sentences for LASER."""
    pool_emb, pool = _laser_reference(model_name, project_root, pool_key, side)
    sims = pool_emb @ q
    sims[(pool[side] == text).to_numpy()] = -np.inf
    if group is not None and "group_id" in pool.columns:
        sims[(pool["group_id"] == group).to_numpy()] = -np.inf
    top = np.argsort(-sims)[:k]
    score = {}
    for j in top:
        score[pool["topic"][j]] = score.get(pool["topic"][j], 0) + 1 + 1e-3 * float(sims[j])
    topic = max(score, key=score.get)
    neighbours = pd.DataFrame({
        "sentence": pool[side].iloc[top].values,
        "topic": pool["topic"].iloc[top].values,
        "similarity": np.round(sims[top], 3),
    })
    return topic, int(round(score[topic])), neighbours


def _laser_steps(model_name, text, side, project_root):
    """Show LASER tokens and first 8 coordinates of the 1024-dim embedding."""
    tok = tokenize_laser_text(text, side=side, project_root=project_root)
    v = encode_laser_texts(model_name, [text], side=side, project_root=project_root)[0]
    st.markdown(f"**1. Tokens** — the sentence is cut into {len(tok)} sub-word pieces (SentencePiece):")
    st.code(" · ".join(tok), language=None, wrap_lines=True)
    st.markdown(f"**2. Vector** — LASER turns the tokens into {len(v)} numbers (1024-dimensional, normalized). First 8:")
    st.code(", ".join(f"{x:+.3f}" for x in v[:8]) + ", …", language=None)
    st.markdown("All nearest neighbours and topic votes are computed via cosine similarity with this 1024-D vector.")


# ---------------------------------------------------------------- tab 6: the vector space
@_fragment
def _space(project_root, res):
    st.markdown(
        "Each dot is one **test sentence**, placed by its **1024-dimensional LASER vector**: circle = Bengali, "
        "triangle = Hindi, colour = its true topic. A thin grey line joins each Bengali sentence to its Hindi translation; "
        "when translation pairs align closely in vector space, the line is very short. "
        "**Left:** the untrained baseline (`laser`). **Right:** after contrastive distillation.\n\n"
        "Contrastive distillation should cluster same-topic sentences together while keeping cross-lingual translation lines short.")

    if not _gpu_ok():
        return st.info("This tab needs a GPU. It works when the dashboard is started through "
                       "`slurm/dashboard.sbatch` on a GPU node; tabs 1-5 work without one.")

    trained_options = ["laser_co", "laser_co_filter", "laser_co_topic"]
    c1, c2, c3 = st.columns([2, 2, 1])
    key = c1.radio("Test sentences", list(SPACE_SETS), format_func=SPACE_SETS.get, key="laser_space_set")
    model = c2.selectbox("Trained model", trained_options, format_func=run_label, key="laser_space_model")
    show_links = c3.checkbox("Translation lines", value=True, key="laser_space_links")

    topics = sorted(_sentences(str(project_root), LIVE_POOLS[key][1], LIVE_POOLS[key][2])["topic"].unique())
    c4, c5 = st.columns([1, 2])
    pick = c4.selectbox("Highlight one topic", ["(all topics)"] + topics, key=f"laser_space_hl_{key}")
    highlight = None if pick == "(all topics)" else pick
    query = c5.text_input("Place your own sentence on both maps (optional, Bengali or Hindi)",
                          key="laser_space_text").strip()

    if not st.checkbox("Draw the maps", key="laser_space_draw"):
        return st.info("Tick 'Draw the maps' to compute them. The first time takes ~30 seconds (models "
                       "are loaded and test sentences encoded); after that results are cached and fast.")

    cols = st.columns(2)
    for col, name in zip(cols, ("laser", model)):
        try:
            pts, links, secs = _laser_tsne(name, str(project_root), key)
            star = _laser_star(name, str(project_root), key, query, pts) if query else None
        except Exception as e:
            col.error(f"Could not draw the map for {run_label(name)}: {e}")
            continue
        col.altair_chart(_space_chart(pts, links, star, run_label(name), show_links, highlight), width="stretch")
        col.caption(f"Map computed in {secs} s; cached for subsequent interactions.")

    st.caption("The 1024 numbers of each LASER vector are projected into 2 dimensions with t-SNE. "
               "A typed sentence is placed at the centroid of its 5 nearest neighbours among the real test vectors. "
               "Hover over any dot to read the sentence; use 'Highlight one topic' to fade other classes.")


# ---------------------------------------------------------------- tab 7: try it (live)
@_fragment
def _live(project_root, res):
    st.markdown(
        "Type or paste a sentence in Bengali or Hindi. The LASER encoder converts it into a "
        "**1024-dimensional normalized vector**, and all nearest-neighbour searches and topic classifications "
        "below are computed directly from that vector.\n\n"
        "You can compare the representations of the untrained baseline vs fine-tuned LASER3-CO models in real time.")

    if not _gpu_ok():
        return st.info("This tab needs a GPU. It works when the dashboard is started through "
                       "`slurm/dashboard.sbatch` on a GPU node; tabs 1-5 work without one.")

    all_laser = ["laser_co_topic", "laser_co", "laser_co_filter", "laser"]
    c1, c2, c3 = st.columns(3)
    side = c1.radio("Input language", list(LANG), format_func=LANG.get, horizontal=True, key="laser_live_side")
    model = c2.selectbox("Model", all_laser, format_func=run_label, key="laser_live_model", index=0)
    ref_opts = [m for m in all_laser if m != model]
    ref = c3.selectbox("Compare with", ref_opts, format_func=run_label, key="laser_live_ref",
                       index=ref_opts.index("laser") if "laser" in ref_opts else 0)

    b1, b2, _ = st.columns([1, 1, 1])
    b1.button("Fill in a random SIB-200 test sentence", on_click=_pick_example, args=(project_root, "sib200"), key="laser_btn_sib")
    b2.button("Fill in a random IN22-Gen sentence", on_click=_pick_example, args=(project_root, "in22_gen"), key="laser_btn_in22")

    text = st.text_area(f"Sentence ({LANG[side]})", key="laser_live_text", height=90)
    if not text.strip():
        return
    text = text.strip()
    gold = st.session_state.get("laser_live_gold")
    if not (gold and gold["text"].strip() == text):
        gold = None

    out = {}
    for name in (model, ref):
        try:
            q = encode_laser_texts(name, [text], side=side, project_root=str(project_root))[0]
            out[name] = {
                "search": _laser_search(name, str(project_root), q, text),
                "votes": {
                    pk: _laser_vote(name, str(project_root), q, text, side, pk,
                                    group=gold["group"] if gold and gold["set"] == LIVE_POOLS[pk][1] else None)
                    for pk in LIVE_POOLS
                },
            }
        except Exception as e:
            st.error(f"Could not run {run_label(name)}: {e}")

    if not out:
        return
    shown = [n for n in (model, ref) if n in out]
    other = "tgt" if side == "src" else "src"

    # 1. Most similar sentences
    st.markdown("---")
    st.subheader("1 · Most similar sentences")
    st.caption("The 5 sentences closest to the input among the 2,731 test sentences in Bengali and Hindi. "
               "For a test sentence, its gold translation in the other language should ideally rank #1.")

    cols = st.columns(len(shown))
    for col, name in zip(cols, shown):
        both, per_lang = out[name]["search"]
        with col:
            st.markdown(f"**{run_label(name)}**")
            if gold:
                hit = per_lang[other]["sentence"].iloc[0] == gold["translation"]
                st.caption(f"Gold {LANG[other]} translation is ranked first ✓" if hit else
                           f"Gold {LANG[other]} translation is not ranked first ✗")
            st.dataframe(both, width="stretch", hide_index=True)
            with st.expander("Step by step: sentence → tokens → vector"):
                _laser_steps(name, text, side, str(project_root))

    # 2. Topic classification via 5-NN voting
    st.markdown("---")
    st.subheader("2 · Topic read from the 5 nearest labelled sentences")
    st.caption("The 5 most similar labelled sentences cast votes; majority label wins. "
               "No retrained classifier head is used — all predictions emerge purely from the embedding geometry.")
    if gold:
        st.markdown(f"True label of this test sentence ({DATA_NAMES[gold['set']]}): **{gold['topic']}**")

    for pk, (label, _, _) in LIVE_POOLS.items():
        st.markdown(f"**{label}**")
        vcols = st.columns(len(shown))
        for col, name in zip(vcols, shown):
            top_topic, votes, neighbours = out[name]["votes"][pk]
            with col:
                st.markdown(f"Predicted topic ({run_label(name)}): **{top_topic}** ({votes} of 5 votes)")
                st.dataframe(neighbours, width="stretch", hide_index=True)
