"""Week 2 — development (gold) set split, then exploratory topic clustering of the training pairs."""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

WEEK_DIR = "week_02_dev_and_clusters"
LAGS = [1, 2, 3, 5, 10, 25, 50, 100]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def _missing(step):
    st.info(f"No results yet — run `{step}` first.")


def _runs(d):
    """{model: [k, ...]} from the summary files present."""
    runs = {}
    for f in sorted(d.glob("summary_*_k*.json")):
        model, k = f.stem.replace("summary_", "").rsplit("_k", 1)
        runs.setdefault(model, []).append(int(k))
    return {m: sorted(ks) for m, ks in runs.items()}


def render(results_root):
    results_root = Path(results_root)
    project_root = results_root.parents[1]          # outputs/results -> project root
    d = results_root / WEEK_DIR
    st.title("Week 2 — Development set & topic clusters")
    st.caption("1,000 pairs are set aside as a hand-labelled development set BEFORE clustering; "
               "clustering uses only the remaining training pairs (Hindi side). "
               "The official test set is not used or shown.")

    notes = d / "notes.md"
    if notes.exists():
        with st.expander("Notes for this week", expanded=True):
            st.markdown(notes.read_text(encoding="utf-8"))

    t1, t2, t3, t4 = st.tabs(["1 · Development set", "2 · Clusters", "3 · Cluster quality",
                              "4 · LaBSE vs E5"])
    with t1:
        _devset(d, project_root)
    with t2:
        _clusters(d)
    with t3:
        _quality(d, project_root)
    with t4:
        _compare(d)


# ---------------------------------------------------------------- tab 1
def _devset(d, project_root):
    rep = _json(d / "split_report.json")
    if not rep:
        return _missing("sbatch slurm/split.sbatch configs/week02.yaml")

    c = st.columns(4)
    c[0].metric("Clean pairs (week 1)", f"{rep['n_pairs']:,}")
    c[1].metric("Development set", f"{rep['n_gold_dev']:,}")
    c[2].metric("Training pool", f"{rep['n_train']:,}")
    c[3].metric("Near-copies set aside", f"{rep['n_excluded_dup']:,}",
                help="Training pairs whose Hindi or Kangri side matches a development pair after removing "
                     "punctuation and spacing. Used nowhere.")

    st.subheader("Split by corpus")
    per = pd.DataFrame(rep["per_source"]).T.reindex(columns=["train", "gold_dev", "excluded_dup"]).fillna(0)
    per = per.astype(int)
    per["dev share %"] = (100 * per["gold_dev"] / per.sum(axis=1)).round(2)
    st.dataframe(per, width="stretch")
    st.caption(f"Sampled per corpus in proportion to its size (seed {rep['seed']}), so both corpora keep "
               "their share in the development set.")

    if rep.get("excluded_dup_examples"):
        with st.expander(f"Near-copies set aside ({len(rep['excluded_dup_examples'])} shown)"):
            st.dataframe(pd.DataFrame(rep["excluded_dup_examples"]), width="stretch", hide_index=True)

    sheet = project_root / "data" / "annotation" / "gold_dev_to_annotate.csv"
    if sheet.exists():
        a = pd.read_csv(sheet, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        st.subheader("Hand-labelling progress")
        c = st.columns(3)
        c[0].metric("Pairs checked (pair_ok)", f"{(a['pair_ok'] != '').sum()} / {len(a)}")
        c[1].metric("Pairs with a topic", f"{(a['topic'] != '').sum()} / {len(a)}")
        bad = a["pair_ok"].str.lower().isin(["n", "p"]).sum()
        c[2].metric("Marked wrong or partial", f"{bad}")
        st.caption("Topics are filled in only after the topic list is fixed.")


# ---------------------------------------------------------------- tab 2
def _clusters(d):
    runs = _runs(d)
    if not runs:
        return _missing("sbatch slurm/cluster.sbatch configs/week02.yaml")

    c1, c2 = st.columns(2)
    model = c1.selectbox("Embedding model", sorted(runs), key="w2_model")
    k = c2.selectbox("Number of clusters (k)", runs[model], key="w2_k")
    s = _json(d / f"summary_{model}_k{k}.json")
    st.caption(f"Silhouette (cosine): {s['silhouette']} · {sum(c['size'] for c in s['clusters']):,} training pairs")

    rows = []
    for c in s["clusters"]:
        total = sum(c["sources"].values())
        rows.append({"cluster": c["cluster"], "size": c["size"], "share %": c["share_pct"],
                     "mendeley %": round(100 * c["sources"].get("mendeley", 0) / total, 1),
                     "chauhan %": round(100 * c["sources"].get("chauhan", 0) / total, 1),
                     "keywords (Hindi side)": ", ".join(c["keywords"])})
    table = pd.DataFrame(rows)
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption("A cluster that is ~100% one corpus may reflect writing style rather than topic. "
               "Also watch for clusters made mostly of questions or dialogue lines.")

    pick = st.selectbox("Read a cluster's example pairs", table["cluster"].tolist(), key="w2_pick",
                        format_func=lambda c: f"cluster {c} — "
                        + table.set_index("cluster").loc[c, "keywords (Hindi side)"][:80])
    ex = next(c for c in s["clusters"] if c["cluster"] == pick)["examples"]
    st.dataframe(pd.DataFrame(ex), width="stretch", hide_index=True)


# ---------------------------------------------------------------- tab 3
@st.cache_data(show_spinner="Computing contiguity by distance…")
def _lag_curve(clusters_path, pairs_path, mtime):
    """Same-cluster rate for lines d apart (both in the training pool), divided by chance."""
    cl = pd.read_csv(clusters_path, sep="\t", dtype={"id": str})
    pairs = pd.read_csv(pairs_path, sep="\t", dtype=str, keep_default_na=False, usecols=["id", "source", "orig_line"])
    df = cl.merge(pairs, on="id", how="left")
    df["orig_line"] = df["orig_line"].astype(int)
    cols = [c for c in cl.columns if c != "id"]
    rows = []
    for src, g in df.groupby("source"):
        g = g.drop_duplicates("orig_line").set_index("orig_line")
        for col in cols:
            s = g[col]
            p = s.value_counts(normalize=True)
            chance = float((p ** 2).sum())
            model, k = col.rsplit("_k", 1)
            for lag in LAGS:
                nb = s.reindex(s.index + lag).to_numpy()
                ok = ~pd.isna(nb)
                if ok.sum() == 0:
                    continue
                same = float((s.to_numpy()[ok] == nb[ok]).mean())
                rows.append({"source": src, "model": model, "k": int(k), "distance": lag,
                             "ratio": round(same / chance, 3), "n_pairs": int(ok.sum())})
    return pd.DataFrame(rows)


def _quality(d, project_root):
    m = _json(d / "cluster_metrics.json")
    if not m:
        return _missing("sbatch slurm/cluster.sbatch configs/week02.yaml")

    st.subheader("Silhouette by k")
    sil = pd.DataFrame(m["silhouette"]).reset_index().rename(columns={"index": "k"})
    sil["k"] = sil["k"].astype(int)
    st.altair_chart(alt.Chart(sil.melt("k", var_name="model", value_name="silhouette")).mark_line(point=True)
                    .encode(x="k:Q", y=alt.Y("silhouette:Q", scale=alt.Scale(zero=False)), color="model:N"),
                    width="stretch")
    st.caption("Compare across k within one model only — the two models spread their vectors differently, "
               "so their silhouette values are not comparable with each other.")

    if m.get("agreement_ari"):
        a, b = m["agreement_models"]
        st.subheader(f"Do {a} and {b} find the same clusters? (adjusted Rand index, same k)")
        st.dataframe(pd.DataFrame([m["agreement_ari"]], index=["ARI"]), width="stretch")
        st.caption("1 = identical, 0 = no more agreement than chance.")

    st.subheader("Do clusters follow the topic blocks? Contiguity by distance")
    clusters_path = d / "clusters.tsv"
    pairs_path = project_root / "data" / "processed" / "pairs_split.tsv"
    if not (clusters_path.exists() and pairs_path.exists()):
        return st.info("clusters.tsv or pairs_split.tsv not found.")
    lag = _lag_curve(str(clusters_path), str(pairs_path), clusters_path.stat().st_mtime)

    ks = sorted(lag["k"].unique())
    k = st.selectbox("k", ks, index=ks.index(12) if 12 in ks else 0, key="w2_lag_k")
    sub = lag[lag["k"] == k]
    ymax = float(sub["ratio"].max()) * 1.05
    cols = st.columns(sub["source"].nunique())
    for col, (src, g) in zip(cols, sub.groupby("source")):
        chart = alt.Chart(g).mark_line(point=True).encode(
            x=alt.X("distance:Q", scale=alt.Scale(type="log"), title="distance between lines (log scale)"),
            y=alt.Y("ratio:Q", title="same-cluster rate ÷ chance", scale=alt.Scale(domain=[0, ymax])),
            color=alt.Color("model:N"), tooltip=["model", "distance", "ratio", "n_pairs"])
        rule = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
        col.markdown(f"**{src.capitalize()}**")
        col.altair_chart(chart + rule, width="stretch")
    st.caption("Dashed line = chance. If the ratio stays well above 1 at 25–50 lines, the clusters follow whole "
               "passages (topics). If it falls to ~1 within a few lines, they mostly follow local style "
               "(same speaker, question/answer format, a repeated name).")
    with st.expander("Table"):
        st.dataframe(lag.pivot_table(index=["source", "model", "k"], columns="distance", values="ratio"),
                     width="stretch")


# ---------------------------------------------------------------- tab 4
def _compare(d):
    runs = _runs(d)
    path = d / "clusters.tsv"
    if len(runs) < 2 or not path.exists():
        return _missing("sbatch slurm/cluster.sbatch configs/week02.yaml")
    models = sorted(runs)
    c1, c2, c3, c4 = st.columns(4)
    ma = c1.selectbox("Model A", models, index=models.index("labse") if "labse" in models else 0, key="w2_ma")
    ka = c2.selectbox("k for A", runs[ma], index=runs[ma].index(16) if 16 in runs[ma] else 0, key="w2_ka")
    mb = c3.selectbox("Model B", models, index=models.index("e5") if "e5" in models else 1, key="w2_mb")
    kb = c4.selectbox("k for B", runs[mb], index=runs[mb].index(12) if 12 in runs[mb] else 0, key="w2_kb")

    cl = pd.read_csv(path, sep="\t", dtype={"id": str})
    kw = {}
    for m, k in ((ma, ka), (mb, kb)):
        s = _json(d / f"summary_{m}_k{k}.json")
        kw[(m, k)] = {c["cluster"]: f"{c['cluster']}: " + ", ".join(c["keywords"][:3]) for c in s["clusters"]}

    ct = pd.crosstab(cl[f"{ma}_k{ka}"], cl[f"{mb}_k{kb}"], normalize="index") * 100
    long = ct.reset_index().melt(id_vars=ct.index.name, var_name="b", value_name="pct")
    long["A"] = long[ct.index.name].map(kw[(ma, ka)])
    long["B"] = long["b"].map(kw[(mb, kb)])
    heat = alt.Chart(long).mark_rect().encode(
        x=alt.X("B:N", title=f"{mb} k={kb}", sort=None),
        y=alt.Y("A:N", title=f"{ma} k={ka}", sort=None),
        color=alt.Color("pct:Q", title="% of A's cluster", scale=alt.Scale(scheme="greys")),
        tooltip=["A", "B", alt.Tooltip("pct:Q", format=".1f")])
    st.altair_chart(heat, width="stretch")
    st.caption("Each row is one cluster of model A; the shading shows where its sentences land in model B. "
               "A row with one dark cell = a topic both models find (more trustworthy). A row spread over many "
               "cells = a grouping only one model makes.")

    best = ct.max(axis=1).rename("best match %").round(1).to_frame()
    best["matches B cluster"] = ct.idxmax(axis=1).map(kw[(mb, kb)])
    best.index = best.index.map(kw[(ma, ka)])
    st.dataframe(best.sort_values("best match %", ascending=False), width="stretch")