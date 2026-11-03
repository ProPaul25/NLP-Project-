"""Project NLP — results dashboard.

Runs as a Slurm job (see slurm/dashboard.sbatch), never on the login node.
Two sections in the sidebar:
  Pilot   — dashboard/pilot/<name>.py : the method tested on a language pair that already has topic labels
  Weeks   — dashboard/weeks/week_NN.py: the Kangri project, week by week
Every page is a module with a render(results_root) function. To add one, create the file and add one line
to PILOT or WEEKS below.
"""
import importlib
import sys
from pathlib import Path

import streamlit as st

# Project_NLP (for LaBSE, E5, Week 1, Week 2) and Project_NLP_2 (for LASER3-CO)
ROOT_2 = Path(__file__).resolve().parents[1]
DEFAULT_NLP = Path("/raid/home/loitongbam/Project_NLP")
ROOT_1 = DEFAULT_NLP if DEFAULT_NLP.exists() else (ROOT_2.parent / "Project_NLP" if (ROOT_2.parent / "Project_NLP").exists() else ROOT_2)

sys.path.insert(0, str(Path(__file__).resolve().parent))

PILOT = {
    "Bengali–Hindi": "pilot.bengali_hindi",
    "LASER3-CO": "pilot.laser3_co",
}
WEEKS = {
    "Week 1 — Data cleaning": "weeks.week_01",
    "Week 2 — Development set & topic clusters": "weeks.week_02",
}
PAGES = {**PILOT, **WEEKS}

st.set_page_config(page_title="Project NLP — Kangri", layout="wide")
st.sidebar.title("Project NLP")
st.sidebar.caption("Topic-based sentence representation · Kangri")

# Two radio groups act as one menu: picking in one group clears the other.
if "nav_pilot" not in st.session_state:
    st.session_state["nav_pilot"] = next(iter(PILOT))
    st.session_state["nav_weeks"] = None


def _picked(this, other):
    if st.session_state[this] is not None:
        st.session_state[other] = None


st.sidebar.subheader("Pilot")
st.sidebar.radio("Pilot", list(PILOT), key="nav_pilot", label_visibility="collapsed",
                 on_change=_picked, args=("nav_pilot", "nav_weeks"))
st.sidebar.subheader("Weekly evaluation · Kangri")
st.sidebar.radio("Weekly evaluation", list(WEEKS), key="nav_weeks", label_visibility="collapsed",
                 on_change=_picked, args=("nav_weeks", "nav_pilot"))

choice = st.session_state["nav_pilot"] or st.session_state["nav_weeks"] or next(iter(PILOT))
module = importlib.import_module(PAGES[choice])
if choice in WEEKS:
    module.render(ROOT_1 / "outputs" / "results")
elif choice == "LASER3-CO":
    module.render(ROOT_2 / "outputs" / "results")
else:
    # "Bengali–Hindi": reads from Project_NLP for benchmarks and Project_NLP_2 for LASER3
    module.render(ROOT_1 / "outputs" / "results")