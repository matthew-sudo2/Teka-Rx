"""Page 2: Results."""

from __future__ import annotations

import streamlit as st
from ui.components import (
    bar_row,
    card,
    clinical_disclaimer,
    empty_state,
    page_header,
    risk_hero,
    stat_card,
)
from ui.model import DISCLAIMER
from ui.style import inject_css

inject_css()

result = st.session_state.get("result")

if not result:
    page_header("Results", "Results", "Clinical medication safety evaluation.")
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    empty_state(
        "No results yet",
        'Run an analysis from the "1 · New check" page to see evaluation results.',
    )
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.page_link("pages/check.py", label="← Go to New check")
    st.stop()

# ---- Header ----
page_header("Step 2 of 2", "Results")
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

prob = result["probability"]
thr = result["threshold"]
is_high = prob >= thr
summary = result["summary"]

# ---- Hero + 3 stat cards stacked ----
hero_col, stats_col = st.columns([1.6, 1], gap="medium")

with hero_col:
    risk_hero(prob, thr)

with stats_col:
    stat_card("Medications matched", str(summary["Matched medications"]))
    stat_card("Highest ROR", f"{summary['Highest ROR']:.2f}")
    stat_card("Boxed warning", summary["Boxed warning"])

st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

# ---- What is driving this score ----
with card():
    st.markdown(
        '<div style="display:flex;justify-content:space-between;'
        'align-items:center;margin-bottom:0.75rem;">'
        '<h2 class="trx-card-title" style="margin:0;">What is driving this score</h2>'
        '<span style="font-size:0.72rem;font-weight:800;color:#53645C;'
        'text-transform:uppercase;letter-spacing:0.07em;">Relative importance</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    contributors = result["contributors"]
    max_pct = max((c["Contribution"] for c in contributors), default=1.0)

    rows_html = "\n".join(
        bar_row(c["Feature"], c["Value"], c["Contribution"], max_pct)
        for c in contributors
    )
    st.markdown(rows_html, unsafe_allow_html=True)

st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

# ---- Interpretation + Recommended step side-by-side ----
if is_high:
    meaning = (
        "The model flagged this regimen. Reporting odds ratios "
        "are elevated or the drugs show complex multi-drug "
        "interaction patterns."
    )
    action = (
        "Prioritize a formal clinical pharmacotherapy review. "
        "Inspect the highlighted drug pairs, cumulative "
        "toxicities and possible therapeutic duplicates."
    )
else:
    meaning = (
        "No high-priority safety signal detected under the frozen "
        "IMRAD model. Regimen parameters fall within baseline "
        "expected tolerance."
    )
    action = (
        "Maintain routine clinical pharmacotherapy monitoring. "
        "Re-evaluate if dosages, active compounds, or patient "
        "renal/hepatic markers change."
    )

c_interp, c_rec = st.columns(2, gap="medium")
with c_interp:
    st.markdown(
        f"""
        <div class="interp-card">
          <h3>Interpretation</h3>
          <p>{meaning}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c_rec:
    st.markdown(
        f"""
        <div class="rec-card">
          <h3>Recommended step</h3>
          <p>{action}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)

# ---- Disclaimer ----
clinical_disclaimer(DISCLAIMER)
