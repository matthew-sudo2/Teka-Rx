"""Page 3: Mapping coverage."""

from __future__ import annotations

import html as _html

import streamlit as st
from ui.components import (
    card,
    empty_state,
    page_header,
    pill,
)
from ui.model import AMBIGUOUS_CUTOFF, MATCHED_CUTOFF
from ui.style import inject_css

inject_css()

mapping = st.session_state.get("mapping")

if not mapping:
    page_header("Data check", "Mapping coverage", "FAERS standard lexicon dictionary resolution.")
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    empty_state(
        "No mapping data yet",
        'Run an analysis from the "1 · New check" page to see mapping results.',
    )
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.page_link("pages/check.py", label="← Go to New check")
    st.stop()

# ---- Header ----
page_header(
    "Data check",
    "Mapping coverage",
    "How each medication you entered was matched to the FAERS standard lexicon.",
)
st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

# ---- Status pills ----
matched_count = sum(1 for m in mapping if m["status"] == "matched")
ambig_count = sum(1 for m in mapping if m["status"] == "ambiguous")
unmatched_count = sum(1 for m in mapping if m["status"] == "unmatched")

pills_html = " ".join([
    pill(f"{matched_count} matched", "green"),
    pill(f"{ambig_count} ambiguous", "amber"),
    pill(f"{unmatched_count} unmatched", "red"),
])
st.markdown(pills_html, unsafe_allow_html=True)
st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

# ---- Table in card ----
with card():
    rows_html = []
    for m in mapping:
        # Status pill
        if m["status"] == "matched":
            status_html = pill("Matched", "green")
            fill_cls = "confidence-fill--green"
        elif m["status"] == "ambiguous":
            status_html = pill("Ambiguous", "amber")
            fill_cls = "confidence-fill--amber"
        else:
            status_html = pill("Unmatched", "red")
            fill_cls = "confidence-fill--red"

        score_pct = int(round(m["score"]))
        in_model = "Included" if m["used"] else "Excluded"

        # Other candidates (exclude primary match)
        suggestions_raw = str(m["suggestions"])
        candidates = [
            s.strip()
            for s in suggestions_raw.split(",")
            if s.strip() and s.strip() != m.get("match", "")
        ]
        other_html = _html.escape(", ".join(candidates)) if candidates else "&mdash;"

        row_str = (
            f"<tr>"
            f"<td>{_html.escape(str(m['input']))}</td>"
            f"<td>{status_html}</td>"
            f"<td>"
            f'<span class="confidence-bar">'
            f'<span class="confidence-track">'
            f'<span class="{fill_cls}" '
            f'style="width:{score_pct}%;display:block;height:100%;border-radius:999px;">'
            f'</span>'
            f"</span>"
            f'<span style="font-weight:700;font-size:0.82rem;">{score_pct}%</span>'
            f"</span>"
            f"</td>"
            f'<td style="font-weight:700;">{in_model}</td>'
            f'<td style="color:#53645C;font-size:0.82rem;">{other_html}</td>'
            f"</tr>"
        )
        rows_html.append(row_str)

    table_inner = "".join(rows_html)
    table_markup = (
        f'<table class="mapping-table">'
        f"<thead>"
        f"<tr>"
        f"<th>Input</th>"
        f"<th>Status</th>"
        f"<th>Confidence</th>"
        f"<th>In model</th>"
        f"<th>Other candidates</th>"
        f"</tr>"
        f"</thead>"
        f"<tbody>{table_inner}</tbody>"
        f"</table>"
        f'<p style="font-size:0.78rem;color:#53645C;margin-top:0.85rem;font-weight:500;">'
        f"Matched at {MATCHED_CUTOFF:.0f}% or higher. "
        f"Ambiguous between {AMBIGUOUS_CUTOFF:.0f}% and {MATCHED_CUTOFF:.0f}%."
        f"</p>"
    )
    st.markdown(table_markup, unsafe_allow_html=True)
