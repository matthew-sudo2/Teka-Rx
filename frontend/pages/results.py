"""Page 2: Results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from ui.model import (
    DISCLAIMER,
    _feature_label,
    compute_shap,
    load_artifacts,
    render_shap_chart,
)
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

# Compute local attribution once for this result.  The bars below and the
# detailed explanation card intentionally use the same Tree SHAP values, so
# the UI never presents global Random Forest importances as if they explained
# this patient's prediction.
artifacts = load_artifacts()
bundle = artifacts["bundle"]
frame = result["frame"]
shap_cache = result.get("shap_explanation")
if shap_cache is None:
    with st.spinner("Calculating local feature attribution…"):
        shap_vals, feat_names, feat_vals = compute_shap(frame, bundle)
    # Keep the expensive TreeExplainer result with this evaluation. Page
    # navigation reruns the script, but should not recompute the explanation.
    shap_cache = {
        "values": np.asarray(shap_vals, dtype=float),
        "names": list(feat_names),
        "feature_values": np.asarray(feat_vals, dtype=float),
    }
    result["shap_explanation"] = shap_cache
    st.session_state["result"] = result
else:
    shap_vals = shap_cache["values"]
    feat_names = shap_cache["names"]
    feat_vals = shap_cache["feature_values"]
shap_df = pd.DataFrame(
    {
        "feature_name": feat_names,
        "feature": [_feature_label(f) for f in feat_names],
        "shap": np.asarray(shap_vals, dtype=float),
        "abs": np.abs(np.asarray(shap_vals, dtype=float)),
    }
).sort_values("abs", ascending=False)
top_shap = shap_df.head(5).copy()
total_abs_shap = float(top_shap["abs"].sum())
if total_abs_shap > 0:
    top_shap["importance_pct"] = top_shap["abs"] / total_abs_shap * 100.0
else:
    top_shap["importance_pct"] = 0.0
raw_values = frame.iloc[0].to_dict()
top_shap["display_value"] = [
    result["summary"].get("Matched medications", "N/A")
    if name == "num_drugs"
    else raw_values.get(name, "N/A")
    for name in top_shap["feature_name"]
]

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

    max_pct = float(top_shap["importance_pct"].max()) if len(top_shap) else 1.0
    rows_html = "\n".join(
        bar_row(
            row["feature"],
            f"{row['display_value']:.2f}" if isinstance(row["display_value"], (int, float, np.number)) else str(row["display_value"]),
            float(row["importance_pct"]),
            max_pct,
        )
        for _, row in top_shap.iterrows()
    )
    st.markdown(rows_html, unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ---- SHAP Feature Attribution (TreeExplainer) ----
with card():
    st.markdown(
        '<div style="display:flex;justify-content:space-between;'
        'align-items:center;margin-bottom:0.75rem;flex-wrap:wrap;gap:8px;">'
        '<div><h2 class="trx-card-title" style="margin:0;">SHAP Model Explainability</h2>'
        '<p style="font-size:0.8rem;color:#72857A;margin:2px 0 0 0;">'
        'Marginal contribution of each clinical feature to the Random Forest model '
        '(<code>imrad_models.joblib</code>).'
        '</p></div>'
        '<div style="display:flex;gap:8px;">'
        '<span style="background:rgba(217,119,10,0.15);color:#D9770A;'
        'border:1px solid rgba(217,119,10,0.3);border-radius:4px;padding:2px 8px;'
        'font-size:0.72rem;font-weight:600;">'
        '● Amber (+): Risk Driver</span>'
        '<span style="background:rgba(16,185,129,0.15);color:#10B981;'
        'border:1px solid rgba(16,185,129,0.3);border-radius:4px;padding:2px 8px;'
        'font-size:0.72rem;font-weight:600;">'
        '● Emerald (-): Baseline / Protective</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Calculating SHAP feature attribution…"):
        shap_fig = render_shap_chart(shap_vals, feat_names, feat_vals, max_display=5)
        st.pyplot(shap_fig, use_container_width=True)
        plt.close(shap_fig)

    # Dynamic top driver breakdown
    top_risk = [
        f"{r['feature']} (+{r['shap']:.3f})"
        for _, r in shap_df[shap_df["shap"] > 0].head(2).iterrows()
    ]
    top_prot = [
        f"{r['feature']} ({r['shap']:.3f})"
        for _, r in shap_df[shap_df["shap"] < 0].head(2).iterrows()
    ]

    risk_text = (
        f"Key features elevating priority: <strong>{', '.join(top_risk)}</strong>."
        if top_risk
        else "No single parameter strongly elevated the priority score above baseline."
    )
    prot_text = (
        f"Key protective / moderating factors: <strong>{', '.join(top_prot)}</strong>."
        if top_prot
        else "No parameters served as strong protective anchors."
    )

    box_risk_style = (
        "background:#FFF9F0;border:1px solid rgba(217,119,10,0.28);"
        "border-radius:6px;padding:8px 12px;"
    )
    box_prot_style = (
        "background:#F1FAF5;border:1px solid rgba(16,185,129,0.28);"
        "border-radius:6px;padding:8px 12px;"
    )
    lbl_risk_style = (
        "font-size:0.72rem;font-weight:700;color:#D9770A;"
        "text-transform:uppercase;letter-spacing:0.04em;"
    )
    lbl_prot_style = (
        "font-size:0.72rem;font-weight:700;color:#10B981;"
        "text-transform:uppercase;letter-spacing:0.04em;"
    )
    p_style = "font-size:0.8rem;color:#53645C;margin:4px 0 0 0;line-height:1.4;"

    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:0.75rem;">
          <div style="{box_risk_style}">
            <span style="{lbl_risk_style}">Primary Risk Catalysts</span>
            <p style="{p_style}">{risk_text}</p>
          </div>
          <div style="{box_prot_style}">
            <span style="{lbl_prot_style}">Baseline / Protective Factors</span>
            <p style="{p_style}">{prot_text}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
