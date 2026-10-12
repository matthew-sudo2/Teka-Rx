"""Page 4: About the model."""

from __future__ import annotations

import streamlit as st
from ui.components import (
    card,
    clinical_disclaimer,
    page_header,
    stat_card,
)
from ui.model import BUILD_COMMANDS, DISCLAIMER, load_artifacts, threshold
from ui.style import inject_css

inject_css()

# ---- Load artifacts for stats ----
try:
    with st.spinner("Loading model metadata…"):
        artifacts = load_artifacts()
except Exception as exc:
    st.error(
        f"Unable to load frozen TekaRx artifacts. "
        f"Build them via:\n{BUILD_COMMANDS}\n\nError details: {exc}"
    )
    st.stop()

bundle = artifacts["bundle"]
dictionary = artifacts["dictionary"]
thr = threshold(bundle)

# ---- Header ----
page_header(
    "Reference",
    "About the model",
    "Architecture, feature pipeline, and limitations of the TekaRx IMRAD scorer.",
)
st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

# ---- 4 stat cards ----
c1, c2, c3, c4 = st.columns(4)
with c1:
    stat_card("Pipeline", "Random Forest")
with c2:
    stat_card("Feature vector", f"{len(bundle['feature_cols'])} dimensions")
with c3:
    stat_card("Operating threshold", f"{thr:.0%}")
with c4:
    stat_card("Dictionary scope", f"{len(dictionary):,} compounds")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ---- Two text cards ----
c_left, c_right = st.columns(2, gap="medium")

with c_left, card("Feature extraction pipeline"):
    st.markdown(
        """
            <p style="font-size:0.88rem;color:#53645C;line-height:1.55;margin:0;font-weight:500;">
              Active medications are parsed and matched against normalized dictionary
              compounds. Extracted features include ROR statistics, cumulative drug
              counts, high-risk ATC classes (nervous, cardiovascular, blood,
              musculoskeletal), polypharmacy interactions, and therapeutic overlaps.
            </p>
            """,
        unsafe_allow_html=True,
    )

with c_right, card("Limitations and intended use"):
    st.markdown(
        """
            <p style="font-size:0.88rem;color:#53645C;line-height:1.55;margin:0;font-weight:500;">
              The model reflects reporting signals present in spontaneous databases.
              It does not account for dosages, patient renal function, or direct
              causation. Free-text matching is approximate. Results serve strictly
              as a research decision-support aid. Always exercise clinical judgment.
            </p>
            """,
        unsafe_allow_html=True,
    )

# ---- Disclaimer ----
clinical_disclaimer(DISCLAIMER)
