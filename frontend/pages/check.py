"""Page 1: New medication check."""

from __future__ import annotations

import streamlit as st
from ui.components import card, empty_state, page_header
from ui.model import (
    BUILD_COMMANDS,
    HIGH_PRIORITY_AGE,
    HIGH_PRIORITY_MEDS,
    HIGH_PRIORITY_SEX,
    MAINTENANCE_AGE,
    MAINTENANCE_MEDS,
    MAINTENANCE_SEX,
    build_features,
    generate_record,
    load_artifacts,
    match_medications,
    predict,
    threshold,
    top_contributors,
)
from ui.style import inject_css

inject_css()

# ---- Load artifacts ----
try:
    with st.spinner("Loading TekaRx model artifacts…"):
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
    "Step 1 of 2",
    "New medication check",
    "Add the patient’s active medications, then run the priority score.",
)
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ---- Card: Add a medication ----
with card("Add a medication", "Search the TekaRx dictionary"):
    meds: list[str] = st.session_state.get("meds", [])
    selected_drug = st.selectbox(
        "Search medication",
        [None, *artifacts["drug_choices"]],
        format_func=lambda value: "Select a medication…" if value is None else value,
        index=0,
        label_visibility="collapsed",
        key="drug_picker",
    )
    if selected_drug is not None and selected_drug not in meds:
        st.session_state["meds"] = [*meds, selected_drug]
        st.rerun()

    # Sample scenario buttons
    st.markdown(
        '<p style="font-size:0.72rem;font-weight:800;color:#53645C;'
        'text-transform:uppercase;letter-spacing:0.07em;margin:1rem 0 0.5rem 0;">'
        "Or try an example</p>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("High-priority polypharmacy", key="ex_high", use_container_width=True):
            st.session_state["pending_age"] = HIGH_PRIORITY_AGE
            st.session_state["pending_sex"] = HIGH_PRIORITY_SEX
            st.session_state["meds"] = list(HIGH_PRIORITY_MEDS)
            st.rerun()
    with c2:
        if st.button("Maintenance monotherapy", key="ex_low", use_container_width=True):
            st.session_state["pending_age"] = MAINTENANCE_AGE
            st.session_state["pending_sex"] = MAINTENANCE_SEX
            st.session_state["meds"] = list(MAINTENANCE_MEDS)
            st.rerun()
    with c3:
        if st.button("Random record", key="ex_random", use_container_width=True):
            age_g, sex_g, meds_g = generate_record(bundle, dictionary)
            valid_sex = sex_g if sex_g in ("Female", "Male", "Unknown") else "Unknown"
            try:
                age_val = int(age_g) if (age_g == age_g and age_g is not None) else 65
            except Exception:
                age_val = 65
            st.session_state["pending_age"] = max(18, min(100, age_val))
            st.session_state["pending_sex"] = valid_sex
            st.session_state["meds"] = meds_g
            st.rerun()

st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

# ---- Card: Active medications ----
with card("Active medications"):
    if meds:
        # Count matched
        _, match_preview = match_medications(meds, dictionary)
        matched_count = sum(1 for m in match_preview if m["status"] == "matched")

        col_count, col_clear = st.columns([3, 1])
        with col_count:
            st.markdown(
                f'<p style="font-size:0.85rem;color:#53645C;font-weight:600;margin:0 0 0.75rem 0;">'
                f"{matched_count} of {len(meds)} matched to dictionary</p>",
                unsafe_allow_html=True,
            )
        with col_clear:
            if st.button("Clear all", key="clear_all_meds", use_container_width=True):
                st.session_state["meds"] = []
                st.rerun()

        # Render active medications in clean chip buttons
        chip_cols = st.columns(min(len(meds), 3))
        for i, drug in enumerate(meds):
            col_target = chip_cols[i % len(chip_cols)]
            with col_target:
                if st.button(
                    f"✕  {drug}",
                    key=f"rm_med_{i}_{drug}",
                    use_container_width=True,
                    help=f"Remove {drug}",
                ):
                    meds.pop(i)
                    st.session_state["meds"] = meds
                    st.rerun()
    else:
        empty_state(
            "No medications added",
            "Use the search above or load a sample scenario.",
        )

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ---- Run button ----
if st.button(
    "Run priority score →",
    type="primary",
    use_container_width=True,
    key="btn_run",
):
    if not meds:
        st.warning("Add at least one medication before running.")
    else:
        with st.spinner("Analyzing regimen…"):
            patient_age = float(st.session_state.get("age", 65))
            patient_sex = str(st.session_state.get("sex", "Female"))

            matched_df, match_info = match_medications(meds, dictionary)
            features_df, summary_stats = build_features(
                patient_age,
                patient_sex,
                matched_df,
                bundle,
            )
            score_prob = predict(features_df, artifacts)
            contrib = top_contributors(bundle, features_df)

            st.session_state["result"] = {
                "probability": score_prob,
                "threshold": thr,
                "matched": matched_df,
                "match_details": match_info,
                "frame": features_df,
                "summary": summary_stats,
                "contributors": contrib,
                "age": patient_age,
                "sex": patient_sex,
                "parsed_meds": meds,
            }
            st.session_state["mapping"] = match_info

        st.switch_page("pages/results.py")
