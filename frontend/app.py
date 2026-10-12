"""TekaRx Streamlit app entrypoint.

Run with: ``streamlit run frontend/app.py``

Uses ``st.navigation`` (Streamlit 1.36+) for multi-page routing.
The sidebar renders brand in the upper corner, navigation links,
and patient inputs with the medication regimen list.
"""

from __future__ import annotations

import streamlit as st
from ui.components import sidebar_brand, sidebar_footer, sidebar_regimen
from ui.style import inject_css

# ---- Page config (must be first Streamlit call) ----
st.set_page_config(
    page_title="TekaRx Medication Priority Review",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ---- Handle pending state updates before widgets are instantiated ----
if "pending_age" in st.session_state:
    st.session_state["age"] = st.session_state.pop("pending_age")
if "pending_sex" in st.session_state:
    st.session_state["sex"] = st.session_state.pop("pending_sex")

# ---- Session state defaults ----
if "age" not in st.session_state:
    st.session_state["age"] = 65
if "sex" not in st.session_state:
    st.session_state["sex"] = "Female"
if "meds" not in st.session_state:
    st.session_state["meds"] = []
if "result" not in st.session_state:
    st.session_state["result"] = None
if "mapping" not in st.session_state:
    st.session_state["mapping"] = None

# ---- Navigation definitions ----
p_check = st.Page("pages/check.py", title="1 · New check", default=True)
p_results = st.Page("pages/results.py", title="2 · Results")
p_mapping = st.Page("pages/mapping.py", title="3 · Mapping coverage")
p_about = st.Page("pages/about.py", title="About the model")

pg = st.navigation(
    [p_check, p_results, p_mapping, p_about],
    position="hidden",
)

# ---- Sidebar (Brand at top, navigation, patient block, regimen, footer) ----
with st.sidebar:
    # 1. Brand in upper corner
    sidebar_brand()

    # 2. Navigation links directly below brand
    st.page_link(p_check, label="1 · New check")
    st.page_link(p_results, label="2 · Results")
    st.page_link(p_mapping, label="3 · Mapping coverage")
    st.page_link(p_about, label="About the model")

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # 3. Patient + Regimen unified card
    with st.container(border=True):
        st.markdown(
            '<p class="sidebar-section-label" style="margin-top:0;">PATIENT</p>',
            unsafe_allow_html=True,
        )
        st.number_input(
            "Age (years)",
            min_value=0,
            max_value=120,
            step=1,
            key="age",
        )
        st.selectbox(
            "Biological sex",
            ("Female", "Male", "Unknown"),
            key="sex",
        )

        meds: list[str] = st.session_state.get("meds", [])
        sidebar_regimen(meds)

    # 4. Footer
    sidebar_footer()

# ---- Run the active page ----
pg.run()
