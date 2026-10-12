"""Shared CSS injection for the TekaRx Streamlit redesign.

Call ``inject_css()`` once from ``app.py`` to apply the full design system.
All CSS selectors are kept here so Streamlit upgrades only need one file fix.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
_CARD_BG = "#FFFFFF"
_BORDER = "#DDE5E1"
_RADIUS = "16px"
_MUTED = "#53645C"
_CHIP_BG = "#E1F3EA"
_CHIP_TEXT = "#064D36"
_AMBER = "#D9770A"
_AMBER_TEXT = "#9A4A06"
_RED_TEXT = "#A1281D"
_RED_BG = "#FDECEA"
_BAR_TRACK = "#E6ECE9"
_GREEN = "#0A7A55"
_GREEN_HOVER = "#086345"
_BG = "#F3F6F4"
_INK = "#10231C"

_CSS = f"""
<style>
/* ---- Google Fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

/* ---- Typography reset ---- */
html, body, [class*='st-'], .stApp,
[data-testid="stSidebar"], [data-testid="stSidebar"] * {{
  font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
}}

/* ---- App background ---- */
.stApp {{
  background-color: {_BG} !important;
}}

/* ---- Hide sidebar collapse control (keyboard_double_arrow) ---- */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarHeader"],
header[data-testid="stHeader"] button[aria-label="Expand sidebar"],
header[data-testid="stHeader"] button[aria-label="Collapse sidebar"],
button[kind="headerNoPadding"],
.st-emotion-cache-iiif1w,
[data-testid="stSidebar"] button[aria-label="Collapse sidebar"],
[data-testid="stSidebar"] button[aria-label="Close sidebar"] {{
  display: none !important;
  visibility: hidden !important;
  width: 0 !important;
  height: 0 !important;
  overflow: hidden !important;
  position: absolute !important;
  pointer-events: none !important;
  margin: 0 !important;
  padding: 0 !important;
}}

/* ---- Main container ---- */
.main .block-container {{
  max-width: 1000px !important;
  padding-top: 2rem !important;
  padding-bottom: 3.5rem !important;
}}

/* ===== Sidebar ===== */
[data-testid="stSidebar"] {{
  background: {_CARD_BG} !important;
  border-right: 1px solid {_BORDER} !important;
}}
[data-testid="stSidebar"] > div:first-child {{
  padding-top: 0.75rem !important;
  padding-left: 1.25rem !important;
  padding-right: 1.25rem !important;
}}

/* Sidebar brand in upper corner */
.sidebar-brand {{
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 0 1rem 0;
  margin-bottom: 0.5rem;
}}
.sidebar-brand-icon {{
  width: 44px; height: 44px; border-radius: 12px;
  background: {_GREEN}; display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; color: #ffffff;
}}
.sidebar-brand-icon svg {{ stroke: #ffffff; }}
.sidebar-brand-name {{
  font-size: 1.35rem; font-weight: 800; color: {_INK};
  letter-spacing: -0.025em; line-height: 1.15; margin: 0;
}}
.sidebar-brand-sub {{
  font-size: 0.8rem; color: {_MUTED}; margin: 0;
  font-weight: 500;
}}

/* Sidebar navigation links */
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {{
  border-radius: 12px !important;
  padding: 0.6rem 0.9rem !important;
  margin-bottom: 4px !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
  color: {_INK} !important;
  text-decoration: none !important;
  transition: all 0.15s ease !important;
}}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {{
  background-color: #EBF3EF !important;
  color: {_GREEN} !important;
}}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"],
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:has([aria-current="page"]) {{
  background-color: {_CHIP_BG} !important;
  color: {_CHIP_TEXT} !important;
}}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] span,
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] p {{
  color: {_CHIP_TEXT} !important;
  font-weight: 800 !important;
}}

/* Sidebar card container (Patient + Regimen) */
[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: #F4F7F5 !important;
  border: 1px solid #E3E9E5 !important;
  border-radius: 16px !important;
  padding: 1.25rem 1.1rem !important;
  margin-top: 0.5rem !important;
  margin-bottom: 1.25rem !important;
  box-shadow: none !important;
}}
[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] > div {{
  gap: 0.75rem !important;
}}

/* Input styling inside sidebar card */
[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] [data-baseweb="input"],
[data-testid="stSidebar"]
div[data-testid="stVerticalBlockBorderWrapper"] [data-baseweb="select"] > div {{
  background-color: #FFFFFF !important;
  border: 1px solid {_BORDER} !important;
  border-radius: 10px !important;
}}
[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] label p {{
  font-size: 0.85rem !important;
  font-weight: 700 !important;
  color: {_INK} !important;
}}

.sidebar-section-label {{
  font-size: 0.72rem; font-weight: 800; color: {_MUTED};
  text-transform: uppercase; letter-spacing: 0.07em;
  margin: 0.5rem 0 0.35rem 0;
}}

/* Sidebar regimen list */
.sidebar-regimen {{
  border: none !important;
  background: transparent !important;
  padding: 0 !important;
  margin-top: 0.75rem !important;
}}
.sidebar-med-item {{
  font-size: 0.95rem; color: {_INK}; padding: 3px 0;
  font-weight: 700;
}}

/* Sidebar footer */
.sidebar-footer {{
  margin-top: 1rem; padding-top: 0.75rem;
  font-size: 0.78rem; color: {_MUTED}; line-height: 1.5;
  font-weight: 500;
}}
.sidebar-footer strong {{
  color: {_INK}; font-weight: 700;
}}

/* ===== Page Header ===== */
.page-header-wrap {{
  margin-bottom: 1.5rem;
}}
.step-label {{
  font-size: 0.72rem; font-weight: 800; color: {_MUTED};
  text-transform: uppercase; letter-spacing: 0.07em;
  margin: 0 0 0.35rem 0;
}}
.page-title {{
  font-size: 2.375rem; font-weight: 800; color: {_INK};
  letter-spacing: -0.025em; margin: 0 0 0.35rem 0;
  line-height: 1.15;
}}
.page-subtitle {{
  font-size: 0.95rem; color: {_MUTED}; margin: 0;
  font-weight: 500; line-height: 1.45;
}}

/* ===== Main Cards (Streamlit Border Containers) ===== */
.main div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: {_CARD_BG} !important;
  border: 1px solid {_BORDER} !important;
  border-radius: {_RADIUS} !important;
  padding: 1.5rem !important;
  margin-bottom: 1rem !important;
  box-shadow: 0 1px 3px rgba(16, 35, 28, 0.04) !important;
}}

.trx-card-title {{
  font-size: 1.25rem; font-weight: 800; color: {_INK};
  letter-spacing: -0.02em; margin: 0 0 0.25rem 0;
}}
.trx-card-subtitle {{
  font-size: 0.85rem; color: {_MUTED}; margin: 0 0 0.85rem 0;
  font-weight: 500;
}}

/* ===== Stat cards ===== */
.stat-card {{
  background: {_CARD_BG} !important;
  border: 1px solid {_BORDER} !important;
  border-radius: {_RADIUS} !important;
  padding: 1.15rem 1.5rem !important;
  box-shadow: 0 1px 3px rgba(16, 35, 28, 0.04) !important;
  margin-bottom: 0.75rem !important;
}}
.stat-card:last-child {{
  margin-bottom: 0 !important;
}}
.stat-card-label {{
  font-size: 0.72rem !important;
  font-weight: 800 !important;
  color: {_MUTED} !important;
  text-transform: uppercase !important;
  letter-spacing: 0.07em !important;
  margin: 0 0 0.35rem 0 !important;
}}
.stat-card-value {{
  font-size: 2rem !important;
  font-weight: 800 !important;
  color: {_INK} !important;
  letter-spacing: -0.02em !important;
  margin: 0 !important;
  font-variant-numeric: tabular-nums !important;
  line-height: 1.1 !important;
}}

/* ===== Pills / badges ===== */
.trx-pill {{
  display: inline-flex !important;
  align-items: center !important;
  gap: 6px !important;
  padding: 0.35rem 0.85rem !important;
  border-radius: 999px !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
}}
.trx-pill--green {{
  background: {_CHIP_BG} !important;
  color: {_CHIP_TEXT} !important;
}}
.trx-pill--amber {{
  background: #FEF3E2 !important;
  color: {_AMBER_TEXT} !important;
}}
.trx-pill--red {{
  background: {_RED_BG} !important;
  color: {_RED_TEXT} !important;
}}
.trx-pill--outline {{
  background: transparent !important;
  border: 1px solid {_BORDER} !important;
  color: {_INK} !important;
}}

/* ===== Medication chips ===== */
.med-chips {{
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 0.5rem !important;
  margin: 0.5rem 0 !important;
}}
.med-chip {{
  display: inline-flex !important;
  align-items: center !important;
  gap: 6px !important;
  background: {_CHIP_BG} !important;
  color: {_CHIP_TEXT} !important;
  border-radius: 999px !important;
  padding: 0.45rem 0.9rem !important;
  font-size: 0.85rem !important;
  font-weight: 700 !important;
  border: 1px solid #c2e5d3 !important;
}}

/* ===== Score hero ===== */
.score-hero-card {{
  background: {_CARD_BG} !important;
  border: 1px solid {_BORDER} !important;
  border-radius: {_RADIUS} !important;
  padding: 1.5rem 1.75rem !important;
  box-shadow: 0 1px 3px rgba(16, 35, 28, 0.04) !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: space-between !important;
  box-sizing: border-box !important;
  min-height: 280px !important;
}}
.score-hero-label {{
  font-size: 0.72rem !important;
  font-weight: 800 !important;
  color: {_MUTED} !important;
  text-transform: uppercase !important;
  letter-spacing: 0.07em !important;
  margin: 0 0 0.5rem 0 !important;
}}
.score-hero-score-row {{
  display: flex !important;
  align-items: center !important;
  gap: 1.25rem !important;
  margin: 0.25rem 0 1.25rem 0 !important;
}}
.score-big {{
  font-size: 4.75rem !important;
  font-weight: 800 !important;
  line-height: 1 !important;
  letter-spacing: -0.03em !important;
  margin: 0 !important;
  font-variant-numeric: tabular-nums !important;
  display: inline-block !important;
}}
.score-big--green {{ color: {_GREEN} !important; }}
.score-big--amber {{ color: {_AMBER} !important; }}

.score-bar-wrap {{
  width: 100% !important;
  margin-top: auto !important;
}}
.score-bar-track {{
  height: 10px !important;
  background: {_BAR_TRACK} !important;
  border-radius: 999px !important;
  position: relative !important;
  overflow: visible !important;
  width: 100% !important;
}}
.score-bar-fill {{
  height: 100% !important;
  border-radius: 999px !important;
  transition: width 0.4s ease !important;
}}
.score-bar-fill--green {{ background: {_GREEN} !important; }}
.score-bar-fill--amber {{ background: {_AMBER} !important; }}
.score-bar-marker {{
  position: absolute !important;
  top: -5px !important;
  width: 3px !important;
  height: 20px !important;
  background: {_INK} !important;
  border-radius: 2px !important;
}}
.score-bar-labels {{
  display: flex !important;
  justify-content: space-between !important;
  font-size: 0.75rem !important;
  color: {_MUTED} !important;
  margin-top: 8px !important;
  font-weight: 600 !important;
}}

/* ===== Contributor bars ===== */
.bar-row {{
  display: grid !important;
  grid-template-columns: minmax(180px, 240px) 1fr auto !important;
  align-items: center !important;
  gap: 16px !important;
  padding: 0.75rem 0 !important;
  border-bottom: 1px solid {_BORDER} !important;
}}
.bar-row:last-child {{ border-bottom: none !important; }}
.bar-label {{
  font-size: 0.88rem !important;
  font-weight: 700 !important;
  color: {_INK} !important;
}}
.bar-track {{
  height: 8px !important;
  background: {_BAR_TRACK} !important;
  border-radius: 999px !important;
  overflow: hidden !important;
}}
.bar-fill {{
  height: 100% !important;
  background: {_GREEN} !important;
  border-radius: 999px !important;
  transition: width 0.4s ease !important;
}}
.bar-stats {{
  font-size: 0.85rem !important;
  font-weight: 700 !important;
  color: {_INK} !important;
  min-width: 95px !important;
  text-align: right !important;
  font-variant-numeric: tabular-nums !important;
}}
.bar-stats .bar-pct {{
  color: {_MUTED} !important;
  font-weight: 600 !important;
}}

/* ===== Interpretation / recommendation ===== */
.interp-rec-spacer {{
  height: 1.5rem !important;
}}
.interp-card {{
  background: {_CARD_BG} !important;
  border: 1px solid {_BORDER} !important;
  border-radius: {_RADIUS} !important;
  padding: 1.5rem 1.75rem !important;
  min-height: 160px !important;
  box-shadow: 0 1px 3px rgba(16, 35, 28, 0.04) !important;
}}
.interp-card h3 {{
  font-size: 1.25rem !important;
  font-weight: 800 !important;
  color: {_INK} !important;
  letter-spacing: -0.02em !important;
  margin: 0 0 0.65rem 0 !important;
}}
.interp-card p {{
  font-size: 0.88rem !important;
  color: {_MUTED} !important;
  line-height: 1.6 !important;
  margin: 0 !important;
  font-weight: 500 !important;
}}
.rec-card {{
  background: {_GREEN} !important;
  border: none !important;
  border-radius: {_RADIUS} !important;
  padding: 1.5rem 1.75rem !important;
  min-height: 160px !important;
  box-shadow: 0 2px 4px rgba(10, 122, 85, 0.15) !important;
}}
.rec-card h3 {{
  font-size: 1.25rem !important;
  font-weight: 800 !important;
  color: #FFFFFF !important;
  letter-spacing: -0.02em !important;
  margin: 0 0 0.65rem 0 !important;
}}
.rec-card p {{
  font-size: 0.88rem !important;
  color: rgba(255,255,255,0.95) !important;
  line-height: 1.6 !important;
  margin: 0 !important;
}}

/* ===== Mapping table ===== */
.mapping-table {{
  width: 100%; border-collapse: collapse; font-size: 0.85rem;
}}
.mapping-table th {{
  text-align: left; padding: 0.75rem 0.75rem;
  font-size: 0.72rem; font-weight: 800; color: {_MUTED};
  text-transform: uppercase; letter-spacing: 0.05em;
  border-bottom: 2px solid {_BORDER};
}}
.mapping-table td {{
  padding: 0.75rem; border-bottom: 1px solid {_BORDER};
  color: {_INK}; vertical-align: middle;
}}
.mapping-table td:first-child {{
  font-weight: 800; font-size: 0.88rem;
}}
.confidence-bar {{
  display: inline-flex; align-items: center; gap: 8px;
}}
.confidence-track {{
  width: 60px; height: 6px; background: {_BAR_TRACK};
  border-radius: 999px; overflow: hidden;
}}
.confidence-fill {{
  height: 100%; border-radius: 999px;
}}
.confidence-fill--green {{ background: {_GREEN}; }}
.confidence-fill--amber {{ background: {_AMBER}; }}
.confidence-fill--red {{ background: {_RED_TEXT}; }}

/* ===== Empty state ===== */
.empty-state {{
  text-align: center; padding: 3rem 2rem;
  border: 2px dashed {_BORDER}; border-radius: {_RADIUS};
  background: {_CARD_BG};
}}
.empty-state h3 {{
  font-size: 1.15rem; font-weight: 800; color: {_INK};
  margin: 0 0 0.35rem 0;
}}
.empty-state p {{
  font-size: 0.88rem; color: {_MUTED}; margin: 0;
  font-weight: 500; line-height: 1.5;
}}

/* ===== Disclaimer ===== */
.clinical-disclaimer {{
  font-size: 0.8rem !important;
  color: {_MUTED} !important;
  text-align: center !important;
  margin-top: 3.5rem !important;
  margin-bottom: 2.5rem !important;
  font-weight: 500 !important;
  line-height: 1.5 !important;
}}

/* ===== Button overrides ===== */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
  height: 56px !important;
  border-radius: 12px !important;
  font-weight: 800 !important;
  font-size: 1rem !important;
  background-color: {_GREEN} !important;
  color: #FFFFFF !important;
  border: none !important;
  box-shadow: 0 2px 4px rgba(10, 122, 85, 0.2) !important;
}}
.stButton > button[kind="primary"]:hover {{
  background-color: {_GREEN_HOVER} !important;
}}

/* Ghost / secondary buttons */
.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {{
  background: #FFFFFF !important;
  border: 1px solid {_BORDER} !important;
  color: {_INK} !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  min-height: 44px !important;
  transition: all 0.15s ease !important;
}}
.stButton > button:not([kind="primary"]):hover {{
  border-color: {_GREEN} !important;
  color: {_GREEN} !important;
  background: #FAFBFB !important;
}}

/* Style remove buttons */
button[key^="rm_med_"] {{
  border-radius: 999px !important;
  background: {_CHIP_BG} !important;
  color: {_CHIP_TEXT} !important;
  border: 1px solid #c2e5d3 !important;
  font-weight: 700 !important;
  min-height: 38px !important;
}}
button[key^="rm_med_"]:hover {{
  background: {_RED_BG} !important;
  color: {_RED_TEXT} !important;
  border-color: #f5c2be !important;
}}

/* ===== Mobile ===== */
@media (max-width: 768px) {{
  .main .block-container {{ max-width: 100% !important; padding: 1rem !important; }}
  .score-hero {{ flex-direction: column; align-items: flex-start; gap: 1rem; }}
  .score-big {{ font-size: 4rem; }}
  .bar-row {{ grid-template-columns: 1fr; gap: 6px; }}
  .page-title {{ font-size: 1.75rem; }}
}}
</style>
"""


def inject_css() -> None:
    """Inject the full design system CSS into the current Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)
