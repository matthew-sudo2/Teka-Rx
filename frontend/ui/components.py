"""Reusable HTML component helpers for the TekaRx Streamlit redesign.

Every page calls these helpers instead of constructing raw HTML.
All user-derived strings are escaped with ``html.escape``.
"""

from __future__ import annotations

import html as _html
from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st

# ---------------------------------------------------------------------------
# SVG icons (inline, no emoji)
# ---------------------------------------------------------------------------

_SVG_SHIELD = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
    'stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" '
    'stroke-linejoin="round">'
    '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
    '<path d="m9 12 2 2 4-4"/>'
    '</svg>'
)


def sidebar_brand() -> None:
    """Render the sidebar brand block in the upper corner."""
    st.markdown(
        f"""
        <div class="sidebar-brand">
          <div class="sidebar-brand-icon" aria-hidden="true">
            {_SVG_SHIELD}
          </div>
          <div>
            <p class="sidebar-brand-name">TekaRx</p>
            <p class="sidebar-brand-sub">Medication safety review</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_regimen(meds: list[str]) -> None:
    """Render the regimen medication list in the sidebar."""
    count = len(meds)
    plural = "S" if count != 1 else ""
    items = "".join(
        f'<div class="sidebar-med-item">{_html.escape(m)}</div>'
        for m in meds
    )
    fallback = (
        '<div class="sidebar-med-item" style="color:#53645C;font-style:italic;font-weight:500;">'
        'No medications added</div>'
    )
    st.markdown(
        f"""
        <div class="sidebar-regimen">
          <p class="sidebar-section-label">REGIMEN &middot; {count} MEDICATION{plural}</p>
          {items if items else fallback}
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_footer() -> None:
    """Render the sidebar footer with model info."""
    st.markdown(
        """
        <div class="sidebar-footer">
          <strong>IMRAD serial frozen model</strong><br>
          Research prototype &middot; not medical advice
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(step: str, title: str, subtitle: str = "") -> None:
    """Render a page header with step label, title, and subtitle."""
    step_html = (
        f'<p class="step-label">{_html.escape(step)}</p>' if step else ""
    )
    sub_html = (
        f'<p class="page-subtitle">{_html.escape(subtitle)}</p>' if subtitle else ""
    )
    st.markdown(
        f"""
        <div class="page-header-wrap">
          {step_html}
          <h1 class="page-title">{_html.escape(title)}</h1>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def card(title: str = "", subtitle: str = "") -> Iterator[None]:
    """Render a styled card container wrapping Streamlit widgets or HTML."""
    with st.container(border=True):
        if title:
            heading = f'<h2 class="trx-card-title">{_html.escape(title)}</h2>'
            st.markdown(heading, unsafe_allow_html=True)
        if subtitle:
            sub = f'<p class="trx-card-subtitle">{_html.escape(subtitle)}</p>'
            st.markdown(sub, unsafe_allow_html=True)
        yield


def card_open(title: str = "", subtitle: str = "") -> None:
    """Deprecated legacy helper: open a card block."""
    if title:
        heading = f'<h2 class="trx-card-title">{_html.escape(title)}</h2>'
        st.markdown(heading, unsafe_allow_html=True)
    if subtitle:
        sub = f'<p class="trx-card-subtitle">{_html.escape(subtitle)}</p>'
        st.markdown(sub, unsafe_allow_html=True)


def card_close() -> None:
    """Deprecated legacy helper: close a card block."""
    pass


def stat_card(label: str, value: str) -> None:
    """Render a single stat card."""
    st.markdown(
        f"""
        <div class="stat-card">
          <p class="stat-card-label">{_html.escape(label)}</p>
          <p class="stat-card-value">{_html.escape(str(value))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, tone: str = "green") -> str:
    """Return HTML for a pill badge. tone: green | amber | red | outline."""
    return (
        f'<span class="trx-pill trx-pill--{_html.escape(tone)}">'
        f"{_html.escape(text)}</span>"
    )


def bar_row(label: str, value: str, pct: float, max_pct: float) -> str:
    """Return HTML for a single contributor bar row."""
    width = min(100.0, (pct / max_pct * 100.0) if max_pct > 0 else 0)
    return (
        f'<div class="bar-row">'
        f'  <span class="bar-label">{_html.escape(label)}</span>'
        f'  <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>'
        f'  <div class="bar-stats">'
        f'{_html.escape(str(value))} · <span class="bar-pct">{pct:.1f}%</span></div>'
        f"</div>"
    )


def risk_hero(score: float, threshold: float) -> None:
    """Render the big score hero with threshold bar below."""
    is_high = score >= threshold
    color_cls = "score-big--amber" if is_high else "score-big--green"
    fill_cls = "score-bar-fill--amber" if is_high else "score-bar-fill--green"
    pill_html = pill(
        "Review priority signal" if is_high else "No priority signal",
        "amber" if is_high else "green",
    )
    pct = int(round(score * 100))
    thr_pct = int(round(threshold * 100))

    st.markdown(
        f"""
        <div class="score-hero-card">
          <p class="score-hero-label">OVERALL RISK SIGNAL</p>
          <div class="score-hero-score-row">
            <span class="score-big {color_cls}">{score:.0%}</span>
            {pill_html}
          </div>
          <div class="score-bar-wrap">
            <div class="score-bar-track">
              <div class="score-bar-fill {fill_cls}" style="width:{pct}%"></div>
              <div class="score-bar-marker" style="left:{thr_pct}%"></div>
            </div>
            <div class="score-bar-labels">
              <span>0%</span>
              <span>Threshold {thr_pct}%</span>
              <span>100%</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def med_chips(meds: list[str]) -> None:
    """Render medication pill chips (read-only)."""
    chips = "".join(
        f'<span class="med-chip">{_html.escape(m)}</span>' for m in meds
    )
    st.markdown(f'<div class="med-chips">{chips}</div>', unsafe_allow_html=True)


def empty_state(title: str, description: str) -> None:
    """Render an empty-state placeholder."""
    st.markdown(
        f"""
        <div class="empty-state">
          <h3>{_html.escape(title)}</h3>
          <p>{_html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clinical_disclaimer(text: str) -> None:
    """Render the clinical disclaimer footer."""
    st.markdown(
        f'<p class="clinical-disclaimer">{_html.escape(text)}</p>',
        unsafe_allow_html=True,
    )
