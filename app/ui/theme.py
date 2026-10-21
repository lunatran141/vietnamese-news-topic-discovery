"""Design token, CSS và theme biểu đồ dùng chung."""

from __future__ import annotations

import streamlit as st

INK = "#111827"
MUTED = "#6B7280"
BORDER = "#E5E7EB"
SURFACE = "#F6F7F9"
ACCENT = "#1F3A5F"
OUTLIER_COLOR = "#D4D4D8"
OUTLIER_OPACITY = 0.4

# Bảng màu phân loại pastel
PASTEL_COLORS = [
    "#A8DADC", "#F4ACB7", "#CDB4DB", "#BDE0FE", "#FFC8DD",
    "#B8E0D2", "#D8B4F8", "#FFD6A5", "#AED9C7", "#F5D6A8",
    "#B6C8E3", "#E8C1D7", "#C4D9B4", "#B8C0E0", "#F1C6B7",
]
CATEGORICAL = PASTEL_COLORS

FONT_STACK = (
    "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)


def color_for(index: int) -> str:
    """Màu của cụm thứ `index` trong bảng màu phân loại."""
    return CATEGORICAL[index % len(CATEGORICAL)]


_CSS = f"""
<style>
html, body, [class*="css"] {{ font-family: {FONT_STACK}; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; color: {INK}; }}
h1 {{ font-size: 1.6rem; font-weight: 650; }}
h2 {{ font-size: 1.2rem; font-weight: 600; }}
h3 {{ font-size: 1.02rem; font-weight: 600; }}

[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}

.ui-eyebrow {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: {MUTED}; margin-bottom: 0.2rem; }}
.ui-title {{ font-size: 1.6rem; font-weight: 650; color: {INK}; margin: 0; }}
.ui-subtitle {{ font-size: 0.9rem; color: {MUTED}; margin-top: 0.25rem; }}
.ui-context {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.6rem; }}
.ui-chip {{ border: 1px solid {BORDER}; border-radius: 4px; padding: 2px 8px;
    font-size: 0.75rem; color: {INK}; background: #FFFFFF; }}
.ui-chip b {{ color: {MUTED}; font-weight: 600; margin-right: 4px; }}
.ui-header {{ padding-bottom: 0.9rem; margin-bottom: 1rem; border-bottom: 1px solid {BORDER}; }}

.kpi {{ border: 1px solid {BORDER}; border-radius: 6px; padding: 0.8rem 1rem;
    background: #FFFFFF; height: 100%; }}
.kpi-label {{ font-size: 0.7rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: {MUTED}; }}
.kpi-value {{ font-size: 1.55rem; font-weight: 650; color: {INK}; line-height: 1.25; }}
.kpi-hint {{ font-size: 0.74rem; color: {MUTED}; margin-top: 2px; }}

.pill {{ display: inline-block; border: 1px solid {BORDER}; background: {SURFACE};
    border-radius: 4px; padding: 2px 8px; margin: 0 4px 6px 0; font-size: 0.78rem;
    font-weight: 500; color: {INK}; }}

.note {{ border-left: 3px solid {ACCENT}; background: {SURFACE}; border-radius: 0 4px 4px 0;
    padding: 0.55rem 0.8rem; font-size: 0.82rem; color: #374151; margin: 0.4rem 0 0.8rem; }}
.note.warn {{ border-left-color: #B45309; }}

.kv {{ display: grid; grid-template-columns: 11rem 1fr; row-gap: 0.35rem;
    font-size: 0.86rem; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 0.8rem 1rem; background: #FFFFFF; }}
.kv .k {{ color: {MUTED}; }}
.kv .v {{ color: {INK}; word-break: break-word; }}

.doc-row {{ border: 1px solid {BORDER}; border-radius: 6px; padding: 0.55rem 0.8rem;
    margin-bottom: 0.4rem; background: #FFFFFF; }}
.doc-title {{ font-size: 0.86rem; font-weight: 600; color: {INK}; }}
.doc-meta {{ font-size: 0.74rem; color: {MUTED}; margin-top: 2px; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 1.5rem; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ height: 2.4rem; padding: 0 0.15rem; font-size: 0.88rem;
    font-weight: 500; color: {MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {INK}; font-weight: 600; }}
.stButton button, .stDownloadButton button {{ border-radius: 4px; font-weight: 500;
    font-size: 0.85rem; border: 1px solid {BORDER}; }}
[data-testid="stExpander"] {{ border: 1px solid {BORDER}; border-radius: 6px; }}
</style>
"""


def inject_css() -> None:
    """Áp CSS toàn cục; gọi một lần ở đầu mỗi lượt chạy."""
    st.markdown(_CSS, unsafe_allow_html=True)


def register_altair_theme() -> None:
    """Đồng bộ màu và font của các biểu đồ Altair còn lại với theme chung."""
    import altair as alt

    def _theme() -> dict:
        return {
            "config": {
                "background": "#FFFFFF",
                "range": {"category": CATEGORICAL},
                "axis": {
                    "labelColor": MUTED, "titleColor": INK, "gridColor": "#F0F1F3",
                    "domainColor": BORDER, "labelFont": "sans-serif",
                },
                "legend": {"labelColor": INK, "titleColor": INK},
                "view": {"stroke": "transparent"},
                "bar": {"color": ACCENT},
            }
        }

    if hasattr(alt, "theme"):  # Altair >= 6
        alt.theme.register("news_topics", enable=True)(_theme)
    else:  # Altair 5
        alt.themes.register("news_topics", _theme)
        alt.themes.enable("news_topics")
