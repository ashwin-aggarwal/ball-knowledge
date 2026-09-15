"""Injects ui/styles.css and exposes palette constants for components.py.

The concept: a 1970s printed trading card. A question is a card back with
the player's name blacked out; the reveal flips to the card front. See
styles.css for the full token system and rationale.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

STYLES_PATH = Path(__file__).parent / "styles.css"

# Mirrors the CSS custom properties in styles.css, for the rare case
# Python needs a color value directly (inline styles, dynamic accents)
# rather than a CSS class.
PALETTE: dict[str, str] = {
    "page_bg": "#17324c",
    "card_stock": "#ece0bb",
    "card_stock_back": "#e2d5a8",
    "ink": "#211d15",
    "ink_red": "#b23a2c",
    "ink_gold": "#c99a35",
    "ink_teal": "#2e6a5c",
}


def inject_theme() -> None:
    css = STYLES_PATH.read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
