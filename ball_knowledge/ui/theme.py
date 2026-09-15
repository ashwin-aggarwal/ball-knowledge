"""Injects ui/styles.css and exposes palette constants for components.py.

styles.css is empty during the Flow phase by design (raw, ugly Streamlit
widgets verify the state machine first); the card aesthetic is built here
in the Design phase.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

STYLES_PATH = Path(__file__).parent / "styles.css"

# Placeholder palette; replaced with the real card-stock palette in the
# Design phase.
PALETTE: dict[str, str] = {
    "background": "#e8dcc0",
    "ink": "#1a1a1a",
    "accent": "#c8102e",
}


def inject_theme() -> None:
    css = STYLES_PATH.read_text()
    if css.strip():
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
