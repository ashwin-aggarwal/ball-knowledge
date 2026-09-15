"""Headshot fetching with disk cache and an HTML fallback.

This is the one place the running app is allowed to touch the network
(stats.nba.com is never called at runtime; only cdn.nba.com, for images).
A cache miss (common for pre-2000 players) is remembered on disk as a
sentinel file so we don't retry the same 404 on every rerun.
"""
from __future__ import annotations

from pathlib import Path

import requests
import streamlit as st

from ball_knowledge.config import HEADSHOT_CACHE_DIR, HEADSHOT_URL_TEMPLATE

REQUEST_TIMEOUT_SECONDS = 10


def _cache_paths(player_id: int) -> tuple[Path, Path]:
    cache_dir = Path(HEADSHOT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{player_id}.png", cache_dir / f"{player_id}.missing"


def fetch_headshot_bytes(player_id: int) -> bytes | None:
    """Return cached/fetched headshot PNG bytes, or None if unavailable."""
    img_path, miss_path = _cache_paths(player_id)
    if img_path.exists():
        return img_path.read_bytes()
    if miss_path.exists():
        return None

    url = HEADSHOT_URL_TEMPLATE.format(player_id=player_id)
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code == 200 and resp.content:
            img_path.write_bytes(resp.content)
            return resp.content
    except requests.exceptions.RequestException:
        pass
    miss_path.write_text("1")
    return None


@st.cache_data(show_spinner=False)
def get_headshot_bytes(player_id: int) -> bytes | None:
    """Session-cached wrapper around fetch_headshot_bytes."""
    return fetch_headshot_bytes(player_id)


def fallback_headshot_html(player_name: str) -> str:
    """An HTML placeholder card face for a player with no CDN headshot."""
    initials = "".join(part[0] for part in player_name.split()[:2]).upper()
    return f"""
    <div class="bk-headshot-fallback">
        <span class="bk-headshot-fallback-initials">{initials}</span>
    </div>
    """
