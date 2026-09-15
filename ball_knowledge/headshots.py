"""Headshot fetching with disk cache and an HTML fallback.

This is the one place the running app is allowed to touch the network
(stats.nba.com is never called at runtime; only cdn.nba.com, for images).
A cache miss (common for pre-2000 players) is remembered on disk as a
sentinel file so we don't retry the same 404 on every rerun.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import requests
import streamlit as st

from ball_knowledge.config import HEADSHOT_CACHE_DIR, HEADSHOT_URL_TEMPLATE

REQUEST_TIMEOUT_SECONDS = 10

# The CDN doesn't 404 for a player with no real photo -- it serves this
# exact generic gray-silhouette placeholder with a 200 status instead.
# Without this check that placeholder would be cached and shown as if it
# were a real headshot. Confirmed by hash across multiple pre-2000 players.
_GENERIC_PLACEHOLDER_MD5 = "e7f284977a4931dedd1cb6ba4c32283e"


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
        is_placeholder = (
            resp.status_code == 200
            and hashlib.md5(resp.content).hexdigest() == _GENERIC_PLACEHOLDER_MD5
        )
        if resp.status_code == 200 and resp.content and not is_placeholder:
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
