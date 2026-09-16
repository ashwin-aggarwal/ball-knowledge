"""Reusable rendering pieces: card, guess strip, scoreboard, handoff gate.

Each render_* function draws pure decoration via st.markdown(unsafe_allow_html).
Native interactive widgets (buttons, selectbox) are placed by app.py itself,
immediately after/around these calls, since HTML injected this way cannot
call back into Python.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import streamlit as st

from ball_knowledge.headshots import fallback_headshot_html, get_headshot_bytes
from ball_knowledge.scoring import ScoredGuess


def _flatten(html: str) -> str:
    """Collapse multi-line/indented HTML to one line.

    st.markdown runs its input through a Markdown parser before applying
    unsafe_allow_html, and indented multi-line HTML can trip that parser's
    indented-code-block rule, leaking a stray closing tag as literal text.
    Rendering everything as a single unbroken line sidesteps it entirely.
    """
    return re.sub(r"\s+", " ", html).strip()


def _markdown(html: str) -> None:
    st.markdown(_flatten(html), unsafe_allow_html=True)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _photo_html(player_id: int, player_name: str, *, css_class: str = "bk-photo") -> str:
    img_bytes = get_headshot_bytes(player_id)
    if img_bytes:
        return (
            f'<div class="{css_class}">'
            f'<img src="data:image/png;base64,{_b64(img_bytes)}" alt="{player_name}" />'
            f"</div>"
        )
    return f'<div class="{css_class}">{fallback_headshot_html(player_name)}</div>'


def render_hero_title() -> None:
    """The lobby wordmark: a vintage sports-logo treatment, not a plain title."""
    _markdown(
        """
        <div class="bk-hero">
          <div class="bk-hero-title">BALL KNOWLEDGE</div>
          <div class="bk-hero-rule"></div>
          <div class="bk-hero-sub">Hot-seat NBA stat trivia</div>
        </div>
        """
    )


def render_card_back(question_text: str, era_caveat: str | None) -> None:
    """Question screen: a card back with the player's name blacked out."""
    caveat_html = f'<div class="bk-caveat">{era_caveat}</div>' if era_caveat else ""
    _markdown(
        f"""
        <div class="bk-stage">
          <div class="bk-card bk-card--back">
            <div class="bk-card-inner">
              <div class="bk-photo bk-photo--mystery">
                <span class="bk-mystery-mark">?</span>
              </div>
              <div class="bk-nameplate">???</div>
              <div class="bk-question">{question_text}</div>
              {caveat_html}
            </div>
          </div>
        </div>
        """
    )


def render_card_front(
    *,
    player_id: int,
    player_name: str,
    value_display: str,
    rank_display: str,
) -> None:
    """Reveal screen: the card front, photo and name and stat revealed."""
    photo_html = _photo_html(player_id, player_name)
    _markdown(
        f"""
        <div class="bk-stage">
          <div class="bk-card bk-card--front">
            <div class="bk-card-inner">
              {photo_html}
              <div class="bk-nameplate">{player_name}</div>
              <div class="bk-stat-callout">{value_display}
                <span class="bk-rank-badge">{rank_display}</span>
              </div>
            </div>
          </div>
        </div>
        """
    )


def render_guess_strip(scored: list[ScoredGuess], *, scoring_mode: str = "rank") -> None:
    """Every guess as a headshot with the guesser's name below, closest first.

    `scoring_mode` matches the question's: "rank" (most templates) labels
    the gap in leaderboard spots; "value" (value_anchor) labels it in the
    stat's own units, since the gap there is a value distance, not a
    position on a leaderboard.
    """
    cards = []
    for s in scored:
        photo_html = _photo_html(s.nba_player_id, s.nba_player_name, css_class="bk-guess-photo")
        variant = " bk-guess-card--exact" if s.is_exact else ""
        if s.is_exact:
            diff_label = "exact match"
        elif scoring_mode == "value":
            diff_label = f"off by {s.diff:,.0f}"
        else:
            spot_word = "spot" if s.diff == 1 else "spots"
            diff_label = f"{s.diff:g} {spot_word} off"
        cards.append(
            f"""
            <div class="bk-guess-card{variant}">
              {photo_html}
              <div class="bk-guess-player">{s.nba_player_name}</div>
              <div class="bk-guess-diff">{diff_label}</div>
              <div class="bk-guess-points">+{s.points}</div>
              <div class="bk-guess-name">{s.guesser_name}</div>
            </div>
            """
        )
    _markdown(f'<div class="bk-guess-strip">{"".join(cards)}</div>')


def render_handoff_gate(name: str) -> None:
    _markdown(
        f"""
        <div class="bk-gate">
          <div class="bk-gate-label">Pass the laptop to</div>
          <div class="bk-gate-name">{name}</div>
        </div>
        """
    )


def render_scoreboard(ranked: list[tuple[str, int]]) -> None:
    rows = "".join(
        f"""
        <div class="bk-scoreboard-row">
          <span class="bk-scoreboard-rank">{i}</span>
          <span class="bk-scoreboard-name">{name}</span>
          <span class="bk-scoreboard-score">{score}</span>
        </div>
        """
        for i, (name, score) in enumerate(ranked, start=1)
    )
    _markdown(f'<div class="bk-scoreboard">{rows}</div>')


def render_crash_card(image_path: str) -> None:
    """The crash screen: a card that got printed wrong."""
    img_bytes = Path(image_path).read_bytes()
    _markdown(
        f"""
        <div class="bk-stage">
          <div class="bk-card bk-card--back bk-crash-card">
            <div class="bk-card-inner">
              <div class="bk-crash-photo">
                <img src="data:image/png;base64,{_b64(img_bytes)}" alt="A card that got printed wrong" />
              </div>
              <div class="bk-nameplate">MISPRINT</div>
            </div>
          </div>
        </div>
        """
    )
