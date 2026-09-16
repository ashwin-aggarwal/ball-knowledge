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


def render_leaderboard_neighbors(
    above,
    below,
    *,
    answer_rank: int,
    answer_name: str,
    answer_value: float,
    value_display_fmt: str = ",.0f",
) -> None:
    """The leaderboard context around the answer: up to N ranks immediately
    above and below it, closest first, meant for the side of the screen.
    `above`/`below` are DataFrames with columns [player_id, full_name,
    value, rank] as returned by questions.leaderboard_neighbors -- either
    can have fewer rows than requested near either end of the leaderboard.
    """

    def _row(rank: object, name: str, value: float, css: str = "") -> str:
        return (
            f'<div class="bk-neighbor-row {css}">'
            f'<span class="bk-neighbor-rank">#{rank}</span>'
            f'<span class="bk-neighbor-name">{name}</span>'
            f'<span class="bk-neighbor-value">{format(value, value_display_fmt)}</span>'
            f"</div>"
        )

    # `above` arrives closest-first (rank descending toward the answer),
    # which is the right contract for questions.leaderboard_neighbors --
    # but wrong for display: the panel should read as one continuous
    # leaderboard slice, rank ascending top to bottom throughout. Reverse
    # just the render order here so #1 sits at the top of the "above"
    # block and the answer's rank sits directly beneath it.
    above_html = "".join(
        _row(r["rank"], r["full_name"], r["value"]) for _, r in above.iloc[::-1].iterrows()
    )
    below_html = "".join(
        _row(r["rank"], r["full_name"], r["value"]) for _, r in below.iterrows()
    )
    answer_html = _row(answer_rank, answer_name, answer_value, css="bk-neighbor-row--answer")

    _markdown(
        f"""
        <div class="bk-neighbors">
          <div class="bk-neighbors-title">On the leaderboard</div>
          <div class="bk-neighbors-section">{above_html}</div>
          {answer_html}
          <div class="bk-neighbors-section">{below_html}</div>
        </div>
        """
    )


def render_guesses_label() -> None:
    """Centered, bold, underlined "Guesses:" label under the reveal card."""
    _markdown('<div class="bk-guesses-label">Guesses:</div>')


def render_guess_strip(
    scored: list[ScoredGuess],
    *,
    scoring_mode: str = "rank",
    value_display_fmt: str = ",.0f",
    target_rank: int | None = None,
) -> None:
    """Every guess as a headshot with the guesser's name below, closest first.

    The displayed gap is always in leaderboard spots (|guess.rank -
    target_rank|), regardless of `scoring_mode` -- even for value_anchor,
    where the round is actually *won* by value distance to the anchor
    (that's what `s.diff`/`s.points` reflect), the label reads in rank
    terms for consistency with every other question type. `target_rank`
    is required to compute that for value-mode questions; rank-mode
    questions already carry it via `s.diff` directly. Every card also
    shows the guessed player's own rank and stat total, regardless of
    scoring mode, so a guess is legible on its own terms even when it
    didn't win the round. `value_display_fmt` is a format-spec (e.g.
    ",.0f" for totals) applied to that value. The closest guess(es) --
    the round's actual winner(s) by the real scoring rule, ties included
    -- get a subtle green card background.
    """
    min_diff = min((s.diff for s in scored), default=None)
    cards = []
    for s in scored:
        photo_html = _photo_html(s.nba_player_id, s.nba_player_name, css_class="bk-guess-photo")
        rank_gap = abs(s.rank - target_rank) if (scoring_mode == "value" and target_rank is not None) else s.diff
        variant = ""
        if min_diff is not None and s.diff == min_diff:
            variant += " bk-guess-card--closest"
        if rank_gap == 0:
            variant += " bk-guess-card--exact"
        if rank_gap == 0:
            diff_label = "exact match"
        else:
            spot_word = "spot" if rank_gap == 1 else "spots"
            diff_label = f"{rank_gap:g} {spot_word} off"
        if s.value is not None:
            stat_line = f"#{s.rank} · {format(s.value, value_display_fmt)}"
        else:
            stat_line = "not on this leaderboard"
        cards.append(
            f"""
            <div class="bk-guess-card{variant}">
              {photo_html}
              <div class="bk-guess-player">{s.nba_player_name}</div>
              <div class="bk-guess-stat-line">{stat_line}</div>
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
