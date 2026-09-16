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


def render_blank_note(blank_players: list[str]) -> None:
    """Names which rows on the collect screen are still empty, since the
    submit button is disabled with no other explanation for why."""
    _markdown(f'<div class="bk-lineup-blank-note">Still waiting on: {", ".join(blank_players)}</div>')


def render_guesses_label() -> None:
    """Centered, bold, underlined "Guesses:" label under the reveal card."""
    _markdown('<div class="bk-guesses-label">Guesses:</div>')


def render_guess_strip(
    scored: list[ScoredGuess],
    *,
    stat_label: str,
    value_display_fmt: str = ",.0f",
) -> None:
    """Every guess as a card: points, raw miss, normalized miss, own rank.

    Reads the way GeoGuessr shows a round result -- for each guess: the
    points it earned (prominent), the raw value miss in the stat's own
    units ("off by 1,840 rebounds"), the same miss normalized into
    leaderboard terms ("about 9 spots off" -- `s.normalized_error`,
    already in that unit for every template, rank-based or value-based,
    thanks to the local-density normalizer in scoring.py), and the
    guessed player's own rank/value so a guess is legible on its own
    terms even when it didn't win the round. The round's closest guess(es)
    (`s.is_closest`, ties included) get a subtle green card background;
    an exact correct-player guess (`s.is_exact`) gets a red-bordered card
    and skips the miss lines entirely, since there's nothing to report.
    """
    cards = []
    for s in scored:
        photo_html = _photo_html(s.nba_player_id, s.nba_player_name, css_class="bk-guess-photo")
        variant = ""
        if s.is_closest:
            variant += " bk-guess-card--closest"
        if s.is_exact:
            variant += " bk-guess-card--exact"

        if s.value is not None:
            stat_line = f"#{s.rank} · {format(s.value, value_display_fmt)}"
        else:
            stat_line = "not on this leaderboard"

        if s.is_exact:
            miss_html = '<div class="bk-guess-diff">exact match</div>'
        elif s.raw_diff is not None:
            raw_miss = f"off by {format(s.raw_diff, value_display_fmt)} {stat_label.lower()}"
            spots = round(s.normalized_error)
            # Deliberately not "N spots off" -- s.normalized_error is a
            # value gap measured against the *local* density of this
            # leaderboard neighborhood, not a literal rank count, and
            # those two diverge exactly where it matters most: near the
            # top of a leaderboard, where consecutive superstars are far
            # apart in value, a small literal rank gap can be "worth"
            # many typical ranks' worth of production. Wording it as a
            # literal spot count reads as simply wrong when a player can
            # see the actual (small) rank gap in the sidebar.
            rank_word = "rank" if spots == 1 else "ranks"
            norm_miss = f"worth about {spots:g} {rank_word} here"
            miss_html = (
                f'<div class="bk-guess-raw-miss">{raw_miss}</div>'
                f'<div class="bk-guess-diff">{norm_miss}</div>'
            )
        else:
            miss_html = ""

        cards.append(
            f"""
            <div class="bk-guess-card{variant}">
              {photo_html}
              <div class="bk-guess-player">{s.nba_player_name}</div>
              <div class="bk-guess-stat-line">{stat_line}</div>
              {miss_html}
              <div class="bk-guess-points">+{s.points}</div>
              <div class="bk-guess-name">{s.guesser_name}</div>
            </div>
            """
        )
    _markdown(f'<div class="bk-guess-strip">{"".join(cards)}</div>')


def render_scoreboard(ranked: list[tuple[str, int]]) -> None:
    """Each player's running total, as a printed score column."""
    rows = "".join(
        f"""
        <div class="bk-scoreboard-row">
          <span class="bk-scoreboard-rank">{i}</span>
          <span class="bk-scoreboard-name">{name}</span>
          <span class="bk-scoreboard-score">{score:,}</span>
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
