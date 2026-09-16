"""Round scoring. Pure functions over plain data, zero Streamlit imports.

GeoGuessr-style: every guess scores 0-1000 on its own accuracy, independent
of what anyone else guessed. A round where everyone misses badly pays
everyone badly; a round where three people are close pays all three well.

The scale problem: raw |value error| cannot drive the curve directly.
Being 400 off on career points is excellent, being 400 off on career
steals is hopeless. So error is normalized first into "ranks worth of
production" using the local density of the leaderboard around the
answer (see `local_rank_density`), then run through an exponential decay
curve (see `round_score`). That normalized unit means the same thing for
every stat and scope, which is the whole point.

One deliberate consequence, not a bug: because density is local, the
same raw miss scores differently depending on where the answer sits on
the leaderboard. Off by 1,000 points near rank 3 scores worse than off
by 1,000 points near rank 200, since ranks are packed far more tightly
near the top. Guessing accurately in a crowded part of the leaderboard
is genuinely harder and should pay more.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

MAX_ROUND_SCORE = 1000
MIN_ROUND_SCORE = 25  # floor for any guess drawn from the eligible pool
CLOSEST_BONUS = 25  # flat bonus for the round's closest guess(es); 0 disables it
TAU = 12.0

# Rank window radius used by the local density normalizer: lo = R - RADIUS,
# hi = R + RADIUS (clamped to the table's edges), a width-50 window by
# default. Small enough to reflect the leaderboard's local shape, large
# enough to survive a few ties without collapsing to a degenerate window.
DEFAULT_WINDOW_RADIUS = 25

# Upper bound on the normalized error before it's exponentiated. e is
# already bounded in practice (density > 0 is guaranteed by
# `resolve_density`, which raises rather than dividing by zero or a
# negative number), so this is pure defense-in-depth against a
# pathologically small density producing an e large enough to be
# numerically silly -- not something the curve is expected to hit often.
MAX_NORMALIZED_ERROR = 1_000.0


@dataclass(frozen=True)
class GuessInput:
    """One guesser's submission for the current question."""

    guesser_name: str
    nba_player_id: int
    nba_player_name: str
    value: float | None
    rank: int


@dataclass(frozen=True)
class ScoredGuess:
    """A guess annotated with its miss (raw and normalized) and points won."""

    guesser_name: str
    nba_player_id: int
    nba_player_name: str
    value: float | None
    rank: int
    raw_diff: float | None  # |value - answer_value| in the stat's own units; None if ineligible
    normalized_error: float  # raw_diff expressed in "ranks worth of production"
    points: int
    is_exact: bool  # guessed the literal answer player
    is_closest: bool  # tied for the round's smallest raw_diff (bonus winner)


class DegenerateLeaderboardError(ValueError):
    """Raised when a question's leaderboard has no usable value spread at
    all (every eligible player tied at the same value) -- there is no
    error normalizer to build a score off of."""


def local_rank_density(pool: pd.DataFrame, target_rank: int, *, window: int = DEFAULT_WINDOW_RADIUS) -> float:
    """Value change per leaderboard rank in a `window`-radius neighborhood
    of `target_rank`, from `pool` (sorted ascending by rank, as returned
    by questions.eligible_pool -- rank 1 first, ties broken stably).

    Indexes `pool` *positionally*: the first row at a given rank always
    sits at that rank's 1-indexed position in a stably rank-sorted table
    (every row ahead of it has a strictly greater value, by definition of
    "min"-method ranking), so `target_rank` is a safe positional index
    even when ties elsewhere in the table cause rank numbers to be
    skipped. Returns 0.0 for a degenerate (zero-width, e.g. a 1-row pool
    or heavy ties collapsing lo==hi) window -- callers must check for
    this and fall back rather than dividing by it.
    """
    n = len(pool)
    if n < 2:
        return 0.0
    lo = max(1, target_rank - window)
    hi = min(n, target_rank + window)
    if hi <= lo:
        return 0.0
    lo_value = float(pool.iloc[lo - 1]["value"])
    hi_value = float(pool.iloc[hi - 1]["value"])
    return (lo_value - hi_value) / (hi - lo)


def _global_density(pool: pd.DataFrame) -> float:
    """Value change per rank averaged across the whole table: a fallback
    normalizer for when the local window is degenerate."""
    n = len(pool)
    if n < 2:
        return 0.0
    return (float(pool.iloc[0]["value"]) - float(pool.iloc[-1]["value"])) / n


def resolve_density(pool: pd.DataFrame, target_rank: int, *, window: int = DEFAULT_WINDOW_RADIUS) -> float:
    """The density to normalize this question's guesses by.

    Tries the local window first; if that's non-positive (degenerate
    window, heavy ties, a tiny pool), falls back to the whole-table
    average. If even that is non-positive -- every eligible player tied
    at the same value -- there is no meaningful "ranks worth of
    production" unit to normalize by, so this raises rather than
    silently dividing by zero or handing back a nonsense score.
    """
    density = local_rank_density(pool, target_rank, window=window)
    if density <= 0:
        density = _global_density(pool)
    if density <= 0:
        raise DegenerateLeaderboardError(
            f"Cannot score this question: {len(pool)} eligible player(s) with no "
            "positive value spread (local and global rank density both "
            "<= 0) -- the leaderboard has nothing to normalize error against."
        )
    return density


def normalized_error(value_guess: float, value_answer: float, density: float) -> float:
    """|value_guess - value_answer| expressed in "ranks worth of production"
    at this question's density, capped for the exponential curve below."""
    e = abs(value_guess - value_answer) / density
    return min(e, MAX_NORMALIZED_ERROR)


def round_score(
    e: float,
    *,
    is_exact: bool = False,
    apply_floor: bool = True,
    max_round_score: int = MAX_ROUND_SCORE,
    min_round_score: int = MIN_ROUND_SCORE,
    tau: float = TAU,
) -> int:
    """The exponential-decay curve: `round(max_round_score * exp(-e / tau))`.

    An exact match short-circuits to `max_round_score` outright rather
    than trusting the curve to land exactly there -- floating point
    should never be able to cost someone a perfect round. `apply_floor`
    gates the `min_round_score` floor: it's meant for guesses drawn from
    the eligible pool ("showing up should not be worth literally zero"),
    not for a guess that isn't even in the pool at all.
    """
    if is_exact:
        return max_round_score
    raw = max_round_score * math.exp(-e / tau)
    score = round(raw)
    if apply_floor:
        score = max(score, min_round_score)
    return score


def score_round(
    guesses: list[GuessInput],
    *,
    pool: pd.DataFrame,
    target_rank: int,
    answer_value: float,
    answer_player_id: int,
    max_round_score: int = MAX_ROUND_SCORE,
    min_round_score: int = MIN_ROUND_SCORE,
    closest_bonus: int = CLOSEST_BONUS,
    tau: float = TAU,
    window: int = DEFAULT_WINDOW_RADIUS,
) -> list[ScoredGuess]:
    """Score every guess independently on its own accuracy (GeoGuessr-style).

    `pool` is the question's eligible-players table (questions.eligible_pool
    output: columns [player_id, full_name, value, rank, gp], sorted
    ascending by rank) -- used only to derive the density normalizer, once,
    shared by every guess this round (the density reflects the
    leaderboard's shape near the *answer*, not the guess). `answer_value`
    is the value guesses are measured against: the anchor value for a
    value_anchor question, the answer player's own value otherwise.
    `answer_player_id` drives the exact-match short-circuit independent of
    any value/rank arithmetic (a guesser guesses a *player*, so "exact"
    means they named that player, not that some computed diff hit zero).

    A guess with no value (a real player who isn't eligible for this
    specific stat/scope -- no recorded value, or below the games floor)
    scores via the curve at the capped normalized error, but never gets
    the eligible-pool floor and can never be the round's closest guess.

    The closest guess(es) this round -- by raw value distance to the
    answer, ties included -- earn `closest_bonus` on top of their curve
    score, clamped so it can never push a score above `max_round_score`
    (an exact guess is already at the ceiling; the bonus exists to make a
    bad round's best guess visibly better than the rest, not to break the
    ceiling every score is displayed against).

    Returns guesses in the same order as `guesses` (unlike the old
    winner-take-all scorer, there's no single "closest first" ordering
    that matters to how this is displayed -- the caller sorts however it
    wants to render).
    """
    if not guesses:
        return []

    # Density is resolved lazily -- and only once, memoized -- rather than
    # eagerly at the top: an exact-match guess with raw_diff 0 needs no
    # density at all (0 / anything is 0), so a round where every guess is
    # exactly correct must never fail just because the underlying
    # leaderboard happens to be degenerate (e.g. a vanishingly small
    # eligible pool). This is what "short-circuited rather than computed"
    # means in practice, not just skipping the final exp().
    density_box: list[float] = []

    def _density() -> float:
        if not density_box:
            density_box.append(resolve_density(pool, target_rank, window=window))
        return density_box[0]

    prelim = []
    for g in guesses:
        is_exact = g.nba_player_id == answer_player_id
        if g.value is None:
            raw_diff = None
            e = MAX_NORMALIZED_ERROR
            eligible = False
        else:
            raw_diff = abs(g.value - answer_value)
            if raw_diff == 0:
                e = 0.0
            else:
                try:
                    e = normalized_error(g.value, answer_value, _density())
                except DegenerateLeaderboardError:
                    if not is_exact:
                        raise
                    # The guessed player is still exactly correct even
                    # though this question's leaderboard has no usable
                    # spread to measure their miss-from-anchor against
                    # (value_anchor only, and only when the anchor itself
                    # sits off the tied value) -- there's no meaningful
                    # "how far off" number here, but the score doesn't
                    # depend on one.
                    e = 0.0
            eligible = True
        score = round_score(
            e,
            is_exact=is_exact,
            apply_floor=eligible,
            max_round_score=max_round_score,
            min_round_score=min_round_score,
            tau=tau,
        )
        prelim.append((g, raw_diff, e, score, is_exact, eligible))

    finite_diffs = [rd for _, rd, _, _, _, _ in prelim if rd is not None]
    min_diff = min(finite_diffs) if finite_diffs else None

    results = []
    for g, raw_diff, e, score, is_exact, eligible in prelim:
        is_closest = min_diff is not None and raw_diff is not None and raw_diff == min_diff
        points = min(max_round_score, score + closest_bonus) if is_closest else score
        results.append(
            ScoredGuess(
                guesser_name=g.guesser_name,
                nba_player_id=g.nba_player_id,
                nba_player_name=g.nba_player_name,
                value=g.value,
                rank=g.rank,
                raw_diff=raw_diff,
                normalized_error=e,
                points=points,
                is_exact=is_exact,
                is_closest=is_closest,
            )
        )
    return results
