"""Round scoring. Pure functions over plain data, zero Streamlit imports.

Scoring is by rank distance: how many leaderboard spots a guessed player
sits from the rank the question actually asked about. Guessing the 52nd
all-time player when the question asked about 50th is "off by 2"; another
guesser naming the 45th all-time player is "off by 5" and loses the round
even though their player's raw stat value might sit closer to the answer's
value than the first guess's does. Rank position, not stat value, is what
players intuitively compare a guess against.
"""
from __future__ import annotations

from dataclasses import dataclass


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
    """A guess annotated with its rank distance from the answer and points won."""

    guesser_name: str
    nba_player_id: int
    nba_player_name: str
    value: float | None
    rank: int
    diff: int
    points: int
    is_exact: bool


def score_round(
    guesses: list[GuessInput],
    answer_rank: int,
    *,
    round_points: int = 1,
    exact_match_bonus_points: int = 2,
) -> list[ScoredGuess]:
    """Score every guess by |guess.rank - answer_rank| and rank by closeness.

    The closest guess (smallest rank distance) earns `round_points`; if
    that closest guess is exact (its player IS the answer, diff == 0) it
    earns `exact_match_bonus_points` instead (a total, not an addition on
    top of `round_points`). All guesses tied for closest score identically.
    Non-winning guesses score 0.

    Returns guesses sorted ascending by diff (closest first).
    """
    if not guesses:
        return []

    scored = [(g, abs(g.rank - answer_rank)) for g in guesses]
    min_diff = min(diff for _, diff in scored)

    results = []
    for guess, diff in scored:
        is_winner = diff == min_diff
        is_exact = is_winner and diff == 0
        if is_exact:
            points = exact_match_bonus_points
        elif is_winner:
            points = round_points
        else:
            points = 0
        results.append(
            ScoredGuess(
                guesser_name=guess.guesser_name,
                nba_player_id=guess.nba_player_id,
                nba_player_name=guess.nba_player_name,
                value=guess.value,
                rank=guess.rank,
                diff=diff,
                points=points,
                is_exact=is_exact,
            )
        )

    results.sort(key=lambda r: r.diff)
    return results
