"""Round scoring. Pure functions over plain data, zero Streamlit imports."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuessInput:
    """One guesser's submission for the current question."""

    guesser_name: str
    nba_player_id: int
    nba_player_name: str
    value: float
    rank: int | None = None


@dataclass(frozen=True)
class ScoredGuess:
    """A guess annotated with its distance from the answer and points won."""

    guesser_name: str
    nba_player_id: int
    nba_player_name: str
    value: float
    diff: float
    rank: int | None
    points: int
    is_exact: bool


def score_round(
    guesses: list[GuessInput],
    answer_value: float,
    *,
    round_points: int = 1,
    exact_match_bonus_points: int = 2,
) -> list[ScoredGuess]:
    """Score every guess against `answer_value` and rank them by closeness.

    The closest guess (smallest abs difference) earns `round_points`; if
    that closest guess is exact (diff == 0) it earns
    `exact_match_bonus_points` instead (a total, not an addition on top of
    `round_points`). All guesses tied for closest score identically.
    Non-winning guesses score 0.

    Returns guesses sorted ascending by diff (closest first).
    """
    if not guesses:
        return []

    scored = [
        (g, abs(g.value - answer_value))
        for g in guesses
    ]
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
                diff=diff,
                rank=guess.rank,
                points=points,
                is_exact=is_exact,
            )
        )

    results.sort(key=lambda r: r.diff)
    return results
