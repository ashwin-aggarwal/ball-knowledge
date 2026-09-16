"""Tests for the GeoGuessr-style per-guess scoring in ball_knowledge/scoring.py."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from ball_knowledge.config import STATS
from ball_knowledge.questions import eligible_pool
from ball_knowledge.scoring import (
    CLOSEST_BONUS,
    MAX_ROUND_SCORE,
    MIN_ROUND_SCORE,
    TAU,
    DegenerateLeaderboardError,
    GuessInput,
    local_rank_density,
    normalized_error,
    resolve_density,
    round_score,
    score_round,
)


def _guess(name, pid, value, rank) -> GuessInput:
    return GuessInput(guesser_name=name, nba_player_id=pid, nba_player_name=f"P{pid}", value=value, rank=rank)


def _linear_pool(n: int = 100, step: float = 10.0) -> pd.DataFrame:
    """A synthetic, perfectly linear leaderboard: rank i has value
    (n - i + 1) * step, so the true density is exactly `step` everywhere
    and every test assertion below has an exact, not approximate, target."""
    ranks = list(range(1, n + 1))
    values = [(n - r + 1) * step for r in ranks]
    return pd.DataFrame(
        {
            "player_id": ranks,
            "full_name": [f"P{r}" for r in ranks],
            "value": values,
            "rank": ranks,
            "gp": [82] * n,
        }
    )


# ---------- local_rank_density / resolve_density ----------


def test_local_rank_density_exact_on_linear_pool() -> None:
    pool = _linear_pool(n=100, step=10.0)
    assert local_rank_density(pool, target_rank=50, window=25) == pytest.approx(10.0)


def test_local_rank_density_clamps_at_top_edge() -> None:
    pool = _linear_pool(n=100, step=10.0)
    # lo clamps to 1, hi = 1 + 25 = 26 -- still a well-defined, exact-density window.
    assert local_rank_density(pool, target_rank=1, window=25) == pytest.approx(10.0)


def test_local_rank_density_clamps_at_bottom_edge() -> None:
    pool = _linear_pool(n=100, step=10.0)
    assert local_rank_density(pool, target_rank=100, window=25) == pytest.approx(10.0)


def test_local_rank_density_degenerate_tiny_pool_returns_zero() -> None:
    pool = _linear_pool(n=1, step=10.0)
    assert local_rank_density(pool, target_rank=1, window=25) == 0.0


def test_resolve_density_falls_back_to_global_when_local_window_is_tied() -> None:
    # A tie block wide enough to cover *both edges* of the rank-50,
    # window=25 window (positions 25 and 75) makes the local window
    # degenerate (density exactly 0, since T[lo] == T[hi]), but the table
    # still has real spread outside the block, so the global fallback
    # must kick in rather than raising.
    n = 100
    values = [100.0] * n
    for i in range(24, 75):  # positions 25..75 (1-indexed): the window's exact span
        values[i] = 50.0
    for i in range(0, 24):
        values[i] = 200.0 - i  # spread above the tie block
    for i in range(75, n):
        values[i] = 40.0 - (i - 75)  # spread below the tie block
    pool = pd.DataFrame(
        {
            "player_id": list(range(1, n + 1)),
            "full_name": [f"P{i}" for i in range(1, n + 1)],
            "value": values,
            "rank": list(range(1, n + 1)),
            "gp": [82] * n,
        }
    )
    local = local_rank_density(pool, target_rank=50, window=25)
    assert local == 0.0  # confirm the window really is degenerate first
    density = resolve_density(pool, target_rank=50, window=25)
    assert density > 0


def test_resolve_density_raises_when_every_value_is_tied() -> None:
    n = 50
    pool = pd.DataFrame(
        {
            "player_id": list(range(1, n + 1)),
            "full_name": [f"P{i}" for i in range(1, n + 1)],
            "value": [100.0] * n,
            "rank": [1] * n,
            "gp": [82] * n,
        }
    )
    with pytest.raises(DegenerateLeaderboardError):
        resolve_density(pool, target_rank=1, window=25)


# ---------- round_score / normalized_error ----------


def test_round_score_exact_short_circuits_regardless_of_e() -> None:
    assert round_score(1e9, is_exact=True) == MAX_ROUND_SCORE


def test_round_score_matches_curve_formula() -> None:
    e = 8.0
    expected = round(MAX_ROUND_SCORE * math.exp(-e / TAU))
    assert round_score(e) == expected


def test_round_score_floor_applies_when_eligible() -> None:
    assert round_score(1000.0, apply_floor=True) == MIN_ROUND_SCORE


def test_round_score_no_floor_when_not_eligible() -> None:
    assert round_score(1000.0, apply_floor=False) < MIN_ROUND_SCORE


def test_normalized_error_is_capped() -> None:
    from ball_knowledge.scoring import MAX_NORMALIZED_ERROR

    e = normalized_error(value_guess=1e12, value_answer=0.0, density=1.0)
    assert e == MAX_NORMALIZED_ERROR


# ---------- score_round: core behaviors ----------


def test_empty_guesses_returns_empty() -> None:
    pool = _linear_pool()
    assert score_round([], pool=pool, target_rank=50, answer_value=510.0, answer_player_id=51) == []


def test_score_monotonically_non_increasing_as_error_grows() -> None:
    pool = _linear_pool(n=200, step=5.0)
    answer_rank = 100
    answer_value = float(pool.iloc[answer_rank - 1]["value"])
    guesses = [
        _guess("g0", 1000, answer_value, answer_rank),  # exact value, distinct id -> not is_exact
        _guess("g1", 1001, answer_value - 5, answer_rank + 1),
        _guess("g2", 1002, answer_value - 25, answer_rank + 5),
        _guess("g3", 1003, answer_value - 100, answer_rank + 20),
        _guess("g4", 1004, answer_value - 500, answer_rank + 100 if answer_rank + 100 <= 200 else 199),
    ]
    scored = score_round(
        guesses, pool=pool, target_rank=answer_rank, answer_value=answer_value, answer_player_id=-1
    )
    points = [s.points for s in scored]
    assert points == sorted(points, reverse=True)


@pytest.mark.parametrize("answer_rank", [1, 50, 100, 199, 200])
def test_exact_answer_always_scores_max(answer_rank: int) -> None:
    pool = _linear_pool(n=200, step=5.0)
    answer_value = float(pool.iloc[answer_rank - 1]["value"])
    guesses = [_guess("g", 777, answer_value, answer_rank)]
    scored = score_round(
        guesses, pool=pool, target_rank=answer_rank, answer_value=answer_value, answer_player_id=777
    )
    assert scored[0].points == MAX_ROUND_SCORE
    assert scored[0].is_exact


def test_exact_answer_scores_max_even_with_tiny_density() -> None:
    # A pool where every value is tied except the answer itself: local and
    # global density both degenerate to 0 -- but the exact-match
    # short-circuit must never even attempt the (undefined) curve.
    n = 30
    pool = pd.DataFrame(
        {
            "player_id": list(range(1, n + 1)),
            "full_name": [f"P{i}" for i in range(1, n + 1)],
            "value": [100.0] * n,
            "rank": [1] * n,
            "gp": [82] * n,
        }
    )
    guesses = [_guess("g", 5, 100.0, 1)]
    scored = score_round(guesses, pool=pool, target_rank=1, answer_value=100.0, answer_player_id=5)
    assert scored[0].points == MAX_ROUND_SCORE


def test_identical_guesses_score_identically() -> None:
    pool = _linear_pool(n=100, step=10.0)
    answer_rank = 50
    answer_value = float(pool.iloc[answer_rank - 1]["value"])
    guess_value = float(pool.iloc[59]["value"])  # 10 ranks off
    guesses = [
        _guess("Alice", 60, guess_value, 60),
        _guess("Bob", 60, guess_value, 60),
    ]
    scored = score_round(
        guesses, pool=pool, target_rank=answer_rank, answer_value=answer_value, answer_player_id=51
    )
    assert scored[0].points == scored[1].points
    assert scored[0].normalized_error == scored[1].normalized_error


def test_ties_in_closeness_all_get_the_bonus() -> None:
    pool = _linear_pool(n=100, step=10.0)
    answer_rank = 50
    answer_value = float(pool.iloc[answer_rank - 1]["value"])
    guess_value = float(pool.iloc[54]["value"])
    guesses = [
        _guess("Alice", 55, guess_value, 55),
        _guess("Bob", 999, guess_value, 55),  # a different player, same value -> tied raw_diff
        _guess("Carol", 200, float(pool.iloc[10]["value"]), 11),  # far off, not tied
    ]
    scored = score_round(
        guesses, pool=pool, target_rank=answer_rank, answer_value=answer_value, answer_player_id=-1
    )
    by_name = {s.guesser_name: s for s in scored}
    assert by_name["Alice"].is_closest
    assert by_name["Bob"].is_closest
    assert not by_name["Carol"].is_closest


def test_closest_bonus_never_pushes_score_above_max() -> None:
    pool = _linear_pool(n=100, step=1.0)
    answer_rank = 50
    answer_value = float(pool.iloc[answer_rank - 1]["value"])
    # A guess right at the answer's value but a *different* player id, so
    # it's the closest (raw_diff 0) without being the is_exact short-circuit.
    guesses = [_guess("Alice", 12345, answer_value, answer_rank)]
    scored = score_round(
        guesses, pool=pool, target_rank=answer_rank, answer_value=answer_value, answer_player_id=-1
    )
    assert scored[0].points <= MAX_ROUND_SCORE


def test_ineligible_guess_has_no_value_gets_no_floor_and_never_closest() -> None:
    pool = _linear_pool(n=50, step=10.0)
    answer_rank = 25
    answer_value = float(pool.iloc[answer_rank - 1]["value"])
    guesses = [
        _guess("Real", 26, float(pool.iloc[25]["value"]), 26),
        GuessInput(guesser_name="Ghost", nba_player_id=99999, nba_player_name="Nobody", value=None, rank=999),
    ]
    scored = score_round(
        guesses, pool=pool, target_rank=answer_rank, answer_value=answer_value, answer_player_id=-1
    )
    by_name = {s.guesser_name: s for s in scored}
    assert by_name["Ghost"].raw_diff is None
    assert not by_name["Ghost"].is_closest
    assert by_name["Ghost"].points < MIN_ROUND_SCORE


def test_rank_1_and_rank_n_score_correctly() -> None:
    pool = _linear_pool(n=100, step=10.0)
    for edge_rank in (1, 100):
        answer_value = float(pool.iloc[edge_rank - 1]["value"])
        exact = score_round(
            [_guess("g", 42, answer_value, edge_rank)],
            pool=pool,
            target_rank=edge_rank,
            answer_value=answer_value,
            answer_player_id=42,
        )
        assert exact[0].points == MAX_ROUND_SCORE

        off_by_five_rank = edge_rank + 5 if edge_rank == 1 else edge_rank - 5
        off_value = float(pool.iloc[off_by_five_rank - 1]["value"])
        off = score_round(
            [_guess("g", 43, off_value, off_by_five_rank)],
            pool=pool,
            target_rank=edge_rank,
            answer_value=answer_value,
            answer_player_id=-1,
        )
        assert 0 < off[0].points < MAX_ROUND_SCORE


def test_scores_stable_across_runs_with_same_inputs() -> None:
    pool = _linear_pool(n=150, step=25.0)
    answer_rank = 80
    answer_value = float(pool.iloc[answer_rank - 1]["value"])
    guesses = [_guess("g", 88, float(pool.iloc[85]["value"]), 86)]
    kwargs = dict(pool=pool, target_rank=answer_rank, answer_value=answer_value, answer_player_id=-1)
    first = score_round(guesses, **kwargs)
    second = score_round(guesses, **kwargs)
    assert [s.points for s in first] == [s.points for s in second]
    assert [s.normalized_error for s in first] == [s.normalized_error for s in second]


def test_scoring_unaffected_by_guess_order() -> None:
    pool = _linear_pool(n=100, step=10.0)
    answer_rank = 50
    answer_value = float(pool.iloc[answer_rank - 1]["value"])
    a = _guess("Alice", 51, float(pool.iloc[50]["value"]), 51)
    b = _guess("Bob", 60, float(pool.iloc[59]["value"]), 60)
    kwargs = dict(pool=pool, target_rank=answer_rank, answer_value=answer_value, answer_player_id=-1)
    forward = {s.guesser_name: s.points for s in score_round([a, b], **kwargs)}
    backward = {s.guesser_name: s.points for s in score_round([b, a], **kwargs)}
    assert forward == backward


def test_closest_bonus_configurable_to_zero() -> None:
    pool = _linear_pool(n=50, step=10.0)
    answer_rank = 25
    answer_value = float(pool.iloc[answer_rank - 1]["value"])
    guesses = [_guess("g", 26, float(pool.iloc[25]["value"]), 26)]
    with_bonus = score_round(
        guesses, pool=pool, target_rank=answer_rank, answer_value=answer_value,
        answer_player_id=-1, closest_bonus=CLOSEST_BONUS,
    )
    without_bonus = score_round(
        guesses, pool=pool, target_rank=answer_rank, answer_value=answer_value,
        answer_player_id=-1, closest_bonus=0,
    )
    assert with_bonus[0].points > without_bonus[0].points


# ---------- The normalizer's whole point: real data, every stat ----------


@pytest.fixture(scope="module")
def real_tables():
    from ball_knowledge.data import load_tables

    try:
        return load_tables()
    except FileNotFoundError:
        pytest.skip("data/*.parquet not built; run scripts/build_dataset.py")


@pytest.mark.parametrize("stat_key", list(STATS.keys()))
def test_five_ranks_off_scores_within_a_tight_band_across_every_stat(stat_key, real_tables) -> None:
    """The test that proves the normalizer works: a guess exactly 5
    leaderboard positions away from the answer should land at a
    similar normalized error (and therefore a similar score) whether
    the stat is points (huge career totals) or steals (tiny ones), and
    regardless of how deep into the leaderboard the answer sits.
    """
    stat_def = STATS[stat_key]
    pool = eligible_pool(
        real_tables.career_totals, real_tables.players, value_col=stat_def.totals_col, min_games=1
    )
    if len(pool) < 60:
        pytest.skip(f"{stat_key}: pool too small ({len(pool)} rows) to test a rank-60 depth")

    errors = []
    for answer_rank in (10, 50, 100, min(200, len(pool) - 10)):
        if answer_rank + 5 > len(pool):
            continue
        answer_row = pool.iloc[answer_rank - 1]
        guess_row = pool.iloc[answer_rank + 4]  # 5 positions further down
        if answer_row["value"] == guess_row["value"]:
            continue  # a tie block here would make "5 ranks off" mean 0 value error; skip
        guesses = [
            _guess("g", int(guess_row["player_id"]), float(guess_row["value"]), int(guess_row["rank"]))
        ]
        scored = score_round(
            guesses,
            pool=pool,
            target_rank=answer_rank,
            answer_value=float(answer_row["value"]),
            answer_player_id=int(answer_row["player_id"]) + 10**9,  # guaranteed not a real id match
        )
        errors.append(scored[0].normalized_error)

    assert errors, f"{stat_key}: no usable non-tied rank-60 window found"
    # Band empirically grounded via scripts/calibrate_scoring.py rather
    # than guessed: real 5-position-off guesses across every stat and a
    # spread of rank depths landed within roughly [1.4, 11.1] (tightest
    # for typical stats, widest for high-volume ones -- pts/fgm/fta/min --
    # right at the shallow end of the leaderboard, where superstars are
    # unusually bunched and a width-25 window is a rougher local-linear
    # approximation). The real point this proves: without normalization,
    # the same "5 ranks off" guess would span raw value errors differing
    # by 100-1000x between a stat like points and one like steals: this
    # collapses that to a single-digit-to-low-double-digit band instead.
    for e in errors:
        assert 1.0 <= e <= 12.0, f"{stat_key}: normalized error {e} far from the true 5-rank gap"
