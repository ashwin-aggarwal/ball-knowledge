import random

import pytest

from ball_knowledge.config import DatasetScope, GameConfig, RankRange, ValueKind
from ball_knowledge.questions import (
    dedupe_best_per_player,
    eligible_players_for_question,
    eligible_pool,
    generate_question,
    resolve_guess_value_and_rank,
    resolve_player_by_name,
)
from conftest import OLD_TIMER_ID, SHORT_CAREER_IDS


def test_eligible_pool_ranks_descending(tables) -> None:
    pool = eligible_pool(tables.career_totals, tables.players, value_col="pts_total", min_games=1)
    assert list(pool["rank"]) == sorted(pool["rank"])
    # Player 1 has the highest pts_total by fixture construction.
    assert pool.iloc[0]["player_id"] == 1
    assert pool.iloc[0]["rank"] == 1


def test_eligible_pool_excludes_null_era_stat(tables) -> None:
    # Player 8 (old-timer) has null stl_total (steals tracked from 1973-74).
    pool = eligible_pool(tables.career_totals, tables.players, value_col="stl_total", min_games=1)
    assert OLD_TIMER_ID not in set(pool["player_id"])
    assert len(pool) == 7


def test_eligible_pool_excludes_below_min_games(tables) -> None:
    pool = eligible_pool(
        tables.career_per_game, tables.players, value_col="pts_per_game", min_games=400
    )
    assert SHORT_CAREER_IDS.isdisjoint(set(pool["player_id"]))
    assert len(pool) == 6


def test_eligible_pool_empty_when_column_missing(tables) -> None:
    pool = eligible_pool(tables.career_totals, tables.players, value_col="nope", min_games=1)
    assert pool.empty


def test_eligible_pool_ties_share_rank_and_skip_next() -> None:
    import pandas as pd

    table = pd.DataFrame(
        {"player_id": [1, 2, 3], "gp": [100, 100, 100], "pts_total": [50.0, 50.0, 40.0]}
    )
    players = pd.DataFrame({"player_id": [1, 2, 3], "full_name": ["A", "B", "C"]})
    pool = eligible_pool(table, players, value_col="pts_total", min_games=1)
    ranks = dict(zip(pool["player_id"], pool["rank"]))
    assert ranks[1] == 1
    assert ranks[2] == 1
    assert ranks[3] == 3  # rank 2 is skipped after the tie


def test_season_record_guess_pool_has_no_duplicate_players(tables, small_game_config) -> None:
    # value_kind=TOTAL: both season rows qualify (season_total_min_games=1),
    # so the raw ranked pool has 2 rows per player...
    raw_pool = eligible_pool(
        tables.season_records, tables.players, value_col="pts_total", min_games=1
    )
    assert len(raw_pool) == 16  # 8 players x 2 qualifying seasons each
    assert raw_pool["player_id"].duplicated().any()

    # ...but the guess pool must dedupe to one entry per player.
    guess_pool = dedupe_best_per_player(raw_pool)
    assert not guess_pool["player_id"].duplicated().any()
    assert len(guess_pool) == 8


def test_season_record_guess_pool_keeps_best_season_value(tables, small_game_config) -> None:
    raw_pool = eligible_pool(
        tables.season_records, tables.players, value_col="pts_total", min_games=1
    )
    guess_pool = dedupe_best_per_player(raw_pool)
    for _, row in guess_pool.iterrows():
        player_rows = raw_pool[raw_pool["player_id"] == row["player_id"]]
        assert row["value"] == player_rows["value"].max()
        assert row["rank"] == player_rows["rank"].min()


def test_eligible_players_for_question_dedupes_for_season_record(tables) -> None:
    config = GameConfig(
        dataset_weights={
            DatasetScope.CAREER_TOTAL: 0.0,
            DatasetScope.CAREER_PER_GAME: 0.0,
            DatasetScope.SEASON_RECORD: 1.0,
        },
        season_record_ranks=RankRange(low=1, high=16, skew=1.0),
        season_total_min_games=1,
        season_per_game_min_games=50,
        season_value_kind_weights={ValueKind.TOTAL: 1.0, ValueKind.PER_GAME: 0.0},
    )
    rng = random.Random(5)
    q = generate_question(tables, config, used_keys=set(), rng=rng)
    guess_pool = eligible_players_for_question(q, tables)
    assert not guess_pool["player_id"].duplicated().any()
    # The answer must still be reachable through the deduped guess pool.
    assert q.answer_player_id in set(guess_pool["player_id"])


def test_generate_question_basic_fields(tables, small_game_config) -> None:
    rng = random.Random(0)
    q = generate_question(tables, small_game_config, used_keys=set(), rng=rng)
    assert q.question_text
    assert q.answer_player_id in set(tables.players["player_id"])
    assert q.target_rank >= 1
    assert q.scope in DatasetScope


def test_generate_question_respects_used_keys(tables, small_game_config) -> None:
    rng = random.Random(1)
    used: set = set()
    questions = []
    for _ in range(20):
        q = generate_question(tables, small_game_config, used_keys=used, rng=rng)
        assert q.key not in used
        used.add(q.key)
        questions.append(q)
    assert len(set(q.key for q in questions)) == len(questions)


def test_generate_question_raises_when_space_exhausted(tables) -> None:
    # A config with a single stat/scope/rank combination possible.
    tiny_config = GameConfig(
        dataset_weights={
            DatasetScope.CAREER_TOTAL: 1.0,
            DatasetScope.CAREER_PER_GAME: 0.0,
            DatasetScope.SEASON_RECORD: 0.0,
        },
        career_total_stats=("pts",),
        career_total_ranks=RankRange(low=1, high=1, skew=1.0),
    )
    rng = random.Random(2)
    used = {(DatasetScope.CAREER_TOTAL.value, ValueKind.TOTAL.value, "pts", 1)}
    with pytest.raises(RuntimeError):
        generate_question(tables, tiny_config, used_keys=used, rng=rng)


def test_era_caveat_present_for_tracked_stat_and_absent_for_untracked(tables, small_game_config) -> None:
    rng = random.Random(0)
    seen_caveat = False
    seen_no_caveat = False
    for _ in range(100):
        q = generate_question(tables, small_game_config, used_keys=set(), rng=rng)
        if q.stat_key in ("stl", "blk", "oreb", "dreb", "tov", "fg3m", "fg3a"):
            assert q.era_caveat is not None
            assert q.era_caveat in q.question_text
            seen_caveat = True
        else:
            assert q.era_caveat is None
            seen_no_caveat = True
        if seen_caveat and seen_no_caveat:
            break
    assert seen_caveat and seen_no_caveat


def test_eligible_players_for_question_matches_answer_pool(tables, small_game_config) -> None:
    rng = random.Random(3)
    q = generate_question(tables, small_game_config, used_keys=set(), rng=rng)
    pool = eligible_players_for_question(q, tables)
    assert q.answer_player_id in set(pool["player_id"])
    match = pool[pool["player_id"] == q.answer_player_id].iloc[0]
    assert match["value"] == q.answer_value
    assert match["rank"] == q.target_rank


def test_season_record_question_carries_answer_season(tables) -> None:
    config = GameConfig(
        dataset_weights={
            DatasetScope.CAREER_TOTAL: 0.0,
            DatasetScope.CAREER_PER_GAME: 0.0,
            DatasetScope.SEASON_RECORD: 1.0,
        },
        season_record_ranks=RankRange(low=1, high=8, skew=1.0),
        season_total_min_games=1,
        season_per_game_min_games=50,
    )
    rng = random.Random(4)
    q = generate_question(tables, config, used_keys=set(), rng=rng)
    assert q.scope is DatasetScope.SEASON_RECORD
    assert q.answer_season in ("2020-21", "2021-22")


def test_resolve_player_by_name_exact_case_insensitive(tables) -> None:
    match = resolve_player_by_name("player 1", tables.players)
    assert match is not None
    assert match["player_id"] == 1

    match_whitespace = resolve_player_by_name("  Player 1  ", tables.players)
    assert match_whitespace is not None
    assert match_whitespace["player_id"] == 1


def test_resolve_player_by_name_no_fuzzy_match(tables) -> None:
    assert resolve_player_by_name("Playr 1", tables.players) is None
    assert resolve_player_by_name("", tables.players) is None
    assert resolve_player_by_name("   ", tables.players) is None
    assert resolve_player_by_name("Someone Nobody Heard Of", tables.players) is None


def test_resolve_guess_value_and_rank_for_eligible_player(tables, small_game_config) -> None:
    rng = random.Random(6)
    q = generate_question(tables, small_game_config, used_keys=set(), rng=rng)
    pool = eligible_players_for_question(q, tables)
    row = pool.iloc[0]
    value, rank = resolve_guess_value_and_rank(int(row["player_id"]), q, tables)
    assert value == row["value"]
    assert rank == row["rank"]


def test_resolve_guess_value_and_rank_for_ineligible_player_is_worse_than_worst(
    tables, small_game_config
) -> None:
    # Career per-game requires 400+ games; players 7 and 8 (gp=200) don't
    # qualify but must still resolve to *some* (worse) rank, not crash.
    from ball_knowledge.config import DatasetScope as DS

    config = small_game_config
    dataset_config = config.dataset_config(DS.CAREER_PER_GAME)
    q = generate_question(
        tables,
        GameConfig(
            dataset_weights={
                DatasetScope.CAREER_TOTAL: 0.0,
                DatasetScope.CAREER_PER_GAME: 1.0,
                DatasetScope.SEASON_RECORD: 0.0,
            },
            career_per_game_stats=("pts",),
            career_per_game_ranks=dataset_config.rank_range,
            career_per_game_min_games=dataset_config.min_games,
        ),
        used_keys=set(),
        rng=random.Random(7),
    )
    pool = eligible_players_for_question(q, tables)
    worst_rank = int(pool["rank"].max())
    value, rank = resolve_guess_value_and_rank(7, q, tables)  # player 7: gp=200, ineligible
    assert value is None
    assert rank == worst_rank + 1


def test_question_text_ordinal_suffixes() -> None:
    from ball_knowledge.questions import _ordinal

    assert _ordinal(1) == "1st"
    assert _ordinal(2) == "2nd"
    assert _ordinal(3) == "3rd"
    assert _ordinal(4) == "4th"
    assert _ordinal(11) == "11th"
    assert _ordinal(12) == "12th"
    assert _ordinal(13) == "13th"
    assert _ordinal(21) == "21st"
    assert _ordinal(102) == "102nd"
