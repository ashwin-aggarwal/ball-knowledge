import random

import pytest

from ball_knowledge.config import DatasetScope, GameConfig, RankRange
from ball_knowledge.questions import (
    DataTables,
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
    # A config with a single stat/scope/rank combination possible, and
    # only the straight_rank template enabled -- value_anchor would
    # otherwise still find a fresh (near-random anchor) key even with
    # the same single stat, since anchor-keyed questions almost never
    # collide with a rank-keyed one sharing the same tuple shape.
    tiny_config = GameConfig(
        dataset_weights={
            DatasetScope.CAREER_TOTAL: 1.0,
            DatasetScope.CAREER_PER_GAME: 0.0,
        },
        career_total_stats=("pts",),
        career_total_ranks=RankRange(low=1, high=1, skew=1.0),
        template_weights={"straight_rank": 1.0, "value_anchor": 0.0, "obscure_spotlight": 0.0},
    )
    rng = random.Random(2)
    used = {(DatasetScope.CAREER_TOTAL.value, "pts", 1)}
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
    dataset_config = small_game_config.dataset_config(DatasetScope.CAREER_PER_GAME)
    q = generate_question(
        tables,
        GameConfig(
            dataset_weights={
                DatasetScope.CAREER_TOTAL: 0.0,
                DatasetScope.CAREER_PER_GAME: 1.0,
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


def test_no_stat_repeats_within_cooldown_across_20_rounds(tables, small_game_config) -> None:
    rng = random.Random(11)
    used: set = set()
    recent_stats: list[str] = []
    recent_scopes: list[str] = []
    cooldown = small_game_config.stat_cooldown_rounds
    for round_number in range(1, 21):
        q = generate_question(
            tables,
            small_game_config,
            used_keys=used,
            recent_stats=recent_stats,
            recent_scopes=recent_scopes,
            round_number=round_number,
            total_rounds=20,
            rng=rng,
        )
        assert q.stat_key not in recent_stats[-cooldown:], (
            f"round {round_number}: {q.stat_key} reused within its "
            f"{cooldown}-round cooldown (recent: {recent_stats[-cooldown:]})"
        )
        used.add(q.key)
        recent_stats.append(q.stat_key)
        recent_scopes.append(q.scope.value)


def test_scope_never_appears_more_than_max_consecutive(tables, small_game_config) -> None:
    rng = random.Random(12)
    used: set = set()
    recent_stats: list[str] = []
    recent_scopes: list[str] = []
    max_consecutive = small_game_config.scope_max_consecutive
    for round_number in range(1, 41):
        q = generate_question(
            tables,
            small_game_config,
            used_keys=used,
            recent_stats=recent_stats,
            recent_scopes=recent_scopes,
            round_number=round_number,
            total_rounds=40,
            rng=rng,
        )
        used.add(q.key)
        recent_stats.append(q.stat_key)
        recent_scopes.append(q.scope.value)

    run = 1
    max_run = 1
    for i in range(1, len(recent_scopes)):
        run = run + 1 if recent_scopes[i] == recent_scopes[i - 1] else 1
        max_run = max(max_run, run)
    assert max_run <= max_consecutive


def test_all_templates_produce_well_formed_questions(tables, small_game_config) -> None:
    from ball_knowledge.config import STATS
    from ball_knowledge.questions import (
        TEMPLATE_OBSCURE_SPOTLIGHT,
        TEMPLATE_STRAIGHT_RANK,
        TEMPLATE_VALUE_ANCHOR,
    )

    rng = random.Random(13)
    seen_templates: set = set()
    for _ in range(300):
        q = generate_question(tables, small_game_config, used_keys=set(), rng=rng)
        seen_templates.add(q.template)
        assert q.question_text
        assert q.scope in DatasetScope
        assert q.stat_key in STATS
        assert q.value_col
        assert q.min_games >= 1
        assert q.scoring_mode in ("rank", "value")
        assert q.target_rank >= 1
        assert q.answer_player_id in set(tables.players["player_id"])
        assert q.answer_player_name
        if q.scoring_mode == "value":
            assert q.anchor_value is not None
        else:
            assert q.anchor_value is None
        if len(seen_templates) == 3:
            break
    assert seen_templates == {TEMPLATE_STRAIGHT_RANK, TEMPLATE_VALUE_ANCHOR, TEMPLATE_OBSCURE_SPOTLIGHT}


def test_value_anchor_never_selects_a_per_game_stat(tables, small_game_config) -> None:
    from ball_knowledge.config import STATS
    from ball_knowledge.questions import TEMPLATE_VALUE_ANCHOR

    rng = random.Random(14)
    seen_value_anchor = False
    for _ in range(300):
        q = generate_question(tables, small_game_config, used_keys=set(), rng=rng)
        if q.template == TEMPLATE_VALUE_ANCHOR:
            seen_value_anchor = True
            assert q.scope is DatasetScope.CAREER_TOTAL
            assert q.value_col == STATS[q.stat_key].totals_col
            assert q.value_kind.value == "total"
    assert seen_value_anchor, "value_anchor template never sampled in 300 attempts"


def test_eligibility_filters_hold_across_all_templates(tables, small_game_config) -> None:
    rng = random.Random(15)
    for _ in range(200):
        q = generate_question(tables, small_game_config, used_keys=set(), rng=rng)
        pool = eligible_players_for_question(q, tables)
        # The answer must come from the min-games-filtered pool, never from
        # a player who fails the eligibility floor -- the floor is applied
        # before ranking, not after, for every template.
        assert q.answer_player_id in set(pool["player_id"])
        assert (pool["gp"] >= q.min_games).all()


def test_generate_question_500_iterations_never_throws_never_duplicates() -> None:
    # The 8-player small_game_config fixture's rank-based combinatorial
    # space (~2 scopes x ~16 stats x 8 ranks) genuinely exhausts well
    # before 500 rounds -- that's a real limit of a deliberately tiny
    # fixture, not a bug, and not representative of the real dataset
    # (thousands of rank-based combos, plus effectively unlimited
    # value_anchor ones). Build a wider synthetic fixture, still fully
    # synthetic/deterministic, just large enough that 500 rounds is a
    # realistic stress test rather than a guaranteed exhaustion.
    import pandas as pd

    from ball_knowledge.config import STATS

    n_players = 80
    stat_keys = list(STATS.keys())
    rows_total, rows_pg, rows_players = [], [], []
    for pid in range(1, n_players + 1):
        gp = 1000
        row_t = {"player_id": pid, "gp": gp}
        row_p = {"player_id": pid, "gp": gp}
        for idx, key in enumerate(stat_keys):
            row_t[f"{key}_total"] = float((n_players + 1 - pid) * 100 + idx)
            if STATS[key].per_game_col is not None:
                row_p[f"{key}_per_game"] = round((n_players + 1 - pid) * 1.0 + idx * 0.01, 3)
        rows_total.append(row_t)
        rows_pg.append(row_p)
        rows_players.append({"player_id": pid, "full_name": f"Wide Player {pid}"})

    wide_tables = DataTables(
        career_totals=pd.DataFrame(rows_total),
        career_per_game=pd.DataFrame(rows_pg),
        players=pd.DataFrame(rows_players),
    )
    wide_config = GameConfig(
        career_per_game_min_games=1,
        career_total_ranks=RankRange(low=1, high=n_players, skew=1.0),
        career_per_game_ranks=RankRange(low=1, high=n_players, skew=1.0),
    )

    rng = random.Random(16)
    used: set = set()
    recent_stats: list[str] = []
    recent_scopes: list[str] = []
    keys_seen = []
    for round_number in range(1, 501):
        q = generate_question(
            wide_tables,
            wide_config,
            used_keys=used,
            recent_stats=recent_stats,
            recent_scopes=recent_scopes,
            round_number=round_number,
            total_rounds=500,
            rng=rng,
        )
        assert q.key not in used
        used.add(q.key)
        keys_seen.append(q.key)
        recent_stats.append(q.stat_key)
        recent_scopes.append(q.scope.value)
    assert len(keys_seen) == len(set(keys_seen))


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
