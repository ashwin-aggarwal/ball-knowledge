"""Question generation over the three stat tables.

Pure and Streamlit-free by design (see file structure notes in the project
brief): this module never imports streamlit, and never imports data.py
(which does, for its @st.cache_data wrappers). data.py is expected to load
the parquet tables and hand the resulting DataFrames to this module.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from ball_knowledge.config import (
    STATS,
    DatasetScope,
    GameConfig,
    RankRange,
    StatDef,
    ValueKind,
)

# Question uniqueness key: (scope, value_kind, stat_key, target_rank).
QuestionKey = tuple[str, str, str, int]

# Max attempts to find a fresh, satisfiable (scope, value_kind, stat, rank)
# combination before giving up. Generous: the combinatorial space is large
# (3 scopes x 2 value kinds x 16 stats x up to ~250 ranks) relative to any
# realistic game length, so exhaustion only happens with tiny test fixtures
# that have deliberately starved the space.
MAX_GENERATION_ATTEMPTS = 500


@dataclass(frozen=True)
class DataTables:
    """Bundles the three stat tables plus players, keyed by player_id."""

    career_totals: pd.DataFrame
    career_per_game: pd.DataFrame
    season_records: pd.DataFrame
    players: pd.DataFrame

    def stat_table(self, scope: DatasetScope) -> pd.DataFrame:
        if scope is DatasetScope.CAREER_TOTAL:
            return self.career_totals
        if scope is DatasetScope.CAREER_PER_GAME:
            return self.career_per_game
        return self.season_records


@dataclass(frozen=True)
class Question:
    key: QuestionKey
    question_text: str
    scope: DatasetScope
    value_kind: ValueKind
    stat_key: str
    stat_label: str
    value_col: str
    min_games: int
    target_rank: int
    answer_player_id: int
    answer_player_name: str
    answer_value: float
    answer_season: str | None
    era_caveat: str | None


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _question_text(
    scope: DatasetScope,
    value_kind: ValueKind,
    stat_label: str,
    target_rank: int,
    era_caveat: str | None,
) -> str:
    rank_str = _ordinal(target_rank)
    label_lower = stat_label.lower()
    if scope is DatasetScope.CAREER_TOTAL:
        text = f"Who ranks {rank_str} all time in career {label_lower}?"
    elif scope is DatasetScope.CAREER_PER_GAME:
        text = f"Who ranks {rank_str} all time in career {label_lower} per game?"
    elif value_kind is ValueKind.TOTAL:
        text = f"Who ranks {rank_str} all time for {label_lower} in a single season?"
    else:
        text = f"Who ranks {rank_str} all time for {label_lower} per game in a single season?"
    if era_caveat:
        text = f"{text} {era_caveat}"
    return text


def eligible_pool(
    table: pd.DataFrame,
    players: pd.DataFrame,
    *,
    value_col: str,
    min_games: int,
) -> pd.DataFrame:
    """Rows of `table` eligible for a given stat/scope, ranked descending.

    Returns columns [player_id, full_name, value, rank, gp] (plus `season`
    when present in `table`), sorted by rank ascending (best first). `rank`
    uses "min" ties (shared rank, next rank skips accordingly), matching
    how a real leaderboard displays ties.
    """
    if value_col not in table.columns:
        return pd.DataFrame(columns=["player_id", "full_name", "value", "rank", "gp"])

    eligible = table[(table["gp"] >= min_games) & table[value_col].notna()].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["player_id", "full_name", "value", "rank", "gp"])

    eligible["rank"] = eligible[value_col].rank(method="min", ascending=False).astype(int)
    eligible = eligible.rename(columns={value_col: "value"})
    eligible = eligible.merge(players[["player_id", "full_name"]], on="player_id", how="left")
    eligible = eligible.sort_values("rank")
    cols = ["player_id", "full_name", "value", "rank", "gp"]
    if "season" in eligible.columns:
        cols.append("season")
    return eligible[cols].reset_index(drop=True)


def _sample_rank(rng: random.Random, available_ranks: list[int], rank_range: RankRange) -> int:
    """Pick a rank from `available_ranks`, skewed toward the low end."""
    in_range = [r for r in available_ranks if rank_range.low <= r <= rank_range.high]
    if not in_range:
        return available_ranks[0]
    in_range.sort()
    u = rng.random() ** rank_range.skew
    idx = min(int(u * len(in_range)), len(in_range) - 1)
    return in_range[idx]


def _pick_scope_and_value_kind(
    rng: random.Random, config: GameConfig
) -> tuple[DatasetScope, ValueKind]:
    scopes = list(config.dataset_weights.keys())
    weights = [config.dataset_weights[s] for s in scopes]
    scope = rng.choices(scopes, weights=weights, k=1)[0]
    if scope is DatasetScope.CAREER_TOTAL:
        return scope, ValueKind.TOTAL
    if scope is DatasetScope.CAREER_PER_GAME:
        return scope, ValueKind.PER_GAME
    kinds = list(config.season_value_kind_weights.keys())
    kind_weights = [config.season_value_kind_weights[k] for k in kinds]
    value_kind = rng.choices(kinds, weights=kind_weights, k=1)[0]
    return scope, value_kind


def generate_question(
    tables: DataTables,
    config: GameConfig,
    used_keys: set[QuestionKey],
    rng: random.Random | None = None,
) -> Question:
    """Generate one question not already present in `used_keys`.

    Raises RuntimeError if no satisfiable, unused combination is found
    within MAX_GENERATION_ATTEMPTS attempts (only realistic with a tiny or
    exhausted question space, e.g. small test fixtures or a very long
    game against a pared-down stat allowlist).
    """
    rng = rng or random.Random()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        scope, value_kind = _pick_scope_and_value_kind(rng, config)
        dataset_config = config.dataset_config(scope, value_kind)
        stat_key = rng.choice(dataset_config.stat_allowlist)
        stat_def: StatDef = STATS[stat_key]

        value_col = (
            stat_def.totals_col if value_kind is ValueKind.TOTAL else stat_def.per_game_col
        )
        table = tables.stat_table(scope)
        pool = eligible_pool(
            table, tables.players, value_col=value_col, min_games=dataset_config.min_games
        )
        if pool.empty:
            continue

        available_ranks = sorted(pool["rank"].unique().tolist())
        target_rank = _sample_rank(rng, available_ranks, dataset_config.rank_range)
        key: QuestionKey = (scope.value, value_kind.value, stat_key, target_rank)
        if key in used_keys:
            continue

        answer_row = pool[pool["rank"] == target_rank].iloc[0]
        era_caveat = stat_def.era_caveat()
        question_text = _question_text(scope, value_kind, stat_def.label, target_rank, era_caveat)

        return Question(
            key=key,
            question_text=question_text,
            scope=scope,
            value_kind=value_kind,
            stat_key=stat_key,
            stat_label=stat_def.label,
            value_col=value_col,
            min_games=dataset_config.min_games,
            target_rank=target_rank,
            answer_player_id=int(answer_row["player_id"]),
            answer_player_name=str(answer_row["full_name"]),
            answer_value=float(answer_row["value"]),
            answer_season=str(answer_row["season"]) if "season" in pool.columns else None,
            era_caveat=era_caveat,
        )

    raise RuntimeError(
        f"Could not generate a fresh question after {MAX_GENERATION_ATTEMPTS} attempts; "
        "the question space may be exhausted."
    )


def dedupe_best_per_player(pool: pd.DataFrame) -> pd.DataFrame:
    """Collapse a ranked pool to one row per player: their best (lowest) rank.

    career_totals/career_per_game already have exactly one row per player,
    so this is a no-op there. season_records has one row per player-season,
    so a prolific player can otherwise appear many times over (once per
    qualifying season) in a guess dropdown built from the raw pool.
    """
    if pool.empty:
        return pool
    best = pool.sort_values("rank").groupby("player_id", as_index=False).first()
    return best.sort_values("rank").reset_index(drop=True)


def eligible_players_for_question(question: Question, tables: DataTables) -> pd.DataFrame:
    """The guess pool for `question`: one entry per player, no duplicates.

    For SEASON_RECORD questions this deliberately differs from the pool
    used to pick the answer (which must consider every qualifying season,
    since the same player can legitimately occupy multiple all-time
    single-season ranks). A guess, though, is one player with one value:
    their own best qualifying season for this stat (see
    dedupe_best_per_player), matching the brief's "a guessed player's
    value is their own best qualifying season" rule.
    """
    table = tables.stat_table(question.scope)
    pool = eligible_pool(
        table, tables.players, value_col=question.value_col, min_games=question.min_games
    )
    if question.scope is DatasetScope.SEASON_RECORD:
        pool = dedupe_best_per_player(pool)
    return pool
