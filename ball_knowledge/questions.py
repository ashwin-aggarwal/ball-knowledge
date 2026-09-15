"""Question generation over the career stat tables.

Pure and Streamlit-free by design (see file structure notes in the project
brief): this module never imports streamlit, and never imports data.py
(which does, for its @st.cache_data wrappers). data.py is expected to load
the parquet tables and hand the resulting DataFrames to this module.

All-time career stats only: no single-season or single-playoff-run
questions. Every DatasetScope is a career aggregate (see config.py).
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

# Question uniqueness key: (scope, stat_key, target_rank). value_kind is
# omitted since it's fixed 1:1 by scope, so including it would be redundant.
QuestionKey = tuple[str, str, int]

# Max attempts to find a fresh, satisfiable (scope, stat, rank) combination
# before giving up. Generous: the combinatorial space is large (2 scopes x
# 16 stats x up to 300 ranks) relative to any realistic game length, so
# exhaustion only happens with tiny test fixtures that have deliberately
# starved the space.
MAX_GENERATION_ATTEMPTS = 500


@dataclass(frozen=True)
class DataTables:
    """Bundles the career stat tables plus players, keyed by player_id."""

    career_totals: pd.DataFrame
    career_per_game: pd.DataFrame
    players: pd.DataFrame

    def stat_table(self, scope: DatasetScope) -> pd.DataFrame:
        if scope is DatasetScope.CAREER_TOTAL:
            return self.career_totals
        return self.career_per_game


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
    era_caveat: str | None


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _question_text(
    scope: DatasetScope,
    stat_label: str,
    target_rank: int,
    era_caveat: str | None,
) -> str:
    rank_str = _ordinal(target_rank)
    label_lower = stat_label.lower()
    if scope is DatasetScope.CAREER_TOTAL:
        text = f"Who ranks {rank_str} all time in career {label_lower}?"
    else:
        text = f"Who ranks {rank_str} all time in career {label_lower} per game?"
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

    Returns columns [player_id, full_name, value, rank, gp], sorted by
    rank ascending (best first). `rank` uses "min" ties (shared rank, next
    rank skips accordingly), matching how a real leaderboard displays ties.
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
    return eligible[["player_id", "full_name", "value", "rank", "gp"]].reset_index(drop=True)


def _sample_rank(rng: random.Random, available_ranks: list[int], rank_range: RankRange) -> int:
    """Pick a rank from `available_ranks`, skewed toward the low end."""
    in_range = [r for r in available_ranks if rank_range.low <= r <= rank_range.high]
    if not in_range:
        return available_ranks[0]
    in_range.sort()
    u = rng.random() ** rank_range.skew
    idx = min(int(u * len(in_range)), len(in_range) - 1)
    return in_range[idx]


def _pick_scope(rng: random.Random, config: GameConfig) -> DatasetScope:
    scopes = list(config.dataset_weights.keys())
    weights = [config.dataset_weights[s] for s in scopes]
    return rng.choices(scopes, weights=weights, k=1)[0]


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
        scope = _pick_scope(rng, config)
        dataset_config = config.dataset_config(scope)
        value_kind = dataset_config.value_kind
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
        key: QuestionKey = (scope.value, stat_key, target_rank)
        if key in used_keys:
            continue

        answer_row = pool[pool["rank"] == target_rank].iloc[0]
        era_caveat = stat_def.era_caveat()
        question_text = _question_text(scope, stat_def.label, target_rank, era_caveat)

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
            era_caveat=era_caveat,
        )

    raise RuntimeError(
        f"Could not generate a fresh question after {MAX_GENERATION_ATTEMPTS} attempts; "
        "the question space may be exhausted."
    )


def eligible_players_for_question(question: Question, tables: DataTables) -> pd.DataFrame:
    """The guess pool for `question`: same filter used to pick its answer."""
    table = tables.stat_table(question.scope)
    return eligible_pool(
        table, tables.players, value_col=question.value_col, min_games=question.min_games
    )


def resolve_player_by_name(name: str, players: pd.DataFrame) -> pd.Series | None:
    """Case/whitespace-insensitive exact match against the full player list.

    Deliberately no fuzzy matching (per the brief: "no fuzzy name matching
    to get wrong") -- a typed name must exactly match a real player's full
    name, ignoring case and surrounding whitespace, to resolve.
    """
    normalized = name.strip().casefold()
    if not normalized:
        return None
    matches = players[players["full_name"].str.strip().str.casefold() == normalized]
    if matches.empty:
        return None
    return matches.iloc[0]


def resolve_guess_value_and_rank(
    player_id: int, question: Question, tables: DataTables
) -> tuple[float | None, int]:
    """A guessed player's (value, rank) for `question`.

    Any real player can be guessed, even one who doesn't qualify for this
    specific stat/scope (no recorded value, or below the games floor) --
    they're scored as one spot past the eligible pool's worst rank, so an
    ineligible guess always loses to a legitimate one but is still
    accepted and recorded rather than rejected outright.
    """
    pool = eligible_players_for_question(question, tables)
    match = pool[pool["player_id"] == player_id]
    if not match.empty:
        row = match.iloc[0]
        return float(row["value"]), int(row["rank"])
    worst_rank = int(pool["rank"].max()) if not pool.empty else question.target_rank
    return None, worst_rank + 1
