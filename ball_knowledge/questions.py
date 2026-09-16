"""Question generation over the career totals table.

Pure and Streamlit-free by design (see file structure notes in the project
brief): this module never imports streamlit, and never imports data.py
(which does, for its @st.cache_data wrappers). data.py is expected to load
the parquet tables and hand the resulting DataFrames to this module.

All-time career counting-stat totals only: no single-season/single-
playoff-run questions, and no per-game/rate stats.

Three templates, chosen per round by weight (config.template_weights):
- straight_rank: "who ranks Nth all time in X" -- the original shape.
- value_anchor: "who's closest to exactly N career X" -- a round number
  on a counting stat. Scored by value distance, not rank.
- obscure_spotlight: straight_rank with its stat forced from
  config.obscure_stats, so personal fouls/turnovers/minutes/games-played/
  free-throw-volume rounds show up deliberately rather than only by
  the luck of a uniform draw.

Stat and scope cooldowns (config.stat_cooldown_rounds,
scope_max_consecutive) and marquee-stat down-weighting
(config.marquee_stats / marquee_weight_share) apply across all
templates. A difficulty arc (config.early_rank_skew / late_rank_skew)
biases early rounds toward shallow, recognizable ranks and later rounds
deeper.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import pandas as pd

from ball_knowledge.config import (
    STATS,
    DatasetScope,
    GameConfig,
    RankRange,
    StatDef,
)

log = logging.getLogger(__name__)

TEMPLATE_STRAIGHT_RANK = "straight_rank"
TEMPLATE_VALUE_ANCHOR = "value_anchor"
TEMPLATE_OBSCURE_SPOTLIGHT = "obscure_spotlight"

# Question uniqueness key. For rank-based templates this is
# (scope, stat_key, target_rank); for value_anchor it's
# (scope, stat_key, int(anchor_value)) -- same shape, different meaning
# in the third slot, which is fine since both mean "don't ask this exact
# question again this session."
QuestionKey = tuple[str, str, int]

# Max attempts to find a fresh, satisfiable question before giving up.
# Generous relative to any realistic game length; exhaustion only
# happens with tiny test fixtures that have deliberately starved the
# space, or a pared-down stat allowlist across a very long game.
MAX_GENERATION_ATTEMPTS = 500


@dataclass(frozen=True)
class DataTables:
    """Bundles the career totals table plus players, keyed by player_id."""

    career_totals: pd.DataFrame
    players: pd.DataFrame

    def stat_table(self, scope: DatasetScope) -> pd.DataFrame:
        return self.career_totals


@dataclass(frozen=True)
class Question:
    key: QuestionKey
    template: str
    question_text: str
    scope: DatasetScope
    stat_key: str
    stat_label: str
    value_col: str
    min_games: int
    scoring_mode: str  # "rank" or "value"
    target_rank: int
    anchor_value: float | None
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


def _question_text_rank(stat_label: str, target_rank: int, era_caveat: str | None) -> str:
    text = f"Who ranks {_ordinal(target_rank)} all time in career {stat_label.lower()}?"
    if era_caveat:
        text = f"{text} {era_caveat}"
    return text


def _question_text_value_anchor(stat_label: str, anchor_value: float, era_caveat: str | None) -> str:
    text = f"Which player sits closest to exactly {anchor_value:,.0f} career {stat_label.lower()}?"
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
    """Rows of `table` eligible for a given stat, ranked descending.

    Returns columns [player_id, full_name, value, rank, gp], sorted by
    rank ascending (best first). `rank` uses "min" ties (shared rank, next
    rank skips accordingly), matching how a real leaderboard displays ties.
    """
    if value_col not in table.columns:
        return pd.DataFrame(columns=["player_id", "full_name", "value", "rank", "gp"])
    if not pd.api.types.is_numeric_dtype(table[value_col]):
        raise TypeError(
            f"eligible_pool: column {value_col!r} is dtype "
            f"{table[value_col].dtype}, not numeric. Ranking a non-numeric "
            "column would silently sort lexicographically (e.g. '9' > '10') "
            "instead of by value -- refusing rather than risking a wrong answer."
        )

    eligible = table[(table["gp"] >= min_games) & table[value_col].notna()].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["player_id", "full_name", "value", "rank", "gp"])

    eligible["rank"] = eligible[value_col].rank(method="min", ascending=False).astype(int)
    # Assign rather than rename: value_col can legitimately be "gp" itself
    # (games played is both the eligibility axis and a guessable stat), in
    # which case a rename would clobber the very "gp" column the return
    # statement below still needs.
    eligible["value"] = eligible[value_col]
    eligible = eligible.merge(players[["player_id", "full_name"]], on="player_id", how="left")
    eligible = eligible.sort_values("rank")
    return eligible[["player_id", "full_name", "value", "rank", "gp"]].reset_index(drop=True)


def leaderboard_neighbors(
    pool: pd.DataFrame, target_rank: int, count: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The up-to-`count` closest ranks above and below `target_rank` in `pool`.

    `pool` is expected already sorted by rank ascending (as returned by
    eligible_pool). Returns (above, below): `above` is the rows with the
    `count` next-best (lower-numbered) distinct ranks, closest-first;
    `below` is the rows with the `count` next-worst (higher-numbered)
    distinct ranks, closest-first. Either can come back with fewer than
    `count` rows near either end of the leaderboard, and a tie can put
    more than one row at a single rank.
    """
    ranks = sorted(pool["rank"].unique().tolist())
    idx = ranks.index(target_rank) if target_rank in ranks else None
    if idx is None:
        return pool.iloc[0:0], pool.iloc[0:0]

    above_ranks = ranks[max(0, idx - count):idx][::-1]  # closest-first
    below_ranks = ranks[idx + 1: idx + 1 + count]

    above = pool[pool["rank"].isin(above_ranks)].copy()
    above["_order"] = above["rank"].map({r: i for i, r in enumerate(above_ranks)})
    above = above.sort_values("_order").drop(columns="_order")

    below = pool[pool["rank"].isin(below_ranks)].copy()
    below["_order"] = below["rank"].map({r: i for i, r in enumerate(below_ranks)})
    below = below.sort_values("_order").drop(columns="_order")

    return above.reset_index(drop=True), below.reset_index(drop=True)


def _sample_rank(rng: random.Random, available_ranks: list[int], rank_range: RankRange) -> int:
    """Pick a rank from `available_ranks`, skewed toward the low end."""
    in_range = [r for r in available_ranks if rank_range.low <= r <= rank_range.high]
    if not in_range:
        return available_ranks[0]
    in_range.sort()
    u = rng.random() ** rank_range.skew
    idx = min(int(u * len(in_range)), len(in_range) - 1)
    return in_range[idx]


def _effective_rank_range(
    base: RankRange, config: GameConfig, round_number: int, total_rounds: int
) -> RankRange:
    """`base` with its skew interpolated along the difficulty arc.

    t=0 at round 1 (early_rank_skew, shallow/easy), t=1 at the final
    round (late_rank_skew, deeper). Setting early == late flattens the
    arc back to a constant skew.
    """
    t = 0.0 if total_rounds <= 1 else (round_number - 1) / (total_rounds - 1)
    t = max(0.0, min(1.0, t))
    skew = config.early_rank_skew + (config.late_rank_skew - config.early_rank_skew) * t
    return RankRange(low=base.low, high=base.high, skew=skew)


def _cooldown_filtered_stats(
    allowlist: tuple[str, ...], recent_stats: list[str], cooldown_rounds: int
) -> tuple[str, ...]:
    """`allowlist` minus any stat used in the last `cooldown_rounds` rounds.

    Relaxes by shrinking the cooldown window one round at a time if that
    would empty the pool (a short allowlist, a long game) -- this must
    always return something non-empty when `allowlist` is non-empty.
    """
    window = cooldown_rounds
    while window > 0:
        excluded = set(recent_stats[-window:]) if recent_stats else set()
        candidates = tuple(s for s in allowlist if s not in excluded)
        if candidates:
            return candidates
        window -= 1
    return allowlist


def _blocked_scope_value(recent_scopes: list[str], max_consecutive: int) -> str | None:
    """The scope value that's occupied every one of the last `max_consecutive`
    rounds, if any -- it cannot legally appear again this round."""
    if max_consecutive > 0 and len(recent_scopes) >= max_consecutive:
        tail = recent_scopes[-max_consecutive:]
        if len(set(tail)) == 1:
            return tail[0]
    return None


def _cooldown_filtered_scopes(
    scopes: list[DatasetScope],
    weights: list[float],
    recent_scopes: list[str],
    max_consecutive: int,
) -> tuple[list[DatasetScope], list[float]]:
    """Drop a scope that's occupied every one of the last `max_consecutive` rounds."""
    blocked = _blocked_scope_value(recent_scopes, max_consecutive)
    if blocked is not None:
        filtered = [(s, w) for s, w in zip(scopes, weights) if s.value != blocked]
        if filtered:
            return [s for s, _ in filtered], [w for _, w in filtered]
    return scopes, weights


def _weighted_stat_choice(
    rng: random.Random, allowlist: tuple[str, ...], config: GameConfig
) -> str:
    """Pick a stat; marquee_stats share marquee_weight_share of the mass."""
    marquee_in = [s for s in allowlist if s in config.marquee_stats]
    rest_in = [s for s in allowlist if s not in config.marquee_stats]
    if not marquee_in or not rest_in:
        return rng.choice(allowlist)
    marquee_w = config.marquee_weight_share / len(marquee_in)
    rest_w = (1 - config.marquee_weight_share) / len(rest_in)
    population = marquee_in + rest_in
    weights = [marquee_w] * len(marquee_in) + [rest_w] * len(rest_in)
    return rng.choices(population, weights=weights, k=1)[0]


def _pick_scope(
    rng: random.Random, config: GameConfig, recent_scopes: list[str]
) -> DatasetScope:
    scopes = list(config.dataset_weights.keys())
    weights = [config.dataset_weights[s] for s in scopes]
    scopes, weights = _cooldown_filtered_scopes(
        scopes, weights, recent_scopes, config.scope_max_consecutive
    )
    return rng.choices(scopes, weights=weights, k=1)[0]


def _pick_template(rng: random.Random, config: GameConfig) -> str:
    names = list(config.template_weights.keys())
    weights = [config.template_weights[n] for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def _value_anchor_step(max_value: float) -> float:
    """A 'round number' step sized to the stat's magnitude (e.g. 1000 for
    a stat maxing in the tens of thousands, 100 for one maxing in the
    thousands)."""
    if max_value <= 0:
        return 1.0
    digits = len(str(int(max_value)))
    return float(10 ** max(digits - 2, 0))


def _build_rank_question(
    tables: DataTables,
    config: GameConfig,
    rng: random.Random,
    recent_stats: list[str],
    recent_scopes: list[str],
    round_number: int,
    total_rounds: int,
    used_keys: set[QuestionKey],
    template: str,
) -> Question | None:
    """Builds straight_rank or obscure_spotlight. Returns None if this
    particular (scope, stat) draw isn't satisfiable right now (empty
    pool) or collides with `used_keys` -- the caller retries."""
    scope = _pick_scope(rng, config, recent_scopes)
    dataset_config = config.dataset_config(scope)

    if template == TEMPLATE_OBSCURE_SPOTLIGHT:
        base_allowlist = tuple(
            s for s in config.obscure_stats if s in dataset_config.stat_allowlist
        )
        if not base_allowlist:
            return None
        allowlist = _cooldown_filtered_stats(base_allowlist, recent_stats, config.stat_cooldown_rounds)
        stat_key = rng.choice(allowlist)
    else:
        allowlist = _cooldown_filtered_stats(
            dataset_config.stat_allowlist, recent_stats, config.stat_cooldown_rounds
        )
        stat_key = _weighted_stat_choice(rng, allowlist, config)

    stat_def: StatDef = STATS[stat_key]
    value_col = stat_def.totals_col
    table = tables.stat_table(scope)
    pool = eligible_pool(
        table, tables.players, value_col=value_col, min_games=dataset_config.min_games
    )
    if pool.empty:
        return None

    rank_range = _effective_rank_range(dataset_config.rank_range, config, round_number, total_rounds)
    available_ranks = sorted(pool["rank"].unique().tolist())
    target_rank = _sample_rank(rng, available_ranks, rank_range)
    key: QuestionKey = (scope.value, stat_key, target_rank)
    if key in used_keys:
        return None

    answer_row = pool[pool["rank"] == target_rank].iloc[0]
    era_caveat = stat_def.era_caveat()
    question_text = _question_text_rank(stat_def.label, target_rank, era_caveat)

    return Question(
        key=key,
        template=template,
        question_text=question_text,
        scope=scope,
        stat_key=stat_key,
        stat_label=stat_def.label,
        value_col=value_col,
        min_games=dataset_config.min_games,
        scoring_mode="rank",
        target_rank=target_rank,
        anchor_value=None,
        answer_player_id=int(answer_row["player_id"]),
        answer_player_name=str(answer_row["full_name"]),
        answer_value=float(answer_row["value"]),
        era_caveat=era_caveat,
    )


def _build_value_anchor_question(
    tables: DataTables,
    config: GameConfig,
    rng: random.Random,
    recent_stats: list[str],
    recent_scopes: list[str],
    used_keys: set[QuestionKey],
) -> Question | None:
    scope = DatasetScope.CAREER_TOTAL
    # This template's scope is fixed, unlike straight_rank/obscure_spotlight
    # which pick among all scopes -- but the scope cooldown still applies
    # globally, so if CAREER_TOTAL is currently blocked *and* some other
    # scope exists to rotate to instead, this template isn't satisfiable
    # this round; the caller retries with another. With only one scope
    # configured (true today), there's no alternative to rotate to, so
    # the cooldown must not block this template at all -- otherwise it
    # would be near-permanently blocked (every scope value here is
    # CAREER_TOTAL, so "the last N rounds were all this scope" becomes
    # true almost immediately and stays true for the rest of the game).
    if (
        len(config.dataset_weights) > 1
        and _blocked_scope_value(recent_scopes, config.scope_max_consecutive) == scope.value
    ):
        return None
    dataset_config = config.dataset_config(scope)
    allowlist = _cooldown_filtered_stats(
        dataset_config.stat_allowlist, recent_stats, config.stat_cooldown_rounds
    )
    stat_key = _weighted_stat_choice(rng, allowlist, config)
    stat_def: StatDef = STATS[stat_key]
    value_col = stat_def.totals_col
    table = tables.stat_table(scope)
    pool = eligible_pool(
        table, tables.players, value_col=value_col, min_games=dataset_config.min_games
    )
    if len(pool) < 2:
        return None

    min_val = float(pool["value"].min())
    max_val = float(pool["value"].max())
    step = _value_anchor_step(max_val)
    anchor = round(rng.uniform(min_val, max_val) / step) * step
    anchor = max(min_val, min(max_val, anchor))

    diffs = (pool["value"] - anchor).abs()
    answer_row = pool.loc[diffs.idxmin()]

    key: QuestionKey = (scope.value, stat_key, int(anchor))
    if key in used_keys:
        return None

    era_caveat = stat_def.era_caveat()
    question_text = _question_text_value_anchor(stat_def.label, anchor, era_caveat)

    return Question(
        key=key,
        template=TEMPLATE_VALUE_ANCHOR,
        question_text=question_text,
        scope=scope,
        stat_key=stat_key,
        stat_label=stat_def.label,
        value_col=value_col,
        min_games=dataset_config.min_games,
        scoring_mode="value",
        target_rank=int(answer_row["rank"]),
        anchor_value=float(anchor),
        answer_player_id=int(answer_row["player_id"]),
        answer_player_name=str(answer_row["full_name"]),
        answer_value=float(answer_row["value"]),
        era_caveat=era_caveat,
    )


def _validate_question(question: Question, tables: DataTables) -> list[str]:
    """Independently re-derive the answer and check it against `question`.

    The answer is produced by lookup, not computation, so this asserts
    the lookup actually returned what was asked for rather than assuming
    it: an answer_value read back straight from the raw table (not the
    already-computed pool the question was built from), a rank/closeness
    recomputed from scratch, and question text that actually names the
    stat it's scoring against. Returns a list of failure descriptions
    (empty means valid) rather than raising, so the caller can log and
    resample instead of ever showing a bad question.
    """
    failures: list[str] = []
    table = tables.stat_table(question.scope)

    raw_row = table[table["player_id"] == question.answer_player_id]
    if raw_row.empty:
        failures.append(
            f"answer_player_id {question.answer_player_id} not found in "
            f"{question.scope.value} table at all"
        )
        return failures
    raw_value = raw_row.iloc[0][question.value_col]
    if pd.isna(raw_value):
        failures.append(
            f"answer player's {question.value_col} is null in the raw table "
            "(should have been excluded by eligibility filtering)"
        )
        return failures
    if abs(float(raw_value) - question.answer_value) > 1e-6:
        failures.append(
            f"answer_value {question.answer_value!r} does not match the raw "
            f"table's independently-read value {raw_value!r} for "
            f"{question.value_col}"
        )

    pool = eligible_pool(
        table, tables.players, value_col=question.value_col, min_games=question.min_games
    )
    match = pool[pool["player_id"] == question.answer_player_id]
    if match.empty:
        failures.append("answer player not present in an independently recomputed eligible pool")
    elif question.scoring_mode == "rank":
        recomputed_rank = int(match.iloc[0]["rank"])
        if recomputed_rank != question.target_rank:
            failures.append(
                f"recomputed rank {recomputed_rank} != question.target_rank {question.target_rank}"
            )
    else:  # "value": the answer must actually be closest to the anchor.
        assert question.anchor_value is not None
        diffs = (pool["value"] - question.anchor_value).abs()
        closest_pid = int(pool.loc[diffs.idxmin(), "player_id"])
        if closest_pid != question.answer_player_id:
            failures.append(
                f"answer player {question.answer_player_id} is not actually closest to "
                f"anchor {question.anchor_value}; recomputation says player {closest_pid} is"
            )

    if question.stat_label.lower() not in question.question_text.lower():
        failures.append(
            f"question_text {question.question_text!r} never mentions stat "
            f"label {question.stat_label!r}"
        )

    return failures


def generate_question(
    tables: DataTables,
    config: GameConfig,
    used_keys: set[QuestionKey],
    recent_stats: list[str] | None = None,
    recent_scopes: list[str] | None = None,
    round_number: int = 1,
    total_rounds: int = 1,
    rng: random.Random | None = None,
) -> Question:
    """Generate one question not already present in `used_keys`.

    `recent_stats`/`recent_scopes` (most-recent-last) drive the stat and
    scope cooldowns; `round_number`/`total_rounds` drive the difficulty
    arc. All four are optional and default to "no history" so existing
    callers/tests that don't care about cooldowns or the arc keep working
    unchanged.

    Raises RuntimeError if no satisfiable, unused question is found within
    MAX_GENERATION_ATTEMPTS attempts (only realistic with a tiny or
    exhausted question space, e.g. small test fixtures or a very long
    game against a pared-down stat allowlist).
    """
    rng = rng or random.Random()
    recent_stats = recent_stats or []
    recent_scopes = recent_scopes or []

    for _ in range(MAX_GENERATION_ATTEMPTS):
        template = _pick_template(rng, config)
        if template == TEMPLATE_VALUE_ANCHOR:
            question = _build_value_anchor_question(
                tables, config, rng, recent_stats, recent_scopes, used_keys
            )
        else:
            question = _build_rank_question(
                tables,
                config,
                rng,
                recent_stats,
                recent_scopes,
                round_number,
                total_rounds,
                used_keys,
                template,
            )
        if question is None:
            continue

        failures = _validate_question(question, tables)
        if not failures:
            return question
        log.warning(
            "Discarding a generated question that failed answer-path validation "
            "(key=%s, template=%s): %s",
            question.key,
            question.template,
            "; ".join(failures),
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
    specific stat (no recorded value, or below the games floor) -- they're
    scored as one spot past the eligible pool's worst rank, so an
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
