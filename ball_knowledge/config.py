"""All tunable knobs for ball-knowledge, in one place.

Nothing in this module imports Streamlit or nba_api. It is pure config so
that questions.py, scoring.py and the build scripts can all depend on it
without dragging in unrelated machinery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DatasetScope(str, Enum):
    """Which table a question is drawn from."""

    CAREER_TOTAL = "career_total"
    CAREER_PER_GAME = "career_per_game"
    SEASON_RECORD = "season_record"


class ValueKind(str, Enum):
    """Whether a question is asked about a total or a per-game average.

    Fixed by scope for CAREER_TOTAL (always TOTAL) and CAREER_PER_GAME
    (always PER_GAME). For SEASON_RECORD either kind is possible from the
    same table, chosen per-question, because a single-season *total* record
    (e.g. a lockout-shortened season) is legitimate with no games floor,
    while a single-season *per-game* record needs a qualifying-games floor
    to stop a 12-game call-up from owning a leaderboard.
    """

    TOTAL = "total"
    PER_GAME = "per_game"


@dataclass(frozen=True)
class StatDef:
    """Describes one statistic as it appears across the three datasets.

    ``totals_col`` and ``per_game_col`` are the column names in
    career_totals.parquet / career_per_game.parquet / season_records.parquet
    (season_records carries both a totals and a per-game column, prefixed
    the same way). ``tracked_since`` is the first season (e.g. "1973-74")
    the NBA recorded this stat league-wide, or None if it has been tracked
    since the league's founding.
    """

    key: str
    label: str
    totals_col: str
    per_game_col: str
    tracked_since: str | None = None
    in_career_totals: bool = True
    in_career_per_game: bool = True
    in_season_records: bool = True

    def era_caveat(self) -> str | None:
        if self.tracked_since is None:
            return None
        return f"{self.label} have been tracked since {self.tracked_since}."


# Canonical stat catalogue. Keys match column name stems used in the
# built parquet files (see scripts/build_dataset.py).
STATS: dict[str, StatDef] = {
    "pts": StatDef("pts", "Points", "pts_total", "pts_per_game"),
    "reb": StatDef("reb", "Total rebounds", "reb_total", "reb_per_game"),
    "oreb": StatDef(
        "oreb", "Offensive rebounds", "oreb_total", "oreb_per_game",
        tracked_since="1973-74",
    ),
    "dreb": StatDef(
        "dreb", "Defensive rebounds", "dreb_total", "dreb_per_game",
        tracked_since="1973-74",
    ),
    "ast": StatDef("ast", "Assists", "ast_total", "ast_per_game"),
    "stl": StatDef(
        "stl", "Steals", "stl_total", "stl_per_game",
        tracked_since="1973-74",
    ),
    "blk": StatDef(
        "blk", "Blocks", "blk_total", "blk_per_game",
        tracked_since="1973-74",
    ),
    "tov": StatDef(
        "tov", "Turnovers", "tov_total", "tov_per_game",
        tracked_since="1977-78",
    ),
    "pf": StatDef("pf", "Personal fouls", "pf_total", "pf_per_game"),
    "min": StatDef("min", "Minutes played", "min_total", "min_per_game"),
    "fgm": StatDef("fgm", "Field goals made", "fgm_total", "fgm_per_game"),
    "fga": StatDef("fga", "Field goals attempted", "fga_total", "fga_per_game"),
    "fg3m": StatDef(
        "fg3m", "Three pointers made", "fg3m_total", "fg3m_per_game",
        tracked_since="1979-80",
    ),
    "fg3a": StatDef(
        "fg3a", "Three pointers attempted", "fg3a_total", "fg3a_per_game",
        tracked_since="1979-80",
    ),
    "ftm": StatDef("ftm", "Free throws made", "ftm_total", "ftm_per_game"),
    "fta": StatDef("fta", "Free throws attempted", "fta_total", "fta_per_game"),
}

# games played is not itself a guessable stat (it's the eligibility axis),
# so it is intentionally excluded from STATS / STAT_ALLOWLIST.


@dataclass(frozen=True)
class RankRange:
    """Inclusive rank sampling range, skewed toward the low (easy) end."""

    low: int
    high: int
    # Exponent > 1 skews sampling toward `low`. 1.0 is uniform.
    skew: float = 2.2


@dataclass(frozen=True)
class DatasetConfig:
    """Fully resolved knobs for one (scope, value_kind) question shape."""

    scope: DatasetScope
    value_kind: ValueKind
    weight: float
    rank_range: RankRange
    stat_allowlist: tuple[str, ...]
    min_games: int


@dataclass(frozen=True)
class GameConfig:
    """Knobs the app or a difficulty preset can override at LOBBY time."""

    min_players: int = 2
    max_players: int = 8
    default_rounds: int = 10
    round_points: int = 1
    exact_match_bonus_points: int = 2  # total awarded for an exact match
    career_per_game_min_games: int = 400
    season_per_game_min_games: int = 58
    season_total_min_games: int = 1

    # Within SEASON_RECORD, how often a question asks about the season's
    # total vs. its per-game average.
    season_value_kind_weights: dict[ValueKind, float] = field(
        default_factory=lambda: {
            ValueKind.TOTAL: 0.5,
            ValueKind.PER_GAME: 0.5,
        }
    )

    career_total_ranks: RankRange = field(
        default_factory=lambda: RankRange(low=1, high=250)
    )
    career_per_game_ranks: RankRange = field(
        default_factory=lambda: RankRange(low=1, high=100)
    )
    season_record_ranks: RankRange = field(
        default_factory=lambda: RankRange(low=1, high=100)
    )

    dataset_weights: dict[DatasetScope, float] = field(
        default_factory=lambda: {
            DatasetScope.CAREER_TOTAL: 0.45,
            DatasetScope.CAREER_PER_GAME: 0.30,
            DatasetScope.SEASON_RECORD: 0.25,
        }
    )

    # Stats eligible per dataset scope. Same catalogue for all three today,
    # but kept separate so a scope can be pared down without touching STATS.
    career_total_stats: tuple[str, ...] = tuple(STATS.keys())
    career_per_game_stats: tuple[str, ...] = tuple(STATS.keys())
    season_record_stats: tuple[str, ...] = tuple(STATS.keys())

    def dataset_config(
        self, scope: DatasetScope, value_kind: ValueKind | None = None
    ) -> DatasetConfig:
        """Resolve full knobs for one (scope, value_kind) question shape.

        `value_kind` is required for SEASON_RECORD (either kind is valid
        there) and ignored for CAREER_TOTAL/CAREER_PER_GAME, whose kind is
        fixed by the scope itself.
        """
        if scope is DatasetScope.CAREER_TOTAL:
            return DatasetConfig(
                scope=scope,
                value_kind=ValueKind.TOTAL,
                weight=self.dataset_weights[scope],
                rank_range=self.career_total_ranks,
                stat_allowlist=self.career_total_stats,
                min_games=1,
            )
        if scope is DatasetScope.CAREER_PER_GAME:
            return DatasetConfig(
                scope=scope,
                value_kind=ValueKind.PER_GAME,
                weight=self.dataset_weights[scope],
                rank_range=self.career_per_game_ranks,
                stat_allowlist=self.career_per_game_stats,
                min_games=self.career_per_game_min_games,
            )
        if value_kind is None:
            raise ValueError("value_kind is required for SEASON_RECORD")
        min_games = (
            self.season_total_min_games
            if value_kind is ValueKind.TOTAL
            else self.season_per_game_min_games
        )
        return DatasetConfig(
            scope=scope,
            value_kind=value_kind,
            weight=self.dataset_weights[scope],
            rank_range=self.season_record_ranks,
            stat_allowlist=self.season_record_stats,
            min_games=min_games,
        )


DEFAULT_GAME_CONFIG = GameConfig()

# --- Build-time config (used only by scripts/build_dataset.py) ---

DEFAULT_SLEEP_SECONDS = 0.8
MAX_RETRIES = 5
REQUEST_TIMEOUT_SECONDS = 30
ALL_TIME_LEADERS_TOP_X = 500

CACHE_DIR = ".cache/nba"
HEADSHOT_CACHE_DIR = ".cache/headshots"
DATA_DIR = "data"

HEADSHOT_URL_TEMPLATE = "https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
