"""All tunable knobs for ball-knowledge, in one place.

Nothing in this module imports Streamlit or nba_api. It is pure config so
that questions.py, scoring.py and the build scripts can all depend on it
without dragging in unrelated machinery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DatasetScope(str, Enum):
    """Which table a question is drawn from.

    All-time career stats only: single-season and single-playoff-run
    questions were cut entirely (they added a qualifying-games floor and
    a season-specific template for comparatively little variety once the
    stat pool itself is wide and well-weighted). Every scope here is a
    career aggregate.
    """

    CAREER_TOTAL = "career_total"
    CAREER_PER_GAME = "career_per_game"


class ValueKind(str, Enum):
    """Whether a question is asked about a total or a per-game average.

    Fixed 1:1 by scope: CAREER_TOTAL is always TOTAL, CAREER_PER_GAME is
    always PER_GAME. Kept as its own type (rather than folded into
    DatasetScope) because question text and formatting logic key off it
    directly.
    """

    TOTAL = "total"
    PER_GAME = "per_game"


@dataclass(frozen=True)
class StatDef:
    """Describes one statistic as it appears across the two datasets.

    ``totals_col`` and ``per_game_col`` are the column names in
    career_totals.parquet / career_per_game.parquet. ``tracked_since`` is
    the first season (e.g. "1973-74") the NBA recorded this stat
    league-wide, or None if it has been tracked since the league's
    founding.
    """

    key: str
    label: str
    totals_col: str
    per_game_col: str | None
    tracked_since: str | None = None

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
    # Games played doubles as the eligibility axis (the `gp` column) *and*
    # a legitimately fun obscure-stat spotlight ("who's played the most
    # games all time") -- but only as a total: "games played per game" is
    # meaningless, so per_game_col is None and it's excluded from any
    # per-game stat allowlist.
    "gp": StatDef("gp", "Games played", "gp", None),
}


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

    min_players: int = 1  # solo play is supported: your closest (only) guess always scores
    max_players: int = 8
    default_rounds: int = 10
    round_points: int = 1
    exact_match_bonus_points: int = 2  # total awarded for an exact match
    career_per_game_min_games: int = 400

    # Widened once single-season questions were cut: with only two scopes
    # left, rank variety within each stat carries more of the game's
    # difficulty range, so career totals can go much deeper than before.
    career_total_ranks: RankRange = field(
        default_factory=lambda: RankRange(low=1, high=300)
    )
    career_per_game_ranks: RankRange = field(
        default_factory=lambda: RankRange(low=1, high=100)
    )

    dataset_weights: dict[DatasetScope, float] = field(
        default_factory=lambda: {
            DatasetScope.CAREER_TOTAL: 0.6,
            DatasetScope.CAREER_PER_GAME: 0.4,
        }
    )

    # Stats eligible per dataset scope. Same catalogue for both today, but
    # kept separate so a scope can be pared down without touching STATS.
    # career_per_game_stats excludes any stat with no per_game_col (just
    # "gp" today) automatically -- "games played per game" is meaningless.
    career_total_stats: tuple[str, ...] = tuple(STATS.keys())
    career_per_game_stats: tuple[str, ...] = tuple(
        k for k, v in STATS.items() if v.per_game_col is not None
    )

    # --- Question variety: cooldowns, weighting, templates, difficulty ---

    # A stat cannot reappear for this many rounds (cooldown is on the stat
    # itself, not the stat+rank pair -- rank 47 and rank 62 in points are
    # formally distinct questions but feel identical to a player). If the
    # cooldown would empty the eligible stat pool (a short allowlist, a
    # long game), it relaxes by one round at a time rather than failing.
    stat_cooldown_rounds: int = 3
    # A dataset scope cannot appear more than this many times in a row.
    scope_max_consecutive: int = 2

    # Marquee stats (the ones people think in) are down-weighted so the
    # game doesn't feel points-heavy under uniform sampling. marquee_stats
    # share marquee_weight_share of the pick probability; everything else
    # in the allowlist splits the remainder.
    marquee_stats: tuple[str, ...] = ("pts", "reb", "ast")
    marquee_weight_share: float = 0.40

    # Deliberately unglamorous stats a template can spotlight -- these
    # leaderboards are full of unexpected names.
    obscure_stats: tuple[str, ...] = ("pf", "tov", "min", "gp", "fta")

    # Which question template a round uses. "straight_rank" is the
    # existing "who ranks Nth" shape; "value_anchor" asks who sits closest
    # to a round counting-stat number (never a per-game stat -- round
    # numbers on rate stats cluster players within hundredths of each
    # other and collapse into a coin flip); "obscure_spotlight" is
    # straight_rank with its stat forced from obscure_stats.
    template_weights: dict[str, float] = field(
        default_factory=lambda: {
            "straight_rank": 0.55,
            "value_anchor": 0.20,
            "obscure_spotlight": 0.25,
        }
    )

    # Difficulty arc: early rounds sample shallower ranks (easier, more
    # recognizable names), later rounds go deeper. Expressed as two rank-
    # sampling skew values interpolated across the game's round count;
    # set them equal to flatten the arc back to a constant skew.
    early_rank_skew: float = 3.0
    late_rank_skew: float = 1.4

    def dataset_config(self, scope: DatasetScope) -> DatasetConfig:
        """Resolve full knobs for one (scope, value_kind) question shape."""
        if scope is DatasetScope.CAREER_TOTAL:
            return DatasetConfig(
                scope=scope,
                value_kind=ValueKind.TOTAL,
                weight=self.dataset_weights[scope],
                rank_range=self.career_total_ranks,
                stat_allowlist=self.career_total_stats,
                min_games=1,
            )
        return DatasetConfig(
            scope=scope,
            value_kind=ValueKind.PER_GAME,
            weight=self.dataset_weights[scope],
            rank_range=self.career_per_game_ranks,
            stat_allowlist=self.career_per_game_stats,
            min_games=self.career_per_game_min_games,
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
