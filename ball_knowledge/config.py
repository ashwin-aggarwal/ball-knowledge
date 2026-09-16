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

    All-time career totals only: no single-season/single-playoff-run
    questions, and no per-game/rate stats -- every question here is a
    counting-stat career total. Kept as an enum (rather than a single
    hardcoded constant) so a future scope (e.g. career playoff totals)
    has an obvious place to go.
    """

    CAREER_TOTAL = "career_total"


@dataclass(frozen=True)
class StatDef:
    """Describes one statistic as it appears in career_totals.parquet.

    ``tracked_since`` is the first season (e.g. "1973-74") the NBA
    recorded this stat league-wide, or None if it has been tracked since
    the league's founding.
    """

    key: str
    label: str
    totals_col: str
    tracked_since: str | None = None

    def era_caveat(self) -> str | None:
        if self.tracked_since is None:
            return None
        return f"{self.label} have been tracked since {self.tracked_since}."


# Canonical stat catalogue. Keys match column name stems used in the
# built parquet files (see scripts/build_dataset.py).
STATS: dict[str, StatDef] = {
    "pts": StatDef("pts", "Points", "pts_total"),
    "reb": StatDef("reb", "Total rebounds", "reb_total"),
    "oreb": StatDef("oreb", "Offensive rebounds", "oreb_total", tracked_since="1973-74"),
    "dreb": StatDef("dreb", "Defensive rebounds", "dreb_total", tracked_since="1973-74"),
    "ast": StatDef("ast", "Assists", "ast_total"),
    "stl": StatDef("stl", "Steals", "stl_total", tracked_since="1973-74"),
    "blk": StatDef("blk", "Blocks", "blk_total", tracked_since="1973-74"),
    "tov": StatDef("tov", "Turnovers", "tov_total", tracked_since="1977-78"),
    "pf": StatDef("pf", "Personal fouls", "pf_total"),
    "min": StatDef("min", "Minutes played", "min_total"),
    "fgm": StatDef("fgm", "Field goals made", "fgm_total"),
    "fga": StatDef("fga", "Field goals attempted", "fga_total"),
    "fg3m": StatDef("fg3m", "Three pointers made", "fg3m_total", tracked_since="1979-80"),
    "fg3a": StatDef("fg3a", "Three pointers attempted", "fg3a_total", tracked_since="1979-80"),
    "ftm": StatDef("ftm", "Free throws made", "ftm_total"),
    "fta": StatDef("fta", "Free throws attempted", "fta_total"),
    "gp": StatDef("gp", "Games played", "gp"),
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
    """Fully resolved knobs for one scope's question shape."""

    scope: DatasetScope
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

    # GeoGuessr-style per-guess scoring (see scoring.py for the curve and
    # the local-rank-density normalizer that makes these units comparable
    # across every stat and scope). Tune TAU only by reading
    # scripts/calibrate_scoring.py's output, not by feel.
    max_round_score: int = 1000
    min_round_score: int = 25  # floor for any guess drawn from the eligible pool
    closest_bonus: int = 25  # flat bonus for the round's closest guess(es); 0 disables it
    score_tau: float = 12.0
    score_window_radius: int = 25  # rank-window radius for the density normalizer

    career_total_ranks: RankRange = field(
        default_factory=lambda: RankRange(low=1, high=300)
    )

    dataset_weights: dict[DatasetScope, float] = field(
        default_factory=lambda: {DatasetScope.CAREER_TOTAL: 1.0}
    )

    career_total_stats: tuple[str, ...] = tuple(STATS.keys())

    # --- Question variety: cooldowns, weighting, templates, difficulty ---

    # A stat cannot reappear for this many rounds (cooldown is on the stat
    # itself, not the stat+rank pair -- rank 47 and rank 62 in points are
    # formally distinct questions but feel identical to a player). If the
    # cooldown would empty the eligible stat pool (a short allowlist, a
    # long game), it relaxes by one round at a time rather than failing.
    stat_cooldown_rounds: int = 3
    # A dataset scope cannot appear more than this many times in a row.
    # A no-op today (only one scope exists), kept for when a second one
    # (e.g. career playoff totals) is added.
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
    # to a round counting-stat number; "obscure_spotlight" is
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
    #
    # Flat (1.0 = uniform) by default: an earlier skew of 3.0 put a 15%
    # chance on rank #1 alone and a 32% chance on top-10 (see the math in
    # _sample_rank -- u = random()**skew concentrates hard near 0 for
    # skew > ~1.5), which read as "always rank 1" in actual play. Raise
    # these above 1.0 again only if a gentler bias toward easier early
    # ranks is wanted; verify the actual rank-1/top-10 probability with
    # scripts/preview_questions.py before shipping a value, not by feel.
    early_rank_skew: float = 1.0
    late_rank_skew: float = 1.0

    # How many leaderboard neighbors above/below the answer to show on
    # the reveal screen (fewer if the answer is near either end).
    reveal_neighbor_count: int = 5

    def dataset_config(self, scope: DatasetScope) -> DatasetConfig:
        """Resolve full knobs for `scope` (only CAREER_TOTAL exists today)."""
        return DatasetConfig(
            scope=scope,
            weight=self.dataset_weights[scope],
            rank_range=self.career_total_ranks,
            stat_allowlist=self.career_total_stats,
            min_games=1,
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
