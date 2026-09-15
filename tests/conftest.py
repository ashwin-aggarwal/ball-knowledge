"""Tiny synthetic fixture tables mirroring the built parquet schema.

Deliberately small and hand-shaped (not a random sample of the real data)
so tests can assert exact ranks, exact eligibility inclusion/exclusion,
and exact era-null behavior without depending on what the real NBA
leaderboards currently look like.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ball_knowledge.config import STATS, DatasetScope, GameConfig, RankRange, ValueKind
from ball_knowledge.questions import DataTables

STAT_KEYS = list(STATS.keys())

# Stats not tracked before certain seasons (see config.STATS tracked_since).
ERA_UNTRACKED_STATS = {"oreb", "dreb", "stl", "blk", "tov", "fg3m", "fg3a"}

# Player 8 is a pre-1973-74 "old-timer": every stat the league didn't track
# yet is null, exactly like the real API returns for that era.
OLD_TIMER_ID = 8

# Players 7 and 8 have short careers (gp below the default career per-game
# floor of 400) to exercise min-games eligibility filtering.
SHORT_CAREER_IDS = {7, 8}

PLAYER_IDS = list(range(1, 9))


def _totals_row(player_id: int) -> dict:
    gp = 200 if player_id in SHORT_CAREER_IDS else 1000
    row: dict = {"player_id": player_id, "gp": gp}
    for idx, key in enumerate(STAT_KEYS):
        if player_id == OLD_TIMER_ID and key in ERA_UNTRACKED_STATS:
            row[f"{key}_total"] = None
        else:
            # Player 1 has the highest value in every stat, descending to
            # player 8 the lowest, with a per-stat offset so stats don't
            # all tie with each other.
            row[f"{key}_total"] = float((9 - player_id) * 100 + idx)
    return row


def _per_game_row(player_id: int) -> dict:
    gp = 200 if player_id in SHORT_CAREER_IDS else 1000
    row: dict = {"player_id": player_id, "gp": gp}
    for idx, key in enumerate(STAT_KEYS):
        if player_id == OLD_TIMER_ID and key in ERA_UNTRACKED_STATS:
            row[f"{key}_per_game"] = None
        else:
            row[f"{key}_per_game"] = round((9 - player_id) * 1.0 + idx * 0.01, 3)
    return row


def _season_rows(player_id: int) -> list[dict]:
    rows = []
    for season, gp, scale in (("2020-21", 70, 0.5), ("2021-22", 20, 0.6)):
        row: dict = {"player_id": player_id, "season": season, "gp": gp}
        for idx, key in enumerate(STAT_KEYS):
            if player_id == OLD_TIMER_ID and key in ERA_UNTRACKED_STATS:
                row[f"{key}_total"] = None
                row[f"{key}_per_game"] = None
            else:
                total = round(((9 - player_id) * 100 + idx) * scale, 2)
                row[f"{key}_total"] = total
                row[f"{key}_per_game"] = round(total / gp, 3)
        rows.append(row)
    return rows


@pytest.fixture
def tables() -> DataTables:
    players = pd.DataFrame(
        {
            "player_id": PLAYER_IDS,
            "full_name": [f"Player {i}" for i in PLAYER_IDS],
            "is_active": [i <= 2 for i in PLAYER_IDS],
            "first_season": ["1959-60" if i == OLD_TIMER_ID else "2000-01" for i in PLAYER_IDS],
            "last_season": ["1972-73" if i == OLD_TIMER_ID else "2023-24" for i in PLAYER_IDS],
            "headshot_available": [i != OLD_TIMER_ID for i in PLAYER_IDS],
        }
    )
    career_totals = pd.DataFrame([_totals_row(pid) for pid in PLAYER_IDS])
    career_per_game = pd.DataFrame([_per_game_row(pid) for pid in PLAYER_IDS])
    season_records = pd.DataFrame(
        [row for pid in PLAYER_IDS for row in _season_rows(pid)]
    )
    return DataTables(
        career_totals=career_totals,
        career_per_game=career_per_game,
        season_records=season_records,
        players=players,
    )


@pytest.fixture
def small_game_config() -> GameConfig:
    """A GameConfig whose thresholds fit the 8-player fixture above.

    Real defaults (400 career games, 58 qualifying season games) would
    filter out every synthetic player, since the fixture only has 8 rows
    total. Rank ranges are similarly narrowed to what the fixture supports.
    """
    return GameConfig(
        career_per_game_min_games=400,  # players 7-8 (gp=200) intentionally excluded
        season_per_game_min_games=50,  # the 2021-22 row (gp=20) intentionally excluded
        season_total_min_games=1,
        career_total_ranks=RankRange(low=1, high=8, skew=1.0),
        career_per_game_ranks=RankRange(low=1, high=8, skew=1.0),
        season_record_ranks=RankRange(low=1, high=8, skew=1.0),
    )
