"""Tiny synthetic fixture tables mirroring the built parquet schema.

Deliberately small and hand-shaped (not a random sample of the real data)
so tests can assert exact ranks, exact eligibility inclusion/exclusion,
and exact era-null behavior without depending on what the real NBA
leaderboards currently look like.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ball_knowledge.config import STATS, GameConfig, RankRange
from ball_knowledge.questions import DataTables

STAT_KEYS = list(STATS.keys())

# Stats not tracked before certain seasons (see config.STATS tracked_since).
ERA_UNTRACKED_STATS = {"oreb", "dreb", "stl", "blk", "tov", "fg3m", "fg3a"}

# Player 8 is a pre-1973-74 "old-timer": every stat the league didn't track
# yet is null, exactly like the real API returns for that era.
OLD_TIMER_ID = 8

# Players 7 and 8 have short careers (few games played) to exercise
# min-games eligibility filtering.
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
    return DataTables(career_totals=career_totals, players=players)


@pytest.fixture
def small_game_config() -> GameConfig:
    """A GameConfig whose rank range fits the 8-player fixture above."""
    return GameConfig(career_total_ranks=RankRange(low=1, high=8, skew=1.0))
