"""Golden-snapshot regression test.

Compares the current data/*.parquet against tests/golden/top25_snapshot.json
-- a top-25 leaderboard per (scope, stat) frozen after
scripts/audit_answers.py passed cleanly against a real, external source
(see scripts/update_golden.py). A real leaderboard change (a player
passes another) will fail this test on purpose: re-run the audit, then
run scripts/update_golden.py to update the snapshot deliberately. A
silent reordering -- from a code change, not a real-world event -- must
never ship unnoticed.

Runs against the real committed dataset (not a synthetic fixture) by
design: the whole point is to pin down what the actual data says.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ball_knowledge.config import DEFAULT_GAME_CONFIG, DatasetScope, STATS
from ball_knowledge.data import load_tables
from ball_knowledge.questions import eligible_pool

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "top25_snapshot.json"


def _load_golden() -> dict[str, list[int]]:
    if not GOLDEN_PATH.exists():
        pytest.skip(f"No golden snapshot at {GOLDEN_PATH}; run scripts/update_golden.py")
    with GOLDEN_PATH.open("r") as f:
        return json.load(f)


def test_golden_snapshot_file_is_nonempty() -> None:
    golden = _load_golden()
    assert len(golden) > 0


@pytest.fixture(scope="module")
def real_tables():
    try:
        return load_tables()
    except FileNotFoundError:
        pytest.skip("data/*.parquet not present; run scripts/build_dataset.py")


def test_top25_matches_golden_snapshot(real_tables) -> None:
    golden = _load_golden()
    config = DEFAULT_GAME_CONFIG
    mismatches = []

    for key, expected_ids in golden.items():
        scope_value, stat_key = key.split(":", 1)
        scope = DatasetScope(scope_value)
        dataset_config = config.dataset_config(scope)
        stat_def = STATS[stat_key]
        value_col = stat_def.totals_col
        table = real_tables.stat_table(scope)
        pool = eligible_pool(
            table, real_tables.players, value_col=value_col, min_games=dataset_config.min_games
        )
        actual_ids = [int(pid) for pid in pool.sort_values("rank").head(len(expected_ids))["player_id"]]
        if actual_ids != expected_ids:
            mismatches.append((key, expected_ids, actual_ids))

    if mismatches:
        lines = [
            f"  {key}: golden={expected} actual={actual}"
            for key, expected, actual in mismatches
        ]
        pytest.fail(
            f"{len(mismatches)} (scope, stat) top-25 snapshot(s) no longer match:\n"
            + "\n".join(lines)
            + "\n\nIf this is a real leaderboard change (a player passed another), "
            "re-run scripts/audit_answers.py to confirm, then scripts/update_golden.py "
            "to update the snapshot deliberately. If not, this is a real regression."
        )
