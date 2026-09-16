#!/usr/bin/env python3
"""Freeze the current top-25 leaderboard per (scope, stat) as a golden snapshot.

Run this deliberately, after scripts/audit_answers.py has passed against
the current data/*.parquet -- never as part of a normal build. Every
future build's tests/test_golden_snapshot.py compares against this file;
a real leaderboard change (a player passes another) will fail that test
until someone re-runs the audit and updates the snapshot on purpose. A
silent reordering should never ship unnoticed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ball_knowledge.config import DatasetScope, DEFAULT_GAME_CONFIG, STATS  # noqa: E402
from ball_knowledge.data import load_tables  # noqa: E402
from ball_knowledge.questions import eligible_pool  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "tests" / "golden" / "top25_snapshot.json"
TOP_N = 25


def main() -> None:
    tables = load_tables()
    config = DEFAULT_GAME_CONFIG
    snapshot: dict[str, list[int]] = {}

    for scope in (DatasetScope.CAREER_TOTAL, DatasetScope.CAREER_PER_GAME):
        dataset_config = config.dataset_config(scope)
        table = tables.stat_table(scope)
        for stat_key in dataset_config.stat_allowlist:
            stat_def = STATS[stat_key]
            value_col = (
                stat_def.totals_col
                if dataset_config.value_kind.value == "total"
                else stat_def.per_game_col
            )
            if value_col is None:
                continue
            pool = eligible_pool(
                table, tables.players, value_col=value_col, min_games=dataset_config.min_games
            )
            top = pool.sort_values("rank").head(TOP_N)
            snapshot[f"{scope.value}:{stat_key}"] = [int(pid) for pid in top["player_id"]]

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN_PATH.open("w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)

    print(f"Wrote {len(snapshot)} (scope, stat) top-{TOP_N} snapshots to {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
