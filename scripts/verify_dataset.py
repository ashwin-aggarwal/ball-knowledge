#!/usr/bin/env python3
"""Structural sanity checks for the built parquet snapshot.

Asserts invariants that must hold regardless of who currently sits atop
any leaderboard (rankings change; we don't hardcode them here): no nulls
in identity columns, no duplicate player ids per table, positive games
played, strictly monotonic ranks when sorting descending, plausible
per-game bounds, and referential integrity against players.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ball_knowledge.config import DATA_DIR, STATS  # noqa: E402

DATA_PATH = Path(DATA_DIR)

# Plausible per-game upper bounds, generous enough to never false-positive
# on a real record but tight enough to catch a units/aggregation bug.
# pts/min/fgm/fga are set just above Wilt Chamberlain's early-60s seasons
# (50.4 PPG in 1961-62, 48.53 MPG the same season on 19.96 FGM/39.5 FGA)
# -- the actual all-time single-season ceilings, not data errors.
PER_GAME_BOUNDS = {
    "pts": 55,
    "reb": 30,
    "oreb": 15,
    "dreb": 20,
    "ast": 15,
    "stl": 5,
    "blk": 6,
    "tov": 8,
    "pf": 6,
    "min": 49,
    "fgm": 22,
    "fga": 42,
    "fg3m": 8,
    "fg3a": 20,
    "ftm": 15,
    "fta": 20,
}

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)
        print(f"  FAIL: {message}")


def verify_identity(name: str, df: pd.DataFrame, id_col: str = "player_id") -> None:
    check(df[id_col].notna().all(), f"{name}: null {id_col}")
    if "full_name" in df.columns:
        check(df["full_name"].notna().all(), f"{name}: null full_name")


def verify_no_dup_player_id(name: str, df: pd.DataFrame) -> None:
    dupe_count = df["player_id"].duplicated().sum()
    check(dupe_count == 0, f"{name}: {dupe_count} duplicate player_id rows")


def verify_games_played(name: str, df: pd.DataFrame) -> None:
    check((df["gp"] > 0).all(), f"{name}: found gp <= 0")


def verify_monotonic_ranks(name: str, df: pd.DataFrame, stat_cols: list[str]) -> None:
    for col in stat_cols:
        if col not in df.columns:
            continue
        non_null = df[df[col].notna()].sort_values(col, ascending=False)
        if len(non_null) < 2:
            continue
        ranks = non_null[col].rank(method="min", ascending=False)
        check(
            ranks.is_monotonic_increasing,
            f"{name}: ranks not monotonic for {col} when sorted descending",
        )


def verify_per_game_bounds(name: str, df: pd.DataFrame, suffix: str) -> None:
    for stat_key, bound in PER_GAME_BOUNDS.items():
        col = f"{stat_key}{suffix}"
        if col not in df.columns:
            continue
        over = df[df[col] > bound]
        check(len(over) == 0, f"{name}: {len(over)} rows exceed plausible bound for {col} (> {bound})")


def verify_referential_integrity(name: str, df: pd.DataFrame, player_ids: set[int]) -> None:
    missing = set(df["player_id"].unique()) - player_ids
    check(len(missing) == 0, f"{name}: {len(missing)} player_ids not present in players.parquet")


def print_top5(df: pd.DataFrame, col: str, label: str) -> None:
    if col not in df.columns:
        print(f"  (skipped: {col} not in table)")
        return
    top = df[df[col].notna()].sort_values(col, ascending=False).head(5)
    cols = ["player_id", col]
    if "full_name" in top.columns:
        cols.insert(1, "full_name")
    print(top[cols].to_string(index=False))


def main() -> None:
    print("Loading parquet files...")
    career_totals = pd.read_parquet(DATA_PATH / "career_totals.parquet")
    career_per_game = pd.read_parquet(DATA_PATH / "career_per_game.parquet")
    season_records = pd.read_parquet(DATA_PATH / "season_records.parquet")
    players = pd.read_parquet(DATA_PATH / "players.parquet")

    total_cols = [f"{k}_total" for k in STATS]
    per_game_cols = [f"{k}_per_game" for k in STATS]

    print("\n=== identity & duplicates ===")
    verify_identity("players", players)
    verify_no_dup_player_id("players", players)
    verify_no_dup_player_id("career_totals", career_totals)
    verify_no_dup_player_id("career_per_game", career_per_game)
    check(
        not season_records.duplicated(subset=["player_id", "season"]).any(),
        "season_records: duplicate (player_id, season) rows",
    )

    print("\n=== games played ===")
    verify_games_played("career_totals", career_totals)
    verify_games_played("career_per_game", career_per_game)
    verify_games_played("season_records", season_records)

    print("\n=== monotonic ranks (career totals) ===")
    verify_monotonic_ranks("career_totals", career_totals, total_cols)

    print("\n=== plausible per-game bounds ===")
    verify_per_game_bounds("career_per_game", career_per_game, "_per_game")
    verify_per_game_bounds("season_records", season_records, "_per_game")

    print("\n=== referential integrity ===")
    player_ids = set(players["player_id"].unique())
    verify_referential_integrity("career_totals", career_totals, player_ids)
    verify_referential_integrity("career_per_game", career_per_game, player_ids)
    verify_referential_integrity("season_records", season_records, player_ids)

    print("\n=== eyeball: top 5 by career points, career rebounds/game, single-season assists ===")
    print("-- career points --")
    print_top5(career_totals.merge(players, on="player_id"), "pts_total", "pts")
    print("-- career rebounds per game --")
    print_top5(career_per_game.merge(players, on="player_id"), "reb_per_game", "reb")
    print("-- single-season assists --")
    print_top5(season_records.merge(players, on="player_id"), "ast_total", "ast")

    print(f"\n{'='*60}")
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
