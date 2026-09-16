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


def verify_numeric_dtypes(name: str, df: pd.DataFrame, cols: list[str]) -> None:
    """A stat column typed as object/string would sort lexicographically
    ('9' > '10') instead of by value, silently producing a wrong rank."""
    for col in cols:
        if col not in df.columns:
            continue
        check(
            pd.api.types.is_numeric_dtype(df[col]),
            f"{name}: column {col} has non-numeric dtype {df[col].dtype}",
        )


def verify_made_le_attempted(name: str, df: pd.DataFrame, made_col: str, att_col: str) -> None:
    if made_col not in df.columns or att_col not in df.columns:
        return
    both = df[df[made_col].notna() & df[att_col].notna()]
    bad = both[both[made_col] > both[att_col]]
    check(len(bad) == 0, f"{name}: {len(bad)} rows have {made_col} > {att_col}")


def verify_oreb_dreb_sum_to_reb(
    name: str, df: pd.DataFrame, suffix: str, careers_fully_in_tracking_era: pd.Series
) -> None:
    """oreb+dreb should equal reb exactly -- but only for a player whose
    *entire* career falls within the OREB/DREB tracking era (1973-74+).
    For a career straddling that boundary, reb_total covers their whole
    career while oreb_total+dreb_total only covers the tracked portion by
    construction (see build_dataset.py) -- that's correct, not a
    violation, and checking it at the career level for a straddling
    player would always show a "deficit" equal to their pre-1973-74
    rebounds. `careers_fully_in_tracking_era` (indexed by player_id)
    restricts the check to players it can't produce a false positive for.
    """
    oreb_col, dreb_col, reb_col = f"oreb{suffix}", f"dreb{suffix}", f"reb{suffix}"
    if not all(c in df.columns for c in (oreb_col, dreb_col, reb_col)):
        return
    eligible_ids = careers_fully_in_tracking_era[careers_fully_in_tracking_era].index
    both = df[
        df["player_id"].isin(eligible_ids)
        & df[oreb_col].notna()
        & df[dreb_col].notna()
        & df[reb_col].notna()
    ]
    diff = (both[oreb_col] + both[dreb_col] - both[reb_col]).abs()
    bad = both[diff > 1e-6]
    check(
        len(bad) == 0,
        f"{name}: {len(bad)} rows (careers fully within 1973-74+) where "
        f"{oreb_col}+{dreb_col} != {reb_col}",
    )


def verify_per_game_reconciles_with_total(
    name: str, df: pd.DataFrame, always_tracked_stats: list[str]
) -> None:
    """total/gp should equal the stored per-game value, for stats tracked
    the league's entire history. Era-gated stats are deliberately excluded:
    their per-game denominator is tracked-seasons-only games, not blanket
    career gp (see build_dataset.py), so this exact check doesn't apply to
    them -- that distinction is exactly the bug scripts/audit_answers.py
    caught previously; this check guards the always-tracked stats where
    the two really must agree."""
    for stat_key in always_tracked_stats:
        total_col, pg_col = f"{stat_key}_total", f"{stat_key}_per_game"
        if total_col not in df.columns or pg_col not in df.columns:
            continue
        both = df[df[total_col].notna() & df[pg_col].notna() & (df["gp"] > 0)]
        recomputed = both[total_col] / both["gp"]
        diff = (recomputed - both[pg_col]).abs()
        bad = both[diff > 1e-4]
        check(
            len(bad) == 0,
            f"{name}: {len(bad)} rows where {pg_col} doesn't reconcile with {total_col}/gp",
        )


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
    players = pd.read_parquet(DATA_PATH / "players.parquet")

    total_cols = [f"{k}_total" for k in STATS]
    per_game_cols = [f"{k}_per_game" for k in STATS if STATS[k].per_game_col is not None]
    always_tracked_stats = [k for k, v in STATS.items() if v.tracked_since is None]

    print("\n=== identity & duplicates ===")
    verify_identity("players", players)
    verify_no_dup_player_id("players", players)
    verify_no_dup_player_id("career_totals", career_totals)
    verify_no_dup_player_id("career_per_game", career_per_game)

    print("\n=== games played ===")
    verify_games_played("career_totals", career_totals)
    verify_games_played("career_per_game", career_per_game)

    print("\n=== numeric dtypes ===")
    verify_numeric_dtypes("career_totals", career_totals, total_cols + ["gp"])
    verify_numeric_dtypes("career_per_game", career_per_game, per_game_cols + ["gp"])

    print("\n=== monotonic ranks (career totals) ===")
    verify_monotonic_ranks("career_totals", career_totals, total_cols)

    print("\n=== plausible per-game bounds ===")
    verify_per_game_bounds("career_per_game", career_per_game, "_per_game")

    print("\n=== per-game reconciles with total/gp (always-tracked stats) ===")
    verify_per_game_reconciles_with_total("career_per_game", career_per_game, always_tracked_stats)

    print("\n=== made <= attempted ===")
    for made, att in [("fgm", "fga"), ("fg3m", "fg3a"), ("ftm", "fta")]:
        verify_made_le_attempted("career_totals", career_totals, f"{made}_total", f"{att}_total")
        verify_made_le_attempted(
            "career_per_game", career_per_game, f"{made}_per_game", f"{att}_per_game"
        )

    print("\n=== offensive + defensive rebounds = total rebounds (careers fully in tracking era) ===")
    fully_tracked = (
        players.set_index("player_id")["first_season"].fillna("") >= "1973-74"
    )
    verify_oreb_dreb_sum_to_reb("career_totals", career_totals, "_total", fully_tracked)
    verify_oreb_dreb_sum_to_reb("career_per_game", career_per_game, "_per_game", fully_tracked)

    print("\n=== referential integrity ===")
    player_ids = set(players["player_id"].unique())
    verify_referential_integrity("career_totals", career_totals, player_ids)
    verify_referential_integrity("career_per_game", career_per_game, player_ids)

    print("\n=== eyeball: top 5 by career points, career rebounds/game, career assists/game ===")
    print("-- career points --")
    print_top5(career_totals.merge(players, on="player_id"), "pts_total", "pts")
    print("-- career rebounds per game --")
    print_top5(career_per_game.merge(players, on="player_id"), "reb_per_game", "reb")
    print("-- career assists per game --")
    print_top5(career_per_game.merge(players, on="player_id"), "ast_per_game", "ast")

    print(f"\n{'='*60}")
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
