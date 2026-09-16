#!/usr/bin/env python3
"""Structural sanity checks for the built parquet snapshot.

Asserts invariants that must hold regardless of who currently sits atop
any leaderboard (rankings change; we don't hardcode them here): no nulls
in identity columns, no duplicate player ids, positive games played,
strictly monotonic ranks when sorting descending, numeric dtypes,
made<=attempted, oreb+dreb==reb (careers fully within the tracking era),
and referential integrity against players.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ball_knowledge.config import DATA_DIR, STATS  # noqa: E402

DATA_PATH = Path(DATA_DIR)

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
    name: str, df: pd.DataFrame, careers_fully_in_tracking_era: pd.Series
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
    if not all(c in df.columns for c in ("oreb_total", "dreb_total", "reb_total")):
        return
    eligible_ids = careers_fully_in_tracking_era[careers_fully_in_tracking_era].index
    both = df[
        df["player_id"].isin(eligible_ids)
        & df["oreb_total"].notna()
        & df["dreb_total"].notna()
        & df["reb_total"].notna()
    ]
    diff = (both["oreb_total"] + both["dreb_total"] - both["reb_total"]).abs()
    bad = both[diff > 1e-6]
    check(
        len(bad) == 0,
        f"{name}: {len(bad)} rows (careers fully within 1973-74+) where "
        "oreb_total+dreb_total != reb_total",
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
    players = pd.read_parquet(DATA_PATH / "players.parquet")

    total_cols = [f"{k}_total" for k in STATS]

    print("\n=== identity & duplicates ===")
    verify_identity("players", players)
    verify_no_dup_player_id("players", players)
    verify_no_dup_player_id("career_totals", career_totals)

    print("\n=== games played ===")
    verify_games_played("career_totals", career_totals)

    print("\n=== numeric dtypes ===")
    verify_numeric_dtypes("career_totals", career_totals, total_cols + ["gp"])

    print("\n=== monotonic ranks ===")
    verify_monotonic_ranks("career_totals", career_totals, total_cols)

    print("\n=== made <= attempted ===")
    for made, att in [("fgm", "fga"), ("fg3m", "fg3a"), ("ftm", "fta")]:
        verify_made_le_attempted("career_totals", career_totals, f"{made}_total", f"{att}_total")

    print("\n=== offensive + defensive rebounds = total rebounds (careers fully in tracking era) ===")
    fully_tracked = players.set_index("player_id")["first_season"].fillna("") >= "1973-74"
    verify_oreb_dreb_sum_to_reb("career_totals", career_totals, fully_tracked)

    print("\n=== referential integrity ===")
    player_ids = set(players["player_id"].unique())
    verify_referential_integrity("career_totals", career_totals, player_ids)

    print("\n=== eyeball: top 5 by career points, rebounds, assists ===")
    print("-- career points --")
    print_top5(career_totals.merge(players, on="player_id"), "pts_total", "pts")
    print("-- career rebounds --")
    print_top5(career_totals.merge(players, on="player_id"), "reb_total", "reb")
    print("-- career assists --")
    print_top5(career_totals.merge(players, on="player_id"), "ast_total", "ast")

    print(f"\n{'='*60}")
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
