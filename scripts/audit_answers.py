#!/usr/bin/env python3
"""Cross-check parquet leaderboards against nba_api's live official leaders.

Local-only verification tool: NOT part of the runtime app, requires
requirements-build.txt (nba_api) and live network access to stats.nba.com.
Run manually, before trusting a build, to catch a wrong answer that would
otherwise ship silently. Self-consistency checks (verify_dataset.py) are
not enough -- a ranking bug can be internally consistent and still wrong,
so this compares against a source we did not build: NBA.com's own
official all-time leaderboards, fetched fresh at audit time.

Exits nonzero if any mismatch is found, so it can gate a build. Prints
every mismatch (rank, expected player+value, actual player+value) --
never summarizes them away.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ball_knowledge.config import DATA_DIR  # noqa: E402

DATA_PATH = Path(DATA_DIR)
TOP_N = 50

# Our stat key -> AllTimeLeadersGrids column name / result-set prefix.
# Same mapping as scripts/build_dataset.py's STAT_API_COL (kept as an
# independent copy deliberately: this script must not trust anything the
# build script assumes).
STAT_API_COL = {
    "pts": "PTS",
    "reb": "REB",
    "ast": "AST",
    "stl": "STL",
    "blk": "BLK",
    "oreb": "OREB",
    "dreb": "DREB",
    "fg3m": "FG3M",
}

# (our table file, our value column suffix, AllTimeLeadersGrids per_mode_simple,
#  min career games floor -- 1 here since career totals has no floor).
CHECKED_SCOPES = [
    ("career_totals.parquet", "_total", "Totals", 1),
]

# The only scope (career totals) is checked above. Kept as a list (rather
# than removed) so a future scope this script can't check against a live
# source has an obvious place to be declared, loudly, rather than
# silently skipped.
UNVERIFIED_SCOPES: list[tuple[str, str]] = []


def fetch_official_top_n(per_mode: str) -> dict[str, list[tuple[int, int, str, float]]]:
    """official[stat_key] = [(rank, player_id, player_name, value), ...].

    Uses the API's own *_RANK field, not list position: a tie (two players
    sharing a rank) leaves a gap in the rank sequence (..., 29, 29, 31, ...),
    and treating list position as rank silently misaligns every subsequent
    comparison past the first tie.
    """
    from nba_api.stats.endpoints import alltimeleadersgrids

    g = alltimeleadersgrids.AllTimeLeadersGrids(topx=TOP_N, per_mode_simple=per_mode, timeout=30)
    data = g.nba_response.get_normalized_dict()
    official: dict[str, list[tuple[int, int, str, float]]] = {}
    for stat_key, api_col in STAT_API_COL.items():
        result_set = data.get(f"{api_col}Leaders", [])
        rows = sorted(result_set, key=lambda r: r[f"{api_col}_RANK"])
        official[stat_key] = [
            (int(r[f"{api_col}_RANK"]), int(r["PLAYER_ID"]), str(r["PLAYER_NAME"]), float(r[api_col]))
            for r in rows
        ]
    return official


def audit_scope(parquet_file: str, value_suffix: str, per_mode: str, min_games: int) -> list[dict]:
    print(
        f"\n=== {parquet_file} (min_games={min_games}) vs. official "
        f"AllTimeLeadersGrids(per_mode_simple={per_mode!r}) ==="
    )
    table = pd.read_parquet(DATA_PATH / parquet_file)
    players = pd.read_parquet(DATA_PATH / "players.parquet")
    merged = table.merge(players[["player_id", "full_name"]], on="player_id")

    official = fetch_official_top_n(per_mode)
    mismatches = []
    gp_by_pid = dict(zip(merged["player_id"], merged["gp"]))

    for stat_key in STAT_API_COL:
        value_col = f"{stat_key}{value_suffix}"
        if value_col not in merged.columns:
            print(f"  {stat_key}: SKIPPED (no column {value_col})")
            continue
        official_rows = official.get(stat_key, [])
        if not official_rows:
            print(f"  {stat_key}: SKIPPED (no official data returned)")
            continue

        eligible = merged[(merged["gp"] >= min_games) & merged[value_col].notna()].copy()
        eligible["rank"] = eligible[value_col].rank(method="min", ascending=False).astype(int)
        our_rank_by_pid = dict(zip(eligible["player_id"], eligible["rank"]))
        our_value_by_pid = dict(zip(eligible["player_id"], eligible[value_col]))

        # NBA.com's own per-game leaderboard applies no visible games floor;
        # ours deliberately does (min_games, a brief-specified config value)
        # to stop a short career from dominating a rate stat. A player who
        # fails OUR floor is expected to be entirely absent from our ranking
        # -- a documented design choice, not a mismatch -- so drop them from
        # the official list, then RE-RANK what's left before comparing.
        # Comparing raw official rank numbers against ours after dropping
        # entries would silently misalign every position past a drop.
        excluded_by_games_floor = 0
        filtered_official = []
        for official_rank, exp_pid, exp_name, exp_val in official_rows:
            if official_rank > TOP_N:
                continue
            if min_games > 1 and gp_by_pid.get(exp_pid, 0) < min_games:
                excluded_by_games_floor += 1
                continue
            filtered_official.append((official_rank, exp_pid, exp_name, exp_val))

        rerank_position = 0
        prev_original_rank = None
        stat_mismatches = 0
        checked_ranks = 0
        for original_rank, exp_pid, exp_name, exp_val in filtered_official:
            checked_ranks += 1
            rerank_position += 1
            # Tie-detection must use the API's own rank field (its ordering
            # reflects full unrounded precision), not the rounded display
            # value: two players can show the identical rounded value
            # ("30.1") while the API still ranks them 1st and 2nd.
            if prev_original_rank is None or original_rank != prev_original_rank:
                official_rerank = rerank_position
            # else: tied with previous -> shares the same official_rerank
            prev_original_rank = original_rank

            our_rank = our_rank_by_pid.get(exp_pid)
            our_val = our_value_by_pid.get(exp_pid)
            if our_rank == official_rerank:
                continue
            stat_mismatches += 1
            mismatches.append(
                {
                    "table": parquet_file,
                    "stat": stat_key,
                    "official_rank": official_rerank,
                    "expected_player": exp_name,
                    "expected_player_id": exp_pid,
                    "expected_value": exp_val,
                    "our_rank": our_rank,
                    "our_value": our_val,
                }
            )
            print(
                f"  MISMATCH {stat_key} rank {official_rerank}: "
                f"expected {exp_name} (id={exp_pid}, official={exp_val:g}) "
                f"but our table has them at rank={our_rank if our_rank is not None else 'MISSING'} "
                f"value={our_val if our_val is not None else '-'}"
            )
        excl_note = f", {excluded_by_games_floor} excluded by our games floor" if excluded_by_games_floor else ""
        if stat_mismatches == 0:
            print(f"  {stat_key}: OK ({checked_ranks} official ranks match position-by-position{excl_note})")

    return mismatches


def main() -> None:
    print(f"Auditing top {TOP_N} against nba_api live data...")
    all_mismatches: list[dict] = []
    for parquet_file, value_suffix, per_mode, min_games in CHECKED_SCOPES:
        all_mismatches.extend(audit_scope(parquet_file, value_suffix, per_mode, min_games))

    if UNVERIFIED_SCOPES:
        print("\n=== Unverified scopes ===")
        for scope_name, reason in UNVERIFIED_SCOPES:
            print(f"  {scope_name}: UNVERIFIED -- {reason}")

    print(f"\n{'=' * 70}")
    if all_mismatches:
        print(f"{len(all_mismatches)} MISMATCH(ES) FOUND across {len(CHECKED_SCOPES)} checked scopes.")
        sys.exit(1)
    print(f"No mismatches in {len(CHECKED_SCOPES)} checked scopes ({len(STAT_API_COL)} stats each).")
    if UNVERIFIED_SCOPES:
        print(f"{len(UNVERIFIED_SCOPES)} scope(s) remain unverified (see above).")


if __name__ == "__main__":
    main()
