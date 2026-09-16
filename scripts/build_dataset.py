#!/usr/bin/env python3
"""One-time local build of the ball-knowledge NBA snapshot.

Pulls candidate players from nba_api's AllTimeLeadersGrids endpoint, then
career totals for each from PlayerCareerStats (season-by-season rows are
fetched too, since they're bundled in the same call, but only used to
compute each player's first/last season -- not written out, since
single-season questions were cut, and per-game/rate stats aren't part of
the game at all). Writes data/career_totals.parquet, data/players.parquet
and data/manifest.json.

Every raw API response is cached to .cache/nba/ as JSON, keyed by endpoint
and parameters, so a rerun after a dropped connection resumes instead of
re-fetching everything. Run with --limit N for a fast smoke test before
committing to the full ~20-40 minute pull.

Requires requirements-build.txt (nba_api). Not part of the runtime app.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ball_knowledge.config import (  # noqa: E402
    ALL_TIME_LEADERS_TOP_X,
    CACHE_DIR,
    DATA_DIR,
    DEFAULT_SLEEP_SECONDS,
    HEADSHOT_URL_TEMPLATE,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    STATS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_dataset")

# Counting-stat leader categories used to define the candidate player pool.
# Percentage categories (FG_PCT, FG3_PCT, FT_PCT) are deliberately excluded:
# they surface tiny-sample efficiency specialists that are irrelevant to a
# counting-stat guessing game and would just bloat the pool.
POOL_LEADER_CATEGORIES = [
    "GPLeaders",
    "PTSLeaders",
    "ASTLeaders",
    "STLLeaders",
    "OREBLeaders",
    "DREBLeaders",
    "REBLeaders",
    "BLKLeaders",
    "FGMLeaders",
    "FGALeaders",
    "TOVLeaders",
    "FG3MLeaders",
    "FG3ALeaders",
    "PFLeaders",
    "FTMLeaders",
    "FTALeaders",
]

# Our stat key -> the API's column name for it in PlayerCareerStats results.
STAT_API_COL = {
    "pts": "PTS",
    "reb": "REB",
    "oreb": "OREB",
    "dreb": "DREB",
    "ast": "AST",
    "stl": "STL",
    "blk": "BLK",
    "tov": "TOV",
    "pf": "PF",
    "min": "MIN",
    "fgm": "FGM",
    "fga": "FGA",
    "fg3m": "FG3M",
    "fg3a": "FG3A",
    "ftm": "FTM",
    "fta": "FTA",
}


class PermanentFetchError(Exception):
    """A call failed after exhausting all retries."""


@dataclass
class BuildStats:
    network_calls: int = 0
    cache_hits: int = 0
    failures: list[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []


def cache_path(cache_dir: Path, endpoint: str, key: str) -> Path:
    return cache_dir / endpoint / f"{key}.json"


def fetch_json_cached(
    *,
    cache_dir: Path,
    endpoint: str,
    key: str,
    fetch_fn,
    sleep_seconds: float,
    max_retries: int,
    stats: BuildStats,
    description: str,
) -> dict[str, Any] | None:
    """Fetch `fetch_fn()` -> normalized dict, cached to disk by (endpoint, key).

    On any failure after `max_retries` attempts, logs the failure (with
    description, endpoint and key) to `stats.failures` and returns None
    rather than raising, so one bad player id cannot crash the whole build.
    Retryable on the next run since nothing is cached for a failed key.
    """
    path = cache_path(cache_dir, endpoint, key)
    if path.exists():
        stats.cache_hits += 1
        with path.open("r") as f:
            return json.load(f)

    backoff = sleep_seconds
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            data = fetch_fn()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                json.dump(data, f)
            stats.network_calls += 1
            time.sleep(sleep_seconds)
            return data
        except (
            requests.exceptions.RequestException,
            json.JSONDecodeError,
            ValueError,
            KeyError,
        ) as exc:
            last_exc = exc
            log.warning(
                "  retry %d/%d for %s (%s key=%s): %s",
                attempt,
                max_retries,
                description,
                endpoint,
                key,
                exc,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

    stats.failures.append(
        {
            "endpoint": endpoint,
            "key": key,
            "description": description,
            "error": str(last_exc),
        }
    )
    log.error(
        "  PERMANENT FAILURE for %s (%s key=%s) after %d attempts: %s",
        description,
        endpoint,
        key,
        max_retries,
        last_exc,
    )
    return None


def fetch_candidate_pool(
    cache_dir: Path, timeout: int, sleep_seconds: float, max_retries: int, stats: BuildStats
) -> set[int]:
    """Union of player ids across all-time counting-stat leader categories."""
    from nba_api.stats.endpoints import alltimeleadersgrids

    def _fetch() -> dict[str, Any]:
        g = alltimeleadersgrids.AllTimeLeadersGrids(
            topx=ALL_TIME_LEADERS_TOP_X, timeout=timeout
        )
        return g.nba_response.get_normalized_dict()

    data = fetch_json_cached(
        cache_dir=cache_dir,
        endpoint="alltimeleadersgrids",
        key=f"topx_{ALL_TIME_LEADERS_TOP_X}",
        fetch_fn=_fetch,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
        stats=stats,
        description="all-time leaders pool",
    )
    if data is None:
        raise SystemExit("Could not fetch the all-time leaders pool; aborting.")

    ids: set[int] = set()
    for category in POOL_LEADER_CATEGORIES:
        for row in data.get(category, []):
            ids.add(int(row["PLAYER_ID"]))
    return ids


def fetch_player_career(
    player_id: int,
    cache_dir: Path,
    timeout: int,
    sleep_seconds: float,
    max_retries: int,
    stats: BuildStats,
) -> dict[str, Any] | None:
    from nba_api.stats.endpoints import playercareerstats

    def _fetch() -> dict[str, Any]:
        p = playercareerstats.PlayerCareerStats(player_id=player_id, timeout=timeout)
        full = p.nba_response.get_normalized_dict()
        # Keep only what we use, to keep cache files small.
        return {
            "CareerTotalsRegularSeason": full.get("CareerTotalsRegularSeason", []),
            "SeasonTotalsRegularSeason": full.get("SeasonTotalsRegularSeason", []),
        }

    return fetch_json_cached(
        cache_dir=cache_dir,
        endpoint="playercareerstats",
        key=str(player_id),
        fetch_fn=_fetch,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
        stats=stats,
        description=f"player {player_id} career stats",
    )


def check_headshot_available(
    player_id: int, cache_dir: Path, timeout: int, sleep_seconds: float
) -> bool:
    """HEAD-check the headshot CDN for this player, cached to disk.

    Not run through the retry/failure-logging machinery above: a headshot
    miss is an expected, common outcome (most pre-2000 players 404), not a
    build failure. Any request problem here is treated as "not available"
    and does not block the build.
    """
    path = cache_dir / "headshot_check" / f"{player_id}.txt"
    if path.exists():
        return path.read_text().strip() == "1"

    url = HEADSHOT_URL_TEMPLATE.format(player_id=player_id)
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        available = resp.status_code == 200
    except requests.exceptions.RequestException as exc:
        log.warning("  headshot check failed for %s: %s", player_id, exc)
        available = False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("1" if available else "0")
    time.sleep(sleep_seconds)
    return available


def dedupe_season_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse multi-team season stints to one row per (player, season).

    nba_api returns one row per team a player suited up for in a season,
    plus a TEAM_ID == 0 "TOT" row with the combined totals when there was
    more than one team. Prefer TOT when present; otherwise there is only
    one row already.
    """
    by_season: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_season.setdefault(row["SEASON_ID"], []).append(row)

    result = []
    for season, season_rows in by_season.items():
        if len(season_rows) == 1:
            result.append(season_rows[0])
            continue
        tot = next((r for r in season_rows if r["TEAM_ID"] == 0), None)
        result.append(tot if tot is not None else season_rows[0])
    return result


def build_tables(
    candidate_ids: list[int],
    careers: dict[int, dict[str, Any]],
    static_by_id: dict[int, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    career_total_rows = []
    player_rows = []

    for pid in candidate_ids:
        data = careers.get(pid)
        if not data:
            continue
        career_list = data["CareerTotalsRegularSeason"]
        if not career_list:
            continue
        career = career_list[0]
        gp = career.get("GP") or 0
        if gp <= 0:
            continue

        row_total: dict[str, Any] = {"player_id": pid, "gp": gp}
        for stat_key, api_col in STAT_API_COL.items():
            row_total[f"{stat_key}_total"] = career.get(api_col)
        career_total_rows.append(row_total)

        # Season data is still fetched (it's bundled in the same
        # PlayerCareerStats call as career totals) and used here only for
        # first/last season -- not written out as its own table, since
        # single-season questions were cut entirely.
        seasons = dedupe_season_rows(data["SeasonTotalsRegularSeason"])
        season_ids = [s["SEASON_ID"] for s in seasons if (s.get("GP") or 0) > 0]

        static_info = static_by_id.get(pid, {})
        player_rows.append(
            {
                "player_id": pid,
                "full_name": static_info.get("full_name", career.get("PLAYER_NAME", "")),
                "is_active": bool(static_info.get("is_active", False)),
                "first_season": min(season_ids) if season_ids else None,
                "last_season": max(season_ids) if season_ids else None,
            }
        )

    career_totals_df = pd.DataFrame(career_total_rows)
    players_df = pd.DataFrame(player_rows)
    return career_totals_df, players_df


def attach_headshot_flags(
    players_df: pd.DataFrame, cache_dir: Path, timeout: int, sleep_seconds: float
) -> pd.DataFrame:
    n = len(players_df)
    flags = []
    start = time.time()
    for i, pid in enumerate(players_df["player_id"], start=1):
        flags.append(check_headshot_available(int(pid), cache_dir, timeout, sleep_seconds))
        if i % 50 == 0 or i == n:
            elapsed = time.time() - start
            rate = elapsed / i
            eta = rate * (n - i)
            log.info("  headshot check %d/%d (eta %ds)", i, n, int(eta))
    players_df = players_df.copy()
    players_df["headshot_available"] = flags
    return players_df


ROW_COUNT_DROP_TOLERANCE = 0.20  # a table shrinking more than 20% aborts the write


def check_row_count_sanity(
    data_dir: Path, new_row_counts: dict[str, int], allow_shrink: bool
) -> None:
    """Refuse to overwrite the dataset if any table shrank suspiciously.

    Compares against the *previous* manifest.json's row_counts, read
    before this build's parquet files are written -- a table that
    suddenly halves in size usually means a partial fetch (dropped
    connection, an early exit) got written as if it were a complete
    build, not a real change in the underlying data. `--limit` runs are
    expected to look like a drop and set `allow_shrink` themselves.
    """
    old_manifest_path = data_dir / "manifest.json"
    if not old_manifest_path.exists():
        return
    with old_manifest_path.open("r") as f:
        old_row_counts = json.load(f).get("row_counts", {})

    problems = []
    for table, new_count in new_row_counts.items():
        old_count = old_row_counts.get(table)
        if not old_count:
            continue
        drop = (old_count - new_count) / old_count
        if drop > ROW_COUNT_DROP_TOLERANCE:
            problems.append(
                f"{table}: {old_count} -> {new_count} rows "
                f"({drop:.0%} drop, tolerance is {ROW_COUNT_DROP_TOLERANCE:.0%})"
            )

    if problems and not allow_shrink:
        message = (
            "Refusing to write: row count(s) dropped more than "
            f"{ROW_COUNT_DROP_TOLERANCE:.0%} versus the previous build "
            "(looks like a partial fetch, not a real change):\n  "
            + "\n  ".join(problems)
            + "\nIf this drop is real and expected (e.g. a deliberate --limit run "
            "or a genuine data cleanup), pass --allow-row-count-shrink."
        )
        log.error(message)
        raise SystemExit(1)
    if problems:
        log.warning(
            "Row count drop(s) allowed via --allow-row-count-shrink:\n  " + "\n  ".join(problems)
        )


def write_manifest(
    data_dir: Path,
    career_totals_df: pd.DataFrame,
    players_df: pd.DataFrame,
    candidate_pool_size: int,
    stats: BuildStats,
) -> None:
    era_caveats = {
        key: stat_def.era_caveat()
        for key, stat_def in STATS.items()
        if stat_def.era_caveat() is not None
    }
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "candidate_pool_size": candidate_pool_size,
        "row_counts": {
            "career_totals": len(career_totals_df),
            "players": len(players_df),
        },
        # Coverage of career first/last seasons across all candidates.
        # No season-by-season table is written (single-season questions
        # were cut), but this still tells us how far back the underlying
        # data reaches.
        "season_coverage": {
            "min_season": (
                players_df["first_season"].min() if not players_df.empty else None
            ),
            "max_season": (
                players_df["last_season"].max() if not players_df.empty else None
            ),
        },
        "endpoints_used": [
            "nba_api.stats.endpoints.alltimeleadersgrids.AllTimeLeadersGrids",
            "nba_api.stats.endpoints.playercareerstats.PlayerCareerStats",
            "nba_api.stats.static.players.get_players",
        ],
        "era_caveats": era_caveats,
        "network_calls": stats.network_calls,
        "cache_hits": stats.cache_hits,
        "permanent_failures": stats.failures,
    }
    with (data_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N candidate players"
    )
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument(
        "--skip-headshot-check",
        action="store_true",
        help="Skip the per-player CDN HEAD-check pass (faster smoke runs)",
    )
    parser.add_argument(
        "--allow-row-count-shrink",
        action="store_true",
        help=(
            "Allow writing even if a table's row count dropped more than "
            f"{ROW_COUNT_DROP_TOLERANCE:.0%} versus the previous build "
            "(expected for a deliberate --limit run)"
        ),
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    stats = BuildStats()

    from nba_api.stats.static import players as static_players

    log.info("Loading static player list (no network call)...")
    static_by_id = {p["id"]: p for p in static_players.get_players()}
    log.info("  %d players in static list", len(static_by_id))

    log.info("Fetching all-time leaders pool (topx=%d)...", ALL_TIME_LEADERS_TOP_X)
    candidate_ids = sorted(
        fetch_candidate_pool(cache_dir, args.timeout, args.sleep, args.max_retries, stats)
    )
    log.info("  candidate pool: %d players", len(candidate_ids))

    if args.limit is not None:
        candidate_ids = candidate_ids[: args.limit]
        log.info("  --limit applied: processing %d players", len(candidate_ids))

    log.info("Fetching career + season stats for each candidate...")
    careers: dict[int, dict[str, Any]] = {}
    n = len(candidate_ids)
    start = time.time()
    for i, pid in enumerate(candidate_ids, start=1):
        data = fetch_player_career(pid, cache_dir, args.timeout, args.sleep, args.max_retries, stats)
        if data is not None:
            careers[pid] = data
        if i % 25 == 0 or i == n:
            elapsed = time.time() - start
            rate = elapsed / i
            eta = rate * (n - i)
            log.info(
                "  %d/%d players (network calls=%d, cache hits=%d, eta %ds)",
                i,
                n,
                stats.network_calls,
                stats.cache_hits,
                int(eta),
            )

    log.info("Building tables...")
    career_totals_df, players_df = build_tables(candidate_ids, careers, static_by_id)
    log.info(
        "  career_totals=%d players=%d",
        len(career_totals_df),
        len(players_df),
    )

    if not args.skip_headshot_check and not players_df.empty:
        log.info("Checking headshot CDN availability for %d players...", len(players_df))
        players_df = attach_headshot_flags(players_df, cache_dir, args.timeout, args.sleep)
    elif not players_df.empty:
        players_df = players_df.copy()
        players_df["headshot_available"] = None

    check_row_count_sanity(
        data_dir,
        {
            "career_totals": len(career_totals_df),
            "players": len(players_df),
        },
        allow_shrink=args.allow_row_count_shrink or args.limit is not None,
    )

    log.info("Writing parquet files to %s ...", data_dir)
    career_totals_df.to_parquet(data_dir / "career_totals.parquet", index=False)
    players_df.to_parquet(data_dir / "players.parquet", index=False)

    write_manifest(
        data_dir,
        career_totals_df,
        players_df,
        len(candidate_ids),
        stats,
    )

    if stats.failures:
        log.warning(
            "%d permanent failures recorded in manifest.json; rerun the build to retry them "
            "(cache means already-succeeded players won't be re-fetched).",
            len(stats.failures),
        )
    log.info("Done. network_calls=%d cache_hits=%d", stats.network_calls, stats.cache_hits)


if __name__ == "__main__":
    main()
