#!/usr/bin/env python3
"""Calibrate the GeoGuessr-style scoring curve against the real dataset.

TAU (and the other scoring constants in config.py) cannot be chosen by
reasoning alone -- this generates a batch of real questions, simulates
guesses at controlled quality levels against each, and prints the
resulting score distribution so a human can read it and decide whether
the curve feels right, rather than trusting the exponent by feel.

Not part of the runtime app. Re-run after any change to TAU,
MAX_ROUND_SCORE, MIN_ROUND_SCORE, CLOSEST_BONUS, or the density window,
and read the output before shipping a new value.
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ball_knowledge.config import DEFAULT_GAME_CONFIG  # noqa: E402
from ball_knowledge.data import load_tables  # noqa: E402
from ball_knowledge.questions import eligible_players_for_question, generate_question  # noqa: E402
from ball_knowledge.scoring import GuessInput, score_round  # noqa: E402

# (label, rank-position offset from the answer, or None for "uniformly random").
QUALITY_LEVELS: list[tuple[str, int | None]] = [
    ("exact", 0),
    ("within 2", 2),
    ("within 5", 5),
    ("within 15", 15),
    ("within 50", 50),
    ("random", None),
]


def _offset_row(pool, target_rank: int, offset: int):
    """The pool row `offset` positions away from `target_rank` (1-indexed),
    clamped to the table's edges and falling back to the other direction
    if the table is too short in the first direction tried."""
    n = len(pool)
    pos = target_rank - 1 + offset
    if pos >= n:
        pos = max(0, target_rank - 1 - offset)
    pos = max(0, min(n - 1, pos))
    return pool.iloc[pos]


def _simulate_guess(pool, target_rank: int, level_offset: int | None, rng: random.Random) -> GuessInput:
    if level_offset is None:
        row = pool.iloc[rng.randrange(len(pool))]
    else:
        row = _offset_row(pool, target_rank, level_offset)
    return GuessInput(
        guesser_name="sim",
        nba_player_id=int(row["player_id"]),
        nba_player_name=str(row["full_name"]),
        value=float(row["value"]),
        rank=int(row["rank"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tables = load_tables()
    config = DEFAULT_GAME_CONFIG
    rng = random.Random(args.seed)

    used_keys: set = set()
    recent_stats: list[str] = []
    recent_scopes: list[str] = []

    # scores[quality][scope] -> list[int]; scores_by_stat[quality][stat] -> list[int]
    scores: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    scores_by_stat: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    print(f"Generating {args.questions} questions (seed={args.seed})...\n")
    for round_number in range(1, args.questions + 1):
        q = generate_question(
            tables,
            config,
            used_keys,
            recent_stats=recent_stats,
            recent_scopes=recent_scopes,
            round_number=round_number,
            total_rounds=args.questions,
            rng=rng,
        )
        used_keys.add(q.key)
        recent_stats.append(q.stat_key)
        recent_scopes.append(q.scope.value)

        pool = eligible_players_for_question(q, tables)
        answer_value = q.anchor_value if q.scoring_mode == "value" else q.answer_value

        for label, offset in QUALITY_LEVELS:
            if label == "exact":
                guess = GuessInput(
                    guesser_name="sim",
                    nba_player_id=q.answer_player_id,
                    nba_player_name=q.answer_player_name,
                    value=q.answer_value,
                    rank=q.target_rank,
                )
            else:
                guess = _simulate_guess(pool, q.target_rank, offset, rng)
            scored = score_round(
                [guess],
                pool=pool,
                target_rank=q.target_rank,
                answer_value=answer_value,
                answer_player_id=q.answer_player_id,
                max_round_score=config.max_round_score,
                min_round_score=config.min_round_score,
                closest_bonus=0,  # bonus depends on other guessers this round; irrelevant here
                tau=config.score_tau,
                window=config.score_window_radius,
            )
            points = scored[0].points
            scores[label][q.scope.value].append(points)
            scores_by_stat[label][q.stat_key].append(points)

    print(f"{'=' * 78}")
    print(
        f"Config: MAX_ROUND_SCORE={config.max_round_score}  "
        f"MIN_ROUND_SCORE={config.min_round_score}  TAU={config.score_tau}  "
        f"window={config.score_window_radius}\n"
    )

    print("-- mean / median score per quality level, per scope --")
    header = f"{'quality':12}" + "".join(f"{s:>16}" for s in sorted({s for d in scores.values() for s in d}))
    print(header)
    overall_mean_by_level: dict[str, float] = {}
    for label, _ in QUALITY_LEVELS:
        by_scope = scores[label]
        all_points = [p for pts in by_scope.values() for p in pts]
        overall_mean_by_level[label] = statistics.mean(all_points) if all_points else float("nan")
        row = f"{label:12}"
        for scope in sorted(by_scope):
            pts = by_scope[scope]
            row += f"{statistics.mean(pts):8.1f}/{statistics.median(pts):<6.0f}"
        print(row)

    print("\n-- flagged scopes (mean deviates >15% from the cross-scope mean for that quality level) --")
    flagged_any = False
    for label, _ in QUALITY_LEVELS:
        overall = overall_mean_by_level[label]
        if not overall:
            continue
        for scope, pts in scores[label].items():
            scope_mean = statistics.mean(pts)
            deviation = abs(scope_mean - overall) / overall if overall else 0.0
            if deviation > 0.15:
                flagged_any = True
                print(
                    f"  {label:12} scope={scope:16} mean={scope_mean:7.1f} vs "
                    f"overall={overall:7.1f} ({deviation:.0%} off)"
                )
    if not flagged_any:
        print("  none")

    print("\n-- mean / median score per quality level, per stat --")
    stat_keys = sorted({s for d in scores_by_stat.values() for s in d})
    for label, _ in QUALITY_LEVELS:
        print(f"\n  {label}:")
        by_stat = scores_by_stat[label]
        for stat in stat_keys:
            pts = by_stat.get(stat)
            if not pts:
                continue
            print(f"    {stat:6} n={len(pts):3}  mean={statistics.mean(pts):7.1f}  median={statistics.median(pts):7.1f}")

    print("\n-- flagged stats (mean deviates >20% from the cross-stat mean for that quality level) --")
    flagged_any = False
    for label, _ in QUALITY_LEVELS:
        by_stat = scores_by_stat[label]
        all_points = [p for pts in by_stat.values() for p in pts]
        if not all_points:
            continue
        overall = statistics.mean(all_points)
        for stat, pts in by_stat.items():
            if not overall:
                continue
            stat_mean = statistics.mean(pts)
            deviation = abs(stat_mean - overall) / overall
            if deviation > 0.20:
                flagged_any = True
                print(
                    f"  {label:12} stat={stat:6} mean={stat_mean:7.1f} vs "
                    f"overall={overall:7.1f} ({deviation:.0%} off)"
                )
    if not flagged_any:
        print("  none")


if __name__ == "__main__":
    main()
