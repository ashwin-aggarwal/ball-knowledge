#!/usr/bin/env python3
"""Preview a simulated game's questions to judge variety by reading, not guessing.

Generates N questions with a fixed seed against the real built dataset,
exactly as a real game would (cooldowns, difficulty arc, and all three
templates active), prints them as a numbered list, then a summary
breaking counts down by stat, scope, and template -- so a skew shows up
as a number in the summary, not something you have to notice by eye.

Not part of the runtime app. Run after any change to the stat weighting,
cooldown, or template config to see the effect before playing a game.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ball_knowledge.config import DEFAULT_GAME_CONFIG  # noqa: E402
from ball_knowledge.data import load_tables  # noqa: E402
from ball_knowledge.questions import generate_question  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tables = load_tables()
    config = DEFAULT_GAME_CONFIG
    rng = random.Random(args.seed)

    used_keys: set = set()
    recent_stats: list[str] = []
    recent_scopes: list[str] = []

    stat_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    marquee_eligible_rounds = 0
    marquee_hits = 0

    print(f"Simulating {args.rounds} rounds (seed={args.seed})...\n")
    for round_number in range(1, args.rounds + 1):
        q = generate_question(
            tables,
            config,
            used_keys,
            recent_stats=recent_stats,
            recent_scopes=recent_scopes,
            round_number=round_number,
            total_rounds=args.rounds,
            rng=rng,
        )
        used_keys.add(q.key)
        recent_stats.append(q.stat_key)
        recent_scopes.append(q.scope.value)
        stat_counts[q.stat_key] += 1
        scope_counts[q.scope.value] += 1
        template_counts[q.template] += 1
        # obscure_spotlight forces its stat from config.obscure_stats, which
        # by construction never overlaps marquee_stats -- marquee weighting
        # never applies there, so it's excluded from this denominator to
        # get a like-for-like comparison against marquee_weight_share.
        if q.template != "obscure_spotlight":
            marquee_eligible_rounds += 1
            if q.stat_key in config.marquee_stats:
                marquee_hits += 1

        print(
            f"{round_number:3}. [{q.template:18}] {q.question_text}\n"
            f"     -> {q.answer_player_name} ({q.answer_value:g})"
        )

    print(f"\n{'=' * 70}")
    print(f"SUMMARY over {args.rounds} rounds")

    print("\n-- by stat --")
    for stat, count in stat_counts.most_common():
        pct = 100 * count / args.rounds
        print(f"  {stat:6} {count:3}  ({pct:5.1f}%)")

    print("\n-- by scope --")
    for scope, count in scope_counts.most_common():
        pct = 100 * count / args.rounds
        print(f"  {scope:16} {count:3}  ({pct:5.1f}%)")

    print("\n-- by template --")
    for template, count in template_counts.most_common():
        pct = 100 * count / args.rounds
        print(f"  {template:18} {count:3}  ({pct:5.1f}%)")

    marquee = set(config.marquee_stats)
    marquee_count = sum(c for s, c in stat_counts.items() if s in marquee)
    marquee_pct = 100 * marquee_count / args.rounds
    print(
        f"\nmarquee stats {sorted(marquee)}, all rounds: {marquee_count}/{args.rounds} "
        f"({marquee_pct:.1f}%)"
    )
    if marquee_eligible_rounds:
        eligible_pct = 100 * marquee_hits / marquee_eligible_rounds
        print(
            f"marquee stats, within straight_rank/value_anchor only "
            f"(obscure_spotlight structurally excludes them): "
            f"{marquee_hits}/{marquee_eligible_rounds} ({eligible_pct:.1f}%) "
            f"-- target {100 * config.marquee_weight_share:.0f}%"
        )


if __name__ == "__main__":
    main()
