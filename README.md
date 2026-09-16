# ball-knowledge

A hot-seat NBA stats guessing game for 1-8 players on one laptop. Each
round asks something like "who ranks 47th all time in career rebounds?"
— everyone guesses in private, then the answer flips like a trading card.

Uses [nba_api](https://github.com/swar/nba_api), an unofficial client for
stats.nba.com. Personal project, not affiliated with the NBA.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The data is already built and committed (`data/*.parquet`), so this just
works — no API key, no setup.

## Rebuild the data (e.g. at the start of a new season)

```bash
pip install -r requirements-build.txt
python scripts/build_dataset.py          # ~15-20 min, resumable if interrupted
python scripts/verify_dataset.py         # sanity checks
python scripts/audit_answers.py          # cross-checks vs. NBA.com's live leaders
```

Add `--limit 25` to `build_dataset.py` for a fast smoke test first.

## Deploy

Push to GitHub, then connect the repo at [share.streamlit.io](https://share.streamlit.io)
with entrypoint `app.py`. Done — the app never calls stats.nba.com at
runtime, so it just serves the committed data.

## Knobs you might want to change

All in `ball_knowledge/config.py`, on `GameConfig`:

- `default_rounds` — rounds per game
- `dataset_weights` — career totals vs. per-game mix
- `template_weights` — straight-rank vs. value-anchor vs. obscure-stat questions
- `marquee_stats` / `marquee_weight_share` — how often points/rebounds/assists show up vs. everything else
- `career_total_ranks` / `career_per_game_ranks` — how deep the leaderboard goes

Run `python scripts/preview_questions.py` after changing any of these to
see 50 sample questions and a variety breakdown before playing.
