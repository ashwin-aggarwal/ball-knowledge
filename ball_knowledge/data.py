"""Parquet loading and caching. The only module allowed to import streamlit
for @st.cache_data (besides the ui/ package and app.py).

The pure loading logic (`load_tables`) takes no Streamlit dependency and is
directly unit-testable; `cached_load_tables` is a thin @st.cache_data
wrapper around it for use from the app. Eligibility filtering itself lives
in questions.py (which must stay Streamlit-free), and is re-exported here
for convenience so UI code only needs to import this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ball_knowledge.config import DATA_DIR
from ball_knowledge.questions import (  # noqa: F401 (re-exported)
    DataTables,
    eligible_players_for_question,
    eligible_pool,
    resolve_guess_value_and_rank,
    resolve_player_by_name,
)

REQUIRED_FILES = (
    "career_totals.parquet",
    "career_per_game.parquet",
    "season_records.parquet",
    "players.parquet",
)


def load_tables(data_dir: str | Path = DATA_DIR) -> DataTables:
    """Load the four parquet tables from `data_dir` into a DataTables bundle."""
    data_path = Path(data_dir)
    missing = [f for f in REQUIRED_FILES if not (data_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset file(s) in {data_path}: {missing}. "
            "Run scripts/build_dataset.py first."
        )
    return DataTables(
        career_totals=pd.read_parquet(data_path / "career_totals.parquet"),
        career_per_game=pd.read_parquet(data_path / "career_per_game.parquet"),
        season_records=pd.read_parquet(data_path / "season_records.parquet"),
        players=pd.read_parquet(data_path / "players.parquet"),
    )


def load_manifest(data_dir: str | Path = DATA_DIR) -> dict:
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {data_dir}")
    with manifest_path.open("r") as f:
        return json.load(f)


@st.cache_data
def cached_load_tables(data_dir: str = DATA_DIR) -> DataTables:
    return load_tables(data_dir)


@st.cache_data
def cached_load_manifest(data_dir: str = DATA_DIR) -> dict:
    return load_manifest(data_dir)
