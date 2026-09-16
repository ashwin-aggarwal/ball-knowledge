import json

import pandas as pd
import pytest

from ball_knowledge.data import eligible_pool, load_manifest, load_tables


def _write_tables(tmp_path, tables) -> None:
    tables.career_totals.to_parquet(tmp_path / "career_totals.parquet", index=False)
    tables.players.to_parquet(tmp_path / "players.parquet", index=False)


def test_load_tables_round_trips(tmp_path, tables) -> None:
    _write_tables(tmp_path, tables)
    loaded = load_tables(tmp_path)
    pd.testing.assert_frame_equal(loaded.career_totals, tables.career_totals)
    pd.testing.assert_frame_equal(loaded.players, tables.players)


def test_load_tables_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tables(tmp_path)


def test_load_tables_partial_files_raises(tmp_path, tables) -> None:
    tables.players.to_parquet(tmp_path / "players.parquet", index=False)
    with pytest.raises(FileNotFoundError, match="career_totals"):
        load_tables(tmp_path)


def test_load_manifest_round_trips(tmp_path) -> None:
    manifest = {"built_at": "2024-01-01T00:00:00Z", "row_counts": {"players": 8}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert load_manifest(tmp_path) == manifest


def test_load_manifest_missing_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path)


def test_reexported_eligible_pool_usable_from_data_module(tables) -> None:
    pool = eligible_pool(tables.career_totals, tables.players, value_col="pts_total", min_games=1)
    assert not pool.empty
