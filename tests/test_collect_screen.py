"""End-to-end tests for the single-screen COLLECT phase, via AppTest.

Runs the real app.py against the real committed dataset (there's no
synthetic-fixture path through Streamlit's widget tree), skipped if the
dataset isn't built.
"""
from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "career_totals.parquet"

pytestmark = pytest.mark.skipif(not DATA_PATH.exists(), reason="data/*.parquet not built")


def _start_at_collect(num_players: int = 2, names: list[str] | None = None):
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    names = names or [f"P{i}" for i in range(num_players)]
    for i, name in enumerate(names):
        at.text_input(key=f"lobby_name_{i}").set_value(name)
    # number_input[0] is "Number of players" (default 4, plenty for these
    # rosters -- left untouched); number_input[1] is "Number of rounds".
    at.number_input[1].set_value(1)
    at.button[0].click().run()  # Start game -> ROUND_INTRO
    at.button[0].click().run()  # Start guessing -> COLLECT
    return at, names


def _select_key(round_number: int, player: str) -> str:
    return f"guess_select_r{round_number}_{player}"


def test_editing_one_player_leaves_others_unchanged() -> None:
    at, names = _start_at_collect(names=["Alice", "Bob", "Carol"])
    options = at.selectbox(key=_select_key(1, "Alice")).options
    at.selectbox(key=_select_key(1, "Alice")).select(options[0]).run()
    at.selectbox(key=_select_key(1, "Bob")).select(options[1]).run()
    at.selectbox(key=_select_key(1, "Carol")).select(options[2]).run()

    # Re-select Alice's row to a different value.
    at.selectbox(key=_select_key(1, "Alice")).select(options[5]).run()

    assert at.selectbox(key=_select_key(1, "Alice")).value == options[5]
    assert at.selectbox(key=_select_key(1, "Bob")).value == options[1]
    assert at.selectbox(key=_select_key(1, "Carol")).value == options[2]


def test_submit_disabled_until_all_filled_then_enables() -> None:
    at, names = _start_at_collect(names=["Alice", "Bob"])
    options = at.selectbox(key=_select_key(1, "Alice")).options
    assert at.button[0].disabled

    at.selectbox(key=_select_key(1, "Alice")).select(options[0]).run()
    assert at.button[0].disabled  # Bob still blank

    at.selectbox(key=_select_key(1, "Bob")).select(options[1]).run()
    assert not at.button[0].disabled


def test_duplicate_guesses_score_identically() -> None:
    at, names = _start_at_collect(names=["Alice", "Bob"])
    q = at.session_state["current_question"]
    at.selectbox(key=_select_key(1, "Alice")).select(q.answer_player_name).run()
    at.selectbox(key=_select_key(1, "Bob")).select(q.answer_player_name).run()
    at.button[0].click().run()  # Reveal answers

    scored = {s.guesser_name: s for s in at.session_state["last_scored"]}
    assert scored["Alice"].points == scored["Bob"].points
    assert scored["Alice"].is_exact and scored["Bob"].is_exact


def test_guesses_survive_a_rerun_from_an_unrelated_widget() -> None:
    at, names = _start_at_collect(names=["Alice", "Bob"])
    options = at.selectbox(key=_select_key(1, "Alice")).options
    at.selectbox(key=_select_key(1, "Alice")).select(options[3]).run()

    at.run()  # a rerun with no widget interaction at all

    assert at.selectbox(key=_select_key(1, "Alice")).value == options[3]


def test_scoring_unaffected_by_entry_order() -> None:
    """Filling the rows in reverse order (Carol/Bob/Alice instead of
    Alice/Bob/Carol) must not swap or misattribute anyone's guess or
    score -- covers the UI-level half of order-independence; scoring.py's
    own order-independence as a pure function is covered separately in
    test_scoring.py.
    """
    at, names = _start_at_collect(names=["Alice", "Bob", "Carol"])
    options = at.selectbox(key=_select_key(1, "Alice")).options
    picks = {"Alice": options[0], "Bob": options[5], "Carol": options[10]}

    for player in reversed(names):  # fill Carol, then Bob, then Alice
        at.selectbox(key=_select_key(1, player)).select(picks[player]).run()

    at.button[0].click().run()  # Reveal answers
    scored = {s.guesser_name: s for s in at.session_state["last_scored"]}
    for player, picked_name in picks.items():
        assert scored[player].nba_player_name == picked_name
