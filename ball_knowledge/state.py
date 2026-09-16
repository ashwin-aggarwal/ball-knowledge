"""Hot-seat game state machine, driven entirely by st.session_state.

LOBBY -> ROUND_INTRO -> COLLECT -> REVEAL -> SCOREBOARD -> (ROUND_INTRO | GAME_OVER)

Every function here mutates st.session_state directly rather than
returning new state, matching Streamlit's rerun-the-whole-script model:
each user action calls one of these, then the script reruns and the
router in app.py re-renders based on the (now updated) phase.
"""
from __future__ import annotations

import random
from enum import Enum
from typing import Any

import streamlit as st

from ball_knowledge.config import DEFAULT_GAME_CONFIG, GameConfig
from ball_knowledge.data import cached_load_tables
from ball_knowledge.questions import DataTables, Question, generate_question
from ball_knowledge.scoring import GuessInput, ScoredGuess, score_round


class Phase(str, Enum):
    LOBBY = "LOBBY"
    ROUND_INTRO = "ROUND_INTRO"
    COLLECT = "COLLECT"
    REVEAL = "REVEAL"
    SCOREBOARD = "SCOREBOARD"
    GAME_OVER = "GAME_OVER"


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("phase", Phase.LOBBY)
    ss.setdefault("players", [])
    ss.setdefault("num_rounds", DEFAULT_GAME_CONFIG.default_rounds)
    ss.setdefault("game_config", DEFAULT_GAME_CONFIG)
    ss.setdefault("scores", {})
    ss.setdefault("current_round", 0)
    ss.setdefault("used_question_keys", set())
    ss.setdefault("recent_stats", [])
    ss.setdefault("recent_scopes", [])
    ss.setdefault("current_question", None)
    ss.setdefault("collect_index", 0)
    ss.setdefault("collect_gate_shown", True)
    ss.setdefault("guess_clear_nonce", 0)
    ss.setdefault("guesses", {})
    ss.setdefault("last_scored", [])
    ss.setdefault("checkpoint", None)
    ss.setdefault("crash_info", None)


def tables() -> DataTables:
    return cached_load_tables()


def start_game(players: list[str], num_rounds: int, game_config: GameConfig | None = None) -> None:
    ss = st.session_state
    ss.players = list(players)
    ss.num_rounds = num_rounds
    ss.game_config = game_config or DEFAULT_GAME_CONFIG
    ss.scores = {p: 0 for p in players}
    ss.current_round = 0
    ss.used_question_keys = set()
    ss.recent_stats = []
    ss.recent_scopes = []
    _start_new_round()


def _start_new_round() -> None:
    ss = st.session_state
    ss.current_round += 1
    rng = random.Random()
    ss.current_question = generate_question(
        tables(),
        ss.game_config,
        ss.used_question_keys,
        recent_stats=ss.recent_stats,
        recent_scopes=ss.recent_scopes,
        round_number=ss.current_round,
        total_rounds=ss.num_rounds,
        rng=rng,
    )
    ss.used_question_keys.add(ss.current_question.key)
    ss.recent_stats.append(ss.current_question.stat_key)
    ss.recent_scopes.append(ss.current_question.scope.value)
    ss.collect_index = 0
    # Solo play has no one to hand the laptop to, so skip the gate screen.
    ss.collect_gate_shown = len(ss.players) > 1
    ss.guess_clear_nonce = 0
    ss.guesses = {}
    ss.last_scored = []
    ss.phase = Phase.ROUND_INTRO


def begin_collecting() -> None:
    st.session_state.phase = Phase.COLLECT


def current_guesser() -> str:
    ss = st.session_state
    return ss.players[ss.collect_index]


def show_guess_input() -> None:
    st.session_state.collect_gate_shown = False


def confirm_guess(guess: GuessInput) -> None:
    ss = st.session_state
    ss.guesses[guess.guesser_name] = guess
    ss.collect_index += 1
    ss.collect_gate_shown = True
    if ss.collect_index >= len(ss.players):
        ss.phase = Phase.REVEAL


def reveal_and_score() -> list[ScoredGuess]:
    """Score the round (idempotent: safe to call on every REVEAL rerun)."""
    ss = st.session_state
    q: Question = ss.current_question
    scored = score_round(
        list(ss.guesses.values()),
        q.target_rank,
        answer_value=q.anchor_value if q.scoring_mode == "value" else None,
        round_points=ss.game_config.round_points,
        exact_match_bonus_points=ss.game_config.exact_match_bonus_points,
    )
    ss.last_scored = scored
    return scored


def apply_scores_and_go_to_scoreboard() -> None:
    ss = st.session_state
    if not ss.last_scored:
        reveal_and_score()
    for s in ss.last_scored:
        ss.scores[s.guesser_name] = ss.scores.get(s.guesser_name, 0) + s.points
    ss.phase = Phase.SCOREBOARD


def advance_after_scoreboard() -> None:
    ss = st.session_state
    if ss.current_round >= ss.num_rounds:
        ss.phase = Phase.GAME_OVER
    else:
        _start_new_round()


def play_again() -> None:
    ss = st.session_state
    ss.scores = {p: 0 for p in ss.players}
    ss.current_round = 0
    ss.used_question_keys = set()
    ss.recent_stats = []
    ss.recent_scopes = []
    _start_new_round()


def reset_all() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


def checkpoint() -> None:
    """Snapshot roster/scores/progress after a phase renders successfully.

    Called from app.py's router after each successful render. Deliberately
    excludes in-flight, potentially-broken state (current_question,
    in-progress guesses) — a crash recovery always lands on SCOREBOARD,
    voiding only the round in progress, never the game.
    """
    ss = st.session_state
    if not ss.players:
        return
    ss.checkpoint = {
        "players": list(ss.players),
        "num_rounds": ss.num_rounds,
        "game_config": ss.game_config,
        "scores": dict(ss.scores),
        "current_round": ss.current_round,
        "used_question_keys": set(ss.used_question_keys),
        "recent_stats": list(ss.recent_stats),
        "recent_scopes": list(ss.recent_scopes),
    }


def recover_to_scoreboard() -> None:
    ss = st.session_state
    cp = ss.checkpoint
    if cp:
        ss.players = cp["players"]
        ss.num_rounds = cp["num_rounds"]
        ss.game_config = cp["game_config"]
        ss.scores = cp["scores"]
        ss.current_round = cp["current_round"]
        ss.used_question_keys = cp["used_question_keys"]
        ss.recent_stats = cp.get("recent_stats", [])
        ss.recent_scopes = cp.get("recent_scopes", [])
        ss.phase = Phase.SCOREBOARD
    else:
        ss.phase = Phase.LOBBY
    ss.crash_info = None
