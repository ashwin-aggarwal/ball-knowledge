"""Entrypoint: routes on game phase. All logic lives in ball_knowledge/."""
from __future__ import annotations

import traceback

import streamlit as st

from ball_knowledge import state
from ball_knowledge.config import DEFAULT_GAME_CONFIG
from ball_knowledge.data import (
    eligible_players_for_question,
    leaderboard_neighbors,
    resolve_guess_value_and_rank,
    resolve_player_by_name,
)
from ball_knowledge.scoring import GuessInput
from ball_knowledge.state import Phase
from ball_knowledge.ui import components
from ball_knowledge.ui.theme import inject_theme

st.set_page_config(page_title="Ball Knowledge", page_icon="🏀")

state.init_state()
inject_theme()

VALUE_FMT = ",.0f"  # every stat left in the game is a career total


def render_lobby() -> None:
    components.render_hero_title()
    cfg = DEFAULT_GAME_CONFIG
    st.write(
        f"Enter {cfg.min_players}-{cfg.max_players} player names (solo play works too), "
        "then start the game."
    )
    num_players = st.number_input(
        "Number of players", min_value=cfg.min_players, max_value=cfg.max_players, value=4, step=1
    )
    names = [
        st.text_input(f"Player {i + 1} name", key=f"lobby_name_{i}")
        for i in range(int(num_players))
    ]
    num_rounds = st.number_input(
        "Number of rounds", min_value=1, max_value=50, value=cfg.default_rounds
    )
    if st.button("Start game"):
        clean_names = [n.strip() for n in names if n.strip()]
        if len(clean_names) < cfg.min_players:
            st.error(f"Need at least {cfg.min_players} player.")
        elif len(clean_names) != len(set(clean_names)):
            st.error("Player names must be unique.")
        else:
            state.start_game(clean_names, int(num_rounds))
            st.rerun()


def render_round_intro() -> None:
    ss = st.session_state
    q = ss.current_question
    st.markdown(f"#### Round {ss.current_round} / {ss.num_rounds}")
    components.render_card_back(q.question_text, q.era_caveat)
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Start guessing", width="stretch"):
            state.begin_collecting()
            st.rerun()


def render_collect() -> None:
    ss = st.session_state
    if ss.collect_gate_shown:
        components.render_handoff_gate(state.current_guesser())
        st.write("Nobody else should see the next screen.")
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            if st.button("I'm ready", width="stretch"):
                state.show_guess_input()
                st.rerun()
        return

    q = ss.current_question
    guesser = state.current_guesser()
    st.markdown(f"#### {guesser}'s guess")
    components.render_card_back(q.question_text, q.era_caveat)
    st.caption("Type any player below.")
    # Keying by (collect_index, clear_nonce) guarantees a fresh widget for
    # each guesser's turn and for each "Clear" click, since Streamlit
    # forbids reassigning a widget's session_state value after it has
    # already been instantiated in the same script run.
    text_key = f"guess_text_{ss.collect_index}_{ss.guess_clear_nonce}"
    typed_name = st.text_input(
        "Who do you think it is?", key=text_key, placeholder="e.g. LeBron James"
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear", disabled=not typed_name, width="stretch"):
            ss.guess_clear_nonce += 1
            st.rerun()
    with col2:
        if st.button("Confirm guess", disabled=not typed_name, width="stretch"):
            match = resolve_player_by_name(typed_name, state.tables().players)
            if match is None:
                st.error(
                    f"No player found named \"{typed_name}\". Check the spelling and try again."
                )
            else:
                value, rank = resolve_guess_value_and_rank(
                    int(match["player_id"]), q, state.tables()
                )
                guess = GuessInput(
                    guesser_name=guesser,
                    nba_player_id=int(match["player_id"]),
                    nba_player_name=str(match["full_name"]),
                    value=value,
                    rank=rank,
                )
                state.confirm_guess(guess)
                st.rerun()


def render_reveal() -> None:
    ss = st.session_state
    q = ss.current_question
    scored = state.reveal_and_score()
    components.render_card_front(
        player_id=q.answer_player_id,
        player_name=q.answer_player_name,
        value_display=f"{format(q.answer_value, VALUE_FMT)} {q.stat_label}",
        rank_display=f"#{q.target_rank}",
    )
    components.render_guesses_label()
    components.render_guess_strip(
        scored, scoring_mode=q.scoring_mode, value_display_fmt=VALUE_FMT, target_rank=q.target_rank
    )
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("See scoreboard", width="stretch"):
            state.apply_scores_and_go_to_scoreboard()
            st.rerun()

    with st.sidebar:
        pool = eligible_players_for_question(q, state.tables())
        above, below = leaderboard_neighbors(
            pool, q.target_rank, DEFAULT_GAME_CONFIG.reveal_neighbor_count
        )
        components.render_leaderboard_neighbors(
            above,
            below,
            answer_rank=q.target_rank,
            answer_name=q.answer_player_name,
            answer_value=q.answer_value,
            value_display_fmt=VALUE_FMT,
        )


def render_scoreboard() -> None:
    ss = st.session_state
    st.markdown("#### Scoreboard")
    ranked = sorted(ss.scores.items(), key=lambda kv: -kv[1])
    components.render_scoreboard(ranked)
    label = "Next round" if ss.current_round < ss.num_rounds else "See final results"
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button(label, width="stretch"):
            state.advance_after_scoreboard()
            st.rerun()


def render_game_over() -> None:
    ss = st.session_state
    st.markdown("#### Game over!")
    ranked = sorted(ss.scores.items(), key=lambda kv: -kv[1])
    components.render_scoreboard(ranked)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Play again", width="stretch"):
            state.play_again()
            st.rerun()
    with col2:
        if st.button("Start over", width="stretch"):
            state.reset_all()
            st.rerun()


PHASE_RENDERERS = {
    Phase.LOBBY: render_lobby,
    Phase.ROUND_INTRO: render_round_intro,
    Phase.COLLECT: render_collect,
    Phase.REVEAL: render_reveal,
    Phase.SCOREBOARD: render_scoreboard,
    Phase.GAME_OVER: render_game_over,
}


def render_crash_screen() -> None:
    """A card that got printed wrong. assets/crash.png is a committed
    static file, never fetched over the network, so this screen never
    depends on whatever just broke.
    """
    ss = st.session_state
    components.render_crash_card("assets/crash.png")
    st.error(f"Something broke: {ss.crash_info['message']}")
    with st.expander("Traceback"):
        st.code(ss.crash_info["traceback"])
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to the scoreboard", width="stretch"):
            state.recover_to_scoreboard()
            st.rerun()
    with col2:
        if st.button("Start over", key="crash_start_over", width="stretch"):
            state.reset_all()
            st.rerun()


def main() -> None:
    ss = st.session_state
    if ss.crash_info is not None:
        render_crash_screen()
        return
    renderer = PHASE_RENDERERS[ss.phase]
    try:
        renderer()
        state.checkpoint()
    except Exception as exc:
        ss.crash_info = {"message": str(exc), "traceback": traceback.format_exc()}
        st.rerun()


main()
