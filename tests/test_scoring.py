from ball_knowledge.scoring import GuessInput, score_round


def _guess(name: str, rank: int, pid: int = 0, value: float = 0.0) -> GuessInput:
    return GuessInput(
        guesser_name=name, nba_player_id=pid, nba_player_name=f"NBA{pid}", value=value, rank=rank
    )


def test_empty_guesses_returns_empty() -> None:
    assert score_round([], answer_rank=50) == []


def test_closest_rank_wins_round_points() -> None:
    # Question asked "who ranks 50th"; Alice guessed the 52nd-ranked
    # player (off by 2), Bob guessed the 48th (also off by 2, tied),
    # Carol guessed the 45th (off by 5, loses even though her guess's
    # raw stat value could be numerically closer to the answer's value).
    guesses = [_guess("Alice", rank=52), _guess("Bob", rank=48), _guess("Carol", rank=45)]
    results = score_round(guesses, answer_rank=50)
    by_name = {r.guesser_name: r for r in results}
    assert by_name["Alice"].points == 1
    assert by_name["Bob"].points == 1
    assert by_name["Carol"].points == 0
    assert by_name["Alice"].diff == 2
    assert by_name["Carol"].diff == 5


def test_exact_rank_match_earns_bonus_not_addition() -> None:
    guesses = [_guess("Alice", rank=50), _guess("Bob", rank=48)]
    results = score_round(guesses, answer_rank=50)
    by_name = {r.guesser_name: r for r in results}
    assert by_name["Alice"].is_exact is True
    assert by_name["Alice"].points == 2  # default exact_match_bonus_points, not 1+2
    assert by_name["Bob"].points == 0


def test_ties_all_score_including_exact_ties() -> None:
    guesses = [_guess("Alice", rank=50), _guess("Bob", rank=50), _guess("Carol", rank=48)]
    results = score_round(guesses, answer_rank=50)
    by_name = {r.guesser_name: r for r in results}
    assert by_name["Alice"].points == 2
    assert by_name["Bob"].points == 2
    assert by_name["Carol"].points == 0


def test_sorted_ascending_by_diff() -> None:
    guesses = [_guess("Alice", rank=70), _guess("Bob", rank=50), _guess("Carol", rank=55)]
    results = score_round(guesses, answer_rank=50)
    assert [r.guesser_name for r in results] == ["Bob", "Carol", "Alice"]
    assert [r.diff for r in results] == [0, 5, 20]


def test_custom_point_values() -> None:
    guesses = [_guess("Alice", rank=50), _guess("Bob", rank=45)]
    results = score_round(guesses, answer_rank=50, round_points=5, exact_match_bonus_points=10)
    by_name = {r.guesser_name: r for r in results}
    assert by_name["Alice"].points == 10
    assert by_name["Bob"].points == 0


def test_all_guesses_wrong_none_exact() -> None:
    guesses = [_guess("Alice", rank=40), _guess("Bob", rank=60)]
    results = score_round(guesses, answer_rank=50)
    assert all(not r.is_exact for r in results)
    assert {r.points for r in results} == {1}  # both tied for closest at diff 10


def test_identity_and_value_fields_pass_through() -> None:
    guesses = [_guess("Alice", rank=50, pid=42, value=123.5)]
    results = score_round(guesses, answer_rank=50)
    assert results[0].nba_player_id == 42
    assert results[0].nba_player_name == "NBA42"
    assert results[0].value == 123.5
    assert results[0].rank == 50
