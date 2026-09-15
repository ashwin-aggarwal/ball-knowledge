from ball_knowledge.scoring import GuessInput, score_round


def _guess(name: str, value: float, pid: int = 0, rank: int | None = None) -> GuessInput:
    return GuessInput(
        guesser_name=name, nba_player_id=pid, nba_player_name=f"NBA{pid}", value=value, rank=rank
    )


def test_empty_guesses_returns_empty() -> None:
    assert score_round([], answer_value=100) == []


def test_closest_guess_wins_round_points() -> None:
    guesses = [_guess("Alice", 90), _guess("Bob", 110), _guess("Carol", 50)]
    results = score_round(guesses, answer_value=100)
    by_name = {r.guesser_name: r for r in results}
    assert by_name["Alice"].points == 1
    assert by_name["Bob"].points == 1  # tie for closest (diff 10 each)
    assert by_name["Carol"].points == 0


def test_exact_match_earns_bonus_not_addition() -> None:
    guesses = [_guess("Alice", 100), _guess("Bob", 90)]
    results = score_round(guesses, answer_value=100)
    by_name = {r.guesser_name: r for r in results}
    assert by_name["Alice"].is_exact is True
    assert by_name["Alice"].points == 2  # default exact_match_bonus_points, not 1+2
    assert by_name["Bob"].points == 0


def test_ties_all_score_including_exact_ties() -> None:
    guesses = [_guess("Alice", 100), _guess("Bob", 100), _guess("Carol", 90)]
    results = score_round(guesses, answer_value=100)
    by_name = {r.guesser_name: r for r in results}
    assert by_name["Alice"].points == 2
    assert by_name["Bob"].points == 2
    assert by_name["Carol"].points == 0


def test_sorted_ascending_by_diff() -> None:
    guesses = [_guess("Alice", 70), _guess("Bob", 100), _guess("Carol", 85)]
    results = score_round(guesses, answer_value=100)
    assert [r.guesser_name for r in results] == ["Bob", "Carol", "Alice"]
    assert [r.diff for r in results] == [0, 15, 30]


def test_custom_point_values() -> None:
    guesses = [_guess("Alice", 100), _guess("Bob", 95)]
    results = score_round(guesses, answer_value=100, round_points=5, exact_match_bonus_points=10)
    by_name = {r.guesser_name: r for r in results}
    assert by_name["Alice"].points == 10
    assert by_name["Bob"].points == 0


def test_all_guesses_wrong_none_exact() -> None:
    guesses = [_guess("Alice", 80), _guess("Bob", 120)]
    results = score_round(guesses, answer_value=100)
    assert all(not r.is_exact for r in results)
    assert {r.points for r in results} == {1}  # both tied for closest at diff 20


def test_rank_and_identity_fields_pass_through() -> None:
    guesses = [_guess("Alice", 100, pid=42, rank=7)]
    results = score_round(guesses, answer_value=100)
    assert results[0].nba_player_id == 42
    assert results[0].nba_player_name == "NBA42"
    assert results[0].rank == 7
