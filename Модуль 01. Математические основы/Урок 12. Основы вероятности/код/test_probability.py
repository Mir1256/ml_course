import pytest

from probability import at_least_one, birthday_exact, birthday_mc


def test_birthday_known_value():
    assert birthday_exact(23, 365) == pytest.approx(0.5073, abs=1e-4)


def test_birthday_mc_close_to_exact():
    assert birthday_mc(30, 365, trials=50_000) == pytest.approx(birthday_exact(30, 365), abs=0.01)


def test_birthday_edge_cases():
    assert birthday_exact(1, 10) == 0.0
    assert birthday_exact(11, 10) == 1.0          # принцип Дирихле


def test_at_least_one():
    assert at_least_one(0.5, 1) == 0.5
    assert at_least_one(0.5, 2) == 0.75
    assert at_least_one(0.0, 1000) == 0.0
