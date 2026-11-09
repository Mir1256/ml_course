"""Импорт модуля выполняет все проверки из gaps.py; тест падает, если хоть одна не выполнена."""


def test_gaps_module_runs():
    import gaps  # noqa: F401
