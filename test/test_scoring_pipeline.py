"""Concurrency contracts tested with events, not timing-dependent sleeps."""

import threading

import pytest

from mhcflurry.scoring_pipeline import scoring_pipeline


def test_next_preparation_and_previous_write_overlap_scoring():
    caller = threading.get_ident()
    second_prepared = threading.Event()
    first_writing = threading.Event()
    second_scoring = threading.Event()

    def first():
        assert threading.get_ident() != caller
        value = yield 1
        first_writing.set()
        assert second_scoring.wait(3), "Previous write did not overlap next score"
        return value

    def second():
        second_prepared.set()
        value = yield 2
        return value

    def score(key, request):
        assert threading.get_ident() == caller
        assert second_prepared.wait(3), "Next preparation did not overlap score"
        if key == "second":
            assert first_writing.wait(3), "Previous sample was not writing during score"
            second_scoring.set()
        return request * 10

    results = dict(scoring_pipeline([("first", first()), ("second", second())], score, 2, 2))
    assert first_writing.is_set()
    assert results == {"first": 10, "second": 20}


def test_multiple_rounds_cached_samples_and_bounded_submission():
    created = []
    completed = []
    calls = []

    def workflow(i):
        if i % 2:
            value = yield i
            value += yield i + 100
        else:
            value = i
        completed.append(i)
        return value

    def items():
        for i in range(20):
            created.append(i)
            assert len(created) - len(completed) <= 3
            yield i, workflow(i)

    def score(key, value):
        calls.append((key, value))
        return value

    results = dict(scoring_pipeline(items(), score))
    assert results == {i: (i if not i % 2 else i * 2 + 100) for i in range(20)}
    assert sorted(calls) == [(i, j) for i in range(1, 20, 2) for j in (i, i + 100)]


@pytest.mark.parametrize("stage", ["prepare", "score", "write"])
def test_stage_errors_propagate_and_close_workflows(stage):
    closed = []

    def workflow():
        try:
            if stage == "prepare":
                raise RuntimeError(stage)
            yield 1
            if stage == "write":
                raise RuntimeError(stage)
        finally:
            closed.append(True)

    def score(*args):
        if stage == "score":
            raise RuntimeError(stage)
        return 1

    with pytest.raises(RuntimeError, match=stage):
        list(scoring_pipeline([(1, workflow())], score))
    assert closed == [True]


def test_consumer_close_drains_started_writes():
    persisted = threading.Event()

    def workflow(i):
        yield i
        persisted.set()
        return i

    pipeline = scoring_pipeline([(i, workflow(i)) for i in range(3)], lambda key, value: value)
    next(pipeline)
    pipeline.close()
    assert persisted.is_set()
    assert not any(thread.name.startswith("processing-cpu") for thread in threading.enumerate())


def test_invalid_capacity_and_duplicate_keys_fail():
    with pytest.raises(ValueError, match="capacities"):
        list(scoring_pipeline([], lambda *a: None, 0))

    def workflow():
        yield 1

    with pytest.raises(ValueError, match="Duplicate"):
        list(scoring_pipeline([(1, workflow()), (1, workflow())], lambda *a: 1))


def test_finished_result_is_yielded_before_refilling():
    """A failure pulling the next workflow must not discard a computed result."""
    def workflow(value):
        return value
        yield  # pragma: no cover - only makes this a generator

    def items():
        yield "a", workflow(1)
        yield "a", workflow(2)

    results = []
    with pytest.raises(ValueError, match="Duplicate"):
        for item in scoring_pipeline(items(), lambda key, value: value, max_in_flight=1):
            results.append(item)
    assert results == [("a", 1)]
