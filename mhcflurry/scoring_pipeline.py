"""Bounded CPU preparation/persistence around one accelerator scoring owner."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def _advance(workflow, value=None):
    try:
        return False, workflow.send(value)
    except StopIteration as finished:
        return True, finished.value


def scoring_pipeline(workflows, score, max_in_flight=3, cpu_workers=2):
    """Run resumable CPU workflows around serial caller-owned scoring.

    Parameters
    ----------
    workflows : iterable of (key, generator)
        Each generator prepares a request, yields it to scoring, receives the
        returned scores, then persists/matches them. It may request additional
        scoring rounds, or return its completed result without requesting scores.
        Generators must own their mutable state and use independent RNGs.
    score : callable
        ``score(key, request)`` runs only on the caller's thread. The accelerator
        and predictor need not be thread-safe.
    max_in_flight : int
        Maximum active sample workflows, including queued work and scored writes.
    cpu_workers : int
        Maximum concurrently running CPU continuations.

    Yields
    ------
    (key, result)
        Completed samples in completion order. Callers restore scientific row
        order using keys, not completion order.

    Notes
    -----
    Backpressure bounds executor submission, not just worker count. On error or
    iterator close, queued work is cancelled and already running CPU stages are
    joined before generators close. Successful atomic writes remain resumable.
    """
    if max_in_flight < 1 or cpu_workers < 1:
        raise ValueError("Pipeline capacities must be positive")
    source = iter(workflows)
    pending = {}
    active = {}
    seen = set()
    exhausted = False
    executor = ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="processing-cpu")

    def fill():
        nonlocal exhausted
        while not exhausted and len(active) < max_in_flight:
            try:
                key, workflow = next(source)
            except StopIteration:
                exhausted = True
                break
            if key in seen:
                workflow.close()
                raise ValueError("Duplicate scoring workflow key: %r" % (key,))
            seen.add(key)
            active[key] = workflow
            pending[executor.submit(_advance, workflow)] = key

    try:
        fill()
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in [f for f in pending if f in completed]:
                key = pending.pop(future)
                done, value = future.result()
                if done:
                    del active[key]
                    fill()
                    yield key, value
                else:
                    prediction = score(key, value)
                    pending[executor.submit(_advance, active[key], prediction)] = key
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        for workflow in active.values():
            workflow.close()
