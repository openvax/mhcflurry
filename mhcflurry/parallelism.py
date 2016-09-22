from concurrent import futures
import logging

DEFAULT_EXECUTOR = None


def set_default_executor(executor):
    global DEFAULT_EXECUTOR
    DEFAULT_EXECUTOR = executor


def get_default_executor():
    global DEFAULT_EXECUTOR
    if DEFAULT_EXECUTOR is None:
        DEFAULT_EXECUTOR = futures.ThreadPoolExecutor(max_workers=1)
    return DEFAULT_EXECUTOR


def map_throw_fast(executor, func, iterable):
    futures = [
        executor.submit(func, arg) for arg in iterable
    ]
    return wait_all_throw_fast(futures)


def wait_all_throw_fast(fs):
    result_dict = {}
    for finished_future in futures.as_completed(fs):
        result = finished_future.result()
        logging.info("%3d / %3d tasks completed" % (
            len(result_dict), len(fs)))
        result_dict[finished_future] = result

    return [result_dict[future] for future in fs]
