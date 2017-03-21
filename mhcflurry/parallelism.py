import logging
from concurrent import futures

DEFAULT_BACKEND = None


class RemoteObjectStub(object):
    def __init__(self, value):
        self.value = value


class ParallelBackend(object):
    """
    Thin wrapper of futures implementations. Designed to support
    concurrent.futures as well as dask.distributed's workalike implementation.
    """
    def __init__(self, executor, module, verbose=1):
        self.executor = executor
        self.module = module
        self.verbose = verbose

    def remote_object(self, value):
        return RemoteObjectStub(value)


class KubefaceParallelBackend(ParallelBackend):
    """
    ParallelBackend that uses kubeface
    """
    def __init__(self, args):
        from kubeface import Client  # pylint: disable=import-error
        self.client = Client.from_args(args)

    def map(self, func, iterable):
        return self.client.map(func, iterable)

    def remote_object(self, value):
        return self.client.remote_object(value)

    def __str__(self):
        return "<Kubeface backend, client=%s>" % self.client


class DaskDistributedParallelBackend(ParallelBackend):
    """
    ParallelBackend that uses dask.distributed
    """
    def __init__(self, scheduler_ip_and_port, verbose=1):
        from dask import distributed  # pylint: disable=import-error
        executor = distributed.Executor(scheduler_ip_and_port)
        ParallelBackend.__init__(self, executor, distributed, verbose=verbose)
        self.scheduler_ip_and_port = scheduler_ip_and_port

    def map(self, func, iterable):
        fs = [
            self.executor.submit(func, arg) for arg in iterable
        ]
        return self.wait(fs)

    def wait(self, fs):
        result_dict = {}
        for finished_future in self.module.as_completed(fs):
            result = finished_future.result()
            logging.info("%3d / %3d tasks completed" % (
                len(result_dict), len(fs)))
            result_dict[finished_future] = result

        return [result_dict[future] for future in fs]

    def __str__(self):
        return "<Dask distributed backend, scheduler=%s, total_cores=%d>" % (
            self.scheduler_ip_and_port,
            sum(self.executor.ncores().values()))


class ConcurrentFuturesParallelBackend(ParallelBackend):
    """
    ParallelBackend that uses Python's concurrent.futures module.
    Can use either threads or processes.
    """
    def __init__(self, num_workers=1, processes=False, verbose=1):
        if processes:
            executor = futures.ProcessPoolExecutor(num_workers)
        else:
            executor = futures.ThreadPoolExecutor(num_workers)
        ParallelBackend.__init__(self, executor, futures, verbose=verbose)
        self.num_workers = num_workers
        self.processes = processes

    def __str__(self):
        return "<Concurrent futures %s parallel backend, num workers = %d>" % (
            ("processes" if self.processes else "threads"), self.num_workers)

    def map(self, func, iterable):
        fs = [
            self.executor.submit(func, arg) for arg in iterable
        ]
        return self.wait(fs)

    def wait(self, fs):
        result_dict = {}
        for finished_future in self.module.as_completed(fs):
            result = finished_future.result()
            logging.info("%3d / %3d tasks completed" % (
                len(result_dict), len(fs)))
            result_dict[finished_future] = result

        return [result_dict[future] for future in fs]


def set_default_backend(backend):
    global DEFAULT_BACKEND
    DEFAULT_BACKEND = backend


def get_default_backend():
    global DEFAULT_BACKEND
    if DEFAULT_BACKEND is None:
        set_default_backend(ConcurrentFuturesParallelBackend())
    return DEFAULT_BACKEND
