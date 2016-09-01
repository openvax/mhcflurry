import multiprocessing
import logging

import joblib.parallel


def configure_joblib(multiprocessing_mode="spawn"):
    """
    Set joblib's default multiprocessing mode.

    The default used in joblib is "fork" which causes a library we use to
    deadlock. This function defaults to setting the multiprocessing mode
    to "spawn", which does not deadlock. On Python 3.4, you can also try
    the "forkserver" mode which does not deadlock and has better
    performance.

    See: https://pythonhosted.org/joblib/parallel.html#bad-interaction-of-multiprocessing-and-third-party-libraries

    Parameters
    -------------
    multiprocessing_mode : string, one of "spawn", "fork", or "forkserver"

    """
    if hasattr(multiprocessing, "get_context"):
        joblib.parallel.DEFAULT_MP_CONTEXT = multiprocessing.get_context(
            multiprocessing_mode)
    else:
        logging.warn(
            "You will probably get deadlocks on Python earlier than 3.4 "
            "if you set n_jobs to anything other than 1.")
