"""
Simple, relatively naive parallel map implementation for HPC clusters.

Used for training MHCflurry models.
"""
import traceback
import sys
import os
import time
import signal
import argparse
import pickle
import subprocess
import shutil

from .local_parallelism import call_wrapped_kwargs
from .class1_affinity_predictor import Class1AffinityPredictor

try:
    from shlex import quote
except ImportError:
    from pipes import quote


def add_cluster_parallelism_args(parser):
    """
    Add commandline arguments controlling cluster parallelism to an argparse
    ArgumentParser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
    """
    group = parser.add_argument_group("Cluster parallelism")
    group.add_argument(
        "--cluster-parallelism",
        default=False,
        action="store_true")
    group.add_argument(
        "--cluster-submit-command",
        default='sh',
        help="Default: %(default)s")
    group.add_argument(
        "--cluster-results-workdir",
        default='./cluster-workdir',
        help="Default: %(default)s")
    group.add_argument(
        "--additional-complete-file",
        default='STDERR',
        help="Additional file to monitor for job completion. Default: %(default)s")
    group.add_argument(
        '--cluster-script-prefix-path',
        help="",
    )
    group.add_argument(
        '--cluster-max-retries',
        type=int,
        help="How many times to rerun failing jobs. Default: %(default)s",
        default=3)


def cluster_results_from_args(
        args,
        work_function,
        work_items,
        constant_data=None,
        input_serialization_method="pickle",
        result_serialization_method="pickle",
        clear_constant_data=False):
    """
    Parallel map configurable using commandline arguments. See the
    cluster_results() function for docs.

    The `args` parameter should be an argparse.Namespace from an argparse parser
    generated using the add_cluster_parallelism_args() function.


    Parameters
    ----------
    args
    work_function
    work_items
    constant_data
    result_serialization_method
    clear_constant_data

    Returns
    -------
    generator
    """
    return cluster_results(
        work_function=work_function,
        work_items=work_items,
        constant_data=constant_data,
        submit_command=args.cluster_submit_command,
        results_workdir=args.cluster_results_workdir,
        additional_complete_file=args.additional_complete_file,
        script_prefix_path=args.cluster_script_prefix_path,
        input_serialization_method=input_serialization_method,
        result_serialization_method=result_serialization_method,
        max_retries=args.cluster_max_retries,
        clear_constant_data=clear_constant_data
    )


def cluster_results(
        work_function,
        work_items,
        constant_data=None,
        submit_command="sh",
        results_workdir="./cluster-workdir",
        additional_complete_file=None,
        script_prefix_path=None,
        input_serialization_method="pickle",
        result_serialization_method="pickle",
        max_retries=3,
        clear_constant_data=False):
    """
    Parallel map on an HPC cluster.

    Returns [work_function(item) for item in work_items] where each invocation
    of work_function is performed as a separate HPC cluster job. Order is
    preserved.

    Optionally, "constant data" can be specified, which will be passed to
    each work_function() invocation as a keyword argument called constant_data.
    This data is serialized once and all workers read it from the same source,
    which is more efficient than serializing it separately for each worker.

    Each worker's input is serialized to a shared NFS directory and the
    submit_command is used to launch a job to process that input. The shared
    filesystem is polled occasionally to watch for results, which are fed back
    to the user.

    Parameters
    ----------
    work_function : A -> B
    work_items : list of A
    constant_data : object
    submit_command : string
        For running on LSF, we use "bsub" here.
    results_workdir : string
        Path to NFS shared directory where inputs and results can be written
    script_prefix_path : string
        Path to script that will be invoked to run each worker. A line calling
        the _mhcflurry-cluster-worker-entry-point command will be appended to
        the contents of this file.
    result_serialization_method : string, one of "pickle" or "save_predictor"
        The "save_predictor" works only when the return type of work_function
        is Class1AffinityPredictor
    max_retries : int
        How many times to attempt to re-launch a failed worker
    clear_constant_data : bool
        If True, the constant data dict is cleared on the launching host after
        it is serialized to disk.

    Returns
    -------
    generator of B
    """

    if input_serialization_method == "dill":
        import dill
        input_serialization_module = dill
    else:
        assert input_serialization_method == "pickle"
        input_serialization_module = pickle

    constant_payload = {
        'constant_data': constant_data,
        'function': work_function,
    }
    if not os.path.exists(results_workdir):
        os.mkdir(results_workdir)

    work_dir = os.path.join(
        os.path.abspath(results_workdir),
        str(int(time.time())))
    os.mkdir(work_dir)

    constant_payload_path = os.path.join(
        work_dir,
        "global_data." + input_serialization_method)
    with open(constant_payload_path, "wb") as fd:
        input_serialization_module.dump(
            constant_payload,
            fd,
            protocol=input_serialization_module.HIGHEST_PROTOCOL)
    print("Wrote:", constant_payload_path)
    if clear_constant_data:
        constant_data.clear()
        print("Cleared constant data to free up memory.")

    if script_prefix_path:
        with open(script_prefix_path) as fd:
            script_prefix = fd.read()
    else:
        script_prefix = "#!/bin/bash"

    result_items = []

    for (i, item) in enumerate(work_items):
        item_workdir = os.path.join(
            work_dir, "work-item.%03d-of-%03d" % (i, len(work_items)))
        os.mkdir(item_workdir)

        item_data_path = os.path.join(
            item_workdir, "data." + input_serialization_method)
        with open(item_data_path, "wb") as fd:
            input_serialization_module.dump(
                item, fd, protocol=input_serialization_module.HIGHEST_PROTOCOL)
        print("Wrote:", item_data_path)

        item_result_path = os.path.join(item_workdir, "result")
        item_error_path = os.path.join(item_workdir, "error.pkl")
        item_finished_path = os.path.join(item_workdir, "COMPLETE")

        item_script_pieces = [
            script_prefix.format(work_item_num=i, work_dir=item_workdir)
        ]
        item_script_pieces.append(" ".join([
            "_mhcflurry-cluster-worker-entry-point",
            "--constant-data", quote(constant_payload_path),
            "--worker-data", quote(item_data_path),
            "--result-out", quote(item_result_path),
            "--error-out", quote(item_error_path),
            "--complete-dir", quote(item_finished_path),
            "--input-serialization-method", input_serialization_method,
            "--result-serialization-method", result_serialization_method,
        ]))
        item_script = "\n".join(item_script_pieces)
        item_script_path = os.path.join(
            item_workdir,
            "run.%d.sh" % i)
        with open(item_script_path, "w") as fd:
            fd.write(item_script)
        print("Wrote:", item_script_path)

        launch_command = " ".join([
            submit_command, "<", quote(item_script_path)
        ])
        subprocess.check_call(launch_command, shell=True)
        print("Invoked", launch_command)

        result_items.append({
            'work_dir': item_workdir,
            'finished_path': item_finished_path,
            'result_path': item_result_path,
            'error_path': item_error_path,
            'retry_num': 0,
            'launch_command': launch_command,
        })

    def result_generator():
        additional_complete_file_path = None
        start = time.time()
        while result_items:
            print("[%0.1f sec elapsed] waiting on %d / %d items." % (
                time.time() - start, len(result_items), len(work_items)))
            while True:
                result_item = None
                for d in result_items:
                    if additional_complete_file:
                        additional_complete_file_path = os.path.join(
                            d['work_dir'], additional_complete_file)
                    if os.path.exists(d['finished_path']):
                        result_item = d
                        break
                    if additional_complete_file and os.path.exists(
                            additional_complete_file_path):
                        result_item = d
                        print("Exists", additional_complete_file_path)
                        break

                if result_item is None:
                    time.sleep(60)
                else:
                    result_items.remove(result_item)
                    break

            complete_dir = result_item['finished_path']
            result_path = result_item['result_path']
            error_path = result_item['error_path']
            retry_num = result_item['retry_num']
            launch_command = result_item['launch_command']

            print("[%0.1f sec elapsed] processing item %s" % (
                time.time() - start, result_item))

            if os.path.exists(error_path) or not os.path.exists(result_path):
                if os.path.exists(error_path):
                    print("Error path exists", error_path)
                    try:
                        with open(error_path, "rb") as fd:
                            exception = pickle.load(fd)
                            print(exception)
                    except Exception as e:
                        exception = RuntimeError(
                            "Error, but couldn't read error path: %s %s" % (
                                type(e), str(e)))
                else:
                    exception = RuntimeError("Error, but no exception saved")
                if not os.path.exists(result_path):
                    print("Result path does NOT exist", result_path)

                if retry_num < max_retries:
                    print("Relaunching", launch_command)
                    attempt_dir = os.path.join(
                        result_item['work_dir'], "attempt.%d" % retry_num)
                    if os.path.exists(complete_dir):
                        shutil.move(complete_dir, attempt_dir)  # directory
                    if additional_complete_file and os.path.exists(
                            additional_complete_file_path):
                        shutil.move(additional_complete_file_path, attempt_dir)
                    if os.path.exists(error_path):
                        shutil.move(error_path, attempt_dir)
                    subprocess.check_call(launch_command, shell=True)
                    print("Invoked", launch_command)
                    result_item['retry_num'] += 1
                    result_items.append(result_item)
                    continue
                else:
                    print("Max retries exceeded", max_retries)
                    raise exception

            if os.path.exists(result_path):
                print("Result path exists", result_path)
                if result_serialization_method == "save_predictor":
                    result = Class1AffinityPredictor.load(result_path)
                elif result_serialization_method == "pickle":
                    with open(result_path, "rb") as fd:
                        result = pickle.load(fd)
                else:
                    raise ValueError(
                        "Unsupported serialization method",
                        result_serialization_method)

                yield result
            else:
                raise RuntimeError("Results do not exist", result_path)

    return result_generator()


parser = argparse.ArgumentParser(
    usage="Entry point for cluster workers")
parser.add_argument(
    "--constant-data",
    required=True,
)
parser.add_argument(
    "--worker-data",
    required=True,
)
parser.add_argument(
    "--result-out",
    required=True,
)
parser.add_argument(
    "--error-out",
    required=True,
)
parser.add_argument(
    "--complete-dir",
)
parser.add_argument(
    "--input-serialization-method",
    choices=("pickle", "dill"),
    default="pickle")
parser.add_argument(
    "--result-serialization-method",
    choices=("pickle", "save_predictor"),
    default="pickle")


def worker_entry_point(argv=sys.argv[1:]):
    """
    Entry point for the worker command.

    Parameters
    ----------
    argv : list of string
    """
    # On sigusr1 print stack trace
    print("To show stack trace, run:\nkill -s USR1 %d" % os.getpid())
    signal.signal(signal.SIGUSR1, lambda sig, frame: traceback.print_stack())

    args = parser.parse_args(argv)

    if args.input_serialization_method == "dill":
        import dill
        input_serialization_module = dill
    else:
        assert args.input_serialization_method == "pickle"
        input_serialization_module = pickle

    with open(args.constant_data, "rb") as fd:
        constant_payload = input_serialization_module.load(fd)

    with open(args.worker_data, "rb") as fd:
        worker_data = input_serialization_module.load(fd)

    kwargs = dict(worker_data)
    if constant_payload['constant_data'] is not None:
        kwargs['constant_data'] = constant_payload['constant_data']

    try:
        result = call_wrapped_kwargs(constant_payload['function'], kwargs)
        if args.result_serialization_method == 'save_predictor':
            result.save(args.result_out)
        else:
            with open(args.result_out, "wb") as fd:
                pickle.dump(result, fd, pickle.HIGHEST_PROTOCOL)
        print("Wrote:", args.result_out)
    except Exception as e:
        print("Exception: ", e)
        with open(args.error_out, "wb") as fd:
            pickle.dump(e, fd, pickle.HIGHEST_PROTOCOL)
        print("Wrote:", args.error_out)
        raise
    finally:
        if args.complete_dir:
            os.mkdir(args.complete_dir)
            print("Created: ", args.complete_dir)
