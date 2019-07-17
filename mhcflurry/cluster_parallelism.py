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
        '--cluster-script-prefix-path',
        help="",
    )
    group.add_argument('--cluster-max-retries', help="", default=3)


def cluster_results_from_args(
        args,
        work_function,
        work_items,
        constant_data=None,
        result_serialization_method="pickle"):
    return cluster_results(
        work_function=work_function,
        work_items=work_items,
        constant_data=constant_data,
        submit_command=args.cluster_submit_command,
        results_workdir=args.cluster_results_workdir,
        script_prefix_path=args.cluster_script_prefix_path,
        result_serialization_method=result_serialization_method
    )


def cluster_results(
        work_function,
        work_items,
        constant_data=None,
        submit_command="sh",
        results_workdir="./cluster-workdir",
        script_prefix_path=None,
        result_serialization_method="pickle",
        max_retries=3):

    constant_payload = {
        'constant_data': constant_data,
        'function': work_function,
    }
    work_dir = os.path.join(
        os.path.abspath(results_workdir),
        str(int(time.time())))
    os.mkdir(work_dir)

    constant_payload_path = os.path.join(work_dir, "global_data.pkl")
    with open(constant_payload_path, "wb") as fd:
        pickle.dump(constant_payload, fd, protocol=pickle.HIGHEST_PROTOCOL)
    print("Wrote:", constant_payload_path)

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

        item_data_path = os.path.join(item_workdir, "data.pkl")
        with open(item_data_path, "wb") as fd:
            pickle.dump(item, fd, protocol=pickle.HIGHEST_PROTOCOL)
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
        start = time.time()
        while result_items:
            while True:
                result_item = None
                for d in result_items:
                    if os.path.exists(item['finished_path']):
                        result_item = d
                        break
                if result_item is None:
                    os.sleep(60)
                else:
                    del result_items[result_item]
                    break

            complete_dir = result_item['finished_path']
            result_path = result_item['result_path']
            error_path = result_item['error_path']
            retry_num = result_item['retry_num']
            launch_command = result_item['launch_command']

            print("[%0.1f sec elapsed] processing item %s" % (
                time.time() - start, result_item))

            if os.path.exists(error_path):
                print("Error path exists", error_path)
                with open(error_path, "rb") as fd:
                    exception = pickle.load(fd)
                    print(exception)
                    if retry_num < max_retries:
                        print("Relaunching", launch_command)
                        attempt_dir = os.path.join(
                            result_item['work_dir'], "attempt.%d" % retry_num)
                        shutil.move(complete_dir, attempt_dir)
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
                print("Result path exists", error_path)
                if result_serialization_method == "save_predictor":
                    result = Class1AffinityPredictor.load(result_path)
                else:
                    assert result_serialization_method == "pickle"
                    with open(result_path, "rb") as fd:
                        result = pickle.load(fd)
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
    "--result-serialization-method",
    choices=("pickle", "save_predictor"),
    default="pickle")


def worker_entry_point(argv=sys.argv[1:]):
    # On sigusr1 print stack trace
    print("To show stack trace, run:\nkill -s USR1 %d" % os.getpid())
    signal.signal(signal.SIGUSR1, lambda sig, frame: traceback.print_stack())

    args = parser.parse_args(argv)

    with open(args.constant_data, "rb") as fd:
        constant_payload = pickle.load(fd)

    with open(args.worker_data, "rb") as fd:
        worker_data = pickle.load(fd)

    kwargs = dict(worker_data)
    if constant_payload['constant_data'] is not None:
        kwargs['constant_data'] = constant_payload['constant_data']

    try:
        result = call_wrapped_kwargs(constant_payload['function'], kwargs)
        if args.result_serialization_method == 'save_predictor':
            result.save(args.result_out)
        else:
            with open(args.out, "wb") as fd:
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

