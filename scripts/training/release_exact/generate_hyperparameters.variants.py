"""
Generate grid of hyperparameters
"""

from sys import stdout, argv
from copy import deepcopy
from yaml import dump, load
import argparse

parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument(
    "production_hyperparameters",
    metavar="data.json",
    help="Production (i.e. standard) hyperparameters grid.")
parser.add_argument(
    "kind",
    choices=('with_flanks', 'no_n_flank', 'no_c_flank', 'no_flank', 'short_flanks'),
    help="Hyperameters variant to output")

args = parser.parse_args(argv[1:])

with open(args.production_hyperparameters) as fd:
    production_hyperparameters_list = load(fd)


def transform(kind, hyperparameters):
    new_hyperparameters = deepcopy(hyperparameters)
    if kind == "no_n_flank" or kind == "no_flank":
        new_hyperparameters["n_flank_length"] = 0
    if kind == "no_c_flank" or kind == "no_flank":
        new_hyperparameters["c_flank_length"] = 0
    if kind == "short_flanks":
        new_hyperparameters["c_flank_length"] = 5
        new_hyperparameters["n_flank_length"] = 5
    return [new_hyperparameters]


result_list = []
for item in production_hyperparameters_list:
    results = transform(args.kind, item)
    for result_item in results:
        if result_item not in result_list:
            result_list.append(result_item)

dump(result_list, stdout)
