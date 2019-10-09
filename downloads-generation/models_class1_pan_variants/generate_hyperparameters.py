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
    choices=('single_hidden', 'no_pretrain'),
    help="Hyperameters variant to output")

args = parser.parse_args(argv[1:])

with open(args.production_hyperparameters) as fd:
    production_hyperparameters_list = load(fd)


def transform_to_single_hidden(hyperparameters):
    result = []
    for size in [64, 128, 256, 1024]:
        new_hyperparameters = deepcopy(hyperparameters)
        new_hyperparameters['layer_sizes'] = [size]
        result.append(new_hyperparameters)
    return result


def transform_to_no_pretrain(hyperparameters):
    result = deepcopy(hyperparameters)
    result['train_data']['pretrain'] = False
    return [result]


TRANSFORMS={
    "single_hidden": transform_to_single_hidden,
    "no_pretrain": transform_to_no_pretrain,
}

transform = TRANSFORMS[args.kind]

result_list = []
for item in production_hyperparameters_list:
    results = transform(item)
    for result_item in results:
        if result_item not in result_list:
            result_list.append(result_item)

dump(result_list, stdout)
