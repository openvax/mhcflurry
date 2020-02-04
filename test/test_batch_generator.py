import logging
logging.getLogger('matplotlib').disabled = True
logging.getLogger('tensorflow').disabled = True

import os
import collections
import time
import cProfile
import pstats

import pandas
import numpy

from mhcflurry.downloads import get_path
from mhcflurry.batch_generator import (
    MultiallelicMassSpecBatchGenerator)
from mhcflurry.regression_target import to_ic50
from mhcflurry import Class1AffinityPredictor

from numpy.testing import assert_equal
from nose.tools import assert_greater, assert_less


def data_path(name):
    '''
    Return the absolute path to a file in the test/data directory.
    The name specified should be relative to test/data.
    '''
    return os.path.join(os.path.dirname(__file__), "data", name)


def test_basic_repeat():
    for _ in range(100):
        test_basic()


def test_basic():
    batch_size = 7
    validation_split = 0.2
    planner = MultiallelicMassSpecBatchGenerator(
        hyperparameters=dict(
            batch_generator_validation_split=validation_split,
            batch_generator_batch_size=batch_size,
            batch_generator_affinity_fraction=0.5))

    exp1_alleles = ["HLA-A*03:01", "HLA-B*07:02", "HLA-C*02:01"]
    exp2_alleles = ["HLA-A*02:01", "HLA-B*27:01", "HLA-C*02:01"]

    df = pandas.DataFrame(dict(
        affinities_mask=([True] * 14) + ([False] * 6),
        experiment_names=([None] * 14) + (["exp1"] * 2) + (["exp2"] * 4),
        alleles_matrix=[["HLA-C*07:01", None, None]] * 10 + [
            ["HLA-A*02:01", None, None],
            ["HLA-A*02:01", None, None],
            ["HLA-A*03:01", None, None],
            ["HLA-A*03:01", None, None],
            exp1_alleles,
            exp1_alleles,
            exp2_alleles,
            exp2_alleles,
            exp2_alleles,
            exp2_alleles,
        ],
        is_binder=[False, True] * 5 + [
            True, True, False, False, True, False, True, False, True, False,
        ]))
    df = pandas.concat([df, df], ignore_index=True)
    df = pandas.concat([df, df], ignore_index=True)

    planner.plan(**df.to_dict("list"))

    assert_equal(
        planner.num_train_batches,
        numpy.ceil(len(df) * (1 - validation_split) / batch_size))
    assert_equal(
        planner.num_test_batches,
        numpy.ceil(len(df) * validation_split / batch_size))

    (train_iter, test_iter) = planner.get_train_and_test_generators(
        x_dict={
            "idx": numpy.arange(len(df)),
        },
        y_list=[])

    for (kind, it) in [("train", train_iter), ("test", test_iter)]:
        for (i, (x_item, y_item)) in enumerate(it):
            idx = x_item["idx"]
            df.loc[idx, "kind"] = kind
            df.loc[idx, "idx"] = idx
            df.loc[idx, "batch"] = i
    df["idx"] = df.idx.astype(int)
    df["batch"] = df.batch.astype(int)

    assert_equal(df.kind.value_counts()["test"], len(df) * validation_split)
    assert_equal(df.kind.value_counts()["train"], len(df) * (1 - validation_split))

    experiment_allele_colocations = collections.defaultdict(int)
    for ((kind, batch_num), batch_df) in df.groupby(["kind", "batch"]):
        if not batch_df.affinities_mask.all():
            # Test each batch has at most one multiallelic ms experiment.
            names = batch_df.loc[
                ~batch_df.affinities_mask
            ].experiment_names.unique()
            assert_equal(len(names), 1)
            (experiment,) = names
            if batch_df.affinities_mask.any():
                # Test experiments are matched to the correct affinity alleles.
                affinity_alleles = batch_df.loc[
                    batch_df.affinities_mask
                ].alleles_matrix.str.get(0).values
                for allele in affinity_alleles:
                    experiment_allele_colocations[(experiment, allele)] += 1

    assert_greater(
        experiment_allele_colocations[('exp1', 'HLA-A*03:01')],
        experiment_allele_colocations[('exp1', 'HLA-A*02:01')])
    assert_less(
        experiment_allele_colocations[('exp2', 'HLA-A*03:01')],
        experiment_allele_colocations[('exp2', 'HLA-A*02:01')])

