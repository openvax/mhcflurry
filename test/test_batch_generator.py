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

from mhcflurry.allele_encoding import MultipleAlleleEncoding
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


def test_large(sample_rate=1.0):
    multi_train_df = pandas.read_csv(
        data_path("multiallelic_ms.benchmark1.csv.bz2"))
    multi_train_df["label"] = multi_train_df.hit
    multi_train_df["is_affinity"] = False

    sample_table = multi_train_df.loc[
        multi_train_df.label == True
    ].drop_duplicates("sample_id").set_index("sample_id").loc[
        multi_train_df.sample_id.unique()
    ]
    grouped = multi_train_df.groupby("sample_id").nunique()
    for col in sample_table.columns:
        if (grouped[col] > 1).any():
            del sample_table[col]
    sample_table["alleles"] = sample_table.hla.str.split()

    pan_train_df = pandas.read_csv(
        get_path(
            "models_class1_pan", "models.combined/train_data.csv.bz2"))
    pan_sub_train_df = pan_train_df
    pan_sub_train_df["label"] = pan_sub_train_df["measurement_value"]
    del pan_sub_train_df["measurement_value"]
    pan_sub_train_df["is_affinity"] = True

    pan_sub_train_df = pan_sub_train_df.sample(frac=sample_rate)
    multi_train_df = multi_train_df.sample(frac=sample_rate)

    pan_predictor = Class1AffinityPredictor.load(
        get_path("models_class1_pan", "models.combined"),
        optimization_level=0,
        max_models=1)

    allele_encoding = MultipleAlleleEncoding(
        experiment_names=multi_train_df.sample_id.values,
        experiment_to_allele_list=sample_table.alleles.to_dict(),
        max_alleles_per_experiment=sample_table.alleles.str.len().max(),
        allele_to_sequence=pan_predictor.allele_to_sequence,
    )
    allele_encoding.append_alleles(pan_sub_train_df.allele.values)
    allele_encoding = allele_encoding.compact()

    combined_train_df = pandas.concat(
        [multi_train_df, pan_sub_train_df], ignore_index=True, sort=True)

    print("Total size", combined_train_df)

    planner = MultiallelicMassSpecBatchGenerator(
        hyperparameters=dict(
            batch_generator_validation_split=0.2,
            batch_generator_batch_size=128,
            batch_generator_affinity_fraction=0.5))

    s = time.time()
    profiler = cProfile.Profile()
    profiler.enable()
    planner.plan(
        affinities_mask=combined_train_df.is_affinity.values,
        experiment_names=combined_train_df.sample_id.values,
        alleles_matrix=allele_encoding.alleles,
        is_binder=numpy.where(
            combined_train_df.is_affinity.values,
            combined_train_df.label.values,
            to_ic50(combined_train_df.label.values)) < 1000.0)
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime").reverse_order().print_stats()
    print(planner.summary())
    print("Planning took [sec]: ", time.time() - s)

    (train_iter, test_iter) = planner.get_train_and_test_generators(
        x_dict={
            "idx": numpy.arange(len(combined_train_df)),
        },
        y_list=[])

    train_batch_sizes = []
    indices_total = numpy.zeros(len(combined_train_df))
    for (kind, it) in [("train", train_iter), ("test", test_iter)]:
        for (i, (x_item, y_item)) in enumerate(it):
            idx = x_item["idx"]
            indices_total[idx] += 1
            batch_df = combined_train_df.iloc[idx]
            if not batch_df.is_affinity.all():
                # Test each batch has at most one multiallelic ms experiment.
                assert_equal(
                    batch_df.loc[~batch_df.is_affinity].sample_id.nunique(), 1)
            if kind == "train":
                train_batch_sizes.append(len(batch_df))

    # At most one short batch.
    assert_less(sum(b != 128 for b in train_batch_sizes), 2)
    assert_greater(
        sum(b == 128 for b in train_batch_sizes), len(train_batch_sizes) - 2)

    # Each point used exactly once.
    assert_equal(
        indices_total, numpy.ones(len(combined_train_df)))
