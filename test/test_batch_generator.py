import pandas
import numpy

from mhcflurry.batch_generator import (
    MultiallelicMassSpecBatchGenerator)

from numpy.testing import assert_equal


def test_basic():
    planner = MultiallelicMassSpecBatchGenerator(
        hyperparameters=dict(
            batch_generator_validation_split=0.2,
            batch_generator_batch_size=10,
            batch_generator_affinity_fraction=0.5))

    exp1_alleles = ["HLA-A*03:01", "HLA-B*07:02", "HLA-C*02:01"]
    exp2_alleles = ["HLA-A*02:01", "HLA-B*27:01", "HLA-C*02:01"]

    df = pandas.DataFrame(dict(
        affinities_mask=([True] * 4) + ([False] * 6),
        experiment_names=([None] * 4) + (["exp1"] * 2) + (["exp2"] * 4),
        alleles_matrix=[
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
        is_binder=[
            True, True, False, False, True, False, True, False, True, False,
        ]))
    planner.plan(**df.to_dict("list"))
    print(planner.summary())

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
    print(df)

    for ((kind, batch_num), batch_df) in df.groupby(["kind", "batch"]):
        if not batch_df.affinities_mask.all():
            # Test each batch has at most one multiallelic ms experiment.
            assert_equal(
                batch_df.loc[
                    ~batch_df.affinities_mask
                ].experiment_names.nunique(), 1)

    #import ipdb;ipdb.set_trace()

