import tempfile
import shutil
import os
import time
import cProfile

import json
from os.path import join
from os import mkdir

from numpy.testing import assert_allclose, assert_equal
from nose.tools import eq_

from . import make_random_peptides

from mhcflurry.class1_allele_specific import scoring
from mhcflurry.measurement_collection import MeasurementCollection
from mhcflurry.class1_allele_specific_ensemble import train_command
from mhcflurry.affinity_measurement_dataset import AffinityMeasurementDataset
from mhcflurry.downloads import get_path
from mhcflurry.amino_acid import common_amino_acid_letters
from mhcflurry \
    .class1_allele_specific_ensemble \
    .class1_ensemble_multi_allele_predictor import (
        Class1EnsembleMultiAllelePredictor,
        get_downloaded_predictor,
        HYPERPARAMETER_DEFAULTS)


def test_basic():
    model_hyperparameters = HYPERPARAMETER_DEFAULTS.models_grid(
        impute=[False, True],
        activation=["tanh"],
        layer_sizes=[[4], [16]],
        embedding_output_dim=[16],
        dropout_probability=[.25],
        n_training_epochs=[20])
    model = Class1EnsembleMultiAllelePredictor(
        ensemble_size=3,
        hyperparameters_to_search=model_hyperparameters)
    print(model)

    dataset = AffinityMeasurementDataset.from_csv(get_path(
        "data_combined_iedb_kim2014", "combined_human_class1_dataset.csv"))
    sub_dataset = AffinityMeasurementDataset(
        dataset._df.ix[
            (dataset._df.allele.isin(["HLA-A0101", "HLA-A0205"])) &
            (dataset._df.peptide.str.len() == 9)
        ])

    mc = MeasurementCollection.from_dataset(sub_dataset)
    print(model.description())
    print("Now fitting.")
    model.fit(mc)

    model2 = Class1EnsembleMultiAllelePredictor(
        ensemble_size=3,
        hyperparameters_to_search=model_hyperparameters)
    model2.fit(mc)
    print(set(model.manifest_df.all_alleles_train_data_hash))
    assert_equal(
        set(model.manifest_df.all_alleles_train_data_hash),
        set(model2.manifest_df.all_alleles_train_data_hash))
    assert_equal(
        set(model.manifest_df.all_alleles_test_data_hash),
        set(model2.manifest_df.all_alleles_test_data_hash))
    assert_equal(
        set(model.manifest_df.all_alleles_imputed_data_hash),
        set(model2.manifest_df.all_alleles_imputed_data_hash))

    print(model.description())
    ic50_pred = model.predict(mc)
    ic50_true = mc.df.measurement_value

    scores = scoring.make_scores(ic50_true, ic50_pred)
    print(scores)
    assert scores['auc'] > 0.85, "Expected higher AUC"

    # test save and restore
    try:
        tmpdir = tempfile.mkdtemp(prefix="mhcflurry-test")
        model.write_fit(
            tmpdir,
            all_models_csv=os.path.join(tmpdir, "models.csv"))
        model2 = Class1EnsembleMultiAllelePredictor.load_fit(
            tmpdir,
            os.path.join(tmpdir, "models.csv"))
    finally:
        shutil.rmtree(tmpdir)

    eq_(model.ensemble_size, model2.ensemble_size)
    eq_(model.supported_alleles, model2.supported_alleles)
    eq_(model.hyperparameters_to_search, model2.hyperparameters_to_search)
    ic50_pred2 = model.predict(mc)
    assert_allclose(ic50_pred, ic50_pred2, rtol=1e-06)


def test_prediction_performance():
    use_profiling = False

    def evaluate(context, s):
        if use_profiling:
            return cProfile.runctx(s, globals(), context, sort="cumtime")
        else:
            return eval(s, globals(), context)

    start = time.time()
    predictor = get_downloaded_predictor()
    print("\nInstantiated ensemble predictor in %0.2f sec" % (time.time() - start))

    start = time.time()
    evaluate({'predictor': predictor}, 'predictor.models_for_allele(("HLA-A*01:01"))')
    print("Loaded ensemble for allele in %0.2f sec" % (time.time() - start))

    start = time.time()
    evaluate({'predictor': predictor}, 'predictor.predict_for_allele("HLA-A*01:01", ["ESDPIVAQY"])')
    print("Generated 1 prediction in %0.2f sec" % (time.time() - start))

    start = time.time()
    peptides = make_random_peptides(10000)
    evaluate(
        {'predictor': predictor, 'peptides': peptides},
        'predictor.predict_for_allele("HLA-A*01:01", peptides)')
    print("Generated %d predictions in %0.2f sec" % (
        len(peptides), time.time() - start))


def test_train_command():
    base_temp_dir = tempfile.mkdtemp()
    temp_dir = join(base_temp_dir, "models_class1_allele_specific_single")
    mkdir(temp_dir)

    def write_json(payload, filename):
        path = join(temp_dir, filename)
        with open(path, 'w') as fd:
            json.dump(payload, fd)
        return path

    models = HYPERPARAMETER_DEFAULTS.models_grid(
        impute=[False, True],
        activation=["tanh"],
        layer_sizes=[[4], [8]],
        embedding_output_dim=[16],
        dropout_probability=[.25],
        n_training_epochs=[10],
        imputer_args=[{"n_burn_in": 2, "n_imputations": 10}],
        impute_min_observations_per_peptide=[1],
        impute_min_observations_per_allele=[1],
    )
    print("Model selection will be over %d models" % len(models))

    bdata2009 = get_path(
        "data_kim2014", "bdata.2009.mhci.public.1.txt")
    mkdir(join(temp_dir, "models"))

    args = [
        '--parallel-backend', 'local-threads',
        "--model-architectures", write_json(models, "models.json"),
        "--train-data", bdata2009,
        "--out-manifest", join(temp_dir, "models.csv"),
        "--out-models", join(temp_dir, "models"),
        "--alleles", "HLA-A0201", "HLA-A0301",
        "--verbose",
        "--num-local-threads", "1",
        "--ensemble-size", "2",
        "--target-tasks", "1",
    ]
    print("Running train command with args: %s " % str(args))

    train_command.run(args)
