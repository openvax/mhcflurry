import numpy as np
np.random.seed(0)

import json
import tempfile
import sys
from os.path import join
from os import mkdir, environ

import pandas

from mhcflurry.class1_allele_specific import cv_and_train_command
from mhcflurry import downloads, predict
from mhcflurry.class1_allele_specific.train import HYPERPARAMETER_DEFAULTS


def test_small_run():
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
        n_training_epochs=[20])

    imputer_args = {
        "imputation_method_name": "mice",
        "n_burn_in": 2,
        "n_imputations": 10,
        "n_nearest_columns": 10,
        "min_observations_per_peptide": 5,
        "min_observations_per_allele": 1000,  # limit the number of alleles
    }

    bdata2009 = downloads.get_path(
        "data_kim2014", "bdata.2009.mhci.public.1.txt")
    bdata_blind = downloads.get_path(
        "data_kim2014", "bdata.2013.mhci.public.blind.1.txt")

    mkdir(join(temp_dir, "models"))

    args = [
        "--model-architectures", write_json(models, "models.json"),
        "--imputer-description", write_json(imputer_args, "imputer.json"),
        "--train-data", bdata2009,
        "--test-data", bdata_blind,
        "--out-cv-results", join(temp_dir, "cv.csv"),
        "--out-production-results", join(temp_dir, "production.csv"),
        "--out-models", join(temp_dir, "models"),
        "--cv-num-folds", "2",
        "--alleles", "HLA-A0201", "HLA-A0301",
        "--verbose",
        "--num-local-threads", "1",
    ]
    print("Running cv_and_train_command with args: %s " % str(args))

    cv_and_train_command.run(args)
    verify_trained_models(base_temp_dir)


def verify_trained_models(base_temp_dir):
    temp_dir = join(base_temp_dir, "models_class1_allele_specific_single")

    cv = pandas.read_csv(join(temp_dir, "cv.csv"))
    production = pandas.read_csv(join(temp_dir, "production.csv"))

    print(cv)
    print(production)

    assert (production.test_auc > 0.85).all(), (
        "Test AUC too low: %s" % production.test_auc)
    assert (production.train_auc > 0.90).all(), (
        "Train AUC too low: %s" % production.train_auc)

    # Swap in our trained models as if they were the downloadable production
    # models for MHCflurry.
    old_dir = environ.get("MHCFLURRY_DOWNLOADS_DIR", "")
    try:
        environ["MHCFLURRY_DOWNLOADS_DIR"] = base_temp_dir
        downloads.configure()

        data = pandas.DataFrame([
            # From Table 2 of:
            #   "HLA-A2-binding peptides cross-react not only within the A2
            #   subgroup but also with other HLA-A-Locus allelic products"
            # http://www.sciencedirect.com/science/article/pii/0198885994902550
            ("HLA-A0201", "GILGFVFTL", True),
            ("HLA-A0201", "EVAPPLLFV", True),
            ("HLA-A0201", "TIAPFGIFGTNY", True),
            ("HLA-A0201", "EPRGSDIAG", False),
            ("HLA-A0201", "YEFGTSSCRL", False),
            ("HLA-A0201", "LLGLPAAEY", False),
        ], columns=("allele", "peptide", "binder"))

        data["prediction"] = predict(data.allele, data.peptide).Prediction
        print(data)
        mean_binder = data.ix[data.binder].prediction.mean()
        mean_nonbinder = data.ix[~data.binder].prediction.mean()

        assert mean_binder < mean_nonbinder, (
            "Known binders predicted mean %f >= known non-binders mean %f" % (
                mean_binder, mean_nonbinder))
    finally:
        environ["MHCFLURRY_DOWNLOADS_DIR"] = old_dir
        downloads.configure()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Make it possible to continue from a previous run since this test
        # takes a while to run.
        base_temp_dir = sys.argv[1]
        print("Verifying models from: %s" % base_temp_dir)
        verify_trained_models(base_temp_dir)
    else:
        test_small_run()
