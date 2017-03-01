import sys
from mhcflurry.class1_allele_specific_ensemble import HYPERPARAMETER_DEFAULTS
import json

models = HYPERPARAMETER_DEFAULTS.models_grid(
    impute=[False, True],
    activation=["tanh"],
    layer_sizes=[[12], [64], [128]],
    embedding_output_dim=[8, 32, 64],
    dropout_probability=[0, .1, .25],
    fraction_negative=[0, .1, .2],
    n_training_epochs=[250],

    # Imputation arguments
    impute_method=["mice"],
    imputer_args=[
        # Arguments specific to imputation method (mice)
        {"n_burn_in": 5, "n_imputations": 50, "n_nearest_columns": 25}
    ],
    impute_min_observations_per_peptide=[1],
    impute_min_observations_per_allele=[1])

sys.stderr.write("Models: %d\n" % len(models))
print(json.dumps(models, indent=4))
