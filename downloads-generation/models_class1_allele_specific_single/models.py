import sys
from mhcflurry.class1_allele_specific.train import HYPERPARAMETER_DEFAULTS
import json

models = HYPERPARAMETER_DEFAULTS.models_grid(
    #impute=[False, True],
    impute=[False],
    activation=["tanh"],
    layer_sizes=[[12], [64], [128]],
    embedding_output_dim=[8, 32, 64],
    dropout_probability=[0, .1, .25],
    # fraction_negative=[0, .1, .2],
    n_training_epochs=[250])

sys.stderr.write("Models: %d\n" % len(models))
print(json.dumps(models, indent=4))
