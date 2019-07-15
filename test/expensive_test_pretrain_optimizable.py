# Expensive test - not run by default.

from mhcflurry import train_pan_allele_models_command
from mhcflurry.downloads import get_path
from mhcflurry.allele_encoding import AlleleEncoding

import pandas
import numpy

PRETRAIN_DATA_PATH = get_path(
    "random_peptide_predictions", "predictions.csv.bz2")

FULL_TRAIN_DF = pandas.read_csv(
        get_path(
            "data_curated",
            "curated_training_data.no_mass_spec.csv.bz2"))
TRAIN_DF = FULL_TRAIN_DF.loc[
    (FULL_TRAIN_DF.peptide.str.len() >= 8) &
    (FULL_TRAIN_DF.peptide.str.len() <= 15)
]
ALLELE_SEQUENCES = pandas.read_csv(
    get_path("allele_sequences", "allele_sequences.csv"),
    index_col=0).sequence
ALLELE_SEQUENCES = ALLELE_SEQUENCES.loc[
    ALLELE_SEQUENCES.index.isin(TRAIN_DF.allele)
]
TRAIN_DF = TRAIN_DF.loc[
    TRAIN_DF.allele.isin(ALLELE_SEQUENCES.index)
]
FOLDS_DF = pandas.DataFrame(index=TRAIN_DF.index)
FOLDS_DF["fold_0"] = True

HYPERPARAMTERS = {
    'activation': 'tanh', 'allele_dense_layer_sizes': [],
    'batch_normalization': False,
    'dense_layer_l1_regularization': 9.999999999999999e-11,
    'dense_layer_l2_regularization': 0.0, 'dropout_probability': 0.5,
    'early_stopping': True, 'init': 'glorot_uniform',
    'layer_sizes': [1024, 512], 'learning_rate': None,
    'locally_connected_layers': [], 'loss': 'custom:mse_with_inequalities',
    'max_epochs': 1, 'min_delta': 0.0, 'minibatch_size': 128,
    'optimizer': 'rmsprop', 'output_activation': 'sigmoid', 'patience': 20,
    'peptide_allele_merge_activation': '',
    'peptide_allele_merge_method': 'concatenate',
    'peptide_amino_acid_encoding': 'BLOSUM62', 'peptide_dense_layer_sizes': [],
    'peptide_encoding': {'alignment_method': 'left_pad_centered_right_pad',
                         'max_length': 15, 'vector_encoding_name': 'BLOSUM62'},
    'random_negative_affinity_max': 50000.0,
    'random_negative_affinity_min': 20000.0, 'random_negative_constant': 25,
    'random_negative_distribution_smoothing': 0.0,
    'random_negative_match_distribution': True, 'random_negative_rate': 0.2,
    'train_data': {'pretrain': True,
                   'pretrain_max_epochs': 1,
                   'pretrain_peptides_per_epoch': 1024,
                   'pretrain_steps_per_epoch': 16},
    'validation_split': 0.1,
}


def test_optimizable():
    predictor = train_pan_allele_models_command.train_model(
        work_item_num=0,
        num_work_items=1,
        architecture_num=0,
        num_architectures=1,
        fold_num=0,
        num_folds=1,
        replicate_num=0,
        num_replicates=1,
        hyperparameters=HYPERPARAMTERS,
        pretrain_data_filename=PRETRAIN_DATA_PATH,
        verbose=1,
        progress_print_interval=5.0,
        predictor=None,
        save_to=None,
        constant_data={
            'train_data': TRAIN_DF,
            'folds_df': FOLDS_DF,
            'allele_encoding': AlleleEncoding(
                alleles=ALLELE_SEQUENCES.index.values,
                allele_to_sequence=ALLELE_SEQUENCES.to_dict()),
        },
    )
    (network,) = predictor.neural_networks
    pretrain_val_loss = network.fit_info[-1]['training_info']["val_loss"][-1]
    print(predictor)
    print(pretrain_val_loss)
    numpy.testing.assert_array_less(pretrain_val_loss, 0.1)


if __name__ == "__main__":
    test_optimizable()
