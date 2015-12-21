# Copyright (c) 2015. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict

import numpy as np
from sklearn.metrics import roc_auc_score

import mhcflurry
from mhcflurry.common import normalize_allele_name
from mhcflurry.data_helpers import indices_to_hotshot_encoding


def encode_allele_dataset(
        allele_dataset,
        max_ic50,
        binary_encoding=False):
    """
    Parameters
    ----------
    allele_dataset : AlleleDataset
        Named tuple with fields "X" and "ic50"
    max_ic50 : float
        Largest IC50 value predictor should return
    binary_encoding : bool (default = False)
        If True, use a binary 1-of-k encoding of amino acids, otherwise
        expect a vector embedding to use integer indices.

    Returns (X, Y_log_ic50, binder_label)
    """
    X_allele = allele_dataset.X
    ic50_allele = allele_dataset.ic50
    if binary_encoding:
        X_allele = indices_to_hotshot_encoding(X_allele, n_indices=20)
    Y_allele = 1.0 - np.minimum(1.0, np.log(ic50_allele) / np.log(max_ic50))
    return (X_allele, Y_allele, ic50_allele)


def encode_allele_datasets(
        allele_datasets,
        max_ic50,
        binary_encoding=False):
    """
    Parameters
    ----------
    allele_dataset : AlleleDataset
        Named tuple with fields "X" and "ic50"
    max_ic50 : float
        Largest IC50 value predictor should return
    binary_encoding : bool (default = False)
        If True, use a binary 1-of-k encoding of amino acids, otherwise
        expect a vector embedding to use integer indices.

    Returns three dictionarys
        - mapping from allele name to X (features)
        - mapping from allele name to Y_log_ic50 (continuous outputs)
        - mapping from allele name to binder_label (binary outputs)
    """
    X_dict = OrderedDict()
    Y_log_ic50_dict = OrderedDict([])
    ic50_dict = OrderedDict([])
    for (allele_name, dataset) in allele_datasets.items():
        allele_name = normalize_allele_name(allele_name)
        (X, Y_log_ic50, Y_ic50) = encode_allele_dataset(
            dataset,
            max_ic50=max_ic50,
            binary_encoding=binary_encoding)
        X_dict[allele_name] = X
        Y_log_ic50_dict[allele_name] = Y_log_ic50
        ic50_dict[allele_name] = Y_ic50
    return (X_dict, Y_log_ic50_dict, ic50_dict)


def f1_score(true_label, label_pred):
    tp = (true_label & label_pred).sum()
    fp = ((~true_label) & label_pred).sum()
    fn = (true_label & (~label_pred)).sum()
    recall = (tp / float(tp + fn)) if (tp + fn) > 0 else 0.0
    precision = (tp / float(tp + fp)) if (tp + fp) > 0 else 0.0
    if (precision + recall) > 0:
        return (2 * precision * recall) / (precision + recall)
    else:
        return 0.0


def score_predictions(predicted_log_ic50, true_label, max_ic50):
    """Computes accuracy, AUC, and F1 score of predictions"""
    auc = roc_auc_score(true_label, predicted_log_ic50)
    ic50_pred = max_ic50 ** (1.0 - predicted_log_ic50)
    label_pred = (ic50_pred <= 500)
    same_mask = true_label == label_pred
    accuracy = np.mean(same_mask)
    f1 = f1_score(true_label, label_pred)
    return accuracy, auc, f1


def train_model_and_return_scores(
        model,
        X_train,
        log_ic50_train,
        X_test,
        binder_label_test,
        n_training_epochs,
        minibatch_size,
        max_ic50):
    model.fit(
        X_train,
        log_ic50_train,
        nb_epoch=n_training_epochs,
        verbose=0,
        batch_size=minibatch_size)
    pred = model.predict(X_test).flatten()
    accuracy, auc, f1_score = score_predictions(
        predicted_log_ic50=pred,
        true_label=binder_label_test,
        max_ic50=max_ic50)
    return (accuracy, auc, f1_score)


def train_model_with_synthetic_data(
        model,
        n_training_epochs,
        max_ic50,
        X_original,
        Y_original,
        X_synth,
        Y_synth,
        original_sample_weights,
        synthetic_sample_weights):
    total_synth_weights = synthetic_sample_weights.sum()
    total_original_weights = original_sample_weights.sum()
    print("Mean Y=%f, Y_synth=%f, weight=%f, weight_synth=%f" % (
        np.mean(Y_original),
        np.mean(Y_synth),
        np.mean(original_sample_weights),
        np.mean(synthetic_sample_weights)))
    combined_weights = np.concatenate([
        original_sample_weights,
        synthetic_sample_weights
    ])
    n_actual_samples, n_actual_dims = X_original.shape
    n_synth_samples, n_synth_dims = X_synth.shape
    assert n_actual_dims == n_synth_dims, \
        "Mismatch between # of actual dims %d and synthetic dims %d" % (
            n_actual_dims, n_synth_dims)
    X_combined = np.vstack([X_original, X_synth])
    n_combined_samples = n_actual_samples + n_synth_samples
    assert X_combined.shape[0] == n_combined_samples, \
        "Expected %d samples but got data array with shape %s" % (
            n_actual_samples + n_synth_samples, X_combined.shape)
    Y_combined = np.concatenate([Y_original, Y_synth])
    assert Y_combined.min() >= 0, \
        "Y should not contain negative numbers! Y.min() = %f" % (
            Y_combined.min(),)
    assert Y_combined.max() <= 1, \
        "Y should have max value 1.0, got Y.max() = %f" % (
            Y_combined.max(),)
    combined_weights = np.concatenate([
        original_sample_weights,
        synthetic_sample_weights
    ])

    assert len(combined_weights) == n_combined_samples

    for epoch in range(n_training_epochs):
        # weights for synthetic points can be shrunk as:
        #  ~ 1 / (1+epoch)**2
        # or
        # 2.0 ** -epoch
        decay_factor = 2.0 ** -epoch
        # if the contribution of synthetic samples is less than a
        # thousandth of the actual data, then stop using it
        synth_contribution = total_synth_weights * decay_factor
        if synth_contribution < total_original_weights / 1000:
            print("Epoch %d, using only actual data" % (epoch + 1,))
            model.fit(
                X_original,
                Y_original,
                sample_weight=original_sample_weights,
                nb_epoch=1,
                verbose=0)
        else:
            print("Epoch %d, synth decay factor = %f" % (
                epoch + 1, decay_factor))
            combined_weights[n_actual_samples:] = (
                synthetic_sample_weights * decay_factor)
            model.fit(
                X_combined,
                Y_combined,
                sample_weight=combined_weights,
                nb_epoch=1,
                verbose=0)
        Y_pred = model.predict(X_original)
        training_mse = ((Y_original - Y_pred) ** 2).mean()
        print(
            "-- Epoch %d/%d Training MSE %0.4f" % (
                epoch + 1,
                n_training_epochs,
                training_mse))


def create_and_evaluate_model_with_synthetic_data(
        X_original,
        Y_original,
        X_synth,
        Y_synth,
        original_sample_weights=None,
        synthetic_sample_weights=None,
        n_training_epochs=150,
        embedding_dim_size=16,
        hidden_layer_size=50,
        dropout_probability=0.0,
        max_ic50=50000.0):
    if original_sample_weights is None:
        original_sample_weights = np.ones(len(X_original), dtype=float)
    if synthetic_sample_weights is None:
        synthetic_sample_weights = np.ones(len(X_synth), dtype=float)
    model = mhcflurry.feedforward.make_embedding_network(
        peptide_length=9,
        embedding_input_dim=20,
        embedding_output_dim=embedding_dim_size,
        layer_sizes=[hidden_layer_size],
        activation="tanh",
        init="lecun_uniform",
        loss="mse",
        output_activation="sigmoid",
        dropout_probability=dropout_probability,
        optimizer=None,
        learning_rate=0.001)
    train_model_with_synthetic_data(
        model=model,
        n_training_epochs=n_training_epochs,
        max_ic50=max_ic50,
        X_original=X_original,
        Y_original=Y_original,
        X_synth=X_synth,
        Y_synth=Y_synth,
        original_sample_weights=original_sample_weights,
        synthetic_sample_weights=synthetic_sample_weights)
