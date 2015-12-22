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

from collections import OrderedDict, namedtuple
import logging
from itertools import groupby

import numpy as np
from sklearn import metrics
from sklearn.cross_validation import LabelKFold
from scipy import stats

import mhcflurry
from mhcflurry.common import normalize_allele_name
from mhcflurry.data import indices_to_hotshot_encoding


PredictionScores = namedtuple("PredictionScores", "tau auc f1 accuracy")


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


def score_predictions(
        predicted_log_ic50,
        true_log_ic50,
        max_ic50):
    """Computes Kendall-tau, AUC, F1 score, and accuracy of predictions

    Parameters
    ----------
    predicted_log_ic50 : np.array

    true_log_ic50 : np.array

    max_ic50 : float

    Returns PredictionScores object with fields (tau, auc, f1, accuracy)
    """
    tau, _ = stats.kendalltau(predicted_log_ic50, true_log_ic50)
    assert not np.isnan(tau)
    true_ic50s = max_ic50 ** (1.0 - np.array(true_log_ic50))
    predicted_ic50s = max_ic50 ** (1.0 - np.array(predicted_log_ic50))

    true_binding_label = true_ic50s <= 500
    if true_binding_label.all() or not true_binding_label.any():
        logging.warn(
            ("Can't compute AUC, F1, accuracy without both"
             " negative and positive ground truth labels"))
        return PredictionScores(
            tau=tau,
            auc=0.5,
            f1=0.0,
            accuracy=0.0)

    auc = metrics.roc_auc_score(true_binding_label, predicted_log_ic50)
    predicted_binding_label = predicted_ic50s <= 500
    if predicted_binding_label.all() or not predicted_binding_label.any():
        logging.warn(
            ("Can't compute AUC, F1, or accuracy without both"
             " positive and negative predicted labels"))
        return PredictionScores(
            tau=tau,
            auc=auc,
            f1=0.0,
            accuracy=0.0)

    f1_score = metrics.f1_score(true_binding_label, predicted_binding_label)

    same_mask = true_binding_label == predicted_binding_label
    accuracy = np.mean(same_mask)
    return PredictionScores(
        tau=tau,
        auc=auc,
        f1=f1_score,
        accuracy=accuracy)


def train_model_and_return_scores(
        model,
        X_train,
        log_ic50_train,
        X_test,
        log_ic50_test,
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
    return score_predictions(
        predicted_log_ic50=pred,
        true_log_ic50=log_ic50_test,
        max_ic50=max_ic50)


def train_model_with_synthetic_data(
        model,
        n_training_epochs,
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

    assert len(combined_weights) == n_combined_samples, \
        "Expected combined_weights to have length %d but got shape = %s" % (
            n_combined_samples,
            combined_weights.shape)

    Y_combined = np.concatenate([Y_original, Y_synth])
    assert Y_combined.min() >= 0, \
        "Y should not contain negative numbers! Y.min() = %f" % (
            Y_combined.min(),)
    assert Y_combined.max() <= 1, \
        "Y should have max value 1.0, got Y.max() = %f" % (
            Y_combined.max(),)

    for epoch in range(n_training_epochs):
        # weights for synthetic points can be shrunk as:
        #  ~ 1 / (1+epoch)**2
        # or
        # 2.0 ** -epoch
        decay_factor = 2.0 ** -epoch
        # if the contribution of synthetic samples is less than a
        # thousandth of the actual data, then stop using it
        synth_contribution = total_synth_weights * decay_factor
        # only use synthetic data if it contributes at least 1/100th of
        # sample weight
        use_synth_data = synth_contribution > (total_original_weights / 100)
        if use_synth_data:
            combined_weights[n_actual_samples:] = (
                synthetic_sample_weights * decay_factor)
            model.fit(
                X_combined,
                Y_combined,
                sample_weight=combined_weights,
                nb_epoch=1,
                verbose=0)
        else:
            model.fit(
                X_original,
                Y_original,
                sample_weight=original_sample_weights,
                nb_epoch=1,
                verbose=0)

        Y_pred = model.predict(X_original)
        training_mse = ((Y_original - Y_pred) ** 2).mean()
        print(
            "-- Epoch %d/%d synth weight=%s, Training MSE %0.4f" % (
                epoch + 1,
                n_training_epochs,
                decay_factor if use_synth_data else 0,
                training_mse))


def peptide_group_indices(original_peptide_sequences):
    """
    Given a list of peptide sequences, some of which are identical,
    generate a sequence of index lists for the occurrence of distinct
    entries.
    """
    indices_and_sequences = enumerate(original_peptide_sequences)
    for _, subiter in groupby(indices_and_sequences, lambda pair: pair[1]):
        yield [idx for (idx, _) in subiter]


def collapse_peptide_group_affinities(
        predictions,
        true_values,
        original_peptide_sequences,
        combine_multiple_predictions_fn=np.median):
    """
    Given predictions and true log-transformed IC50 values for 9mers which may
    have been either elongated or shortened from peptide sequences of other
    lengths, collapse the log-transformed IC50 values to a single value
    per original peptide sequence.
    """
    assert len(predictions) == len(true_values), \
        ("Expected predictions (%d elements) to have same length"
         " as true values (%d elements)") % (len(predictions), len(true_values))
    assert len(predictions) == len(original_peptide_sequences), \
        ("Expected predictions (%d elements) to have same length"
         " as peptide sequences (%d elements)") % (
            len(predictions), len(original_peptide_sequences))

    collapsed_predictions = []
    collapsed_true_values = []
    for peptide_indices in peptide_group_indices(original_peptide_sequences):
        if len(peptide_indices) == 1:
            idx = peptide_indices[0]
            collapsed_predictions.append(predictions[idx])
            collapsed_true_values.append(true_values[idx])
        else:
            collapsed_predictions.append(
                combine_multiple_predictions_fn(predictions[peptide_indices]))
            collapsed_true_values.append(
                combine_multiple_predictions_fn(
                    true_values[peptide_indices]))
    return collapsed_predictions, collapsed_true_values


def kfold_cross_validation_of_model_fn_with_synthetic_data(
        make_model_fn,
        n_training_epochs,
        max_ic50,
        X_train,
        Y_train,
        source_peptides_train,
        X_synth,
        Y_synth,
        original_sample_weights,
        synthetic_sample_weights,
        n_cross_validation_folds,
        combine_multiple_predictions_fn=np.median):
    """
    Given a function which generates fresh copies of a model, use k-fold
    cross validation (stratified by the list of source peptide sequences) to
    evaluate the predictive performance of this model type.

    Returns a list of pairs, the first element is a trained model and the second
    element is a PredictionScores object with the following fields:
        ("tau", "f1", "auc", "accuracy")
    """
    assert len(X_train) == len(Y_train)
    assert len(source_peptides_train) == len(X_train)
    # randomly shuffle the training data first
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train, Y_train, weights_train = \
        X_train[indices], Y_train[indices], original_sample_weights[indices]
    source_peptides_train = [source_peptides_train[i] for i in indices]

    # we need to do cross-validation carefully since some of our training
    # samples may be extracted from the same longer peptide.
    # To avoid training vs. test contamination we do k-fold cross validation
    # stratified by the "source" peptides of each sample.
    cv_iterator = enumerate(
        LabelKFold(
            labels=source_peptides_train,
            n_folds=n_cross_validation_folds))
    results = []
    for fold_number, (fold_train_index, fold_test_index) in cv_iterator:
        model = make_model_fn()
        X_train_fold, X_test_fold = \
            X_train[fold_train_index], X_train[fold_test_index]
        weights_train_fold = weights_train[fold_train_index]
        Y_train_fold, Y_test_fold = \
            Y_train[fold_train_index], Y_train[fold_test_index]
        peptides_train_fold = [
            source_peptides_train[i]
            for i in fold_train_index
        ]
        peptides_test_fold = [
            source_peptides_train[i]
            for i in fold_test_index
        ]

        train_model_with_synthetic_data(
            model=model,
            n_training_epochs=n_training_epochs,
            X_original=X_train_fold,
            Y_original=Y_train_fold,
            X_synth=X_synth,
            Y_synth=Y_synth,
            original_sample_weights=weights_train_fold,
            synthetic_sample_weights=synthetic_sample_weights)
        Y_pred = model.predict(X_test_fold).flatten()
        # since some 'epitopes' are actually substrings of longer peptides
        # we need to coalesce those predictions by calling the function
        # `combine_multiple_predictions_fn` on the set of predictions
        # associated with one longer peptide
        collapsed_predictions, collapsed_true_values = \
            collapse_peptide_group_affinities(
                predictions=Y_pred,
                true_values=Y_test_fold,
                original_peptide_sequences=peptides_test_fold,
                combine_multiple_predictions_fn=combine_multiple_predictions_fn)
        scores = score_predictions(
            predicted_log_ic50=collapsed_predictions,
            true_log_ic50=collapsed_true_values,
            max_ic50=max_ic50)
        print("::: CV fold %d/%d (n_samples=%d, n_unique=%d): %s\n\n" % (
            fold_number + 1,
            n_cross_validation_folds,
            len(peptides_train_fold),
            len(set(peptides_train_fold)),
            scores))
        results.append((model, scores))
    return results


def average_prediction_scores_list(model_scores_list):
    """
    Given a list of (model, PredictionScores) pairs,
    returns a single PredictionScores object whose fields are
    the average across the list.
    """
    n = len(model_scores_list)
    return PredictionScores(
        tau=sum(x.tau for (_, x) in model_scores_list) / n,
        auc=sum(x.auc for (_, x) in model_scores_list) / n,
        f1=sum(x.f1 for (_, x) in model_scores_list) / n,
        accuracy=sum(x.accuracy for (_, x) in model_scores_list) / n)


def kfold_cross_validation_of_model_params_with_synthetic_data(
        X_original,
        Y_original,
        source_peptides_original,
        X_synth=None,
        Y_synth=None,
        original_sample_weights=None,
        synthetic_sample_weights=None,
        n_training_epochs=150,
        n_cross_validation_folds=5,
        embedding_dim_size=16,
        hidden_layer_size=50,
        dropout_probability=0.0,
        activation="tanh",
        max_ic50=50000.0):
    """
    Returns:
        - PredictionScores object with average Kendall-tau, AUC, F1,
        and accuracy across all cross-validation folds.
        - list of all (model, PredictionScores) objects from cross-validation
        folds
    """
    n_unique_samples = len(set(source_peptides_original))

    if n_cross_validation_folds > n_unique_samples:
        n_cross_validation_folds = n_unique_samples

    if original_sample_weights is None:
        original_sample_weights = np.ones(len(X_original), dtype=float)

    # if synthetic data is missing then make an empty array so all the
    # downstream code still works
    if X_synth is None:
        X_synth = np.array([[]])

    # if only the target values for synthetic data are missing then
    # treat all synthetic entries as negative examples
    if Y_synth is None:
        Y_synth = np.zeros(len(X_synth), dtype=float)

    if synthetic_sample_weights is None:
        synthetic_sample_weights = np.ones(len(X_synth), dtype=float)

    def make_model():
        return mhcflurry.feedforward.make_embedding_network(
            peptide_length=9,
            embedding_input_dim=20,
            embedding_output_dim=embedding_dim_size,
            layer_sizes=[hidden_layer_size],
            activation=activation,
            init="lecun_uniform",
            loss="mse",
            output_activation="sigmoid",
            dropout_probability=dropout_probability,
            optimizer=None,
            learning_rate=0.001)

    model_scores_list = kfold_cross_validation_of_model_fn_with_synthetic_data(
        make_model_fn=make_model,
        n_training_epochs=n_training_epochs,
        max_ic50=max_ic50,
        X_train=X_original,
        Y_train=Y_original,
        source_peptides_train=source_peptides_original,
        X_synth=X_synth,
        Y_synth=Y_synth,
        original_sample_weights=original_sample_weights,
        synthetic_sample_weights=synthetic_sample_weights,
        n_cross_validation_folds=n_cross_validation_folds)
    average_scores = average_prediction_scores_list(model_scores_list)
    return average_scores, model_scores_list
