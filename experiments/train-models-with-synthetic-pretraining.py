#!/usr/bin/env python
#
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


import mhcflurry
import numpy as np
import sklearn.metrics


def get_extra_data(allele, train_datasets, expanded_predictions):
    original_dataset = train_datasets[allele]
    original_peptides = set(original_dataset.peptides)
    expanded_allele_affinities = expanded_predictions[allele]
    extra_affinities = {
        k: v
        for (k, v) in expanded_allele_affinities.items()
        if k not in original_peptides
    }
    extra_peptides = list(extra_affinities.keys())
    extra_values = list(extra_affinities.values())
    extra_X = mhcflurry.data_helpers.index_encoding(extra_peptides, peptide_length=9)
    extra_Y = np.array(extra_values)
    return extra_X, extra_Y


def data_augmentation(
        X,
        Y,
        extra_X,
        extra_Y,
        fraction=0.5,
        niters=10,
        extra_sample_weight=0.1,
        nb_epoch=50,
        nn=True,
        hidden_layer_size=5,
        max_ic50=50000.0):
    n = len(Y)
    aucs = []
    f1s = []
    n_originals = []
    for _ in range(niters):
        mask = np.random.rand(n) <= fraction
        X_train = X[mask]
        X_test = X[~mask]
        Y_train = Y[mask]
        Y_test = Y[~mask]
        test_ic50 = max_ic50 ** (1 - Y_test)
        test_label = test_ic50 <= 500
        if test_label.all() or not test_label.any():
            continue
        n_original = mask.sum()
        print("Keeping %d original training samples" % n_original)
        X_train_combined = np.vstack([X_train, extra_X])
        Y_train_combined = np.concatenate([Y_train, extra_Y])
        print("Combined shape: %s" % (X_train_combined.shape,))
        assert len(X_train_combined) == len(Y_train_combined)
        # initialize weights to count synthesized and actual data equally
        # but weight on synthesized points will decay across training epochs
        weight = np.ones(len(Y_train_combined))
        model = mhcflurry.feedforward.make_embedding_network(
            layer_sizes=[hidden_layer_size],
            embedding_output_dim=10,
            activation="tanh")
        for i in range(nb_epoch):
            # decay weight as training progresses
            weight[n_original:] = (
                extra_sample_weight
                if extra_sample_weight is not None
                else 1.0 / (i + 1) ** 2
            )
            model.fit(
                X_train_combined,
                Y_train_combined,
                sample_weight=weight,
                shuffle=True,
                nb_epoch=1,
                verbose=0)
        pred = model.predict(X_test)

        pred_ic50 = max_ic50 ** (1 - pred)
        pred_label = pred_ic50 <= 500
        mse = sklearn.metrics.mean_squared_error(Y_test, pred)
        auc = sklearn.metrics.roc_auc_score(test_label, pred)

        if pred_label.all() or not pred_label.any():
            f1 = 0
        else:
            f1 = sklearn.metrics.f1_score(test_label, pred_label)
        print("MSE=%0.4f, AUC=%0.4f, F1=%0.4f" % (mse, auc, f1))
        n_originals.append(n_original)
        aucs.append(auc)
        f1s.append(f1)
    return aucs, f1s, n_originals
