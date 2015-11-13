import argparse

import fancyimpute
import numpy as np
import mhcflurry

from dataset_paths import PETERS2009_CSV_PATH

parser = argparse.ArgumentParser()

parser.add_argument(
    "--binding-data-csv",
    default=PETERS2009_CSV_PATH)

parser.add_argument(
    "--fill-missing-values",
    action="store_true",
    default=False)


parser.add_argument(
    "--min-overlap-weight",
    default=2.0,
    help="Minimum overlap weight between pair of alleles")

parser.add_argument(
    "--max-ic50",
    default=50000.0)

parser.add_argument(
    "--output-csv",
    required=True)

def binary_encode(X, n_indices=20):
    n_cols = X.shape[1]
    X_encode = np.zeros( (len(X), n_indices * n_cols), dtype=float)
    for i in range(len(X)):
        for col_idx in range(n_cols):
            X_encode[i, col_idx*n_indices + X[i, col_idx]] = True
    return X_encode

def prepare_incomplete_matrix(datasets_dict, prefix=None, allele_name_length=5):
    if allele_name_length:
        datasets_dict = {k:d for (k,d) in datasets_dict.items() if len(k) == allele_name_length}
    if prefix:
        datasets_dict = {k:d for (k,d) in datasets_dict.items() if k.startswith(prefix)}
    if len(datasets_dict) == 0:
        raise ValueError("No alleles matched criteria")
    all_peptides = set([])
    allele_list = []
    for k,d in sorted(datasets_dict.items()):
        print(k, len(d.peptides))
        allele_list.append(k)
        for p in d.peptides:
            all_peptides.add(p)
    peptide_list = list(sorted(all_peptides))
    allele_order = {a:i for (i,a) in enumerate(allele_list)}
    peptide_order = {p:i for (i,p) in enumerate(peptide_list)}
    n_alleles = len(allele_order)
    n_peptides = len(peptide_order)
    incomplete_affinity_matrix = np.ones((n_alleles, n_peptides), dtype=float) * np.nan
    for k,d in datasets_dict.items():
        if k not in allele_order:
            continue
        i = allele_order[k]
        for p,y in zip(d.peptides, d.Y):
            j = peptide_order[p]
            incomplete_affinity_matrix[i,j] = y
    print("Total # alleles = %d, peptides = %d" % (n_alleles, n_peptides))
    return incomplete_affinity_matrix, allele_order, peptide_order

def get_extra_data(allele, train_datasets, expanded_predictions):
    original_dataset = train_datasets[allele]
    original_peptides = set(original_dataset.peptides)
    expanded_allele_affinities = expanded_predictions[allele]
    extra_affinities = {k: v for (k, v) in expanded_allele_affinities.items() if k not in original_peptides}
    extra_peptides = list(extra_affinities.keys())
    extra_values = list(extra_affinities.values())
    extra_X = mhcflurry.data_helpers.index_encoding(extra_peptides, peptide_length=9)
    extra_Y = np.array(extra_values)
    return extra_X, extra_Y

def WHO_KNOWS_WHAT():


    smoothing = 0.005
    exponent = 2
    all_predictions = defaultdict(dict)
    for allele, allele_idx in allele_order.items():
        for peptide in peptide_order.keys():
            predictions = reverse_lookups[peptide]
            total_value = 0.0
            total_denom = 0.0
            for (other_allele, y) in predictions:
                key = (allele,other_allele)
                if key not in completed_sims_dict:
                    continue
                sim = completed_sims_dict[key]
                weight = sim ** exponent
                total_value += weight * y
                total_denom += weight
            if total_denom > 0.0:
                all_predictions[allele][peptide] = total_value / (smoothing + total_denom)


def data_augmentation(X, Y, extra_X, extra_Y,
                      fraction=0.5,
                      niters=10,
                      extra_sample_weight=0.1,
                      nb_epoch=50,
                      nn=True,
                      hidden_layer_size=5):
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
        test_ic50 = 20000 ** (1-Y_test)
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
        if nn:
            model = mhcflurry.feedforward.make_embedding_network(
                layer_sizes=[hidden_layer_size],
                embedding_output_dim=10,
                activation="tanh")
            for i in range(nb_epoch):
                # decay weight as training progresses
                weight[n_original:] = extra_sample_weight if extra_sample_weight is not None else 1.0 / (i+1)**2
                model.fit(
                    X_train_combined,
                    Y_train_combined,
                    sample_weight=weight,
                    shuffle=True,
                    nb_epoch=1,
                    verbose=0)
            pred = model.predict(X_test)
        else:
            model = sklearn.linear_model.Ridge(alpha=5)
            X_train_combined_binary = binary_encode(X_train_combined)
            X_test_binary = binary_encode(X_test)
            model.fit(X_train_combined_binary, Y_train_combined, sample_weight=weight)
            pred = model.predict(X_test_binary)
        pred_ic50 = 20000 ** (1-pred)
        pred_label =  pred_ic50 <= 500
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


if __name__ == "__main__":
    args = parser.parse_args()

    datasets = mhcflurry.data_helpers.load_data(
        args.binding_data_csv,
        binary_encoding=True,
        max_ic50=args.max_ic50)

    sims = {}
    overlaps = {}
    weights = {}
    for allele_name_a, da in train_datasets.items():
        if len(allele_name_a) != 5:
            continue
        y_dict_a = {peptide: y for (peptide,y) in zip(da.peptides, da.Y)}
        seta = set(da.peptides)
        for allele_name_b, db in train_datasets.items():
            if len(allele_name_b) != 5:
                continue
            y_dict_b = {peptide: y for (peptide,y) in zip(db.peptides, db.Y)}
            setb = set(db.peptides)
            intersection = seta.intersection(setb)
            overlaps[a,b] = len(intersection)
            total = 0.0
            weight = 0.0
            for peptide in intersection:
                ya = y_dict_a[peptide]
                yb = y_dict_b[peptide]
                minval = min(ya, yb)
                maxval = max(ya, yb)
                total += minval
                weight += maxval
            weights[a,b] = weight
            if weight > min_weight:
                sims[allele_name_a, allele_name_b] = total / weight
            else:
                sims[allele_name_a, allele_name_b] = np.nan

    if args.fill_missing_values:
        sims_array = np.zeros((n_alleles, n_alleles), dtype=float)
        for i, a in enumerate(allele_names):
            for j, b in enumerate(allele_names):
                sims_array[i,j] = sims[(a,b)]
