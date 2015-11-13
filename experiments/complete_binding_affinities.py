
"""
Use a convex solver to complete the full matrix of all pMHC affinities
"""

import mhcflurry
import dataset_paths
import numpy as np
import cvxpy


def compute_overlaps_similarities(datasets, min_weight=2.0):
    sims = {}
    overlaps = {}
    weights = {}
    for a, da in datasets.items():
        if len(a) != 5:
            continue
        y_dict_a = {peptide: y for (peptide, y) in zip(da.peptides, da.Y)}
        seta = set(da.peptides)
        for b, db in train_datasets.items():
            if len(b) != 5:
                continue

            y_dict_b = {peptide: y for (peptide, y) in zip(db.peptides, db.Y)}
            setb = set(db.peptides)
            intersection = seta.intersection(setb)
            overlaps[(a, b)] = len(intersection)
            total = 0.0
            weight = 0.0
            for peptide in intersection:
                ya = y_dict_a[peptide]
                yb = y_dict_b[peptide]
                minval = min(ya, yb)
                maxval = max(ya, yb)
                total += minval
                weight += maxval
            weights[(a, b)] = weight
            if weight > min_weight:
                sims[(a, b)] = total / weight
            else:
                sims[(a, b)] = np.nan
    return sims, overlaps, weights


def create_sims_array(sims_dict):
    allele_names_set = {a for (a, _) in sims_dict.keys()}
    allele_names_list = list(sorted(allele_names_set))
    n_alleles = len(allele_names_list)
    sims_array = np.zeros((n_alleles, n_alleles), dtype=float)
    for i, a in enumerate(allele_names_list):
        for j, b in enumerate(allele_names_list):
            sims_array[i, j] = sims_dict[(a, b)]
    return sims_array, allele_names_list


def prepare_incomplete_affinity_matrix(datasets_dict, prefix=None, allele_name_length=5):
    if allele_name_length:
        datasets_dict = {k: d for (k, d) in datasets_dict.items() if len(k) == allele_name_length}
    if prefix:
        datasets_dict = {k: d for (k, d) in datasets_dict.items() if k.startswith(prefix)}
    if len(datasets_dict) == 0:
        raise ValueError("No alleles matched criteria")
    all_peptides = set([])
    allele_list = []
    for k, d in sorted(datasets_dict.items()):
        print(k, len(d.peptides))
        allele_list.append(k)
        for p in d.peptides:
            all_peptides.add(p)
    peptide_list = list(sorted(all_peptides))
    allele_order = {a: i for (i, a) in enumerate(allele_list)}
    peptide_order = {p: i for (i, p) in enumerate(peptide_list)}
    n_alleles = len(allele_order)
    n_peptides = len(peptide_order)
    incomplete_affinity_matrix = np.ones((n_alleles, n_peptides), dtype=float) * np.nan
    for k, d in datasets_dict.items():
        if k not in allele_order:
            continue
        i = allele_order[k]
        for p, y in zip(d.peptides, d.Y):
            j = peptide_order[p]
            incomplete_affinity_matrix[i, j] = y
    print("Total # alleles = %d, peptides = %d" % (n_alleles, n_peptides))
    return incomplete_affinity_matrix, allele_list, allele_order, peptide_list, peptide_order


def create_matrix_completion_convex_problem(
        n_alleles,
        n_peptides,
        known_mask,
        known_values,
        tolerance=0.001):

    A = cvxpy.Variable(n_alleles, n_peptides, name="A")
    # nuclear norm minimization should create a low-rank matrix
    objective = cvxpy.Minimize(cvxpy.norm(A, "nuc"))

    masked_A = cvxpy.mul_elemwise(known_mask, A)
    masked_known = cvxpy.mul_elemwise(known_mask, known_values)
    diff = masked_A - masked_known
    close_to_data = diff ** 2 <= tolerance
    lower_bound = (A >= 0)
    upper_bound = (A <= 1)
    constraints = [close_to_data, lower_bound, upper_bound]
    problem = cvxpy.Problem(objective, constraints)
    return A, problem

if __name__ == "__main__":
    train_datasets = mhcflurry.data_helpers.load_allele_datasets(
        dataset_paths.PETERS2009_CSV_PATH,
        binary_encoding=False,
        max_ic50=20000)

    allele_sims, allele_overlap_counts, allele_overlap_weights = \
        compute_overlaps_similarities(train_datasets)
    allele_sims_array, allele_names_list = create_sims_array(allele_sims)
    incomplete_affinity_matrix, allele_list, allele_order, peptide_list, peptide_order = \
        prepare_incomplete_affinity_matrix(train_datasets)
    n_peptides = len(peptide_order)
    n_alleles = len(allele_order)

    affinity_nan_mask = np.isnan(incomplete_affinity_matrix)
    print("# missing entries: %d/%d" % (
        affinity_nan_mask.sum(),
        n_peptides * n_alleles))
    affinity_ok_mask = (~affinity_nan_mask).astype(int)
    zeroed_affinity_matrix = incomplete_affinity_matrix.copy()
    zeroed_affinity_matrix[affinity_nan_mask] = 0.0

    completed_matrix_variable, convex_problem = \
        create_matrix_completion_convex_problem(
            n_alleles=n_alleles,
            n_peptides=n_peptides,
            known_mask=affinity_ok_mask,
            known_values=zeroed_affinity_matrix,
            tolerance=0.001)
    convex_problem.solve(verbose=True, solver=cvxpy.SCS)
    completed_affinities = completed_matrix_variable.value
    np.save("completed_pmhc_affinity_matrix.npy", completed_affinities)
    with open("completed_pmhc_affinity_matrix_rows.txt", "w") as f:
        for allele in allele_names_list:
            f.write("%s\n" % allele)
    with open("completed_pmhc_affinity_matrix_columns.txt", "w") as f:
        for peptide in peptide_list:
            f.write("%s\n" % peptide)
