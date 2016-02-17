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

from sklearn.cross_validation import KFold


def generate_cross_validation_datasets(
        allele_to_peptide_to_affinity,
        n_folds=4):
    for allele, dataset in sorted(allele_to_peptide_to_affinity.items()):
        peptides = list(dataset.keys())
        affinities = list(dataset.values())
        n_samples = len(peptides)
        print("Generating similarities for folds of %s data (n=%d)" % (
            allele,
            n_samples))

        if n_samples < n_folds:
            print("Too few samples (%d) for %d-fold cross-validation" % (
                n_samples,
                n_folds))
            continue

        kfold_iterator = enumerate(
            KFold(n_samples, n_folds=n_folds, shuffle=True))

        for fold_number, (train_indices, test_indices) in kfold_iterator:
            train_peptides = [peptides[i] for i in train_indices]
            train_affinities = [affinities[i] for i in train_indices]
            test_peptide_set = set([peptides[i] for i in test_indices])
            # copy the affinity data for all alleles other than this one
            fold_affinity_dict = {
                allele_key: affinity_dict
                for (allele_key, affinity_dict)
                in allele_to_peptide_to_affinity.items()
                if allele_key != allele
            }
            # include an affinity dictionary for this allele which
            # only uses training data
            fold_affinity_dict[allele] = {
                train_peptide: train_affinity
                for (train_peptide, train_affinity)
                in zip(train_peptides, train_affinities)
            }
            allele_similarities, overlap_counts, overlap_weights = \
                compute_allele_similarities(
                    allele_to_peptide_to_affinity=fold_affinity_dict,
                    min_weight=0.1)
            this_allele_similarities = allele_similarities[allele]

            yield (
                allele,
                dataset,
                fold_number,
                this_allele_similarities,
                test_peptide_set
            )