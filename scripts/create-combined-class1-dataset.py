"""
Combine 2013 Kim/Peters NetMHCpan dataset[*] with more recent IEDB entries

* = "Dataset size and composition impact the reliability..."
"""
import pickle
import pandas as pd
from collections import Counter

if __name__ == "__main__":
    with open("iedb_human_class1_assay_datasets.pickle", "r'") as f:
        iedb_datasets = pickle.load(f)
    nielsen_data = pd.read_csv("bdata.20130222.mhci.public.1.txt", sep="\t")
    print("Size of 2013 Nielsen dataset: %d" % len(nielsen_data))
    new_allele_counts = Counter()
    combined_columns = {
        "species": list(nielsen_data["species"]),
        "mhc": list(nielsen_data["mhc"]),
        "peptide": list(nielsen_data["sequence"]),
        "peptide_length": list(nielsen_data["peptide_length"]),
        "meas": list(nielsen_data["meas"]),
    }

    for assay, assay_dataset in sorted(iedb_datasets.items(), key=lambda x: len(x[1])):
        joined = nielsen_data.merge(
            assay_dataset,
            left_on=["mhc", "sequence"],
            right_on=["mhc", "peptide"],
            how="outer")

        if len(joined) == 0:
            continue
        # drop NaN binding values and entries without values in both datasets

        left_missing = joined["meas"].isnull()
        right_missing = joined["value"].isnull()
        overlap_filter_mask = ~(left_missing | right_missing)
        filtered = joined[overlap_filter_mask]
        n_overlap = len(filtered)
        if n_overlap == 0:
            continue
        # let's count what fraction of this IEDB assay is within 1% of the values in the
        # Nielsen dataset
        similar_values = (
            (filtered["value"] - filtered["meas"]).abs() <= (filtered["meas"] / 100.0))
        fraction_similar = similar_values.mean()
        print("Assay=%s, count=%d" % (assay, len(assay_dataset)))
        print("  # entries w/ values in both data sets: %d" % n_overlap)
        print("  fraction similar binding values=%0.4f" % fraction_similar)
        new_peptides = joined[left_missing & ~right_missing]
        if fraction_similar > 0.9:
            combined_columns["mhc"].extend(new_peptides["mhc"])
            combined_columns["peptide"].extend(new_peptides["peptide"])
            combined_columns["peptide_length"].extend(new_peptides["peptide"].str.len())
            combined_columns["meas"].extend(new_peptides["value"])
            # TODO: make this work for non-human data
            combined_columns["species"].extend(["human"] * len(new_peptides))
            for allele in new_peptides["mhc"]:
                new_allele_counts[allele] += 1

    combined_df = pd.DataFrame(
        combined_columns,
        columns=["species", "mhc", "peptide", "peptide_length", "meas"])
    print("New entry allele distribution")
    for (allele, count) in new_allele_counts.most_common():
        print("%s: %d" % (allele, count))
    print("Combined DataFrame size: %d (+%d)" % (
            len(combined_df),
            len(combined_df) - len(nielsen_data)))
    combined_df.to_csv("combined_human_class1_dataset.csv", index=False)
