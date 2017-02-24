from ..dataset import Dataset

from sklearn.model_selection import StratifiedKFold

import pandas


COLUMNS = [
    "allele",
    "peptide",
    "measurement_type",
    "measurement_source",
    "measurement_value",
    "weight",
]

MEASUREMENT_TYPES = [
    "affinity",
    "ms_hit",
]

MEASUREMENT_SOURCES = [
    "in_vitro_affinity_assay",
    "imputed",
    "ms_hit",
    "ms_decoy",
]


class MeasurementCollection(object):
    def __init__(self, df, check=True):
        if check:
            for col in COLUMNS:
                assert col in df.columns, col

            for measurement_type in df.measurement_type.unique():
                assert measurement_type in MEASUREMENT_TYPES, measurement_type
        self.df = df[COLUMNS]

    def select_allele(self, allele):
        return MeasurementCollection(
            self.df.ix[self.df.allele == allele],
            check=False)

    def half_splits(self, num, random_state=None):
        """
        Parameters
        -------------
        num : int
            Number of pairs to return

        random_state : int, optional

        Returns
        -------------
        list of (MeasurementCollection, MeasurementCollection) pairs
        Each pair gives a disjoint train and test split.
        """
        assert num > 0
        results = []
        while True:
            cv = StratifiedKFold(
                n_splits=2,
                shuffle=True,
                random_state=(
                    None if random_state is None
                    else random_state + len(results)))
            stratification_groups = self.df.allele + self.df.measurement_type
            (indices1, indices2) = next(
                cv.split(self.df, stratification_groups))
            mc1 = MeasurementCollection(self.df.iloc[indices1], check=False)
            mc2 = MeasurementCollection(self.df.iloc[indices2], check=False)
            for pair in [(mc1, mc2), (mc2, mc1)]:
                results.append(pair)
                if len(results) == num:
                    return results

    def to_dataset(
            self,
            include_ms=False,
            ms_hit_affinity=1.0,
            ms_decoy_affinity=20000):
        if include_ms:
            dataset = Dataset(pandas.DataFrame({
                "allele": self.df.allele,
                "peptide": self.df.peptide,
                "affinity": [
                    row.measurement_value if row.measurement_type == "affinity"
                    else (
                        ms_hit_affinity if row.value > 0
                        else ms_decoy_affinity)
                    for (_, row) in self.df
                ],
                "sample_weight": self.df.weight,
            }))
        else:
            df = self.df.ix[
                (self.df.measurement_type == "affinity") &
                (self.df.measurement_source == "in_vitro_affinity_assay")
            ]
            dataset = Dataset(pandas.DataFrame({
                "allele": df.allele,
                "peptide": df.peptide,
                "affinity": df.measurement_value,
                "sample_weight": df.weight,
            }))
        return dataset

    def impute(self, **kwargs):
        dataset = self.to_datset(include_ms=False)
        result_df = dataset.impute_missing_values(**kwargs).to_dataframe()
        result_df["measurement_type"] = "affinity"
        result_df["measurement_source"] = "imputed"
        result_df["measurement_value"] = result_df.affinity
        result_df["weight"] = result_df.sample_weight
        return MeasurementCollection(result_df)
