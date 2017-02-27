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

    @staticmethod
    def from_dataset(dataset):
        dataset_df = dataset.to_dataframe()
        df = dataset_df.reset_index(drop=True)[["allele", "peptide"]].copy()
        df["measurement_type"] = "affinity"
        df["measurement_source"] = "in_vitro_affinity_assay"
        df["measurement_value"] = dataset_df.affinity.values
        df["weight"] = dataset_df.sample_weight.values
        return MeasurementCollection(df)

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
                    for (_, row) in self.df.iterrows()
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

    def impute(
            self,
            impute_method="mice",
            impute_log_transform=True,
            impute_min_observations_per_peptide=1,
            impute_min_observations_per_allele=1):

        dataset = self.to_dataset(include_ms=False)
        result_df = dataset.impute_missing_values(
            imputation_method=impute_method,
            log_transform=impute_log_transform,
            min_observations_per_peptide=impute_min_observations_per_peptide,
            min_observations_per_allele=impute_min_observations_per_allele,
        ).to_dataframe()
        result_df["measurement_type"] = "affinity"
        result_df["measurement_source"] = "imputed"
        result_df["measurement_value"] = result_df.affinity
        result_df["weight"] = result_df.sample_weight
        return MeasurementCollection(result_df)
