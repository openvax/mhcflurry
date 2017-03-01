from ..dataset import Dataset

from sklearn.model_selection import StratifiedKFold

import pandas

from ..imputation_helpers import imputer_from_name

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
    """
    A measurement collection is a set of observations for allele/peptide pairs.
    A single measurement collection may have both MS hits and affinity
    measurements.

    This is more general than a Dataset since it supports MS hits. It is also
    simpler, as the user is expected to manipulate the underlying dataframe.
    Later we may want to retire Dataset or combine it with this class.
    """

    def __init__(self, df, check=True):
        if check:
            for col in COLUMNS:
                assert col in df.columns, col

            for measurement_type in df.measurement_type.unique():
                assert measurement_type in MEASUREMENT_TYPES, measurement_type
        self.df = df[COLUMNS]

    @staticmethod
    def from_dataset(dataset):
        """
        Given a Dataset, return a MeasurementCollection
        """
        dataset_df = dataset.to_dataframe()
        df = dataset_df.reset_index(drop=True)[["allele", "peptide"]].copy()
        df["measurement_type"] = "affinity"
        df["measurement_source"] = "in_vitro_affinity_assay"
        df["measurement_value"] = dataset_df.affinity.values
        df["weight"] = dataset_df.sample_weight.values
        return MeasurementCollection(df)

    def select_allele(self, allele):
        """
        Return a new MeasurementCollection containing only observations for the
        specified allele.
        """
        assert isinstance(allele, str), type(allele)
        assert len(self.df) > 0
        assert allele in self.df.allele.unique()
        return MeasurementCollection(
            self.df.ix[self.df.allele == allele],
            check=False)

    def half_splits(self, num, random_state=None):
        """
        Split the MeasurementCollection into disjoint pairs of
        MeasurementCollection instances, each containing half the observations.

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
                cv.split(self.df.values, stratification_groups))
            assert len(indices1) > 0
            assert len(indices2) > 0
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
        """
        Return a Dataset containing the observations in the collection.
        Mass-spec data are converted to affinities according to
        ms_hit_affinity and ms_decoy_affinity.

        Parameters
        -------------
        include_ms : bool
            If True then mass spec data is included; otherwise it is dropped

        ms_hit_affinity : float
            nM affinity to assign to mass-spec hits (relevant only if
            include_ms=True)

        ms_decoy_affinity : float
            nM affinity to assign to mass-spec decoys (relevant only if
            include_ms=True)

        Returns
        -------------
        Dataset instance
        """
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
            impute_min_observations_per_allele=1,
            imputer_args={}):
        """
        Return a new MeasurementCollection after applying imputation to
        this collection. The imputed collection will have the
        observations in the current collection plus the imputed data.
        """
        assert len(self.df) > 0

        dataset = self.to_dataset(include_ms=False)
        imputer = imputer_from_name(impute_method, **imputer_args)
        result_df = dataset.impute_missing_values(
            imputation_method=imputer,
            log_transform=impute_log_transform,
            min_observations_per_peptide=impute_min_observations_per_peptide,
            min_observations_per_allele=impute_min_observations_per_allele
        ).to_dataframe()
        assert len(result_df) > 0
        result_df["measurement_type"] = "affinity"
        result_df["measurement_source"] = "imputed"
        result_df["measurement_value"] = result_df.affinity
        result_df["weight"] = result_df.sample_weight
        return MeasurementCollection(result_df)
