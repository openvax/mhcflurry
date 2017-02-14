import logging
from copy import copy

import pandas
from numpy import log, exp, nanmean, array

from ...dataset import Dataset
from ...class1_allele_specific import Class1BindingPredictor
from ...common import normalize_allele_name, dataframe_cryptographic_hash

from .presentation_component_model import PresentationComponentModel
from ..decoy_strategies import SameTranscriptsAsHits
from ..percent_rank_transform import PercentRankTransform


MHCFLURRY_DEFAULT_HYPERPARAMETERS = dict(
    embedding_output_dim=8,
    dropout_probability=0.25)


class MHCflurryTrainedOnHits(PresentationComponentModel):
    """
    Final model input that is a mhcflurry predictor trained on mass-spec
    hits and, optionally, affinity measurements (for example from IEDB).

    Parameters
    ------------

    predictor_name : string
        used on column name. Example: 'vanilla'

    experiment_to_alleles : dict: string -> string list
        Normalized allele names for each experiment.

    experiment_to_expression_group : dict of string -> string
        Maps experiment names to expression groups.

    transcripts : pandas.DataFrame
        Index is peptide, columns are expression groups, values are
        which transcript to use for the given peptide.
        Not required if decoy_strategy specified.

    peptides_and_transcripts : pandas.DataFrame
        Dataframe with columns 'peptide' and 'transcript'
        Not required if decoy_strategy specified.

    decoy_strategy : decoy_strategy.DecoyStrategy
        how to pick decoys. If not specified peptides_and_transcripts and
        transcripts must be specified.

    fallback_predictor : function: (allele, peptides) -> predictions
        Used when missing an allele.

    iedb_dataset : mhcflurry.Dataset
        IEDB data for this allele. If not specified no iedb data is used.

    decoys_per_hit : int

    mhcflurry_hyperparameters : dict

    hit_affinity : float
        nM affinity to use for hits

    decoy_affinity : float
        nM affinity to use for decoys

    random_peptides_for_percent_rank : list of string
        If specified, then percentile rank will be calibrated and emitted
        using the given peptides.
    """

    def __init__(
            self,
            predictor_name,
            experiment_to_alleles,
            experiment_to_expression_group=None,
            transcripts=None,
            peptides_and_transcripts=None,
            decoy_strategy=None,
            fallback_predictor=None,
            iedb_dataset=None,
            decoys_per_hit=10,
            mhcflurry_hyperparameters=MHCFLURRY_DEFAULT_HYPERPARAMETERS,
            hit_affinity=100,
            decoy_affinity=20000,
            random_peptides_for_percent_rank=None,
            **kwargs):

        PresentationComponentModel.__init__(self, **kwargs)
        self.predictor_name = predictor_name
        self.experiment_to_alleles = experiment_to_alleles
        self.fallback_predictor = fallback_predictor
        self.iedb_dataset = iedb_dataset
        self.mhcflurry_hyperparameters = mhcflurry_hyperparameters
        self.hit_affinity = hit_affinity
        self.decoy_affinity = decoy_affinity

        self.allele_to_model = None

        if decoy_strategy is None:
            assert peptides_and_transcripts is not None
            assert transcripts is not None
            self.decoy_strategy = SameTranscriptsAsHits(
                experiment_to_expression_group=experiment_to_expression_group,
                peptides_and_transcripts=peptides_and_transcripts,
                peptide_to_expression_group_to_transcript=transcripts,
                decoys_per_hit=decoys_per_hit)
        else:
            self.decoy_strategy = decoy_strategy

        if random_peptides_for_percent_rank is None:
            self.percent_rank_transforms = None
            self.random_peptides_for_percent_rank = None
        else:
            self.percent_rank_transforms = {}
            self.random_peptides_for_percent_rank = array(
                random_peptides_for_percent_rank)

    def combine_ensemble_predictions(self, column_name, values):
        # Geometric mean
        return exp(nanmean(log(values), axis=1))

    def stratification_groups(self, hits_df):
        return [
            self.experiment_to_alleles[e][0]
            for e in hits_df.experiment_name
        ]

    def column_name_affinity(self):
        return "mhcflurry_%s_affinity" % self.predictor_name

    def column_name_percentile_rank(self):
        return "mhcflurry_%s_percentile_rank" % self.predictor_name

    def column_names(self):
        columns = [self.column_name_affinity()]
        if self.percent_rank_transforms is not None:
            columns.append(self.column_name_percentile_rank())
        return columns

    def requires_fitting(self):
        return True

    def fit_percentile_rank_if_needed(self, alleles):
        for allele in alleles:
            if allele not in self.percent_rank_transforms:
                logging.info('fitting percent rank for allele: %s' % allele)
                self.percent_rank_transforms[allele] = PercentRankTransform()
                self.percent_rank_transforms[allele].fit(
                    self.predict_affinity_for_allele(
                        allele,
                        self.random_peptides_for_percent_rank))

    def fit(self, hits_df):
        assert 'experiment_name' in hits_df.columns
        assert 'peptide' in hits_df.columns
        if 'hit' in hits_df.columns:
            assert (hits_df.hit == 1).all()

        grouped = hits_df.groupby("experiment_name")
        for (experiment_name, sub_df) in grouped:
            self.fit_to_experiment(experiment_name, sub_df.peptide.values)

        # No longer required after fitting.
        self.decoy_strategy = None
        self.iedb_dataset = None

    def fit_to_experiment(self, experiment_name, hit_list):
        assert len(hit_list) > 0
        if self.allele_to_model is None:
            self.allele_to_model = {}

        alleles = self.experiment_to_alleles[experiment_name]
        if len(alleles) != 1:
            raise ValueError("Monoallelic data required")

        (allele,) = alleles
        mhcflurry_allele = normalize_allele_name(allele)
        assert allele not in self.allele_to_model, \
            "TODO: Support training on >1 experiments with same allele " \
            + str(self.allele_to_model)

        extra_hits = hit_list = set(hit_list)

        iedb_dataset_df = None
        if self.iedb_dataset is not None:
            iedb_dataset_df = (
                self.iedb_dataset.get_allele(mhcflurry_allele).to_dataframe())
            extra_hits = hit_list.difference(set(iedb_dataset_df.peptide))
            print("Using %d / %d ms hits not in iedb in augmented model" % (
                len(extra_hits),
                len(hit_list)))

        decoys = self.decoy_strategy.decoys_for_experiment(
            experiment_name, hit_list)

        df = pandas.DataFrame({"peptide": sorted(set(hit_list).union(decoys))})
        df["allele"] = mhcflurry_allele
        df["species"] = "human"
        df["affinity"] = ((
            ~df.peptide.isin(hit_list))
            .astype(float) * (
                self.decoy_affinity - self.hit_affinity) + self.hit_affinity)
        df["sample_weight"] = 1.0
        df["peptide_length"] = 9

        if self.iedb_dataset is not None:
            df = df.append(iedb_dataset_df, ignore_index=True)

        dataset = Dataset(
            df.sample(frac=1))  # shuffle dataframe
        print("Train data: ", dataset)
        model = Class1BindingPredictor(
            **self.mhcflurry_hyperparameters)
        model.fit_dataset(dataset, verbose=True)
        self.allele_to_model[allele] = model

    def predict_affinity_for_allele(self, allele, peptides):
        if self.cached_predictions is None:
            cache_key = None
            cached_result = None
        else:
            cache_key = (
                allele,
                dataframe_cryptographic_hash(pandas.Series(peptides)))
            cached_result = self.cached_predictions.get(cache_key)
        if cached_result is not None:
            print("Cache hit in predict_affinity_for_allele: %s %s %s" % (
                allele, str(self), id(cached_result)))
            return cached_result
        else:
            print("Cache miss in predict_affinity_for_allele: %s %s" % (
                allele, str(self)))

        if allele in self.allele_to_model:
            result = self.allele_to_model[allele].predict(peptides)
        elif self.fallback_predictor:
            print(
                "MHCflurry: falling back on %s, "
                "available alleles: %s" % (
                    allele, ' '.join(self.allele_to_model)))
            result = self.fallback_predictor(allele, peptides)
        else:
            raise ValueError("No model for allele: %s" % allele)

        if self.cached_predictions is not None:
            self.cached_predictions[cache_key] = result
        return result

    def predict_for_experiment(self, experiment_name, peptides):
        assert self.allele_to_model is not None, "Must fit first"

        peptides_deduped = pandas.unique(peptides)
        print(len(peptides_deduped))

        alleles = self.experiment_to_alleles[experiment_name]
        predictions = pandas.DataFrame(index=peptides_deduped)
        for allele in alleles:
            predictions[allele] = self.predict_affinity_for_allele(
                allele, peptides_deduped)

        result = {
            self.column_name_affinity(): (
                predictions.min(axis=1).ix[peptides].values)
        }
        if self.percent_rank_transforms is not None:
            self.fit_percentile_rank_if_needed(alleles)
            percentile_ranks = pandas.DataFrame(index=peptides_deduped)
            for allele in alleles:
                percentile_ranks[allele] = (
                    self.percent_rank_transforms[allele]
                    .transform(predictions[allele].values))
            result[self.column_name_percentile_rank()] = (
                percentile_ranks.min(axis=1).ix[peptides].values)
        assert all(len(x) == len(peptides) for x in result.values()), (
            "Result lengths don't match peptide lengths. peptides=%d, "
            "peptides_deduped=%d, %s" % (
                len(peptides),
                len(peptides_deduped),
                ", ".join(
                    "%s=%d" % (key, len(value))
                    for (key, value) in result.items())))
        return result

    def get_fit(self):
        return {
            'model': 'MHCflurryTrainedOnMassSpec',
            'allele_to_model': self.allele_to_model,
        }

    def restore_fit(self, fit_info):
        fit_info = dict(fit_info)
        self.allele_to_model = fit_info.pop('allele_to_model')

        model = fit_info.pop('model')
        assert model == 'MHCflurryTrainedOnMassSpec', model
        assert not fit_info, "Extra info in fit: %s" % str(fit_info)

    def clone(self):
        result = copy(self)
        result.reset_cache()
        result.allele_to_model = copy(result.allele_to_model)
        return result
