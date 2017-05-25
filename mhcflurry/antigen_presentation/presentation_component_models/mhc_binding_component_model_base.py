import logging

import pandas
from numpy import array

from ...common import dataframe_cryptographic_hash

from .presentation_component_model import PresentationComponentModel
from ..decoy_strategies import SameTranscriptsAsHits
from ..percent_rank_transform import PercentRankTransform


class MHCBindingComponentModelBase(PresentationComponentModel):
    """
    Base class for single-allele MHC binding predictors.

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

    iedb_dataset : mhcflurry.AffinityMeasurementDataset
        IEDB data for this allele. If not specified no iedb data is used.

    decoys_per_hit : int

    random_peptides_for_percent_rank : list of string
        If specified, then percentile rank will be calibrated and emitted
        using the given peptides.

    **kwargs : dict
        passed to PresentationComponentModel()
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
            random_peptides_for_percent_rank=None,
            **kwargs):

        PresentationComponentModel.__init__(self, **kwargs)
        self.predictor_name = predictor_name
        self.experiment_to_alleles = experiment_to_alleles
        self.fallback_predictor = fallback_predictor
        self.iedb_dataset = iedb_dataset

        self.fit_alleles = set()

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

    def stratification_groups(self, hits_df):
        return [
            self.experiment_to_alleles[e][0]
            for e in hits_df.experiment_name
        ]

    def column_name_value(self):
        return "%s_value" % self.predictor_name

    def column_name_percentile_rank(self):
        return "%s_percentile_rank" % self.predictor_name

    def column_names(self):
        columns = [self.column_name_value()]
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

    def fit_allele(self, allele, hit_list, decoys_list):
        raise NotImplementedError()

    def predict_allele(self, allele, peptide_list):
        raise NotImplementedError()

    def supports_predicting_allele(self, allele):
        raise NotImplementedError()

    def fit_to_experiment(self, experiment_name, hit_list):
        assert len(hit_list) > 0
        alleles = self.experiment_to_alleles[experiment_name]
        if len(alleles) != 1:
            raise ValueError("Monoallelic data required")

        (allele,) = alleles
        decoys = self.decoy_strategy.decoys_for_experiment(
            experiment_name, hit_list)

        self.fit_allele(allele, hit_list, decoys)
        self.fit_alleles.add(allele)

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

        if self.supports_predicting_allele(allele):
            result = self.predict_allele(allele, peptides)
        elif self.fallback_predictor:
            print("Falling back on allee %s" % allele)
            result = self.fallback_predictor(allele, peptides)
        else:
            raise ValueError("No model for allele: %s" % allele)

        if self.cached_predictions is not None:
            self.cached_predictions[cache_key] = result
        return result

    def predict_for_experiment(self, experiment_name, peptides):
        peptides_deduped = pandas.unique(peptides)
        print(len(peptides_deduped))

        alleles = self.experiment_to_alleles[experiment_name]
        predictions = pandas.DataFrame(index=peptides_deduped)
        for allele in alleles:
            predictions[allele] = self.predict_affinity_for_allele(
                allele, peptides_deduped)

        result = {
            self.column_name_value(): (
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
