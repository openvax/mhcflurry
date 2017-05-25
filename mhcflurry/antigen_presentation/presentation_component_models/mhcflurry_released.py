import logging

import numpy
import pandas

from mhcnames import normalize_allele_name

from ..percent_rank_transform import PercentRankTransform
from ...encodable_sequences import EncodableSequences
from .presentation_component_model import PresentationComponentModel
from ...class1_affinity_prediction.class1_affinity_predictor import (
    Class1AffinityPredictor)


class MHCflurryReleased(PresentationComponentModel):
    """
    Final model input that uses the standard downloaded MHCflurry models.

    Parameters
    ------------
    experiment_to_alleles : dict: string -> string list
        Normalized allele names for each experiment.

    random_peptides_for_percent_rank : list of string
        If specified, then percentile rank will be calibrated and emitted
        using the given peptides.

    predictor : Class1EnsembleMultiAllelePredictor-like object
        Predictor to use.
    """

    def __init__(
            self,
            experiment_to_alleles,
            random_peptides_for_percent_rank=None,
            predictor=None,
            predictor_name="mhcflurry_released",
            **kwargs):
        PresentationComponentModel.__init__(self, **kwargs)
        self.experiment_to_alleles = experiment_to_alleles
        if predictor is None:
            predictor = Class1AffinityPredictor.load()
        self.predictor = predictor
        self.predictor_name = predictor_name
        if random_peptides_for_percent_rank is None:
            self.percent_rank_transforms = None
            self.random_peptides_for_percent_rank = None
        else:
            self.percent_rank_transforms = {}
            self.random_peptides_for_percent_rank = numpy.array(
                random_peptides_for_percent_rank)

    def column_names(self):
        columns = [self.predictor_name + '_affinity']
        if self.percent_rank_transforms is not None:
            columns.append(self.predictor_name + '_percentile_rank')
        return columns

    def requires_fitting(self):
        return False

    def fit_percentile_rank_if_needed(self, alleles):
        for allele in alleles:
            if allele not in self.percent_rank_transforms:
                logging.info('fitting percent rank for allele: %s' % allele)
                self.percent_rank_transforms[allele] = PercentRankTransform()
                self.percent_rank_transforms[allele].fit(
                    self.predictor.predict(
                        allele=allele,
                        peptides=self.random_peptides_for_percent_rank))

    def predict_min_across_alleles(self, alleles, peptides):
        alleles = list(set([
            normalize_allele_name(allele)
            for allele in alleles
        ]))
        peptides = EncodableSequences.create(peptides)
        df = pandas.DataFrame()
        df["peptide"] = peptides.sequences
        for allele in alleles:
            df[allele] = self.predictor.predict(peptides, allele=allele)
        result = {
            self.predictor_name + '_affinity': (
                df[list(df.columns)[1:]].min(axis=1))
        }
        if self.percent_rank_transforms is not None:
            self.fit_percentile_rank_if_needed(alleles)
            percentile_ranks = pandas.DataFrame(index=df.index)
            for allele in alleles:
                percentile_ranks[allele] = (
                    self.percent_rank_transforms[allele]
                    .transform(df[allele].values))
            result[self.predictor_name + '_percentile_rank'] = (
                percentile_ranks.min(axis=1).values)

        for (key, value) in result.items():
            assert len(value) == len(peptides), (len(peptides), result)
        return result

    def predict_for_experiment(self, experiment_name, peptides):
        alleles = self.experiment_to_alleles[experiment_name]
        return self.predict_min_across_alleles(alleles, peptides)
