import logging

import numpy
import pandas

from ...common import normalize_allele_name
from ...predict import predict
from ..percent_rank_transform import PercentRankTransform
from .presentation_component_model import PresentationComponentModel


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
    """

    def __init__(
            self,
            experiment_to_alleles,
            random_peptides_for_percent_rank=None,
            **kwargs):
        PresentationComponentModel.__init__(self, **kwargs)
        self.experiment_to_alleles = experiment_to_alleles
        if random_peptides_for_percent_rank is None:
            self.percent_rank_transforms = None
            self.random_peptides_for_percent_rank = None
        else:
            self.percent_rank_transforms = {}
            self.random_peptides_for_percent_rank = numpy.array(
                random_peptides_for_percent_rank)

    def column_names(self):
        columns = ['mhcflurry_released_affinity']
        if self.percent_rank_transforms is not None:
            columns.append('mhcflurry_released_percentile_rank')
        return columns

    def requires_fitting(self):
        return False

    def fit_percentile_rank_if_needed(self, alleles):
        for allele in alleles:
            if allele not in self.percent_rank_transforms:
                logging.info('fitting percent rank for allele: %s' % allele)
                self.percent_rank_transforms[allele] = PercentRankTransform()
                self.percent_rank_transforms[allele].fit(
                    predict(
                        [allele],
                        self.random_peptides_for_percent_rank)
                    .Prediction.values)

    def predict_min_across_alleles(self, alleles, peptides):
        alleles = [
            normalize_allele_name(allele)
            for allele in alleles
        ]
        df = predict(alleles, numpy.unique(numpy.array(peptides)))
        pivoted = df.pivot(index='Peptide', columns='Allele')
        pivoted.columns = pivoted.columns.droplevel()
        result = {
            'mhcflurry_released_affinity': (
                pivoted.min(axis=1).ix[peptides].values)
        }
        if self.percent_rank_transforms is not None:
            self.fit_percentile_rank_if_needed(alleles)
            percentile_ranks = pandas.DataFrame(index=pivoted.index)
            for allele in alleles:
                percentile_ranks[allele] = (
                    self.percent_rank_transforms[allele]
                    .transform(pivoted[allele].values))
            result['mhcflurry_released_percentile_rank'] = (
                percentile_ranks.min(axis=1).ix[peptides].values)
        return result

    def predict_for_experiment(self, experiment_name, peptides):
        alleles = self.experiment_to_alleles[experiment_name]
        return self.predict_min_across_alleles(alleles, peptides)
