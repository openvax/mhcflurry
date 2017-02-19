from .presentation_component_model import PresentationComponentModel

from ...common import assert_no_null


class FixedAffinityPredictions(PresentationComponentModel):
    """
    Parameters
    ------------

    experiment_to_alleles : dict: string -> string list
        Normalized allele names for each experiment.

    panel : pandas.Panel
        Dimensions should be:
            - "value", "percentile_rank" (IC50 and percent rank)
            - peptide (string)
            - allele (string)

    name : string
        Used to name output columns and in debug messages
    """

    def __init__(
            self,
            experiment_to_alleles,
            panel,
            name='precomputed',
            **kwargs):
        PresentationComponentModel.__init__(self, **kwargs)
        self.experiment_to_alleles = experiment_to_alleles
        for key in panel.items:
            assert_no_null(panel[key])
        self.panel = panel
        self.name = name

    def column_names(self):
        return [
            "%s_affinity" % self.name,
            "%s_percentile_rank" % self.name
        ]

    def requires_fitting(self):
        return False

    def predict_min_across_alleles(self, alleles, peptides):
        return {
            ("%s_affinity" % self.name): (
                self.panel
                .value[alleles]
                .min(axis=1)
                .ix[peptides].values),
            ("%s_percentile_rank" % self.name): (
                self.panel
                .percentile_rank[alleles]
                .min(axis=1)
                .ix[peptides].values)
        }

    def predict_for_experiment(self, experiment_name, peptides):
        alleles = self.experiment_to_alleles[experiment_name]
        return self.predict_min_across_alleles(alleles, peptides)
