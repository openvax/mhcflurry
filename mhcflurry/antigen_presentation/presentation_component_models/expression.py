from .presentation_component_model import PresentationComponentModel

from ...common import assert_no_null


class Expression(PresentationComponentModel):
    """
    Model input for transcript expression.

    Parameters
    ------------

    experiment_to_expression_group : dict of string -> string
        Maps experiment names to expression groups.

    expression_values : pandas.DataFrame
        Columns should be expression groups. Indices should be peptide.

    """

    def __init__(
            self, experiment_to_expression_group, expression_values, **kwargs):
        PresentationComponentModel.__init__(self, **kwargs)
        assert all(
            group in expression_values.columns
            for group in experiment_to_expression_group.values())

        assert_no_null(expression_values)

        self.experiment_to_expression_group = experiment_to_expression_group
        self.expression_values = expression_values

    def column_names(self):
        return ["expression"]

    def requires_fitting(self):
        return False

    def predict_for_experiment(self, experiment_name, peptides):
        expression_group = self.experiment_to_expression_group[experiment_name]
        return {
            "expression": (
                self.expression_values.ix[peptides, expression_group]
                .values)
        }
