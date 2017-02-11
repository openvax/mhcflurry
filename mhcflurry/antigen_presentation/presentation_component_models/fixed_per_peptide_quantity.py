from .presentation_component_model import PresentationComponentModel

from ...common import assert_no_null


class FixedPerPeptideQuantity(PresentationComponentModel):
    """
    Model input for arbitrary fixed (i.e. not fitted) quantities that
    depend only on the peptide. Motivating example: Mike's cleavage
    predictions.

    Parameters
    ------------

    name : string
        Name for this final model input. Used in debug messages.

    df : pandas.DataFrame
        index must be named 'peptide'. The columns of the dataframe are
        the columns emitted by this final modle input.
    """

    def __init__(self, name, df, **kwargs):
        PresentationComponentModel.__init__(self, **kwargs)
        self.name = name
        assert df.index.name == "peptide"
        assert_no_null(df)
        self.df = df

    def column_names(self):
        return list(self.df.columns)

    def requires_fitting(self):
        return False

    def predict(self, peptides_df):
        sub_df = self.df.ix[peptides_df.peptide]
        return dict(
            (col, sub_df[col].values)
            for col in self.column_names())
