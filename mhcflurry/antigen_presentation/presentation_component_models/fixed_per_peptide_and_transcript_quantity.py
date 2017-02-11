import logging

from .presentation_component_model import PresentationComponentModel

from ...common import assert_no_null


class FixedPerPeptideAndTranscriptQuantity(PresentationComponentModel):
    """
    Model input for arbitrary fixed (i.e. not fitted) quantities that
    depend only on the peptide and the transcript it comes from, which
    is taken to be the most-expressed transcript in the experiment.

    Motivating example: netChop cleavage predictions.

    Parameters
    ------------

    name : string
        Name for this final model input. Used in debug messages.

    experiment_to_expression_group : dict of string -> string
        Maps experiment names to expression groups.

    top_transcripts : pandas.DataFrame
        Columns should be expression groups. Indices should be peptide. Values
        should be transcript names.

    df : pandas.DataFrame
        Must have columns 'peptide' and 'transcript'. Remaining columns are
        the values emitted by this model input.

    """

    def __init__(
            self,
            name,
            experiment_to_expression_group,
            top_transcripts,
            df,
            **kwargs):
        PresentationComponentModel.__init__(self, **kwargs)
        self.name = name
        self.experiment_to_expression_group = experiment_to_expression_group
        self.top_transcripts = top_transcripts.copy()

        self.df = df.drop_duplicates(['peptide', 'transcript'])

        # This hack seems to be faster than using a multindex.
        self.df.index = self.df.peptide.str.cat(self.df.transcript, sep=":")
        del self.df["peptide"]
        del self.df["transcript"]
        assert_no_null(self.df)

        df_set = set(self.df.index)
        missing = set()

        for expression_group in self.top_transcripts.columns:
            self.top_transcripts[expression_group] = (
                self.top_transcripts.index.str.cat(
                    self.top_transcripts[expression_group],
                    sep=":"))
            missing.update(
                set(self.top_transcripts[expression_group]).difference(df_set))
        if missing:
            logging.warn(
                "%s: missing %d (peptide, transcript) pairs from df: %s" % (
                    self.name,
                    len(missing),
                    sorted(missing)[:1000]))

    def column_names(self):
        return list(self.df.columns)

    def requires_fitting(self):
        return False

    def predict_for_experiment(self, experiment_name, peptides):
        expression_group = self.experiment_to_expression_group[experiment_name]
        indices = self.top_transcripts.ix[peptides, expression_group]
        assert len(indices) == len(peptides)
        sub_df = self.df.ix[indices]
        assert len(sub_df) == len(peptides)
        result = {}
        for col in self.column_names():
            result_series = sub_df[col]
            num_nulls = result_series.isnull().sum()
            if num_nulls > 0:
                logging.warning("%s: mean-filling for %d nulls" % (
                    self.name, num_nulls))
                result_series = result_series.fillna(self.df[col].mean())
            result[col] = result_series.values
        return result
