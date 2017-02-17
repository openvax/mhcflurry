import numpy

from .decoy_strategy import DecoyStrategy


class SameTranscriptsAsHits(DecoyStrategy):
    """
    Decoy strategy that selects decoys from the same transcripts the
    hits come from. The transcript for each hit is taken to be the
    transcript containing the hit with the the highest expression for
    the given experiment.

    Parameters
    ------------
    experiment_to_expression_group : dict of string -> string
        Maps experiment names to expression groups.

    peptides_and_transcripts: pandas.DataFrame
        Must have columns 'peptide' and 'transcript', index unimportant.

    peptide_to_expression_group_to_transcript : pandas.DataFrame
        Indexed by peptides, columns are expression groups. Values
        give transcripts to use.

    decoys_per_hit : int
    """
    def __init__(
            self,
            experiment_to_expression_group,
            peptides_and_transcripts,
            peptide_to_expression_group_to_transcript,
            decoys_per_hit=10):
        DecoyStrategy.__init__(self)
        assert decoys_per_hit > 0
        self.experiment_to_expression_group = experiment_to_expression_group
        self.peptides_and_transcripts = peptides_and_transcripts
        self.peptide_to_expression_group_to_transcript = (
            peptide_to_expression_group_to_transcript)
        self.decoys_per_hit = decoys_per_hit

    def decoys_for_experiment(self, experiment_name, hit_list):
        assert len(hit_list) > 0, "No hits for %s" % experiment_name
        expression_group = self.experiment_to_expression_group[experiment_name]
        transcripts = self.peptide_to_expression_group_to_transcript.ix[
            hit_list, expression_group
        ]
        assert len(transcripts) > 0, experiment_name

        universe = self.peptides_and_transcripts.ix[
            self.peptides_and_transcripts.transcript.isin(transcripts) &
            (~ self.peptides_and_transcripts.peptide.isin(hit_list))
        ].peptide.values
        assert len(universe) > 0, experiment_name

        return numpy.random.choice(
            universe,
            replace=True,
            size=self.decoys_per_hit * len(hit_list))
