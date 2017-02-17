import pandas


class DecoyStrategy(object):
    """
    A mechanism for selecting decoys (non-hit peptides) given hits (
    peptides detected via mass-spec).

    Subclasses should override either decoys() or decoys_for_experiment().
    Whichever one is not overriden is implemented using the other.
    """

    def __init__(self):
        pass

    def decoys(self, hits_df):
        """
        Given a df of hits with columns 'experiment_name' and 'peptide',
        return a df with the same structure giving decoys.

        Subclasses should override either this or decoys_for_experiment()
        """

        assert 'experiment_name' in hits_df.columns
        assert 'peptide' in hits_df.columns
        assert len(hits_df) > 0
        grouped = hits_df.groupby("experiment_name")
        dfs = []
        for (experiment_name, sub_df) in grouped:
            decoys = self.decoys_for_experiment(
                experiment_name,
                sub_df.peptide.values)
            df = pandas.DataFrame({
                'peptide': decoys,
            })
            df["experiment_name"] = experiment_name
            dfs.append(df)
        return pandas.concat(dfs, ignore_index=True)

    def decoys_for_experiment(self, experiment_name, hit_list):
        """
        Return decoys for a single experiment.

        Parameters
        ------------
        experiment_name : string

        hit_list : list of string
            List of hits

        """
        # prevent infinite recursion:
        assert self.decoys is not DecoyStrategy.decoys

        hits_df = pandas.DataFrame({'peptide': hit_list})
        hits_df["experiment_name"] = experiment_name
        return self.decoys(hits_df)
