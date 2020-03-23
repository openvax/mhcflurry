import logging
import math

import numpy
import pandas

from .hyperparameters import HyperparameterDefaults
from .common import amino_acid_distribution, random_peptides


class RandomNegativePeptides(object):
    """
    Generate random negative (peptide, allele) pairs. These are used during
    model training, where they are resampled at each epoch.
    """

    hyperparameter_defaults = HyperparameterDefaults(
        random_negative_rate=0.0,
        random_negative_constant=0,
        random_negative_match_distribution=True,
        random_negative_distribution_smoothing=0.0,
        random_negative_method="recommended",
        random_negative_binder_threshold=None,
        random_negative_lengths=[8,9,10,11,12,13,14,15])
    """
    Hyperperameters for random negative peptides.
    
    Number of random negatives will be:
        random_negative_rate * (num measurements) + random_negative_constant
        
    where the exact meaning of (num measurements) depends on the particular
    random_negative_method in use.
    
    If random_negative_match_distribution is True, then the amino acid
    frequencies of the training data peptides are used to generate the
    random peptides.
    
    Valid values for random_negative_method are:
        "by_length": used for allele-specific prediction. See description in
            `RandomNegativePeptides.plan_by_length` method.
        "by_allele": used for pan-allele prediction. See
            `RandomNegativePeptides.plan_by_allele` method.
        "by_allele_equalize_nonbinders": used for pan-allele prediction. See
            `RandomNegativePeptides.plan_by_allele_equalize_nonbinders` method.
        "recommended": the default. Use by_length if the predictor is allele-
            specific and by_allele if it's pan-allele.    
        
    """

    def __init__(self, **hyperparameters):
        self.hyperparameters = self.hyperparameter_defaults.with_defaults(
            hyperparameters)
        self.plan_df = None
        self.aa_distribution = None

    def plan(self, peptides, affinities, alleles=None, inequalities=None):
        """
        Calculate the number of random negatives for each allele and peptide
        length. Call this once after instantiating the object.

        Parameters
        ----------
        peptides : list of string
        affinities : list of float
        alleles : list of string, optional
        inequalities : list of string (">", "<", or "="), optional

        Returns
        -------
        pandas.DataFrame indicating number of random negatives for each length
        and allele.
        """
        numpy.testing.assert_equal(len(peptides), len(affinities))
        if alleles is not None:
            numpy.testing.assert_equal(len(peptides), len(alleles))
        if inequalities is not None:
            numpy.testing.assert_equal(len(peptides), len(inequalities))

        peptides = pandas.Series(peptides, copy=False)
        peptide_lengths = peptides.str.len()

        if self.hyperparameters['random_negative_match_distribution']:
            self.aa_distribution = amino_acid_distribution(
                peptides.values,
                smoothing=self.hyperparameters[
                    'random_negative_distribution_smoothing'
                ])
            logging.info(
                "Using amino acid distribution for random negative:\n%s" % (
                    str(self.aa_distribution.to_dict())))

        df_all = pandas.DataFrame({
            'length': peptide_lengths,
            'affinity': affinities,
        })
        df_all["allele"] = "" if alleles is None else alleles
        df_all["inequality"] = "=" if inequalities is None else inequalities

        df_binders = None
        df_nonbinders = None
        if self.hyperparameters['random_negative_binder_threshold']:
            df_nonbinders = df_all.loc[
                (df_all.inequality != "<") &
                (df_all.affinity > self.hyperparameters[
                    'random_negative_binder_threshold'
                ])
            ]
            df_binders = df_all.loc[
                (df_all.inequality != ">") &
                (df_all.affinity <= self.hyperparameters[
                    'random_negative_binder_threshold'
                ])
            ]

        method = self.hyperparameters['random_negative_method']
        if method == 'recommended':
            # by_length for allele-specific prediction and by_allele for pan.
            method = (
                "by_length"
                if alleles is None else
                "by_allele")

        function = {
            'by_length': self.plan_by_length,
            'by_allele': self.plan_by_allele,
            'by_allele_equalize_nonbinders':
                self.plan_by_allele_equalize_nonbinders,
        }[method]
        function(df_all, df_binders, df_nonbinders)
        assert self.plan_df is not None
        logging.info("Random negative plan [%s]:\n%s", method, self.plan_df)
        return self.plan_df

    def plan_by_length(self, df_all, df_binders=None, df_nonbinders=None):
        """
        Generate a random negative plan using the "by_length" policy.

        Parameters are as in the `plan` method. No return value.

        Used for allele-specific predictors. Does not work well for pan-allele.

        Different numbers of random negatives per length. Alleles are sampled
        proportionally to the number of times they are used in the training
        data.
        """
        assert list(df_all.allele.unique()) == [""], (
            "by_length only recommended for allele specific prediction")

        df = df_all if df_binders is None else df_binders
        lengths = self.hyperparameters['random_negative_lengths']

        length_to_num_random_negative = {}
        length_counts = df.length.value_counts().to_dict()
        for length in lengths:
            length_to_num_random_negative[length] = int(
                length_counts.get(length, 0) *
                self.hyperparameters['random_negative_rate'] +
                self.hyperparameters['random_negative_constant'])

        plan_df = pandas.DataFrame(index=sorted(df.allele.unique()))
        for length in lengths:
            plan_df[length] = length_to_num_random_negative[length]
        self.plan_df = plan_df.astype(int)

    def plan_by_allele(self, df_all, df_binders=None, df_nonbinders=None):
        """
        Generate a random negative plan using the "by_allele" policy.

        Parameters are as in the `plan` method. No return value.

        For each allele, a particular number of random negatives are used
        for all lengths. Across alleles, the number of random negatives
        varies; within an allele, the number of random negatives for each
        length is a constant
        """
        allele_to_num_per_length = {}
        total_random_peptides_per_length = 0
        df = df_all if df_binders is None else df_binders
        lengths = self.hyperparameters['random_negative_lengths']
        all_alleles = df_all.allele.unique()
        for allele in all_alleles:
            sub_df = df.loc[df.allele == allele]
            num_for_allele = len(sub_df) * (
                self.hyperparameters['random_negative_rate']
            ) + self.hyperparameters['random_negative_constant']
            num_per_length = int(math.ceil(
                num_for_allele / len(lengths)))
            total_random_peptides_per_length += num_per_length
            allele_to_num_per_length[allele] = num_per_length

        plan_df = pandas.DataFrame(index=sorted(df.allele.unique()))
        for length in lengths:
            plan_df[length] = plan_df.index.map(allele_to_num_per_length)
        self.plan_df = plan_df.astype(int)

    def plan_by_allele_equalize_nonbinders(
            self, df_all, df_binders, df_nonbinders):
        """
        Generate a random negative plan using the
        "by_allele_equalize_nonbinders" policy.

        Parameters are as in the `plan` method. No return value.

        Requires that the random_negative_binder_threshold hyperparameter is set.

        In a first step, the number of random negatives selected by the
        "by_allele" method are added (see `plan_by_allele`). Then, the total
        number of non-binders are calculated for each allele and length. This
        total includes non-binder measurements in the training data plus the
        random negative peptides added in the first step. In a second step,
        additional random negative peptides are added so that for each allele,
        all peptide lengths have the same total number of non-binders.
        """
        assert df_binders is not None
        assert df_nonbinders is not None

        lengths = self.hyperparameters['random_negative_lengths']

        self.plan_by_allele(df_all, df_binders, df_nonbinders)
        first_pass_plan = self.plan_df
        self.plan_df = None

        new_plan = first_pass_plan.copy()
        new_plan[:] = numpy.nan

        for (allele, first_pass_per_length) in first_pass_plan.iterrows():
            real_nonbinders_by_length = df_nonbinders.loc[
                df_nonbinders.allele == allele
            ].length.value_counts().reindex(lengths).fillna(0)
            total_nonbinders_by_length = (
                real_nonbinders_by_length + first_pass_per_length)
            new_plan.loc[allele] = first_pass_per_length + (
                total_nonbinders_by_length.max() - total_nonbinders_by_length)

        self.plan_df = new_plan.astype(int)

    def get_alleles(self):
        """
        Get the list of alleles corresponding to each random negative peptide
        as returned by `get_peptides`. This does NOT change and can be safely
        called once and reused.

        Returns
        -------
        list of string
        """
        assert self.plan_df is not None, "Call plan() first"
        alleles = []
        for allele, row in self.plan_df.iterrows():
            alleles.extend([allele] * int(row.sum()))
        assert len(alleles) == self.get_total_count()
        return alleles

    def get_peptides(self):
        """
        Get the list of random negative peptides. This will be different each
        time the method is called.

        Returns
        -------
        list of string

        """
        assert self.plan_df is not None, "Call plan() first"
        peptides = []
        for allele, row in self.plan_df.iterrows():
            for (length, num) in row.items():
                peptides.extend(
                    random_peptides(
                        num,
                        length=length,
                        distribution=self.aa_distribution))
        assert len(peptides) == self.get_total_count()
        return peptides

    def get_total_count(self):
        """
        Total number of planned random negative peptides.

        Returns
        -------
        int
        """
        return self.plan_df.sum().sum()