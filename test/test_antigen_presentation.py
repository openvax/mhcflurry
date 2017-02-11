from nose.tools import eq_, assert_less

import numpy
from numpy import testing
import pandas
from mhcflurry import amino_acid
from mhcflurry.antigen_presentation import presentation_component_models


def make_random_peptides(num, length=9):
    return [
        ''.join(peptide_sequence)
        for peptide_sequence in
        numpy.random.choice(
            amino_acid.common_amino_acid_letters, size=(num, length))
    ]


PEPTIDES = make_random_peptides(100, 9)

TRANSCRIPTS = [
    "transcript-%d" % i
    for i in range(1, 10)
]

EXPERIMENT_TO_ALLELES = {
    'exp1': ['HLA-A*01:01'],
    'exp2': ['HLA-A*02:01', 'HLA-B*51:01'],
}

EXPERIMENT_TO_EXPRESSION_GROUP = {
    'exp1': 'group1',
    'exp2': 'group2',
}

EXPERESSION_GROUPS = sorted(set(EXPERIMENT_TO_EXPRESSION_GROUP.values()))

TRANSCIPTS_DF = pandas.DataFrame(index=PEPTIDES, columns=EXPERESSION_GROUPS)
TRANSCIPTS_DF[:] = numpy.random.choice(TRANSCRIPTS, size=TRANSCIPTS_DF.shape)

PEPTIDES_AND_TRANSCRIPTS_DF = TRANSCIPTS_DF.stack().to_frame().reset_index()
PEPTIDES_AND_TRANSCRIPTS_DF.columns = ["peptide", "group", "transcript"]
del PEPTIDES_AND_TRANSCRIPTS_DF["group"]


def hit_criterion(experiment_name, peptide):
    return 'A' in peptide


PEPTIDES_DF = pandas.DataFrame({"peptide": PEPTIDES})
PEPTIDES_DF["experiment_name"] = "exp1"
PEPTIDES_DF["hit"] = [
    hit_criterion(row.experiment_name, row.peptide)
    for _, row in
    PEPTIDES_DF.iterrows()
]

HITS_DF = PEPTIDES_DF.ix[PEPTIDES_DF.hit].reset_index().copy()
del HITS_DF["hit"]


def test_mhcflurry_trained_on_hits():
    model = presentation_component_models.MHCflurryTrainedOnHits(
        "basic",
        experiment_to_alleles=EXPERIMENT_TO_ALLELES,
        experiment_to_expression_group=EXPERIMENT_TO_EXPRESSION_GROUP,
        transcripts=TRANSCIPTS_DF,
        peptides_and_transcripts=PEPTIDES_AND_TRANSCRIPTS_DF,
        random_peptides_for_percent_rank=make_random_peptides(10000, 9),
    )
    model.fit(HITS_DF)

    peptides = PEPTIDES_DF.copy()
    predictions = model.predict(peptides)
    peptides["affinity"] = predictions["mhcflurry_basic_affinity"]
    peptides["percent_rank"] = predictions["mhcflurry_basic_percentile_rank"]
    assert_less(
        peptides.affinity[peptides.hit].mean(),
        peptides.affinity[~peptides.hit].mean())
    assert_less(
        peptides.percent_rank[peptides.hit].mean(),
        peptides.percent_rank[~peptides.hit].mean())
