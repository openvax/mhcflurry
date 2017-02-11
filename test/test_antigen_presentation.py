from nose.tools import eq_, assert_less

import numpy
from numpy.testing import assert_allclose
import pandas
from mhcflurry import amino_acid
from mhcflurry.antigen_presentation import (
    decoy_strategies,
    percent_rank_transform,
    presentation_component_models,
    presentation_model)


######################
# Helper functions

def make_random_peptides(num, length=9):
    return [
        ''.join(peptide_sequence)
        for peptide_sequence in
        numpy.random.choice(
            amino_acid.common_amino_acid_letters, size=(num, length))
    ]


def hit_criterion(experiment_name, peptide):
    # Peptides with 'A' are always hits. Easy for model to learn.
    return 'A' in peptide


######################
# Small test dataset

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

PEPTIDES_DF = pandas.DataFrame({"peptide": PEPTIDES})
PEPTIDES_DF["experiment_name"] = "exp1"
PEPTIDES_DF["hit"] = [
    hit_criterion(row.experiment_name, row.peptide)
    for _, row in
    PEPTIDES_DF.iterrows()
]
print("Hit rate: %0.3f" % PEPTIDES_DF.hit.mean())

HITS_DF = PEPTIDES_DF.ix[PEPTIDES_DF.hit].reset_index().copy()
del HITS_DF["hit"]

######################
# Tests


def test_percent_rank_transform():
    model = percent_rank_transform.PercentRankTransform()
    model.fit(numpy.arange(1000))
    assert_allclose(
        model.transform([-2, 0, 50, 100, 2000]),
        [0.0, 0.0, 5.0, 10.0, 100.0],
        err_msg=str(model.__dict__))


def test_mhcflurry_trained_on_hits():
    mhcflurry_model = presentation_component_models.MHCflurryTrainedOnHits(
        "basic",
        experiment_to_alleles=EXPERIMENT_TO_ALLELES,
        experiment_to_expression_group=EXPERIMENT_TO_EXPRESSION_GROUP,
        transcripts=TRANSCIPTS_DF,
        peptides_and_transcripts=PEPTIDES_AND_TRANSCRIPTS_DF,
        random_peptides_for_percent_rank=make_random_peptides(10000, 9),
    )
    mhcflurry_model.fit(HITS_DF)

    peptides = PEPTIDES_DF.copy()
    predictions = mhcflurry_model.predict(peptides)
    peptides["affinity"] = predictions["mhcflurry_basic_affinity"]
    peptides["percent_rank"] = predictions["mhcflurry_basic_percentile_rank"]
    assert_less(
        peptides.affinity[peptides.hit].mean(),
        peptides.affinity[~peptides.hit].mean())
    assert_less(
        peptides.percent_rank[peptides.hit].mean(),
        peptides.percent_rank[~peptides.hit].mean())


def test_presentation_model():
    mhcflurry_model = presentation_component_models.MHCflurryTrainedOnHits(
        "basic",
        experiment_to_alleles=EXPERIMENT_TO_ALLELES,
        experiment_to_expression_group=EXPERIMENT_TO_EXPRESSION_GROUP,
        transcripts=TRANSCIPTS_DF,
        peptides_and_transcripts=PEPTIDES_AND_TRANSCRIPTS_DF,
        random_peptides_for_percent_rank=make_random_peptides(1000, 9),
    )

    decoys = decoy_strategies.UniformRandom(
        make_random_peptides(1000, 9),
        decoys_per_hit=50)

    terms = {
        'A_ms': (
            [mhcflurry_model],
            ["log1p(mhcflurry_basic_affinity)"]),
    }

    models = presentation_model.build_presentation_models(
        terms,
        ["A_ms"],
        decoy_strategy=decoys)
    eq_(len(models), 1)

    model = models["A_ms"]
    model.fit(HITS_DF.ix[HITS_DF.experiment_name == "exp1"])

    peptides = PEPTIDES_DF.copy()
    peptides["prediction"] = model.predict(peptides)
    assert_less(
        peptides.prediction[~peptides.hit].mean(),
        peptides.prediction[peptides.hit].mean())
