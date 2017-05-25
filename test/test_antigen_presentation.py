import pickle

from nose.tools import eq_, assert_less

import numpy
from numpy.testing import assert_allclose
import pandas
from mhcflurry.antigen_presentation import (
    decoy_strategies,
    percent_rank_transform,
    presentation_component_models,
    presentation_model)

from mhcflurry.amino_acid import COMMON_AMINO_ACIDS
from mhcflurry.common import random_peptides


######################
# Helper functions

def hit_criterion(experiment_name, peptide):
    # Peptides with 'A' are always hits. Easy for model to learn.
    return 'A' in peptide


######################
# Small test dataset

PEPTIDES = random_peptides(1000, 9)
OTHER_PEPTIDES = random_peptides(1000, 9)

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

AA_COMPOSITION_DF = pandas.DataFrame({
    'peptide': sorted(set(PEPTIDES).union(set(OTHER_PEPTIDES))),
})
for aa in sorted(COMMON_AMINO_ACIDS):
    AA_COMPOSITION_DF[aa] = AA_COMPOSITION_DF.peptide.str.count(aa)

AA_COMPOSITION_DF.index = AA_COMPOSITION_DF.peptide
del AA_COMPOSITION_DF['peptide']

HITS_DF = PEPTIDES_DF.ix[PEPTIDES_DF.hit].reset_index().copy()

# Add some duplicates:
HITS_DF = pandas.concat([HITS_DF, HITS_DF.iloc[:10]], ignore_index=True)
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


def mhcflurry_basic_model():
    return presentation_component_models.MHCflurryTrainedOnHits(
        predictor_name="mhcflurry_affinity",
        experiment_to_alleles=EXPERIMENT_TO_ALLELES,
        experiment_to_expression_group=EXPERIMENT_TO_EXPRESSION_GROUP,
        transcripts=TRANSCIPTS_DF,
        peptides_and_transcripts=PEPTIDES_AND_TRANSCRIPTS_DF,
        random_peptides_for_percent_rank=OTHER_PEPTIDES,
    )


def mhcflurry_released_model():
    return presentation_component_models.MHCflurryReleased(
        predictor_name="mhcflurry_ensemble",
        experiment_to_alleles=EXPERIMENT_TO_ALLELES,
        random_peptides_for_percent_rank=OTHER_PEPTIDES,
        fit_cache_policy="strong",
        predictions_cache_policy="strong")


def test_mhcflurry_trained_on_hits():
    mhcflurry_model = mhcflurry_basic_model()
    mhcflurry_model.fit(HITS_DF)

    peptides = PEPTIDES_DF.copy()
    predictions = mhcflurry_model.predict(peptides)
    peptides["affinity"] = predictions["mhcflurry_affinity_value"]
    peptides["percent_rank"] = predictions[
        "mhcflurry_affinity_percentile_rank"
    ]
    assert_less(
        peptides.affinity[peptides.hit].mean(),
        peptides.affinity[~peptides.hit].mean())
    assert_less(
        peptides.percent_rank[peptides.hit].mean(),
        peptides.percent_rank[~peptides.hit].mean())


def compare_predictions(peptides_df, model1, model2):
    predictions1 = model1.predict(peptides_df)
    predictions2 = model2.predict(peptides_df)
    failed = False
    for i in range(len(peptides_df)):
        if abs(predictions1[i] - predictions2[i]) > .0001:
            failed = True
            print(
                "Compare predictions: mismatch at index %d: "
                "%f != %f, row: %s" % (
                    i,
                    predictions1[i],
                    predictions2[i],
                    str(peptides_df.iloc[i])))
    assert not failed


def test_presentation_model():
    mhcflurry_model = mhcflurry_basic_model()
    mhcflurry_ie_model = mhcflurry_released_model()

    aa_content_model = (
        presentation_component_models.FixedPerPeptideQuantity(
            "aa composition",
            numpy.log1p(AA_COMPOSITION_DF)))

    decoys = decoy_strategies.UniformRandom(
        OTHER_PEPTIDES,
        decoys_per_hit=50)

    terms = {
        'A_ie': (
            [mhcflurry_ie_model],
            ["log1p(mhcflurry_ensemble_affinity)"]),
        'A_ms': (
            [mhcflurry_model],
            ["log1p(mhcflurry_affinity_value)"]),
        'P': (
            [aa_content_model],
            list(AA_COMPOSITION_DF.columns)),
    }

    for kwargs in [{}, {'ensemble_size': 3}]:
        models = presentation_model.build_presentation_models(
            terms,
            ["A_ms", "A_ms + P", "A_ie + P"],
            decoy_strategy=decoys,
            **kwargs)
        eq_(len(models), 3)

        unfit_model = models["A_ms"]
        model = unfit_model.clone()
        model.fit(HITS_DF.ix[HITS_DF.experiment_name == "exp1"])

        peptides = PEPTIDES_DF.copy()
        peptides["prediction"] = model.predict(peptides)
        print(peptides)
        print("Hit mean", peptides.prediction[peptides.hit].mean())
        print("Decoy mean", peptides.prediction[~peptides.hit].mean())

        assert_less(
            peptides.prediction[~peptides.hit].mean(),
            peptides.prediction[peptides.hit].mean())

        model2 = pickle.loads(pickle.dumps(model))
        compare_predictions(peptides, model, model2)

        model3 = unfit_model.clone()
        assert not model3.has_been_fit
        model3.restore_fit(model2.get_fit())
        compare_predictions(peptides, model, model3)

        better_unfit_model = models["A_ms + P"]
        model = better_unfit_model.clone()
        model.fit(HITS_DF.ix[HITS_DF.experiment_name == "exp1"])
        peptides["prediction_better"] = model.predict(peptides)
        assert_less(
            peptides.prediction_better[~peptides.hit].mean(),
            peptides.prediction[~peptides.hit].mean())
        assert_less(
            peptides.prediction[peptides.hit].mean(),
            peptides.prediction_better[peptides.hit].mean())

        another_unfit_model = models["A_ie + P"]
        model = another_unfit_model.clone()
        model.fit(HITS_DF.ix[HITS_DF.experiment_name == "exp1"])
        model.predict(peptides)
