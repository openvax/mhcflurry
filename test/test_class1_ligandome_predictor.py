"""

Idea:

- take an allele where MS vs. no-MS trained predictors are very different. One
    possiblility is DLA-88*501:01 but human would be better
- generate synethetic multi-allele MS by combining single-allele MS for differnet
   alleles, including the selected allele
- train ligandome predictor based on the no-ms pan-allele models on theis
  synthetic dataset
- see if the pan-allele predictor learns the "correct" motif for the selected
  allele, i.e. updates to become more similar to the with-ms pan allele predictor.


"""

from sklearn.metrics import roc_auc_score
import pandas
import argparse
import sys

from numpy.testing import assert_, assert_equal, assert_allclose
import numpy
from random import shuffle

from sklearn.metrics import roc_auc_score

from mhcflurry import Class1AffinityPredictor,Class1NeuralNetwork
from mhcflurry.allele_encoding import MultipleAlleleEncoding
from mhcflurry.class1_ligandome_predictor import Class1LigandomePredictor
from mhcflurry.downloads import get_path
from mhcflurry.regression_target import from_ic50

from mhcflurry.testing_utils import cleanup, startup
from mhcflurry.amino_acid import COMMON_AMINO_ACIDS

COMMON_AMINO_ACIDS = sorted(COMMON_AMINO_ACIDS)

PAN_ALLELE_PREDICTOR_NO_MASS_SPEC = None
PAN_ALLELE_MOTIFS_WITH_MASS_SPEC_DF = None
PAN_ALLELE_MOTIFS_NO_MASS_SPEC_DF = None


def setup():
    global PAN_ALLELE_PREDICTOR_NO_MASS_SPEC
    global PAN_ALLELE_MOTIFS_WITH_MASS_SPEC_DF
    global PAN_ALLELE_MOTIFS_NO_MASS_SPEC_DF
    startup()
    PAN_ALLELE_PREDICTOR_NO_MASS_SPEC = Class1AffinityPredictor.load(
        get_path("models_class1_pan", "models.no_mass_spec"),
        optimization_level=0,
        max_models=1)

    PAN_ALLELE_MOTIFS_WITH_MASS_SPEC_DF = pandas.read_csv(
        get_path(
            "models_class1_pan",
            "models.with_mass_spec/frequency_matrices.csv.bz2"))
    PAN_ALLELE_MOTIFS_NO_MASS_SPEC_DF = pandas.read_csv(
        get_path(
            "models_class1_pan",
            "models.no_mass_spec/frequency_matrices.csv.bz2"))


def teardown():
    global PAN_ALLELE_PREDICTOR_NO_MASS_SPEC
    global PAN_ALLELE_MOTIFS_WITH_MASS_SPEC_DF
    global PAN_ALLELE_MOTIFS_NO_MASS_SPEC_DF

    PAN_ALLELE_PREDICTOR_NO_MASS_SPEC = None
    PAN_ALLELE_MOTIFS_WITH_MASS_SPEC_DF = None
    PAN_ALLELE_MOTIFS_NO_MASS_SPEC_DF = None
    cleanup()


def sample_peptides_from_pssm(pssm, count):
    result = pandas.DataFrame(
        index=numpy.arange(count),
        columns=pssm.index,
        dtype=object,
    )

    for (position, vector) in pssm.iterrows():
        result.loc[:, position] = numpy.random.choice(
            pssm.columns,
            size=count,
            replace=True,
            p=vector.values)

    return result.apply("".join, axis=1)


def scramble_peptide(peptide):
    lst = list(peptide)
    shuffle(lst)
    return "".join(lst)


def test_synthetic_allele_refinement():
    refine_allele = "HLA-C*01:02"
    alleles = [
        "HLA-A*02:01", "HLA-B*27:01", "HLA-C*07:01",
        "HLA-A*03:01", "HLA-B*15:01", refine_allele
    ]
    peptides_per_allele = [
        2000, 1000, 500,
        1500, 1200, 800,
    ]

    allele_to_peptides = dict(zip(alleles, peptides_per_allele))

    length = 9

    train_with_ms = pandas.read_csv(
        get_path("data_curated", "curated_training_data.with_mass_spec.csv.bz2"))
    train_no_ms = pandas.read_csv(get_path("data_curated",
        "curated_training_data.no_mass_spec.csv.bz2"))

    def filter_df(df):
        df = df.loc[
            (df.allele.isin(alleles)) &
            (df.peptide.str.len() == length)
        ]
        return df

    train_with_ms = filter_df(train_with_ms)
    train_no_ms = filter_df(train_no_ms)

    ms_specific = train_with_ms.loc[
        ~train_with_ms.peptide.isin(train_no_ms.peptide)
    ]

    train_peptides = []
    train_true_alleles = []
    for allele in alleles:
        peptides = ms_specific.loc[ms_specific.allele == allele].peptide.sample(
            n=allele_to_peptides[allele])
        train_peptides.extend(peptides)
        train_true_alleles.extend([allele] * len(peptides))

    hits_df = pandas.DataFrame({"peptide": train_peptides})
    hits_df["true_allele"] = train_true_alleles
    hits_df["hit"] = 1.0

    decoys_df = hits_df.copy()
    decoys_df["peptide"] = decoys_df.peptide.map(scramble_peptide)
    decoys_df["true_allele"] = ""
    decoys_df["hit"] = 0.0

    train_df = pandas.concat([hits_df, decoys_df], ignore_index=True)

    predictor = Class1LigandomePredictor(
        PAN_ALLELE_PREDICTOR_NO_MASS_SPEC,
        max_ensemble_size=1,
        max_epochs=100,
        patience=5)

    allele_encoding = MultipleAlleleEncoding(
        experiment_names=["experiment1"] * len(train_df),
        experiment_to_allele_list={
            "experiment1": alleles,
        },
        max_alleles_per_experiment=6,
        allele_to_sequence=PAN_ALLELE_PREDICTOR_NO_MASS_SPEC.allele_to_sequence,
    ).compact()

    pre_predictions = predictor.predict(
        peptides=train_df.peptide.values,
        allele_encoding=allele_encoding)

    (model,) = PAN_ALLELE_PREDICTOR_NO_MASS_SPEC.class1_pan_allele_models
    expected_pre_predictions = from_ic50(
        model.predict(
            peptides=numpy.repeat(train_df.peptide.values, len(alleles)),
            allele_encoding=allele_encoding.allele_encoding,
    )).reshape((-1, len(alleles)))

    train_df["pre_max_prediction"] = pre_predictions.max(1)
    pre_auc = roc_auc_score(train_df.hit.values, train_df.pre_max_prediction.values)
    print("PRE_AUC", pre_auc)

    #import ipdb ; ipdb.set_trace()

    assert_allclose(pre_predictions, expected_pre_predictions)

    predictor.fit(
        peptides=train_df.peptide.values,
        labels=train_df.hit.values,
        allele_encoding=allele_encoding
    )

    predictions = predictor.predict(
        peptides=train_df.peptide.values,
        allele_encoding=allele_encoding,
    )

    train_df["max_prediction"] = predictions.max(1)
    train_df["predicted_allele"] = pandas.Series(alleles).loc[
        predictions.argmax(1).flatten()
    ].values

    print(predictions)

    auc = roc_auc_score(train_df.hit.values, train_df.max_prediction.values)
    print("AUC", auc)

    import ipdb ; ipdb.set_trace()

    return predictions



"""
def test_simple_synethetic(
        num_peptide_per_allele_and_length=100, lengths=[8,9,10,11]):
    alleles = [
        "HLA-A*02:01", "HLA-B*52:01", "HLA-C*07:01",
        "HLA-A*03:01", "HLA-B*57:02", "HLA-C*03:01",
    ]
    cutoff = PAN_ALLELE_MOTIFS_DF.cutoff_fraction.min()
    peptides_and_alleles = []
    for allele in alleles:
        sub_df = PAN_ALLELE_MOTIFS_DF.loc[
            (PAN_ALLELE_MOTIFS_DF.allele == allele) &
            (PAN_ALLELE_MOTIFS_DF.cutoff_fraction == cutoff)
        ]
        assert len(sub_df) > 0, allele
        for length in lengths:
            pssm = sub_df.loc[
                sub_df.length == length
            ].set_index("position")[COMMON_AMINO_ACIDS]
            peptides = sample_peptides_from_pssm(pssm, num_peptide_per_allele_and_length)
            for peptide in peptides:
                peptides_and_alleles.append((peptide, allele))

    hits_df = pandas.DataFrame(
        peptides_and_alleles,
        columns=["peptide", "allele"]
    )
    hits_df["hit"] = 1

    decoys = hits_df.copy()
    decoys["peptide"] = decoys.peptide.map(scramble_peptide)
    decoys["hit"] = 0.0

    train_df = pandas.concat([hits_df, decoys], ignore_index=True)

    return train_df
    return
    pass
"""

parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument(
    "--alleles",
    nargs="+",
    default=None,
    help="Which alleles to test")

if __name__ == '__main__':
    # If run directly from python, leave the user in a shell to explore results.
    setup()
    args = parser.parse_args(sys.argv[1:])
    result = test_synthetic_allele_refinement()

    # Leave in ipython
    import ipdb  # pylint: disable=import-error
    ipdb.set_trace()
