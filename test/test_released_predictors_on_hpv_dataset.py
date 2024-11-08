"""
Test accuracy on HPV benchmark used in MHCflurry Cell Systems 2018 paper.

The study that generated this dataset has now been published
(Bonsack et al 2019, DOI: 10.1158/2326-6066.CIR-18-0584), and the authors
request that any work based on the HPV dataset cite this paper.
"""
from . import initialize
initialize()

import os
import pandas
import pytest
from sklearn.metrics import roc_auc_score

from mhcflurry import Class1AffinityPredictor
from mhcflurry.downloads import get_path

from mhcflurry.testing_utils import cleanup, startup

def data_path(name):
    '''
    Return the absolute path to a file in the test/data directory.
    The name specified should be relative to test/data.
    '''
    return os.path.join(os.path.dirname(__file__), "data", name)


DF = pandas.read_csv(data_path("hpv_predictions.csv"))


# Define a fixture to initialize and clean up predictors
@pytest.fixture(scope="module")
def predictors():
    startup()
    predictors_dict = {
        'allele-specific': Class1AffinityPredictor.load(get_path("models_class1", "models")),
        'pan-allele': Class1AffinityPredictor.load(get_path("models_class1_pan", "models.combined")),
    }
    yield predictors_dict
    cleanup()


def test_on_hpv(predictors, df=DF):
    scores_df = []
    for (name, predictor) in predictors.items():
        print("Running", name)
        df[name] = predictor.predict(df.peptide, alleles=df.allele)

    for name in df.columns[8:]:
        for nm_cutoff in [2000, 5000, 50000]:
            labels = df["affinity"] < nm_cutoff
            auc = roc_auc_score(labels.values, -1 * df[name].values)
            scores_df.append((name, "auc-%dnM" % nm_cutoff, auc))
    scores_df = pandas.DataFrame(
        scores_df,
        columns=["predictor", "metric", "score"])
    scores_df = scores_df.pivot(
        index="metric", columns="predictor", values="score")

    print("Predictions")
    print(df)

    print("Scores")
    print(scores_df)

    mean_scores = scores_df.mean()
    assert mean_scores["allele-specific"] > mean_scores["netmhcpan4"]
    assert mean_scores["pan-allele"] > mean_scores["netmhcpan4"]
