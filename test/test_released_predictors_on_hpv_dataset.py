# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test accuracy on HPV benchmark used in MHCflurry Cell Systems 2018 paper.

The study that generated this dataset has now been published
(Bonsack et al 2019, DOI: 10.1158/2326-6066.CIR-18-0584), and the authors
request that any work based on the HPV dataset cite this paper.
"""

import os
import pandas
import pytest
from sklearn.metrics import roc_auc_score

pytestmark = [pytest.mark.slow, pytest.mark.downloads]

def data_path(name):
    '''
    Return the absolute path to a file in the test/data directory.
    The name specified should be relative to test/data.
    '''
    return os.path.join(os.path.dirname(__file__), "data", name)


DF = pandas.read_csv(data_path("hpv_predictions.csv"))


def test_on_hpv(released_affinity_predictors, df=DF):
    scores_df = []
    for (name, predictor) in released_affinity_predictors.items():
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
