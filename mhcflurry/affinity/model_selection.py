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

"""Model-selection helpers for class I affinity predictors."""

import numpy
import pandas


def model_select(
        predictor,
        predictor_class,
        score_function,
        alleles=None,
        min_models=1,
        max_models=10000):
    """
    Perform model selection using a user-specified scoring function.

    This works only with allele-specific models, not pan-allele models.

    Model selection is done using a "step up" variable selection procedure,
    in which models are repeatedly added to an ensemble until the score
    stops improving.

    Parameters
    ----------
    score_function : Class1AffinityPredictor -> float function
        Scoring function

    alleles : list of string, optional
        If not specified, model selection is performed for all alleles.

    min_models : int, optional
        Min models to select per allele

    max_models : int, optional
        Max models to select per allele

    Returns
    -------
    Class1AffinityPredictor : predictor containing the selected models
    """

    if alleles is None:
        alleles = predictor.supported_alleles

    dfs = []
    allele_to_allele_specific_models = {}

    def model_is_selectable(model):
        max_epochs = model.hyperparameters.get("max_epochs", 1)
        return max_epochs is None or int(max_epochs) > 0

    for allele in alleles:
        df = pandas.DataFrame({
            'model': predictor.allele_to_allele_specific_models[allele]
        })
        df["model_num"] = df.index
        df["allele"] = allele
        df["selected"] = False
        df["selectable"] = df.model.map(model_is_selectable)

        round_num = 1

        while (
                not df.loc[df.selectable, "selected"].all()
                and sum(df.selected) < max_models):
            score_col = "score_%2d" % round_num
            prev_score_col = "score_%2d" % (round_num - 1)

            existing_selected = list(df[df.selected].model)
            df[score_col] = [
                numpy.nan if row.selected or not row.selectable else
                score_function(
                    predictor_class(
                        allele_to_allele_specific_models={
                            allele: [row.model] + existing_selected
                        }
                    )
                )
                for (_, row) in df.iterrows()
            ]

            if not numpy.isfinite(df[score_col]).any():
                break

            if round_num > min_models and (
                    df[score_col].max() < df[prev_score_col].max()):
                break

            # In case of a tie, pick a model at random.
            (best_model_index,) = df.loc[
                (df[score_col] == df[score_col].max())
            ].sample(1).index
            df.loc[best_model_index, "selected"] = True
            round_num += 1

        dfs.append(df)
        allele_to_allele_specific_models[allele] = list(
            df.loc[df.selected].model)

    df = pandas.concat(dfs, ignore_index=True)

    new_predictor = predictor_class(
        allele_to_allele_specific_models,
        metadata_dataframes={
            "model_selection": df,
        })
    return new_predictor
