# Copyright (c) 2015. Mount Sinai School of Medicine
#
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

from collections import OrderedDict

import numpy as np
from scipy.stats import mannwhitneyu

from model_configs import ModelConfig


def hyperparameter_performance(df):
    print("\n=== Hyperparameter Score 25th/50th/75th Percentiles ===")
    for hyperparameter_name in ModelConfig._fields:
        print("\n%s" % hyperparameter_name)
        groups = df.groupby(hyperparameter_name)
        for hyperparameter_value, group in groups:
            aucs = list(group.groupby("config_idx")["auc_mean"].mean())
            f1_scores = list(group.groupby("config_idx")["f1_mean"].mean())
            auc_25th = np.percentile(aucs, 10.0)
            auc_50th = np.percentile(aucs, 50.0)
            auc_75th = np.percentile(aucs, 90.0)

            f1_25th = np.percentile(f1_scores, 10.0)
            f1_50th = np.percentile(f1_scores, 50.0)
            f1_75th = np.percentile(f1_scores, 90.0)
            print(
                "-- %s (%d): AUC=%0.4f/%0.4f/%0.4f, F1=%0.4f/%0.4f/%0.4f" % (
                    hyperparameter_value,
                    len(aucs),
                    auc_25th, auc_50th, auc_75th,
                    f1_25th, f1_50th, f1_75th))


def hyperparameter_score_difference_hypothesis_tests(df):
    for hyperparameter_name in ModelConfig._fields:
        print("\n%s" % hyperparameter_name)
        combined_scores_by_param = OrderedDict()
        groups = df.groupby(hyperparameter_name)
        for hyperparameter_value, group in groups:
            raw_scores = group.groupby("config_idx")[["auc_mean", "f1_mean"]]
            combined_scores = []
            aucs = []
            f1_scores = []
            for config_idx, subgroup in raw_scores:
                subgroup_aucs = subgroup["auc_mean"]
                subgroup_f1_scores = subgroup["f1_mean"]
                combined_score = np.sqrt(
                    (subgroup_aucs * subgroup_f1_scores)).mean()
                auc = subgroup_aucs.mean()
                f1_score = subgroup_f1_scores.mean()
                aucs.append(auc)
                f1_scores.append(f1_score)
                combined_scores.append(combined_score)
            combined_scores_by_param[hyperparameter_value] = combined_scores

            print(
                ("-- %s (%d) AUC median = %0.4f, F1 median = %0.4f"
                 " combined score median: %0.4f, max = %0.4f") % (
                    hyperparameter_value,
                    len(combined_scores),
                    np.median(aucs),
                    np.median(f1_scores),
                    np.median(combined_scores),
                    np.max(combined_scores)))
        done = set([])
        for (value1, scores1) in combined_scores_by_param.items():
            for (value2, scores2) in combined_scores_by_param.items():
                if value1 == value2:
                    continue
                if frozenset({value1, value2}) in done:
                    continue
                U, p = mannwhitneyu(scores1, scores2)
                # AUC = Area under ROC curve
                # probability of always first hyperparam
                # causing us to correct rank a pair of classifiers
                AUC = U / (len(scores1) * len(scores2))
                print (">>> %s%s vs. %s%s, AUC=%0.4f, p=%0.20f" % (
                    value1, "*" if AUC > 0.5 else "",
                    value2, "*" if AUC < 0.5 else "",
                    AUC,
                    p))
                done.add(frozenset({value1, value2}))
