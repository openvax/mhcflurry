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

import numpy as np

from model_configs import ModelConfig


def hyperparameter_performance(df):
    print("\n=== Hyperparameters ===")
    for hyperparameter_name in ModelConfig._fields:
        print("\n%s" % hyperparameter_name)
        groups = df.groupby(hyperparameter_name)
        for hyperparameter_value, group in groups:
            aucs = list(group["auc_mean"])
            f1_scores = list(group["f1_mean"])
            unique_configs = group["config_idx"].unique()
            auc_25th = np.percentile(aucs, 25.0)
            auc_50th = np.percentile(aucs, 50.0)
            auc_75th = np.percentile(aucs, 75.0)

            f1_25th = np.percentile(f1_scores, 25.0)
            f1_50th = np.percentile(f1_scores, 50.0)
            f1_75th = np.percentile(f1_scores, 75.0)

            print(
                "-- %s (%d): AUC=%0.4f/%0.4f/%0.4f, F1=%0.4f/%0.4f/%0.4f" % (
                    hyperparameter_value,
                    len(unique_configs),
                    auc_25th, auc_50th, auc_75th,
                    f1_25th, f1_50th, f1_75th))
