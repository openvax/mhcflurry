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

from argparse import ArgumentParser

import pandas as pd
import numpy as np

from summarize_model_results import hyperparameter_performance

parser = ArgumentParser()

parser.add_argument(
    "--results-filename",
    required=True,
    help="CSV with results from hyperparameter search")


def infer_dtypes(df):
    column_names = list(df.columns)
    for column_name in column_names:
        column_values = np.array(df[column_name])
        if "object" in str(column_values.dtype):
            if any("." in value for value in column_values):
                df[column_name] = column_values.astype(float)
            elif all(value.isdigit() for value in column_values):
                df[column_name] = column_values.astype(int)
    return df

if __name__ == "__main__":
    args = parser.parse_args()
    results = pd.read_csv(args.results_filename, sep=",", header=0)
    results = infer_dtypes(results)
    hyperparameter_performance(results)
