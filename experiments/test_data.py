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

from collections import defaultdict, OrderedDict
from os import listdir
from os.path import join

import pandas as pd
from mhcflurry.common import normalize_allele_name


def load_test_data(
        dirpaths,
        sep="\s+",
        ic50_base=10.0,
        comment_char="B",
        dataset_name="blind"):
    """
    Load all allele-specific datasets from the given path assuming filenames
    have the form:
        pred.PREDICTOR_NAME.CV_METHOD.ALLELE-LENGTH.xls
    Example:
        pred.netmhc.blind.HLA-A-3201-9.xls
        pred.blind.smmpmbec_cpp.Mamu-A-02-9.xls
    where ALLELE could be HLA-A-0201 and LENGTH is an integer

    Combines all loaded files into a single DataFrame.

    If `column_per_predictor` is True then reshape the DataFrame to have
    multiple prediction columns, one per distinct predictor.

    If ic50_base is not None, then transform IC50 using ic50_base ** pred
    """

    # dictionary mapping from (allele, sequence) to dictionary of binding
    # predictions and the actual measuremnt called "meas"
    test_datasets = {}
    predictor_names = set([])

    for dirpath in dirpaths:
        for filename in listdir(dirpath):
            filepath = join(dirpath, filename)
            dot_parts = filename.split(".")
            dot_parts
            if len(dot_parts) != 5:
                print("Skipping %s" % filepath)
                continue
            prefixes = dot_parts[:-2]
            interesting_prefixes = {
                substring
                for substring in prefixes
                if substring not in {"pred", "test", dataset_name}
            }
            if len(interesting_prefixes) != 1:
                print("Can't infer predictor name for %s" % filepath)
                continue
            predictor_name = list(interesting_prefixes)[0]
            suffix, ext = dot_parts[-2:]
            dash_parts = suffix.split("-")
            if len(dash_parts) < 2:
                print("Skipping %s due to incorrect format" % filepath)
                continue
            predictor_names.add(predictor_name)
            print("Reading %s" % filepath)
            allele = normalize_allele_name("-".join(dash_parts[:-1]))
            length = int(dash_parts[-1])
            df = pd.read_csv(filepath, sep=sep, comment=comment_char)
            df["dirpath"] = dirpath
            df["predictor"] = predictor_name
            df["allele"] = allele
            df["length"] = length
            if ic50_base is not None:
                df["pred"] = ic50_base ** df["pred"]
                df["meas"] = ic50_base ** df["meas"]

            if allele not in test_datasets:
                test_datasets[allele] = defaultdict(OrderedDict)

            dataset_dict = test_datasets[allele]
            for _, row in df.iterrows():
                sequence = row["sequence"]
                dataset_dict[sequence]["length"] = length
                dataset_dict[sequence]["meas"] = row["meas"]
                dataset_dict[sequence][predictor_name] = row["pred"]
    test_dataframes = {
        allele: pd.DataFrame.from_dict(
            ic50_values, orient="index")
        for (allele, ic50_values) in test_datasets.items()
    }
    return test_dataframes, predictor_names
