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

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals
)
from collections import OrderedDict

import numpy as np
import pandas as pd


class ScoreCollection(object):
    """
    Collect model scores across multiple alleles and hyperparameters
    """
    def __init__(
            self,
            score_names=["auc", "accuracy", "f1"],
            statistics=[np.mean, np.median, np.std, np.min, np.max]):
        self.score_names = score_names
        self.statistics = statistics
        self.result_dict = OrderedDict([
            ("allele_name", []),
            ("dataset_size", []),
        ])
        for score_name in self.score_names:
            for statistic in self.statistics:
                key = "%s_%s" % (score_name, statistic.__name__)
                self.result_dict[key] = []

    def add(self, allele_name, **scores):
        self.result_dict["allele_name"].append(allele_name)
        assert len(scores) > 0
        assert set(scores.keys()) == set(self.score_names), \
            "Expected scores to be %s but got %s" % (
                self.score_names, list(scores.keys()))
        lengths = [len(score_values) for score_values in scores.values()]
        assert all(length == lengths[0] for length in lengths), \
            "Length mismatch between scoring functions"
        self.result_dict["dataset_size"].append(lengths[0])
        for (name, values) in scores.items():
            for statistic in self.statistics:
                key = "%s_%s" % (name, statistic.__name__)
                self.result_dict[key].append(statistic(values))

    def dataframe(self):
        return pd.DataFrame(self.result_dict)
