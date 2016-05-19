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
)
from collections import OrderedDict

import numpy as np


class ScoreSet(object):
    """
    Useful for keeping a collection of score dictionaries
    which map name->score type->list of values.
    """
    def __init__(self, verbose=True, index="name"):
        self.groups = {}
        self.verbose = verbose
        if isinstance(index, (list, tuple)):
            index = ",".join("%s" % item for item in index)
        self.index = index

    def add_many(self, group, **kwargs):
        for (k, v) in sorted(kwargs.items()):
            self.add(group, k, v)

    def add(self, group, score_type, value):
        if isinstance(group, (list, tuple)):
            group = ",".join("%s" % item for item in group)
        if group not in self.groups:
            self.groups[group] = {}
        if score_type not in self.groups[group]:
            self.groups[group][score_type] = []
        self.groups[group][score_type].append(value)
        if self.verbose:
            print("--> %s:%s %0.4f" % (group, score_type, value))

    def score_types(self):
        result = set([])
        for (g, d) in sorted(self.groups.items()):
            for score_type in sorted(d.keys()):
                result.add(score_type)
        return list(sorted(result))

    def _reduce_scores(self, reduce_fn):
        score_types = self.score_types()
        return {
            group:
                OrderedDict([
                    (score_type, reduce_fn(score_dict[score_type]))
                    for score_type
                    in score_types
                ])
            for (group, score_dict)
            in self.groups.items()
        }

    def averages(self):
        return self._reduce_scores(np.mean)

    def stds(self):
        return self._reduce_scores(np.std)

    def to_csv(self, filename):
        with open(filename, "w") as f:
            header_list = [self.index]
            score_types = self.score_types()
            for score_type in score_types:
                header_list.append(score_type)
                header_list.append(score_type + "_std")

            header_line = ",".join(header_list) + "\n"
            if self.verbose:
                print(header_line)
            f.write(header_line)

            score_averages = self.averages()
            score_stds = self.stds()

            for name in sorted(score_averages.keys()):
                line_elements = [name]
                for score_type in score_types:
                    line_elements.append(
                        "%0.4f" % score_averages[name][score_type])
                    line_elements.append(
                        "%0.4f" % score_stds[name][score_type])
                line = ",".join(line_elements) + "\n"
                if self.verbose:
                    print(line)
                f.write(line)
