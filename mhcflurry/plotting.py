# Copyright (c) 2015-2016. Mount Sinai School of Medicine
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

import seaborn
from matplotlib import pyplot
from matplotlib import rc
import numpy as np

def plot_nsamples_vs_metrics_with_imputation(
        results_df,
        base_filename,
        metrics=["auc", "f1", "tau"],
        titles={
            "tau": "Kendall's $\\tau$",
            "auc": "AUC",
            "f1": "$F_1$ score",
        },
        figsize=(6.5, 3.5),
        groupby_columns=["num_samples", "impute"],
        dpi=None):

    pyplot.figure(figsize=figsize)
    seaborn.set_style("whitegrid")

    for (j, score_name) in enumerate(metrics):
        pyplot.subplot2grid((1, 4), (0, j))
        groups = results_df.groupby(groupby_columns)
        groups_score = groups[score_name].mean().to_frame().reset_index()
        groups_score["std_error"] = \
            groups[score_name].std().to_frame().reset_index()[score_name]

        for impute in [True, False]:
            sub = groups_score[groups_score.impute == impute]
            color = seaborn.get_color_cycle()[0] if impute else seaborn.get_color_cycle()[1]
            pyplot.errorbar(
                x=sub.num_samples.values,
                y=sub[score_name].values,
                yerr=sub.std_error.values,
                label=("with" if impute else "without") + " imputation",
                color=color)
        if j == 1:
            pyplot.xlabel("Training set size")
        pyplot.xscale("log")
        pyplot.title(titles[score_name])

        if score_name == "auc":
            pyplot.ylim(ymin=0.5, ymax=1.0)
        if score_name == "f1":
            pyplot.ylim(ymin=0, ymax=1)
        if score_name == "tau":
            pyplot.ylim(ymin=0, ymax=0.6)
            pyplot.yticks(np.arange(0, 0.61, 0.15))

    pyplot.legend(
        bbox_to_anchor=(1.1, 1),
        loc=2,
        borderaxespad=0.,
        fancybox=True,
        frameon=True,
        fontsize="small")

    pyplot.tight_layout()

    if dpi:
        rc("savefig", dpi=dpi)

    # Put the legend out of the figure
    image_filename = base_filename + ".png"
    print("Writing PNG to %s" % image_filename)
    pyplot.savefig(image_filename)

    pdf_filename = base_filename + ".pdf"
    print("Writing PDF to %s" % pdf_filename)
    pyplot.savefig(pdf_filename)
