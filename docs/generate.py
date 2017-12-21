"""
Generate certain RST files used in documentation.
"""

import sys
import argparse
from textwrap import wrap

import pypandoc
import pandas
from keras.utils.vis_utils import plot_model

from mhcflurry import __version__
from mhcflurry.downloads import get_path
from mhcflurry.class1_affinity_predictor import Class1AffinityPredictor

parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument(
    "--cv-summary-csv",
    metavar="FILE.csv",
    default=get_path(
        "cross_validation_class1", "summary.all.csv", test_exists=False),
    help="Cross validation scores summary. Default: %(default)s",
)
parser.add_argument(
    "--class1-models-dir",
    metavar="DIR",
    default=get_path(
        "models_class1", "models", test_exists=False),
    help="Class1 models. Default: %(default)s",
)
parser.add_argument(
    "--out-models-cv-rst",
    metavar="FILE.rst",
    help="rst output file",
)
parser.add_argument(
    "--out-models-info-rst",
    metavar="FILE.rst",
    help="rst output file",
)
parser.add_argument(
    "--out-models-architecture-png",
    metavar="FILE.png",
    help="png output file",
)
parser.add_argument(
    "--out-models-supported-alleles-rst",
    metavar="FILE.png",
    help="png output file",
)


def go(argv):
    args = parser.parse_args(argv)

    predictor = None

    if args.out_models_supported_alleles_rst:
        # Supported alleles rst
        if predictor is None:
            predictor = Class1AffinityPredictor.load(args.class1_models_dir)
        with open(args.out_models_supported_alleles_rst, "w") as fd:
            fd.write(
                "Models released with the current version of MHCflurry (%s) "
                "support peptides of "
                "length %d-%d and the following %d alleles:\n\n::\n\n\t%s\n\n" % (
                    __version__,
                    predictor.supported_peptide_lengths[0],
                    predictor.supported_peptide_lengths[1],
                    len(predictor.supported_alleles),
                    "\n\t".join(
                        wrap(", ".join(predictor.supported_alleles)))))
            print("Wrote: %s" % args.out_models_supported_alleles_rst)

    if args.out_models_architecture_png:
        # Architecture diagram
        if predictor is None:
            predictor = Class1AffinityPredictor.load(args.class1_models_dir)
        network = predictor.neural_networks[0].network()
        plot_model(
            network,
            to_file=args.out_models_architecture_png,
            show_layer_names=True,
            show_shapes=True)
        print("Wrote: %s" % args.out_models_architecture_png)

    if args.out_models_info_rst:
        # Architecture information rst
        if predictor is None:
            predictor = Class1AffinityPredictor.load(args.class1_models_dir)
        network = predictor.neural_networks[0].network()
        lines = []
        network.summary(print_fn=lines.append)

        with open(args.out_models_info_rst, "w") as fd:
            fd.write("Layers and parameters summary: ")
            fd.write("\n\n::\n\n")
            for line in lines:
                fd.write("    ")
                fd.write(line)
                fd.write("\n")
            print("Wrote: %s" % args.out_models_info_rst)

    if args.out_models_cv_rst:
        # Models cv output
        df = pandas.read_csv(args.cv_summary_csv)
        sub_df = df.loc[
            df.kind == "ensemble"
            ].sort_values("allele").dropna().copy().reset_index(drop=True)
        sub_df["Allele"] = sub_df.allele
        sub_df["CV Training Size"] = sub_df.train_size.astype(int)
        sub_df["AUC"] = sub_df.auc
        sub_df["F1"] = sub_df.f1
        sub_df["Kendall Tau"] = sub_df.tau
        sub_df = sub_df[sub_df.columns[-5:]]
        html = sub_df.to_html(
            index=False,
            float_format=lambda v: "%0.3f" % v,
            justify="left")
        rst = pypandoc.convert_text(html, format="html", to="rst")

        with open(args.out_models_cv_rst, "w") as fd:
            fd.write(
                "Showing estimated performance for %d alleles." % len(sub_df))
            fd.write("\n\n")
            fd.write(rst)
            print("Wrote: %s" % args.out_models_cv_rst)

if __name__ == "__main__":
    go(sys.argv[1:])