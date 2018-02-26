"""
Generate certain RST files used in documentation.
"""

import sys
import argparse
import json
from textwrap import wrap
from collections import OrderedDict, defaultdict
from os.path import join

import pypandoc
import pandas
from keras.utils.vis_utils import plot_model
from tabulate import tabulate

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
    "--class1-unselected-models-dir",
    metavar="DIR",
    default=get_path(
        "models_class1_unselected", "models", test_exists=False),
    help="Class1 unselected models. Default: %(default)s",
)
parser.add_argument(
    "--out-alleles-info-rst",
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
    unselected_predictor = None

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
        raise NotImplementedError()  # for now
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
        if unselected_predictor is None:
            unselected_predictor = Class1AffinityPredictor.load(
                args.class1_unselected_models_dir)

        config_to_network = {}
        config_to_alleles = {}
        for (allele, networks) in unselected_predictor.allele_to_allele_specific_models.items():
            for network in networks:
                config = json.dumps(network.hyperparameters)
                if config not in config_to_network:
                    config_to_network[config] = network
                config_to_alleles[config] = []

        for (allele, networks) in predictor.allele_to_allele_specific_models.items():
            for network in networks:
                config = json.dumps(network.hyperparameters)
                assert config in config_to_network
                config_to_alleles[config].append(allele)

        all_hyperparameters = [
            network.hyperparameters for network in config_to_network.values()
        ]
        hyperparameter_keys =  all_hyperparameters[0].keys()
        assert all(
            hyperparameters.keys() == hyperparameter_keys
            for hyperparameters in all_hyperparameters)

        constant_hyperparameter_keys = [
            k for k in hyperparameter_keys
            if all([
                hyperparameters[k] == all_hyperparameters[0][k]
                for hyperparameters in all_hyperparameters
            ])
        ]
        constant_hypeparameters = dict(
            (key, all_hyperparameters[0][key])
            for key in sorted(constant_hyperparameter_keys)
        )

        def write_hyperparameters(fd, hyperparameters):
            rows = []
            for key in sorted(hyperparameters.keys()):
                rows.append((key, json.dumps(hyperparameters[key])))
            fd.write("\n")
            fd.write(
                tabulate(rows, ["Hyperparameter", "Value"], tablefmt="grid"))

        with open(args.out_models_info_rst, "w") as fd:
            fd.write("Hyperparameters shared by all %d architectures:\n" %
                len(config_to_network))
            write_hyperparameters(fd, constant_hypeparameters)
            fd.write("\n")

            configs = sorted(
                config_to_alleles,
                key=lambda s: len(config_to_alleles[s]),
                reverse=True)

            for (i, config) in enumerate(configs):
                network = config_to_network[config]
                lines = []
                network.network().summary(print_fn=lines.append)

                specific_hyperparameters = dict(
                    (key, value)
                    for (key, value) in network.hyperparameters.items()
                    if key not in constant_hypeparameters)

                def name_component(key, value):
                    if key == "locally_connected_layers":
                        return "lc=%d" % len(value)
                    elif key == "train_data":
                        return value["subset"] + "-data"
                    elif key == "layer_sizes":
                        (value,) = value
                        key = "size"
                    elif key == "dense_layer_l1_regularization":
                        if value == 0:
                            return "no-reg"
                        key = "reg"
                    return "%s=%s" % (key, value)

                def sort_key(component):
                    if "lc" in component:
                        return (1, component)
                    if "reg" in component:
                        return (2, component)
                    return (0, component)

                components = [
                    name_component(key, value)
                    for (key, value) in specific_hyperparameters.items()
                ]
                name = ",".join(sorted(components, key=sort_key))

                fd.write("Architecture %d / %d %s\n" % (
                    (i + 1, len(config_to_network), name)))
                fd.write("+" * 40)
                fd.write("\n")
                fd.write(
                    "Selected in the ensembles for %d alleles: *%s*.\n\n" % (
                        len(config_to_alleles[config]),
                        ", ".join(
                            sorted(config_to_alleles[config]))))
                write_hyperparameters(
                    fd,
                    specific_hyperparameters)
                fd.write("\n\n::\n\n")
                for line in lines:
                    fd.write("    ")
                    fd.write(line)
                    fd.write("\n")
        print("Wrote: %s" % args.out_models_info_rst)

    if args.out_alleles_info_rst:
        # Models cv output
        df = pandas.read_csv(
            join(args.class1_models_dir, "unselected_summary.csv.bz2"))

        train_df = pandas.read_csv(
            join(args.class1_unselected_models_dir, "train_data.csv.bz2"))

        quantitative_train_measurements_by_allele = train_df.loc[
            train_df.measurement_type == "quantitative"
        ].allele.value_counts()

        train_measurements_by_allele = train_df.allele.value_counts()

        df = df.sort_values("allele").copy()

        df["scoring"] = df.unselected_score_plan.str.replace(
            "\\(\\|[0-9.]+\\|\\)", "")
        df["models selected"] = df["num_models"]

        df["sanitized_scoring"] = df.scoring.map(
            lambda s: s.replace("mass-spec", "").replace("mse", "").replace("(", "").replace(")", "").strip()
        )

        df["mass spec scoring"] = df.sanitized_scoring.map(
            lambda s: s.split(",")[0] if "," in s else ""
        )
        df["mean square error scoring"] = df.sanitized_scoring.map(
            lambda s: s.split(",")[-1]
        )
        df["unselected percentile"] = df.unselected_accuracy_score_percentile

        df["train data (all)"] = df.allele.map(train_measurements_by_allele)
        df["train data (quantitative)"] = df.allele.map(
            quantitative_train_measurements_by_allele)

        def write_df(df, fd):
            rows = [
                row for (_, row) in df.iterrows()
            ]
            fd.write("\n")
            fd.write(
                tabulate(rows,
                    [col.replace("_", " ") for col in df.columns],
                    tablefmt="grid"))
            fd.write("\n\n")

        with open(args.out_alleles_info_rst, "w") as fd:
            fd.write("Supported alleles\n")
            fd.write("+" * 80)
            fd.write("\n\n")

            common_cols = [
                "allele",
                "train data (all)",
                "train data (quantitative)",
                "mass spec scoring",
                "mean square error scoring",
                "unselected percentile",
                "unselected_score",
                "unselected_score_scrambled_mean",
            ]

            sub_df = df.loc[df.retained][common_cols + [
                "models selected",
            ]]
            write_df(sub_df, fd)

            fd.write("Rejected alleles\n")
            fd.write("+" * 80)
            fd.write("\n\n")
            fd.write(
                "Training for the following alleles was attempted but the "
                "resulting models were excluded due to inadequate performance on "
                "held out data.")
            fd.write("\n\n")

            sub_df = df.loc[~df.retained][common_cols].sort_values("allele")
            write_df(sub_df, fd)
            print("Wrote: %s" % args.out_alleles_info_rst)

if __name__ == "__main__":
    go(sys.argv[1:])