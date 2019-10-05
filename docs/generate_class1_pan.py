"""
Generate certain RST files used in documentation.
"""
from __future__ import print_function
import sys
import argparse
from collections import OrderedDict, defaultdict
import os
from os.path import join, exists
from os import mkdir

import pandas
import logomaker

from matplotlib import pyplot

from mhcflurry.downloads import get_path
from mhcflurry.amino_acid import COMMON_AMINO_ACIDS

AMINO_ACIDS = sorted(COMMON_AMINO_ACIDS)

parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument(
    "--class1-models-dir-with-ms",
    metavar="DIR",
    default=get_path(
        "models_class1_pan", "models.with_mass_spec", test_exists=False),
    help="Class1 models. Default: %(default)s",
)
parser.add_argument(
    "--class1-models-dir-no-ms",
    metavar="DIR",
    default=get_path(
        "models_class1_pan", "models.no_mass_spec", test_exists=False),
    help="Class1 models. Default: %(default)s",
)
parser.add_argument(
    "--logo-cutoff",
    default=0.01,
    type=float,
    help="Fraction of top to use for motifs",
)
parser.add_argument(
    "--length-cutoff",
    default=0.01,
    type=float,
    help="Fraction of top to use for length distribution",
)
parser.add_argument(
    "--length-distribution-lengths",
    nargs="+",
    default=[8, 9, 10, 11, 12, 13, 14, 15],
    type=int,
    help="Peptide lengths for length distribution plots",
)
parser.add_argument(
    "--motif-lengths",
    nargs="+",
    default=[8, 9, 10, 11],
    type=int,
    help="Peptide lengths for motif plots",
)
parser.add_argument(
    "--out-dir",
    metavar="DIR",
    required=True,
    help="Directory to write RSTs and images to",
)
parser.add_argument(
    "--max-alleles",
    default=None,
    type=int,
    metavar="N",
    help="Only use N alleles (for testing)",
)


def model_info(models_dir):
    length_distributions_df = pandas.read_csv(
        join(models_dir, "length_distributions.csv.bz2"))
    frequency_matrices_df = pandas.read_csv(
        join(models_dir, "frequency_matrices.csv.bz2"))
    train_data_df = pandas.read_csv(
        join(models_dir, "train_data.csv.bz2"))

    distribution = frequency_matrices_df.loc[
        (frequency_matrices_df.cutoff_fraction == 1.0), AMINO_ACIDS
    ].mean(0)

    normalized_frequency_matrices = frequency_matrices_df.copy()
    normalized_frequency_matrices.loc[:, AMINO_ACIDS] = (
            normalized_frequency_matrices[AMINO_ACIDS] / distribution)

    observations_per_allele = (
        train_data_df.groupby("allele").peptide.nunique().to_dict())

    return {
        'length_distributions': length_distributions_df,
        'normalized_frequency_matrices': normalized_frequency_matrices,
        'observations_per_allele': observations_per_allele,
    }


def write_logo(
        normalized_frequency_matrices,
        allele,
        lengths,
        cutoff,
        models_label,
        out_dir):

    fig = pyplot.figure(figsize=(8,10))

    for (i, length) in enumerate(lengths):
        ax = pyplot.subplot(len(lengths), 1, i + 1)
        matrix = normalized_frequency_matrices.loc[
            (normalized_frequency_matrices.allele == allele) &
            (normalized_frequency_matrices.length == length) &
            (normalized_frequency_matrices.cutoff_fraction == cutoff)
        ].set_index("position")[AMINO_ACIDS]
        if matrix.shape[0] == 0:
            return None

        matrix = (matrix.T / matrix.sum(1)).T  # row normalize

        ss_logo = logomaker.Logo(
            matrix,
            width=.8,
            vpad=.05,
            fade_probabilities=True,
            stack_order='small_on_top',
            ax=ax,
        )
        pyplot.title(
            "%s %d-mer (%s)" % (allele, length, models_label), y=0.85)
        pyplot.xticks(matrix.index.values)
    pyplot.tight_layout()
    name = "%s.motifs.%s.png" % (
        allele.replace("*", "-").replace(":", "-"), models_label)
    filename = os.path.abspath(join(out_dir, name))
    pyplot.savefig(filename)
    print("Wrote: ", filename)
    fig.clear()
    pyplot.close(fig)
    return name


def write_length_distribution(
        length_distributions_df, allele, lengths, cutoff, models_label, out_dir):
    length_distribution = length_distributions_df.loc[
        (length_distributions_df.allele == allele) &
        (length_distributions_df.cutoff_fraction == cutoff)
    ]
    if length_distribution.shape[0] == 0:
        return None

    length_distribution = length_distribution.set_index(
        "length").reindex(lengths).fillna(0.0).reset_index()

    fig = pyplot.figure(figsize=(8, 2))
    length_distribution.plot(x="length", y="fraction", kind="bar", color="black")
    pyplot.title("%s (%s)" % (allele, models_label))
    pyplot.xlabel("")
    pyplot.xticks(rotation=0)
    pyplot.gca().get_legend().remove()
    name = "%s.lengths.%s.png" % (
        allele.replace("*", "-").replace(":", "-"), models_label)

    filename = os.path.abspath(join(out_dir, name))
    pyplot.savefig(filename)
    print("Wrote: ", filename)
    fig.clear()
    pyplot.close(fig)
    return name


def go(argv):
    args = parser.parse_args(argv)

    if not exists(args.out_dir):
        mkdir(args.out_dir)

    predictors = [
        ("with_mass_spec", args.class1_models_dir_with_ms),
        ("no_mass_spec", args.class1_models_dir_no_ms),
    ]
    info_per_predictor = OrderedDict()
    alleles = set()
    for (label, models_dir) in predictors:
        if not models_dir:
            continue
        info_per_predictor[label] = model_info(models_dir)
        alleles.update(
            info_per_predictor[label]["normalized_frequency_matrices"].allele.unique())

    lines = []

    def w(*pieces):
        lines.extend(pieces)

    w('Motifs and length distributions from the pan-allele predictor')
    w('=' * 80, "")

    w(
        "Length distributions and binding motifs were calculated by ranking a "
        "large set of random peptides (an equal number of peptides for each "
        "length 8-15) by predicted affinity for each allele. "
        "For length distribution, the top %g%% of peptides were collected and "
        "their length distributions plotted. For sequence motifs, sequence "
        "logos for the top %g%% "
        "peptides for each length are shown.\n" % (
            args.length_cutoff * 100.0,
            args.logo_cutoff * 100.0,
        ))

    w(".. contents:: :local:", "")


    def image(name):
        if name is None:
            return ""
        return '.. image:: %s\n' % name

    alleles = sorted(alleles, key=lambda a: ("HLA" not in a, a))
    if args.max_alleles:
        alleles = alleles[:args.max_alleles]

    for allele in alleles:
        w(allele, "-" * 80, "")
        for (label, info) in info_per_predictor.items():
            length_distribution = info["length_distributions"]
            normalized_frequency_matrices = info["normalized_frequency_matrices"]

            length_distribution_image_path = write_length_distribution(
                length_distributions_df=length_distribution,
                allele=allele,
                lengths=args.length_distribution_lengths,
                cutoff=args.length_cutoff,
                out_dir=args.out_dir,
                models_label=label)
            if not length_distribution_image_path:
                continue

            w(
                "*" + (
                    "With mass-spec" if label == "with_mass_spec" else "Affinities only")
                + "*\n")
            w("Training observations (unique peptides): %d" % (
                info['observations_per_allele'].get(allele, 0)))
            w("\n")
            w(image(length_distribution_image_path))
            w(image(write_logo(
                normalized_frequency_matrices=normalized_frequency_matrices,
                allele=allele,
                lengths=args.motif_lengths,
                cutoff=args.logo_cutoff,
                out_dir=args.out_dir,
                models_label=label,
            )))
        w("")

    document_path = join(args.out_dir, "allele_motifs.rst")
    with open(document_path, "w") as fd:
        for line in lines:
            fd.write(line)
            fd.write("\n")
    print("Wrote", document_path)


if __name__ == "__main__":
    go(sys.argv[1:])
