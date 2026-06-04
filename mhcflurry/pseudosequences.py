"""Class I pseudosequence artifact names and lookup helpers.

This module is the canonical registry for pseudosequence CSV filenames used by
MHCflurry training and model serialization.

Canonical pseudosequence files:

* ``pseudosequences.netmhcpan.34aa.csv``: NetMHCpan-derived 34 amino acid
  pseudosequences.
* ``pseudosequences.mhcflurry.37aa.csv``: MHCflurry-generated 37 amino acid
  pseudosequences used by older public pan-allele model bundles.
* ``pseudosequences.mhcflurry.39aa.csv``: MHCflurry-generated 39 amino acid
  pseudosequences from the aligned full-sequence pipeline.

Compatibility aliases:

* ``allele_sequences.csv``: legacy runtime/model artifact filename.
* ``class1_pseudosequences.csv``: legacy NetMHCpan 34aa filename.
* ``allele_sequences.no_differentiation.csv``: legacy pan-variant experiment
  output; not a canonical pseudosequence artifact name.

Trained model directories should ship the exact pseudosequence CSV used during
training. The saved weights depend on both the representation width and the
position definition.

Expected locations:

* ``downloads-generation/*/pseudosequences.netmhcpan.34aa.csv``: checked-in
  generation seed tables.
* standalone ``allele_sequences`` download: should contain
  ``pseudosequences.mhcflurry.39aa.csv`` plus ``allele_sequences.csv`` as a
  compatibility copy.
* trained model directories: should contain ``allele_sequences.csv`` and, for
  newly-saved models, the matching canonical ``pseudosequences.*.*aa.csv``.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from glob import glob
from os.path import basename, exists, join

import pandas


LEGACY_ALLELE_SEQUENCES_FILENAME = "allele_sequences.csv"
LEGACY_CLASS1_PSEUDOSEQUENCES_FILENAME = "class1_pseudosequences.csv"
LEGACY_NO_DIFFERENTIATION_FILENAME = "allele_sequences.no_differentiation.csv"
PSEUDOSEQUENCE_GLOB = "pseudosequences.*.*aa.csv"


@dataclass(frozen=True)
class PseudosequenceDefinition:
    """A named pseudosequence CSV definition."""

    source: str
    length: int
    filename: str
    description: str


PSEUDOSEQUENCE_DEFINITIONS = (
    PseudosequenceDefinition(
        source="netmhcpan",
        length=34,
        filename="pseudosequences.netmhcpan.34aa.csv",
        description="NetMHCpan-derived 34 amino acid pseudosequences.",
    ),
    PseudosequenceDefinition(
        source="mhcflurry",
        length=37,
        filename="pseudosequences.mhcflurry.37aa.csv",
        description=(
            "MHCflurry-generated 37 amino acid pseudosequences used by older "
            "public pan-allele model bundles."
        ),
    ),
    PseudosequenceDefinition(
        source="mhcflurry",
        length=39,
        filename="pseudosequences.mhcflurry.39aa.csv",
        description=(
            "MHCflurry-generated 39 amino acid pseudosequences from the "
            "aligned full-sequence pipeline."
        ),
    ),
)

PSEUDOSEQUENCE_DEFINITIONS_BY_LENGTH = {
    definition.length: definition
    for definition in PSEUDOSEQUENCE_DEFINITIONS
}

PSEUDOSEQUENCE_FILENAMES_BY_LENGTH = {
    length: definition.filename
    for (length, definition) in PSEUDOSEQUENCE_DEFINITIONS_BY_LENGTH.items()
}

PSEUDOSEQUENCE_FILENAME_PREFERENCE = (
    LEGACY_ALLELE_SEQUENCES_FILENAME,
    PSEUDOSEQUENCE_FILENAMES_BY_LENGTH[39],
    PSEUDOSEQUENCE_FILENAMES_BY_LENGTH[37],
    PSEUDOSEQUENCE_FILENAMES_BY_LENGTH[34],
    LEGACY_CLASS1_PSEUDOSEQUENCES_FILENAME,
)

LEGACY_FILENAMES = {
    "allele_sequences": LEGACY_ALLELE_SEQUENCES_FILENAME,
    "class1_pseudosequences": LEGACY_CLASS1_PSEUDOSEQUENCES_FILENAME,
    "no_differentiation": LEGACY_NO_DIFFERENTIATION_FILENAME,
}


def pseudosequence_length(allele_to_sequence):
    """Return the common pseudosequence length, or ``None`` if ambiguous."""
    if not allele_to_sequence:
        return None
    lengths = set()
    for sequence in allele_to_sequence.values():
        if pandas.isnull(sequence):
            continue
        lengths.add(len(str(sequence)))
    if len(lengths) == 1:
        return lengths.pop()
    return None


def pseudosequence_filename_for_length(length):
    """Return the canonical pseudosequence filename for a representation length."""
    if length is None:
        return None
    return PSEUDOSEQUENCE_FILENAMES_BY_LENGTH.get(int(length))


def pseudosequence_filename_for_mapping(allele_to_sequence):
    """Return the canonical pseudosequence filename for a saved mapping."""
    return pseudosequence_filename_for_length(
        pseudosequence_length(allele_to_sequence))


def pseudosequence_filename_candidates(models_dir):
    """
    Pseudosequence files accepted when loading a saved predictor.

    ``allele_sequences.csv`` and ``class1_pseudosequences.csv`` are legacy
    artifact filenames. The ``pseudosequences.*.*aa.csv`` names make the source
    and representation width explicit for newly-generated artifacts.
    """
    result = []
    for filename in PSEUDOSEQUENCE_FILENAME_PREFERENCE:
        if exists(join(models_dir, filename)):
            result.append(filename)
    for path in sorted(glob(join(models_dir, PSEUDOSEQUENCE_GLOB))):
        filename = basename(path)
        if filename not in result:
            result.append(filename)
    return result


def pseudosequence_path(directory, length, fallback_legacy=True):
    """Return the preferred pseudosequence path in ``directory``.

    When the canonical ``pseudosequences.*.<length>aa.csv`` is absent and
    ``fallback_legacy`` is true, returns the legacy
    ``allele_sequences.csv`` path and logs a warning so callers don't
    silently read a different sequence set than requested.
    """
    filename = pseudosequence_filename_for_length(length)
    if filename is None:
        raise ValueError("No canonical pseudosequence filename for %saa" % length)
    path = join(directory, filename)
    if exists(path) or not fallback_legacy:
        return path
    legacy_path = join(directory, LEGACY_ALLELE_SEQUENCES_FILENAME)
    logging.warning(
        "Canonical pseudosequence file %s not found in %s; falling back to "
        "legacy %s. The legacy file may carry a different representation "
        "width than requested (%saa).",
        filename,
        directory,
        LEGACY_ALLELE_SEQUENCES_FILENAME,
        length,
    )
    return legacy_path


def _run_filename(args):
    filename = pseudosequence_filename_for_length(args.length)
    if filename is None:
        raise SystemExit("No canonical pseudosequence filename for %saa" % (
            args.length,))
    print(filename)


def _run_legacy(args):
    print(LEGACY_FILENAMES[args.name])


def _run_path(args):
    print(pseudosequence_path(
        args.directory,
        args.length,
        fallback_legacy=args.fallback_legacy))


def _run_list(_args):
    for definition in PSEUDOSEQUENCE_DEFINITIONS:
        print("%s\t%s\t%s" % (
            definition.length,
            definition.source,
            definition.filename,
        ))


def main(argv=None, prog=None):
    """Command-line access for shell generation scripts."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    filename_parser = subparsers.add_parser(
        "filename",
        help="Print the canonical pseudosequence filename for a length.")
    filename_parser.add_argument("--length", required=True, type=int)
    filename_parser.set_defaults(func=_run_filename)

    legacy_parser = subparsers.add_parser(
        "legacy",
        help="Print a legacy compatibility filename.")
    legacy_parser.add_argument("name", choices=sorted(LEGACY_FILENAMES))
    legacy_parser.set_defaults(func=_run_legacy)

    path_parser = subparsers.add_parser(
        "path",
        help="Print the preferred pseudosequence path in a directory.")
    path_parser.add_argument("--directory", required=True)
    path_parser.add_argument("--length", required=True, type=int)
    path_parser.add_argument(
        "--fallback-legacy",
        action="store_true",
        help="Use allele_sequences.csv if the canonical file is absent.")
    path_parser.set_defaults(func=_run_path)

    list_parser = subparsers.add_parser(
        "list",
        help="List canonical pseudosequence definitions.")
    list_parser.set_defaults(func=_run_list)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
