"""
Adapted from pyensembl, github.com/openvax/pyensembl
Original implementation by Alex Rubinsteyn.

The worse sin in bioinformatics is to write your own FASTA parser.
We're doing this to avoid adding another dependency to MHCflurry, however.
"""

from __future__ import print_function, division, absolute_import

from gzip import GzipFile
import logging

from six import binary_type, PY3

import pandas


def read_fasta_to_dataframe(filename):
    reader = FastaParser()
    rows = reader.iterate_over_file(filename)
    return pandas.DataFrame(
        rows,
        columns=["sequence_id", "sequence"])

class FastaParser(object):
    """
    FastaParser object consumes lines of a FASTA file incrementally.
    """
    def __init__(self):
        self.current_id = None
        self.current_lines = []

    def iterate_over_file(self, fasta_path):
        """
        Generator that yields identifiers paired with sequences.
        """
        with self.open_file(fasta_path) as f:
            for line in f:
                line = line.rstrip()

                if len(line) == 0:
                    continue

                # have to slice into a bytes object or else get a single integer
                first_char = line[0:1]

                if first_char == b">":
                    previous_entry = self._current_entry()
                    self.current_id = self._parse_header_id(line)

                    if len(self.current_id) == 0:
                        logging.warning(
                            "Unable to parse ID from header line: %s", line)

                    self.current_lines = []

                    if previous_entry is not None:
                        yield previous_entry

                elif first_char == b";":
                    # semicolon are comment characters
                    continue
                else:
                    self.current_lines.append(line)

        # the last sequence is still in the lines buffer after we're done with
        # the file so make sure to yield it
        id_and_seq = self._current_entry()
        if id_and_seq is not None:
            yield id_and_seq

    def _current_entry(self):
        # when we hit a new entry, if this isn't the first
        # entry of the file then put the last one in the dictionary
        if self.current_id:
            if len(self.current_lines) == 0:
                logging.warning("No sequence data for '%s'", self.current_id)
            else:
                sequence = b"".join(self.current_lines)
                if PY3:
                    # only decoding into an ASCII str for Python 3 since
                    # the binary sequence type for Python 2 is already 'str'
                    # and the unicode representation is inefficient
                    # (using either 16 or 32 bits per character depends on build)
                    sequence = sequence.decode("ascii")
                return self.current_id, sequence

    @staticmethod
    def open_file(fasta_path):
        """
        Open either a text file or compressed gzip file as a stream of bytes.
        """
        if fasta_path.endswith("gz") or fasta_path.endswith("gzip"):
            return GzipFile(fasta_path, 'rb')
        else:
            return open(fasta_path, 'rb')

    @staticmethod
    def _parse_header_id(line):
        """
        Pull the transcript or protein identifier from the header line
        which starts with '>'
        """
        if type(line) is not binary_type:
            raise TypeError("Expected header line to be of type %s but got %s" % (
                binary_type, type(line)))

        if len(line) <= 1:
            raise ValueError("No identifier on FASTA line")

        # split line at first space to get the unique identifier for
        # this sequence
        space_index = line.find(b" ")
        if space_index >= 0:
            identifier = line[1:space_index]
        else:
            identifier = line[1:]

        return identifier.decode("ascii")