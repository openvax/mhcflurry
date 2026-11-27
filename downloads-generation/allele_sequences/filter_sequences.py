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

"""
Filter and combine class I sequence fastas.
"""

import sys
import argparse

from mhcflurry.common import normalize_allele_name
from mhcflurry.fasta import FastaParser

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "fastas",
    nargs="+",
    help="Unaligned fastas")

parser.add_argument(
    "--out",
    required=True,
    help="Fasta output")


def _format_fasta_header(record_id, description):
    """Match Biopython's FASTA header rule when description is prefixed."""
    if description.startswith(record_id):
        return description
    return "%s %s" % (record_id, description)


def _write_fasta(records, fd, width=60):
    """Write ``(header, sequence)`` tuples in wrapped FASTA format."""
    for header, sequence in records:
        fd.write(">%s\n" % header)
        for offset in range(0, len(sequence), width):
            fd.write(sequence[offset:offset + width] + "\n")


def run(argv=None):
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    print(args)

    total = 0
    order = []
    name_to_record = {}
    reader = FastaParser()
    for fasta in args.fastas:
        for description, sequence in reader.iterate_over_file(
                fasta, full_descriptions=True):
            total += 1
            if len(sequence) < 50:
                print("-- Skipping '%s', sequence too short" % (
                    description,))
                continue

            parts = description.split()
            record_id = parts[0]
            candidate_strings = [description]
            if len(parts) > 1:
                candidate_strings.extend([
                    parts[1],
                    " ".join(parts[1:]),
                ])
            if description.startswith("sp|") and len(parts) > 6:
                # From a description like:
                # sp|P01897|HA1L_MOUSE H-2 class I histocompatibility antigen, L-D alpha chain OS=Mus musculus OX=10090 GN=H2-L PE=1 SV=2
                # make a string to parse like:
                # H-2-L-D
                candidate_strings.insert(0, parts[1] + "-" + parts[6])
            name = None
            for candidate_string in candidate_strings:
                name = normalize_allele_name(
                    candidate_string, raise_on_error=False)
                if name is not None:
                    break
            if name is None:
                print("Skipping '%s'" % (description,))
                continue

            if '*' not in name:
                raise ValueError("Normalization gave name without a '*' for '%s': %s" % (
                    description, name))

            print("Parsed '%s' as %s" % (description, name))
            header = _format_fasta_header(record_id, name + " " + description)

            if name in name_to_record:
                _, old_sequence = name_to_record[name]
                if len(old_sequence) < len(sequence):
                    print("-- Replacing old record (%d aa) with new (%d aa)" % (
                        len(old_sequence),
                        len(sequence)))
                    name_to_record[name] = (header, sequence)
                else:
                    print("-- Skipping, already seen")
            else:
                order.append(name)
                name_to_record[name] = (header, sequence)

    records = [name_to_record[name] for name in order]

    with open(args.out, "w", encoding="utf-8") as fd:
        _write_fasta(records, fd)

    print("Wrote %d / %d sequences: %s" % (len(records), total, args.out))


if __name__ == '__main__':
    run()
