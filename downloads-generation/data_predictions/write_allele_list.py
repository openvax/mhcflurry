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
"""
import sys
import argparse
import os

import pandas
import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "input",
    metavar="FILE.csv",
    help="CSV of annotated mass spec hits")
parser.add_argument(
    "--out",
    metavar="OUT.txt",
    help="Out file path")


def run():
    args = parser.parse_args(sys.argv[1:])

    df = pandas.read_csv(args.input)
    print("Read peptides", df.shape, *df.columns.tolist())

    df = df.loc[df.mhc_class == "I"]

    hla_sets = df.hla.unique()
    all_hla = set()
    for hla_set in hla_sets:
        all_hla.update(hla_set.split())

    all_hla = pandas.Series(sorted(all_hla))
    all_hla.to_csv(args.out, index=False, header=False)
    print("Wrote [%d alleles]: %s" % (len(all_hla), os.path.abspath(args.out)))


if __name__ == '__main__':
    run()
