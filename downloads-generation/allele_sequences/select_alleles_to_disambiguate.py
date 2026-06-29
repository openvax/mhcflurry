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
Select alleles to disambiguate

"""
from __future__ import print_function

import sys
import argparse

import pandas

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "train_data",
    help="Path to training data CSV. Must have column: allele")
parser.add_argument(
    "--min-count",
    type=int,
    metavar="N",
    help="Keep only alleles with at least N measurements")
parser.add_argument(
    "--out",
    help="Result file.")


def run():
    args = parser.parse_args(sys.argv[1:])
    print(args)

    df = pandas.read_csv(args.train_data)
    if args.min_count:
        allele_counts = df.allele.value_counts()
        df = df.loc[
            df.allele.map(allele_counts) > args.min_count
        ]

    df.drop_duplicates("allele").allele.to_csv(
        args.out, header=False, index=False)
    print("Wrote: ", args.out)


if __name__ == '__main__':
    run()
