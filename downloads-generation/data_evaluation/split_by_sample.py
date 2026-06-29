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
Split a big csv by a particular column (sample id)
"""
import sys
import argparse
import re

import pandas


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "data",
    metavar="CSV")
parser.add_argument(
    "--out",
    help="Out pattern (%s will be replaced by sample)",
    metavar="CSV")
parser.add_argument(
    "--out-samples",
    help="Out sample list",
    metavar="CSV")
parser.add_argument(
    "--split-column",
    help="Column to split by",
    default="sample_id")


def run():
    args = parser.parse_args(sys.argv[1:])
    df = pandas.read_csv(args.data)
    print("Read data with shape", df.shape)

    names = []
    for (i, (sample, sub_df)) in enumerate(df.groupby(args.split_column)):
        name = re.sub(r'[^\w\d-]', '', sample) + (".%d" % i)
        dest = args.out % name
        sub_df.to_csv(dest, index=False)
        print("Wrote [%d rows]" % len(sub_df), dest)
        names.append(name)

    if args.out_samples:
        pandas.Series(names).to_csv(args.out_samples, index=False, header=False)
        print("Wrote", args.out_samples)

if __name__ == '__main__':
    run()
