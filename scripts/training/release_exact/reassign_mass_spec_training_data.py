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
Reassign affinity values for mass spec data.
"""
from mhcflurry.cli.reassign_mass_spec_training_data import (
    make_parser,
    reassign_mass_spec_training_data,
)


parser = make_parser()


def go(args):
    return reassign_mass_spec_training_data(
        args.data,
        ms_only=args.ms_only,
        drop_negative_ms=args.drop_negative_ms,
        set_measurement_value=args.set_measurement_value,
        out_csv=args.out_csv,
        verbose=args.verbose)


if __name__ == "__main__":
    go(parser.parse_args())
