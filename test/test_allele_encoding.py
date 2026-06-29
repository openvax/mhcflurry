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

import time

from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.amino_acid import BLOSUM62_MATRIX
from numpy.testing import assert_equal


def test_allele_encoding_speed():
    encoding = AlleleEncoding(
        ["A*02:01", "A*02:03", "A*02:01"],
        {
            "A*02:01": "AC",
            "A*02:03": "AE",
        }
    )
    start = time.time()
    encoding1 = encoding.fixed_length_vector_encoded_sequences("BLOSUM62")
    assert_equal(
        [
            [BLOSUM62_MATRIX["A"], BLOSUM62_MATRIX["C"]],
            [BLOSUM62_MATRIX["A"], BLOSUM62_MATRIX["E"]],
            [BLOSUM62_MATRIX["A"], BLOSUM62_MATRIX["C"]],
        ], encoding1)
    print("Simple encoding in %0.2f sec." % (time.time() - start))
    print(encoding1)

    encoding = AlleleEncoding(
        ["A*02:01", "A*02:03", "A*02:01"] * int(1e5),
        {
            "A*02:01": "AC" * 16,
            "A*02:03": "AE" * 16,
        }
    )
    start = time.time()
    encoding1 = encoding.fixed_length_vector_encoded_sequences("BLOSUM62")
    print("Long encoding in %0.2f sec." % (time.time() - start))
