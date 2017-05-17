# Copyright (c) 2016. Mount Sinai School of Medicine
#
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

import cProfile

import mhcflurry
import mhcflurry.class1_affinity_prediction
import mhcflurry.class1_allele_specific_ensemble


predictors = [
    mhcflurry.class1_affinity_prediction.get_downloaded_predictor(),
    mhcflurry.class1_allele_specific_ensemble.get_downloaded_predictor(),
]


def predict_and_check(allele, peptide, expected_range=(0, 500)):
    for predictor in predictors:
        (prediction,) = predictor.predict_for_allele(allele, [peptide])
        assert prediction >= expected_range[0], (predictor, prediction)
        assert prediction <= expected_range[1], (predictor, prediction)


def test_A1_Titin_epitope():
    # Test the A1 Titin epitope ESDPIVAQY from
    #   Identification of a Titin-Derived HLA-A1-Presented Peptide
    #   as a Cross-Reactive Target for Engineered MAGE A3-Directed
    #   T Cells
    predict_and_check("HLA-A*01:01", "ESDPIVAQY")


def test_A1_MAGE_epitope():
    # Test the A1 MAGE epitope EVDPIGHLY from
    #   Identification of a Titin-Derived HLA-A1-Presented Peptide
    #   as a Cross-Reactive Target for Engineered MAGE A3-Directed
    #   T Cells
    predict_and_check("HLA-A*01:01", "EVDPIGHLY")


def test_A2_HIV_epitope():
    # Test the A2 HIV epitope SLYNTVATL from
    #    The HIV-1 HLA-A2-SLYNTVATL Is a Help-Independent CTL Epitope
    predict_and_check("HLA-A*02:01", "SLYNTVATL")


if __name__ == "__main__":
    test_A1_Titin_epitope()
    test_A1_MAGE_epitope()
    test_A2_HIV_epitope()
