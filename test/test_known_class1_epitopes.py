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

from mhcflurry.class1_allele_specific import class1_single_model_multi_allele_predictor


def test_A1_Titin_epitope():
    # Test the A1 Titin epitope ESDPIVAQY from
    #   Identification of a Titin-Derived HLA-A1-Presented Peptide
    #   as a Cross-Reactive Target for Engineered MAGE A3-Directed
    #   T Cells
    model = class1_single_model_multi_allele_predictor.from_allele_name("HLA-A*01:01")
    ic50s = model.predict(["ESDPIVAQY"])
    print(ic50s)
    assert len(ic50s) == 1
    ic50 = ic50s[0]
    assert ic50 <= 500, ic50


def test_A1_MAGE_epitope():
    # Test the A1 MAGE epitope EVDPIGHLY from
    #   Identification of a Titin-Derived HLA-A1-Presented Peptide
    #   as a Cross-Reactive Target for Engineered MAGE A3-Directed
    #   T Cells
    model = class1_single_model_multi_allele_predictor.from_allele_name("HLA-A*01:01")
    ic50s = model.predict(["EVDPIGHLY"])
    print(ic50s)
    assert len(ic50s) == 1
    ic50 = ic50s[0]
    assert ic50 <= 500, ic50


def test_A2_HIV_epitope():
    # Test the A2 HIV epitope SLYNTVATL from
    #    The HIV-1 HLA-A2-SLYNTVATL Is a Help-Independent CTL Epitope
    model = class1_single_model_multi_allele_predictor.from_allele_name("HLA-A*02:01")
    ic50s = model.predict(["SLYNTVATL"])
    print(ic50s)
    assert len(ic50s) == 1
    ic50 = ic50s[0]
    assert ic50 <= 500, ic50


if __name__ == "__main__":
    test_A1_Titin_epitope()
    test_A1_MAGE_epitope()
    test_A2_HIV_epitope()
