#!/usr/bin/env python

# Copyright (c) 2015. Mount Sinai School of Medicine
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

from __future__ import (
    print_function,
    division,
    absolute_import,
)
import argparse

from bottle import post, request, run, get

from mhcflurry.common import (
    split_uppercase_sequences,
    split_allele_names,
)
from mhcflurry.class1 import predict, supported_alleles


parser = argparse.ArgumentParser()

parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", default=80, type=int)
parser.add_argument("--debug", default=False, action="store_true")


@post('/')
def get_binding_value():
    peptides_string = request.forms.get('peptide')
    if peptides_string is None:
        return "ERROR: no peptide given"
    peptides_list = split_uppercase_sequences(peptides_string)
    alleles_string = request.forms.get('allele')
    if alleles_string is None:
        return "ERROR: no allele given"
    alleles_list = split_allele_names(alleles_string)
    try:
        result_df = predict(alleles=alleles_list, peptides=peptides_list)
    except ValueError as e:
        return "ERROR: %s" % e.args[0]
    return result_df.to_csv(sep="\t", index=False, float_format="%0.4f")


@get('/alleles')
def get_supported_alleles():
    peptide_lengths = "8,9,10,11,12"
    strings = [
        "%s\t%s" % (allele, peptide_lengths)
        for allele in supported_alleles()
    ]
    return "\n".join(strings)


if __name__ == "__main__":
    args = parser.parse_args()
    run(host=args.host, port=args.port, debug=args.debug, server="cherrypy")
