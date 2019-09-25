from mhcflurry import amino_acid
from nose.tools import eq_
from numpy.testing import assert_equal
import numpy
import pandas
import math

from mhcflurry import Class1NeuralNetwork
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.common import random_peptides


def test_random_negative_peptides_by_allele():
    network = Class1NeuralNetwork(
        random_negative_method="by_allele",
        random_negative_binder_threshold=500,
        random_negative_rate=1.0,
        random_negative_constant=2)

    allele_to_sequence = {
        "HLA-A*02:01": "AAAAA",
        "HLA-B*44:02": "CCCCC",
        "HLA-C*07:02": "EEEEE",
    }
    data_rows = [
        ("HLA-A*02:01", "SIINFEKL", 400, "="),
        ("HLA-A*02:01", "SIINFEKLL", 300, "="),
        ("HLA-A*02:01", "SIINFEKLL", 300, "="),
        ("HLA-A*02:01", "SIINFEKLQ", 1000, "="),
        ("HLA-A*02:01", "SIINFEKLZZ", 12000, ">"),
    ]
    for peptide in random_peptides(1000, length=9):
        data_rows.append(("HLA-B*44:02", peptide, 100, "="))
    for peptide in random_peptides(1000, length=9):
        data_rows.append(("HLA-B*44:02", peptide, 1000, "="))
    for peptide in random_peptides(5, length=10):
        data_rows.append(("HLA-B*44:02", peptide, 100, "="))

    data = pandas.DataFrame(
        data_rows,
        columns=["allele", "peptide", "affinity", "inequality"])
    data["length"] = data.peptide.str.len()

    peptides = EncodableSequences.create(data.peptide.values)
    allele_encoding = AlleleEncoding(data.allele.values, allele_to_sequence)

    results = network.random_negatives_generator(
        peptides,
        data.affinity.values,
        allele_encoding,
        data.inequality.values
    )
    result_allele_encoding = next(results)
    first_peptide_sample = next(results)
    second_peptide_sample = next(results)

    result_df1 = pandas.DataFrame({
        "allele": result_allele_encoding.alleles,
        "peptide": first_peptide_sample.sequences,
    })
    result_df1["length"] = result_df1.peptide.str.len()
    random_negatives = result_df1.groupby(["allele", "length"]).peptide.count().unstack()
    real_data = data.groupby(["allele", "length"]).peptide.count().unstack().fillna(0)
    real_binders = data.loc[
        data.affinity <= 500
    ].groupby(["allele", "length"]).peptide.count().unstack().fillna(0)
    real_nonbinders = data.loc[
        data.affinity > 500
    ].groupby(["allele", "length"]).peptide.count().unstack().fillna(0)
    total_nonbinders = random_negatives + real_nonbinders

    assert (random_negatives.loc["HLA-A*02:01"] == 1.0).all()
    assert (random_negatives.loc["HLA-B*44:02"] == math.ceil(1007 / 8)).all(), (
        random_negatives.loc["HLA-B*44:02"], math.ceil(1007 / 8))



def test_random_negative_peptides_by_allele():
    network = Class1NeuralNetwork(
        random_negative_method="by_allele_equalize_nonbinders",
        random_negative_binder_threshold=500,
        random_negative_rate=1.0,
        random_negative_constant=2)

    allele_to_sequence = {
        "HLA-A*02:01": "AAAAA",
        "HLA-B*44:02": "CCCCC",
        "HLA-C*07:02": "EEEEE",
    }
    data_rows = [
        ("HLA-A*02:01", "SIINFEKL", 400, "="),
        ("HLA-A*02:01", "SIINFEKLL", 300, "="),
        ("HLA-A*02:01", "SIINFEKLL", 300, "="),
        ("HLA-A*02:01", "SIINFEKLQ", 1000, "="),
        ("HLA-A*02:01", "SIINFEKLZZ", 12000, ">"),
    ]
    for peptide in random_peptides(1000, length=9):
        data_rows.append(("HLA-B*44:02", peptide, 100, "="))
    for peptide in random_peptides(1000, length=9):
        data_rows.append(("HLA-B*44:02", peptide, 1000, "="))
    for peptide in random_peptides(5, length=10):
        data_rows.append(("HLA-B*44:02", peptide, 100, "="))

    data = pandas.DataFrame(
        data_rows,
        columns=["allele", "peptide", "affinity", "inequality"])
    data["length"] = data.peptide.str.len()

    peptides = EncodableSequences.create(data.peptide.values)
    allele_encoding = AlleleEncoding(data.allele.values, allele_to_sequence)

    results = network.random_negatives_generator(
        peptides,
        data.affinity.values,
        allele_encoding,
        data.inequality.values
    )
    result_allele_encoding = next(results)
    first_peptide_sample = next(results)
    second_peptide_sample = next(results)

    result_df1 = pandas.DataFrame({
        "allele": result_allele_encoding.alleles,
        "peptide": first_peptide_sample.sequences,
    })
    result_df1["length"] = result_df1.peptide.str.len()
    random_negatives = result_df1.groupby(["allele", "length"]).peptide.count().unstack()
    real_data = data.groupby(["allele", "length"]).peptide.count().unstack().fillna(0)
    real_binders = data.loc[
        data.affinity <= 500
    ].groupby(["allele", "length"]).peptide.count().unstack().fillna(0)
    real_nonbinders = data.loc[
        data.affinity > 500
    ].groupby(["allele", "length"]).peptide.count().unstack().fillna(0)
    for length in random_negatives.columns:
        if length not in real_nonbinders.columns:
            real_nonbinders[length] = 0
    total_nonbinders = random_negatives + real_nonbinders

    assert (total_nonbinders.loc["HLA-A*02:01"] == 2.0).all()
    assert (total_nonbinders.loc["HLA-B*44:02"] == 1126).all()
