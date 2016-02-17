from mhcflurry.peptide_encoding import (
    all_kmers,
    extend_peptide,
    shorten_peptide,
    fixed_length_from_many_peptides,
)
from nose.tools import eq_


def test_all_kmers():
    kmers = all_kmers(2, alphabet=["A", "B"])
    assert len(kmers) == 4, kmers
    eq_(set(kmers), {"AA", "AB", "BA", "BB"})


def test_all_kmers_string_alphabet():
    kmers = all_kmers(2, alphabet="AB")
    assert len(kmers) == 4, kmers
    eq_(set(kmers), {"AA", "AB", "BA", "BB"})


def test_extend_peptide_all_positions():
    # insert 0 or 1 at every position
    results = extend_peptide(
        "111",
        desired_length=4,
        start_offset=0,
        end_offset=0,
        insert_amino_acid_letters="01")

    expected = [
        "0111",
        "1111",
        "1011",
        "1111",
        "1101",
        "1111",
        "1110",
        "1111",
    ]
    eq_(results, expected)


def test_shorten_peptide_all_positions():
    # insert 0 or 1 at every position
    results = shorten_peptide(
        "012",
        desired_length=2,
        start_offset=0,
        end_offset=0,
        insert_amino_acid_letters="012")

    expected = [
        "12",
        "02",
        "01"
    ]
    eq_(results, expected)


def test_shorten_peptide_all_positions_except_first():
    # insert 0 or 1 at every position
    results = shorten_peptide(
        "012",
        desired_length=2,
        start_offset=1,
        end_offset=0,
        insert_amino_acid_letters="012")

    expected = [
        "02",
        "01",
    ]
    eq_(results, expected)


def test_shorten_peptide_all_positions_except_last():
    # insert 0 or 1 at every position
    results = shorten_peptide(
        "012",
        desired_length=2,
        start_offset=0,
        end_offset=1,
        insert_amino_acid_letters="012")

    expected = [
        "12",
        "02",
    ]
    eq_(results, expected)


def test_fixed_length_from_many_peptides():
    kmers, original, counts = fixed_length_from_many_peptides(
        peptides=["ABC", "A"],
        desired_length=2,
        start_offset_extend=0,
        end_offset_extend=0,
        start_offset_shorten=0,
        end_offset_shorten=0,
        insert_amino_acid_letters="ABC")
    print(kmers)
    print(original)
    print(counts)
    eq_(len(kmers), len(original))
    eq_(len(kmers), len(counts))
    eq_(kmers, ["BC", "AC", "AB", "AA", "BA", "CA", "AA", "AB", "AC"])
    eq_(original, ["ABC", "ABC", "ABC", "A", "A", "A", "A", "A", "A"])
    eq_(counts, [3, 3, 3, 6, 6, 6, 6, 6, 6])
