from mhcflurry import predict


def test_predict_A1_Titin_epitope():
    result_df = predict(alleles=["HLA-A*01:01"], peptides=["ESDPIVAQY"])
    assert len(result_df) == 1
    row = result_df.ix[0]
    ic50 = row["Prediction"]
    assert ic50 <= 500


def test_predict_duplicates_and_empty():
    result_df = predict(
        alleles=["HLA-A*01:01", "HLA-A01:01"],
        peptides=["ESDPIVAQY", "ESDPIVAQY"])
    assert len(set(result_df.Prediction)) == 1
    assert len(set(result_df.Allele)) == 1
    assert len(set(result_df.Peptide)) == 1

    result_df = predict(
        alleles=["HLA-A*01:01"],
        peptides=[])
    assert len(result_df) == 0