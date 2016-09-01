from mhcflurry import predict


def test_predict_A1_Titin_epitope():
    result_df = predict(alleles=["HLA-A*01:01"], peptides=["ESDPIVAQY"])
    assert len(result_df) == 1
    row = result_df.ix[0]
    ic50 = row["Prediction"]
    assert ic50 <= 700
