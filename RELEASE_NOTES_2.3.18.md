# MHCflurry 2.3.18

Test-only fix for pandas 3.1 (#510). `test_predict_scan_command.py` read
`mhcflurry-predict-scan` output with `converters={"c_flank": str}`, but pandas
3.1 turns an empty cell into NaN before converters run. A peptide that ends
at its sequence's C-terminus has an empty `c_flank`, so `test_fasta` and
`test_fasta_percentile` failed. The test now reads the output with
`keep_default_na=False`, as `mhcflurry-predict` reads its input since 2.3.16,
checks that only flank cells may be empty, and checks that an empty `c_flank`
survives the round trip.

`mhcflurry-predict-scan` output, the prediction models and numerical
prediction behavior are unchanged.
