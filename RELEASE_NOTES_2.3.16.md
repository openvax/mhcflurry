# MHCflurry 2.3.16

`mhcflurry-predict` now accepts an input CSV in which a flank cell is empty,
as it is for a peptide at a protein terminus. pandas previously read the empty
cell as NaN, and prediction failed with `ValueError: Sequence of strings is
required`. Flank strings such as `NA` and `NAN`, which pandas also read as NaN,
are now kept as sequence.

The prediction models and numerical prediction behavior are unchanged.
