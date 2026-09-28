# MHCflurry 2.3.2

Browse model and data releases and select historical weights directly from the
CLI. Help, version, and download-discovery commands no longer import the
numerical model stack before displaying their output.

- Add `mhcflurry downloads releases`, `downloads list`, and detailed
  `downloads info DOWNLOAD`, with descriptions, shared archive mappings,
  installed/source status, and JSON output.
- Support `--release` consistently for catalogue inspection and downloading.
  Add `--model-release` to `predict` and `predict-scan`, mutually exclusive with
  `--models`. Show resolved paths before optional environment overrides.
- Document MHC allele inputs and N/C flanks first, followed by no-flank
  comparisons. Embed stable-comparison figures, include NetMHCpan 4.0/4.1/4.2
  BA and EL, and move detailed paired-bootstrap tables into a separate report.
- Make the comparison PDF the primary download and improve navigation to
  model selection and compact percentile calibration.

The default weights remain **2.3.0**. This release does not change trained
weights, numerical prediction methods, or evaluation rows and scores.

[View all comparison figures (PDF)](https://github.com/openvax/mhcflurry/releases/download/2.3.0/mhcflurry-2.3.0-model-comparison-v3.pdf).
[Tables and source data (.tar.gz)](https://github.com/openvax/mhcflurry/releases/download/2.3.0/model-comparison.20260928-v3.tar.gz).
