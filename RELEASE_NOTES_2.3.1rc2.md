# MHCflurry 2.3.1rc2

This training and evaluation prerelease retains the published public weights.

- Expand and freeze processing evaluation pools for unique 10:1 negatives,
  preserving every held-out hit and the fixed affinity caliper. Saved cohorts
  are checksummed and shared by every processing comparator (#435).
- Record the actual allele-sequence CSV and checksum on fresh training runs
  (#429).
- Include every requested NetMHCpan version and both BA/EL scores in paired
  comparisons and curves (#421). An explicit common-coverage mode evaluates
  every predictor on exactly the same rows and records exclusions.

These changes support controlled retraining; they do not claim improved model
accuracy or publish replacement model weights.
