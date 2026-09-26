# Retraining comparison readiness

The without-replacement matcher exposes insufficient fixed processing pools
(#435). Expand only unresolved sample/length pools using a frozen public
affinity reference, preserve every held-out hit, and save immutable scored
rounds and unique 10:1 assignments. Keep presentation on its original cohort.
Validate cohort provenance and row identities before scoring all processing
predictors against those same assignments. Never select recipes using these
held-out labels.

Fix release input provenance on an empty download cache (#429) by fetching
allele sequences before recording their actual CSV checksum. Include every
requested external predictor in paired intervals and curves (#421). Report
an explicitly common-coverage table alongside full-cohort metrics and missing
coverage; do not compare different row subsets in one headline table.

Validation: meaningful regression fixtures for expansion/exhaustion, all-hit
preservation, reference/cache corruption, exact-row comparators, fresh input
provenance and all NetMHCpan versions. Run lint, full tests and CI before merge.
Model accuracy remains a separate empirical gate after the bounded smoke run
and full clean-source retraining.
