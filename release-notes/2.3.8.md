# MHCflurry 2.3.8

This maintenance release preserves training provenance and makes evaluation
overlap claims more precise. Default weights remain 2.3.0; frozen-weight
predictions are unchanged.

- New affinity curation preserves study/sample/assay identities, source hashes
  and raw row references, including every contributor to a deduplicated row.
- New holdout policies exclude known whole source samples or studies in addition
  to peptide–MHC pairs. Legacy policies retain their original scope.
- `mhcflurry train release-holdout audit-samples` audits all compared models'
  components and pretraining/training/development/selection sources, resolves
  explicit specimen aliases, and exports one shared cohort with hashes/counts.
- Missing historical or external lineage remains unresolved. Passing the old
  peptide/sample-ID checks is never presented as complete sample-disjoint proof.

This release does not retrain or republish weights and does not retrospectively
certify historical sample separation. See issue #444 for that remaining evidence
limitation and #469 for the planned processing-quality investigation.
