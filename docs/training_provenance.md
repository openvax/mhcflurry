# Auditing training-sample overlap

Zero peptide–MHC overlap does **not** establish biological-sample separation:
other peptides from the same specimen may have contributed to training or model
selection. The historical affinity bundles lack enough sample provenance to
certify this separation. Upgrading the code cannot reconstruct those identities.
External predictors' training overlap also remains uncertified without their
complete lineage. Existing benchmark results retain these limitations.

## Preserve lineage during new training

New affinity curation adds `source_provenance`, a JSON array of source records.
Each record preserves available study, sample and assay IDs, the raw input
file's SHA-256, and its zero-based row index after parsing the CSV header.
Deduplication unions every contributing record while retaining the previous
measurement, row order and weighting. Reassignment, training metadata and
selection metadata retain this column. Archive the original sources as well:
hashes and row indices identify evidence; they do not replace it.

Missing IDs remain unknown. An assay ID is not a biological sample ID.
`complete` on a source record means contributor retention, not known specimen
identity. Historical deduplicated tables must not be relabeled as complete
sample tables merely because one study or sample can be recovered.

New release-holdout policies also write `affinity_source_samples.csv`, containing
the union of monoallelic and selected presentation evaluation samples. The
affinity preparation script applies it in addition to the pMHC exclusions.
Direct use is:

```shell
mhcflurry class1-reassign-mass-spec-training-data curated.csv \
    --exclude-pmhcs holdout/affinity_pmhcs.csv \
    --exclude-source-samples holdout/affinity_source_samples.csv \
    --sample-aliases specimen_aliases.csv --out-csv training.csv
```

Any overlapping contributor removes the entire measurement. If training has a
known study but no sample identity, the whole matching study is excluded. An
exclusion lacking a study conservatively matches its sample ID in every study.
Unresolvable training rows remain explicitly unresolved; retaining them cannot
support a verified-disjoint claim. The ordinary `release-holdout validate`
command checks the recorded exclusions and labels whole-sample proof unresolved.
Old policy files remain readable with their original, narrower scope.

## Audit every compared model together

Freeze one cohort and inventory every compared MHCflurry model, including
affinity, processing and presentation components. Inventory all biological
sources used for pretraining, training, stopping/development and ensemble
selection, including the lineage of any teacher used to generate synthetic
pretraining targets. An empty stage list explicitly means that stage used no
data; it does not mean its data are unavailable.

Run the audit before claiming sample-disjoint evaluation:

```shell
mhcflurry train release-holdout audit-samples \
    --inventory lineage.json --cohort frozen_cohort.csv.gz \
    --sample-metadata cohort_studies.csv --aliases specimen_aliases.csv \
    --out-dir results/sample_audit
```

The output directory must be new. The command writes `sample_audit.csv`,
`sample_disjointness.json` and `cohort.csv.gz`, with hashes, sample/row/label
counts, and reasons for overlap or uncertainty. It fails if any input sample is
overlapping or unresolved against any inventoried MHCflurry model. Use
`--report-only` to inspect incomplete historical evidence without that exit
failure. External evidence has a separate status and does not silently become
verified when MHCflurry passes.

The JSON also records the generating function and arguments, package/source
hashes and Python/pandas versions. The audit is deterministic and uses no random
seed. Freeze the inventory and source files while it runs.

The exported cohort contains only samples verified against **all** inventoried
MHCflurry models. Use that one cohort for every comparator, retaining all its
positive and negative rows. Disclose changed counts and hashes; never filter
each predictor independently. Recheck peptide–MHC overlap separately. A benchmark
already used for development remains a regression benchmark, even after overlap
filtering; use an untouched confirmation cohort for new generalization claims.

## Inventory and identity formats

This example describes one affinity-only model. A full presentation model uses
`"kind": "presentation"` and requires `affinity`, `processing` and `presentation`
component entries. Add every compared release as a separate model. External
models use `"kind": "external"`; unavailable components/artifacts yield an
explicit unresolved result.

```json
{
  "schema_version": 1,
  "identity_review": "Describe the reviewed study namespaces, specimen aliases and shared specimens here.",
  "models": [{
    "name": "candidate-affinity",
    "kind": "affinity",
    "artifacts": ["affinity-model-bundle.tar.bz2"],
    "components": {
      "affinity": {
        "complete": true,
        "evidence": "Describe how these files cover this model's complete lineage.",
        "pretraining": [],
        "training": [{"path": "train_data.csv.bz2"}],
        "development": [],
        "selection": [{"path": "model_selection_data.csv.bz2"}]
      }
    }
  }]
}
```

Paths are relative to the inventory file. Include an immutable model archive or
all model files in `artifacts`; their hashes bind the report to those weights.
The tool verifies the supplied files, not the truth of a completeness declaration.
Supply review evidence and leave `complete: false` when any lineage is missing.

Data entries default to the `source_provenance` column. For reviewed raw sample
tables, an entry may instead specify `"identity_mode": "sample_table"`, with
`"study_column": "pmid"` and `"sample_column": "sample_id"` (defaults:
`study_id` and `sample_id`). This mode is unsuitable for historical affinity
tables that already discarded contributors.

`cohort_studies.csv` has `study_id,sample_id` columns; it is optional if the cohort
already has both. Numeric study IDs mean PubMed IDs and normalize to `pmid:ID`.
Other study IDs need explicit namespaces and reviewed correspondence. Sample
IDs retain leading zeroes. Alias CSV columns, in order, are
`study_id,sample_id,canonical_study_id,canonical_sample_id`. Blank sample IDs on
both sides map a study alias; filled IDs map a shared specimen, including across
studies. Conflicting mappings and cycles are errors. Different spelling or a
different publication is not sufficient evidence of a different specimen.
