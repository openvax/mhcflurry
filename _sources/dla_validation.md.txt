# DLA evidence and canine evaluation

MHCflurry can execute DLA predictions without having independently validated
canine performance. `supported_alleles` describes execution support. The
commands below expose the other evidence separately and leave missing evidence
unknown. They do not change prediction values or model weights.

## Inspect one model bundle

```shell
mhcflurry eval allele-capabilities \
  --models-dir /path/to/models_class1_presentation/models \
  --alleles 'DLA-88*01:01' 'DLA-88*003:02' 'DLA-88*501:01' \
  --lengths 7 8 9 10 11 12 13 14 15 16 \
  --out capabilities.json
```

The Python API returns the same JSON-serializable dictionary:

```python
from mhcflurry.allele_capabilities import capability_report

report = capability_report(
    "/path/to/models_class1_presentation/models",
    ["DLA-88*01:01", "DLA-88*003:02"], lengths=[8, 9, 15, 16])
```

Each requested allele/length records its canonical identity, actual model key,
MHCgnomes species identity, model-input sequence SHA-256 and length, and support
for binding, processing and presentation **without flanks**. MHCgnomes may
identify an MHC species group; this is never substituted for the experimental
host species. The model-input sequence may be a pseudosequence, not a full
protein. Configuration support is not a runtime probe or accuracy measurement.
Unsupported names/lengths do not acquire a human or nearest-sequence allele.
Processing is allele-independent, so it can remain available when a requested
allele is unsupported.

Training evidence counts all rows in each available bundled table, including
decoys, by canonical allele and peptide length. Explicit `source_species` and
`host_species` columns are retained when present. Their absence is `null`, even
for DLA-labeled rows: a DLA molecule can be expressed in a human host. A present
table with zero matching rows is distinguished from a missing table or column.
These counts do not inventory every training/development/selection source.

Percentile-transform availability is separate from its reference population and
validation. The report leaves the latter unknown. The inspected bundle's
`GENERATE.sh` records presentation calibration using random peptides matched to
training-table amino-acid composition, 50 single-allele genotypes, 10,000 peptides
per length and seed 42. This recipe does not establish a canine reference
population or calibration on canine experimental outcomes.

`model.sha256` hashes a sorted compact JSON mapping from every relative file
name to its SHA-256. Component weights, manifests, calibration and historical
metadata are therefore bound together. Keep output files outside the model
directory, and freeze inputs while either command runs. Existing output files
or directories are not overwritten.

Symlinks inside the model bundle are rejected with an error naming the path.
Use a copy containing regular files and directories so every loaded component
is included in the fingerprint.

## Reproduce the 2026 observation audit

The source is [Kaabinejadian et al., iScience 2026,
PMID 42199926](https://pubmed.ncbi.nlm.nih.gov/42199926/),
[DOI 10.1016/j.isci.2026.115975](https://doi.org/10.1016/j.isci.2026.115975).
Raw mass spectra are deposited as
[PXD074485](https://www.ebi.ac.uk/pride/archive/projects/PXD074485).
Download the original Data S1–S6 workbooks (`mmc2.xlsx` through `mmc7.xlsx`)
from the article. The command checks their pinned SHA-256 values before parsing.
These files also matched the MD5 values in the article's PMC full-text XML.
Install the optional Excel reader in the evaluation environment:

```shell
python -m pip install openpyxl==3.1.5
mkdir -p work/dla-supplements
for n in 2 3 4 5 6 7; do
  curl --fail --location \
    "https://ars.els-cdn.com/content/image/1-s2.0-S2589004226013507-mmc${n}.xlsx" \
    --output "work/dla-supplements/mmc${n}.xlsx"
done
mhcflurry eval dla \
  --models-dir /path/to/models_class1_presentation/models \
  --supplements-dir work/dla-supplements \
  --out-dir results/dla-2026
```

The output contains `observations.csv.gz`, `scores.csv.gz`, `capabilities.json`,
`report.json`, an explicitly incomplete `lineage.json`, and the existing
sample-audit outputs under `sample_audit/`. Source file, worksheet and Excel row
are preserved. The report records source/model/code hashes, dependency versions,
configuration and the bootstrap seed. Its linked capability report retains
unknown empirical validation and identifies the exploratory evaluation.

The original observation sheets contain 3,580 monoallelic DLA observations from
engineered **human HCT116 cells**, and 12,779 canine tumor observations from four
dogs (12,181 distinct canine peptide strings). Human comparison/TAA worksheets
are excluded. Lola's H58A and BB7.6 captures share one donor identity. HCT116
transductants retain separate capture/allele labels while sharing their parental
biological identity. All original lengths remain in the output, with explicit
missing-score reasons for unsupported lengths or residues.

For each capture, the report includes length distributions, nine-mer
position-specific residue counts, execution counts and score medians by length.
Affinity is in nM; processing and presentation are model scores on [0, 1].
Tumor affinity is the minimum predicted affinity over the reported genotype;
`predicted_best_allele` is a model assignment, never an experimental restriction.
No tumor observation is duplicated across its typed alleles. If any genotype
allele is unsupported, that genotype's binding/presentation scores stay missing.
Only no-flank processing is scored, because the workbook's multiple protein
accessions do not establish a unique flank context.

The descriptive bootstrap resamples canine donors with replacement (seed 490,
2,000 replicates), after deduplicating each donor's peptide observations across
antibody captures. Its estimand is the mean of within-donor score medians on
executable peptides. The intervals are not accuracy estimates. Three human-host
transductants do not supply independent biological replicates per DLA allele.

No negatives or sampled background are introduced. AP, precision-at-N, AUROC,
quantitative binding accuracy, isolated processing accuracy, and calibration
remain unevaluated. MS nonobservations must not be labeled measured negatives.
Future ranking evaluations need a frozen, species/context-appropriate sampled
background, declared sampling prevalence and length/motif diagnostics, with
uncertainty at the biological sampling unit.

## Historical lineage limits

The recipe audits all alleles and labels in every available bundled training
table for exact and I/L-collapsed peptide overlap. The I/L denominator remains
distinct **original** peptide strings. Peptide overlap does not establish the
same donor, study or peptide–DLA pair. Removing overlapping strings cannot
establish sample separation.

The generated inventory deliberately marks every component incomplete. Unknown
pretraining/development/selection stages are omitted, rather than recorded as
empty lists (which mean reviewed as unused). The existing
{doc}`training_provenance` audit excludes unresolved samples from its exported
cohort. The exploratory score file retains the entire original cohort and
overlap flags, and is not a held-out benchmark.

To claim independent canine performance, recover and review the complete
affinity, processing and presentation lineage, including synthetic-target
teachers, stopping/development and ensemble selection. Bind every source and
model artifact to an inventory, resolve study/donor aliases, and run the
existing sample audit plus peptide overlap checks. If historical identities
cannot be recovered, retrain with explicit study/donor/peptide exclusions and
preserved contributor provenance. Use an untouched confirmation cohort after
developing against these observations. Assess human-host DLA binding evidence,
endogenous canine processing, presentation and percentile calibration separately.

This implements the reporting and reproducible historical audit requested in
[MHCflurry #490](https://github.com/openvax/mhcflurry/issues/490).
The mhctools legacy dictionary-key normalization defect is tracked separately
in [mhctools #544](https://github.com/openvax/mhctools/issues/544).

## Executed report: October 8, 2026

The inspected local bundle is under model catalogue `2.3.0`. Its `info.txt`
records September 28, 2026, MHCflurry `2.3.1rc3`, training source
`ac253f8c859dc687f7fbce9f4840fb55d5acf2c5`, and workflow
`pr433-completion-20260928`. Its full file-tree fingerprint is:

```text
7bcbbd0846a0642c67b367ed51320f92ff747f6f15c2fec8251bd07f678a37bd
```

The [machine-readable report](_static/dla-2026/report.json) and
[capability report](_static/dla-2026/capabilities.json) bind these results to that
bundle. The [observations](_static/dla-2026/observations.csv.gz),
[scores and overlap flags](_static/dla-2026/scores.csv.gz), and
[sample audit](_static/dla-2026/sample_audit.csv) preserve the evaluated rows.
The capability report covers the seven DLA alleles typed in this study.

| Bundled training table | All rows | Exact canine overlap | I/L-collapsed canine overlap |
|---|---:|---:|---:|
| Affinity | 811,134 | 2,610 / 12,181 | 2,686 / 12,181 |
| Processing with flanks | 399,392 | 1,144 / 12,181 | 1,183 / 12,181 |
| Processing without flanks | 399,392 | 1,144 / 12,181 | 1,183 / 12,181 |

Among the seven study alleles, the affinity table contains 1,963 rows labeled
`DLA-88*501:01` and zero for each of the other six. Neither processing table has
rows labeled with these alleles. Host/source species and complete contributor
provenance are unavailable in these tables. The sample audit retains **zero
rows and zero biological samples** as verified disjoint.

| Observation capture | Original rows | Rows scored for each modality | Median affinity (nM) | Median processing score | Median presentation score |
|---|---:|---:|---:|---:|---:|
| HCT116 / DLA-88*003:02 | 811 | 626 | 11,268.2 | 0.464 | 0.023 |
| HCT116 / DLA-88*012:01 | 912 | 733 | 1,714.1 | 0.518 | 0.160 |
| HCT116 / DLA-88*501:01 | 1,857 | 1,524 | 164.5 | 0.461 | 0.663 |
| Lola / H58A | 207 | 185 | 108.6 | 0.614 | 0.860 |
| Lola / BB7.6 | 330 | 281 | 211.8 | 0.531 | 0.669 |
| 163828A / BB7.6 | 1,671 | 1,377 | 221.3 | 0.495 | 0.585 |
| Lily / H58A | 60 | 58 | 93.3 | 0.489 | 0.777 |
| Bogey / BB7.6 | 10,511 | 8,003 | 5,088.8 | 0.395 | 0.042 |

All unscored rows in this run exceed the model's maximum peptide length of 15.
The observed score differences across alleles and captures are exploratory;
they do not estimate sensitivity, specificity, calibrated probability, or
canine population coverage. Length-specific medians, motif counts and the
four-donor descriptive bootstrap are in the JSON report.

For a compact motif check, the two most frequent residues at positions 2 and 9
of the **unfiltered original nine-mer observations** are shown below. These
are residue frequencies, not the paper's GibbsCluster-selected motifs or a
measure of model accuracy. Tumor rows retain their unassigned mixture.

| Capture | Nine-mers | Position 2 | Position 9 |
|---|---:|---|---|
| HCT116 / DLA-88*003:02 | 300 | D 59.3%, E 19.0% | I 32.3%, V 26.0% |
| HCT116 / DLA-88*012:01 | 357 | A 24.1%, P 24.1% | L 58.3%, V 20.2% |
| HCT116 / DLA-88*501:01 | 676 | I 38.9%, V 31.2% | L 45.9%, V 26.0% |
| Lola / H58A | 97 | I 48.5%, V 23.7% | L 54.6%, V 14.4% |
| Lola / BB7.6 | 116 | I 41.4%, V 27.6% | L 47.4%, V 20.7% |
| 163828A / BB7.6 | 755 | I 34.4%, L 27.5% | L 49.4%, V 17.1% |
| Lily / H58A | 32 | I 53.1%, L 21.9% | V 40.6%, L 37.5% |
| Bogey / BB7.6 | 2,573 | F 19.7%, L 15.2% | L 40.7%, V 15.7% |
