# Evaluation of the 2.3.0 weights

The release's `latest-433-full` evaluation label identifies the final full
presentation predictor. Results below are equal-patient means on ten patients,
with 16,400 positives and 164,731 negatives (181,131 total rows). Every model
receives the same rows and labels; no predictor chooses its own negatives.

## Full presentation

| Input | Model | AP | PPV@N | AUROC |
|---|---|---:|---:|---:|
| Peptide and genotype | MHCflurry 2.3.0 | 0.7893 | 0.7373 | 0.9501 |
| Peptide and genotype | MHCflurry 2.2.1 | 0.7713 | 0.7258 | 0.9409 |
| Peptide and genotype | NetMHCpan 4.2 EL | 0.7832 | 0.7307 | 0.9455 |
| Peptide and genotype | MixMHCpred 3.0 | 0.7765 | 0.7281 | 0.9310 |
| Also native flanks | MHCflurry 2.3.0 | 0.7986 | 0.7450 | 0.9524 |
| Also native flanks | MHCflurry 2.2.1 | 0.7766 | 0.7297 | 0.9439 |

Paired 95% patient-bootstrap intervals for **2.3.0 minus 2.2.1**, using 10,000
resamples and seed 42:

| Input | ΔAP [95% CI] | ΔPPV@N [95% CI] | ΔAUROC [95% CI] |
|---|---:|---:|---:|
| Peptide and genotype | +0.01797 [0.00488, 0.03004] | +0.01148 [−0.00246, 0.02440] | +0.00921 [0.00758, 0.01085] |
| Also native flanks | +0.02205 [0.00921, 0.03425] | +0.01528 [0.00354, 0.02586] | +0.00846 [0.00640, 0.01046] |

The no-flank AP and PPV@N differences versus NetMHCpan 4.2 EL and MixMHCpred 3.0
have intervals including zero. With flanks, AP intervals favor 2.3.0, but those
external predictors receive no extra flank input. Intervals are not adjusted
for multiple comparisons. These results do not establish universal superiority.

## Cohort and overlap

A separate benchmark builder sampled source-protein/length-matched negative
windows, with seed 20260928 and an initial ten negatives per positive. Sampling
did not use scores from either the old or new predictor. The final overlap
audit removed an additional 380 rows (101 positives and 279 negatives) from
every comparator without resampling. The revised cohort SHA256 is
`9f0fbc92fd44ac58438102ae7496687b67f109685d8f1fecfc820a7262bf46cc`;
the previous cohort SHA256 was
`81d18501616358b3c4b0f87d6e405ed50ea67dfefa31d6f8097d39818d658b41`.

The cohort is disjoint from the inventoried MHCflurry training peptide sources.
Unavailable or incomplete training records, especially for external predictors,
prevent claiming disjointness from every model ever trained. Ten patients limit
precision and generalizability; the release changed both recipes and data, so
its gains cannot be attributed solely to newer data. PPV@N uses the retained
positive count in each patient, with expected precision across score ties.

## Components and artifacts

Affinity-only ligand-ranking AP increased from 0.6971 to 0.7224 versus public
2.2.1. Processing-only AP decreased: 0.4144 versus 0.5399 with flanks and 0.4243
versus 0.5794 without flanks. These are component rankings on the presentation
cohort, not independent quantitative affinity or isolated processing endpoints.
The full presentation improvements therefore do not imply that every component
improved. Task AP values are never averaged.

The with-flanks predictor contains four short-flank and four cleavage-boundary
networks. The separately trained long-flank ensemble is a diagnostic and is not
used in that full predictor. The saved presentation-percentile mapping preserves
the raw-score AP, PPV@N and AUROC on the complete revised cohort.

[Download the comparison tables, row-level scores, audits and figures](https://github.com/openvax/mhcflurry/releases/download/2.3.0/model-comparison.20260928.tar.gz).
The archive includes public 2.1.5/2.2.0/2.2.1, prior full models, NetMHCpan
4.0/4.1/4.2 BA and EL, and MixMHCpred 3.0, with paired intervals for all
comparisons. MixMHCpred 2.0.2 is reported only on its supported subset and is not
mixed into the full-cohort table. Model archive checksums and training/packaging
provenance accompany the [2.3.0 release](https://github.com/openvax/mhcflurry/releases/tag/2.3.0).
