# MHCflurry 3.0 Plan

## Goals

MHCflurry 3.0 will retrain all models on updated data with broader allele
coverage, using mhcseqs for allele sequences instead of the legacy
alignment-derived pseudosequences.

## Allele sequences: switch to mhcseqs

Replace the current pseudosequence pipeline (`downloads-generation/allele_sequences/`)
with [mhcseqs](https://github.com/pirl-unc/mhcseqs), which provides
alignment-free structural groove decomposition for 54,000+ MHC alleles
across 500+ species.

**Current state (2.2.0):** 20,252 alleles with 34-39 aa pseudosequences
derived from ClustalO multiple sequence alignment of full-length proteins,
then position selection to match a reference pseudosequence set.

**With mhcseqs:** 26,000+ class I alleles with groove-ok status, including
~2,000 new human classical HLA-A/B/C alleles, plus major gains in fish
(+657), bird (+518), murine (+422), and other species.

Tasks:
- [ ] Add `mhcseqs` as a dependency
- [ ] Replace pseudosequence generation with mhcseqs groove sequences
- [ ] Decide on sequence representation: use full groove (groove1 + groove2,
  ~180 aa) or derive a contact-position subset for model input
- [ ] Handle the 223 alleles in the current file but not in mhcseqs
  (tracked in pirl-unc/mhcseqs#23 -- mostly macaque genes, horse, rat)
- [ ] Include non-classical genes (HLA-E, F, G, MICA, MICB) where grooves
  are available
- [ ] Validate that models trained on mhcseqs groove sequences reproduce
  or improve upon 2.2.0 accuracy on existing benchmarks

## Data curation

Curate updated training data for both binding affinity and mass-spec
identified ligands.

Tasks:
- [ ] Update IEDB affinity data pull (current curated data was last
  regenerated for the 2.0 release cycle)
- [ ] Update mass-spec ligand data pull
- [ ] Integrate any new public MS datasets published since last curation
- [ ] Review and update data filtering / QC pipeline
- [ ] Check for allele name normalization issues with new mhcgnomes + mhcseqs

## Model training

Retrain pan-allele models on the updated data with the new allele
representation.

Tasks:
- [ ] Decide whether to change network architecture or keep current
  (convolutional + pan-allele embedding)
- [ ] Train pan-allele binding affinity models
- [ ] Train antigen processing models
- [ ] Train presentation models
- [ ] Run benchmark evaluation against 2.2.0 models and other tools
- [ ] Calibrate percentile ranks for expanded allele set

## Fixes from 2.2.0

- [ ] H2-D*q missing from model bundle allele_sequences.csv (known issue,
  sequence exists in global file but was dropped during model generation)
- [ ] Consider fixing in 2.2.1 patch or just include in 3.0

## Dependency changes

| Dependency | 2.2.0 | 3.0 (proposed) |
|------------|-------|----------------|
| mhcseqs | -- | new, >= TBD |
| mhcgnomes | >= 3.0.1 | >= 3.14.0 (mhcseqs requirement) |
| torch | >= 2.0.0 | >= 2.0.0 (no change expected) |
| numpy | >= 1.22.4 | >= 1.22.4 (no change expected) |
| pandas | >= 2.0 | >= 2.0 (no change expected) |
| Python | >= 3.10 | >= 3.10 (no change expected) |

## Open questions

- Should the allele representation input to the model change from a
  ~34 aa pseudosequence to the full ~180 aa groove, or should we select
  contact positions from the groove? Longer input = more information but
  larger embedding, slower training.
- Should non-classical alleles (E, F, G, MICA/B) be included in the
  default predictor even though binding data may be sparse?
- Is there value in class II support in this release cycle?
