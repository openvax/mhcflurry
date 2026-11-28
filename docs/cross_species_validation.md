# Predicting on new species and MHC mutants

MHCflurry's pan-allele models predict peptide binding from a compact
**pseudosequence** — the 39 amino-acid positions of an MHC class I protein that
line its peptide-binding pocket and directly contact the bound peptide. Because
the model reads those 39 contact positions rather than memorizing allele names,
it can make predictions for MHC alleles (and entire species) that were never
seen during training, as long as we know their 39 contact residues.

By default, MHCflurry ships with pseudosequences for about 20,000 alleles
across 47 vertebrate species. Installing the optional
[`mhcseqs`](https://github.com/pirl-unc/mhcseqs) companion package — a curated
database and structural aligner for vertebrate MHC protein sequences — expands
coverage to **over 220 vertebrate species** and lets you extract pseudosequences
directly from any raw MHC class I protein sequence or point mutant, without
installing external alignment tools like `clustalo`.

Existing `2.3.0` model weights and predictions on previously supported alleles
are completely unchanged.

## Install `mhcseqs` support

```shell
pip install "mhcflurry[mhcseqs]>=2.5,<2.6"
```

MHCflurry uses [`mhcgnomes`](https://github.com/pirl-unc/mhcgnomes) to normalize
allele names across species (for example, `HLA-A0201` → `HLA-A*02:01`, as well
as sequence-backed UniProt identifiers for non-model organisms).

## Predict on a new allele or pocket mutant

You can register any MHC class I heavy-chain protein sequence on a loaded
predictor at runtime. Signal peptides are stripped automatically before
extracting the 39 contact residues:

```python
from mhcflurry import Class1AffinityPredictor

predictor = Class1AffinityPredictor.load()

# 1. Register a novel heavy-chain sequence (for example, a feline allele)
predictor.add_allele_from_sequence("FLA-E*003:01", cat_heavy_chain_seq)
predictor.predict(["APRGPGRAF"], allele="FLA-E*003:01")

# 2. Score a single-residue point mutant (mature protein numbering)
predictor.add_allele_from_sequence(
    "FLA-E*018:01_W167S",
    cat_e018_seq,
    mutations=("W167S",),
)

# 3. Extend a predictor with all alleles in mhcseqs while keeping existing
#    2.3.0 pseudosequences unchanged
predictor.extend_alleles_from_mhcseqs(prefer_existing=True)
```

The same operations are available from the command line:

```shell
# Extract a 39-aa pseudosequence from a raw sequence (with optional mutation)
mhcflurry pseudosequences extract --sequence "$SEQ" --allele "FLA-E*018:01" --mutation W167S

# Build an extended allele_sequences.csv table from mhcseqs
mhcflurry pseudosequences build --existing-csv allele_sequences.csv --prefer-existing --out extended.csv
```

## Checking for binding-pocket insertions

Most mammalian MHC class I proteins share the same length across their
peptide-binding region, so all 39 contact positions line up cleanly. Some
non-model species (such as certain bats, birds, amphibians, and marsupials)
have extra inserted or deleted amino acids inside the binding pocket.

When `mhcseqs >= 2.9` is installed, MHCflurry aligns each input sequence to the
standard IMGT MHC coordinate system and **logs a warning** whenever:

- a groove contains internal insertions that cannot be represented in a fixed
  39-residue pseudosequence, or
- a requested point mutation falls outside the 39 contact positions (meaning it
  will not change predicted binding).

To inspect the alignment details programmatically, call
`mhcflurry.pseudosequences.extract_class1_pseudosequence_with_alignment(...)`,
which returns both the 39-aa pseudosequence and a structured `GrooveAlignment`
object.

## What to expect on unseen species

We evaluated frozen `2.3.0` weights (with zero retraining) across newly added
vertebrate species, engineered pocket mutants, and audited livestock/dog
mass-spectrometry cohorts using [`hitlist`](https://github.com/pirl-unc/hitlist)
(an open benchmark corpus of MHC binding and elution datasets) and IEDB:

- **Strong transfer across canonical mammalian and avian grooves:** Unseen
  carnivore, equine, avian, and marsupial alleles reach `0.77–0.98` AUROC
  against length-matched decoys, and engineered pocket mutants outperform their
  wild-type grooves on mutant-specific ligands (`Pig SLA-1*04:01 R156A`,
  `Cat FLA-E*018:01 W167S`).
- **Human-trained antigen processing transfers across species:** Supplying real
  source-protein flanking sequences improves presentation ranking even in
  marsupials (Tasmanian devil: `0.749 → 0.806` AUROC, `+30%` relative PPV@N)
  and on `21,332` strictly unseen Cattle `BoLA-I` MS ligands (`0.902–0.960`
  AUROC).
- **Known failure mode — pocket insertions:** Alleles with multi-residue
  insertions inside a peptide-binding pocket (such as black flying fox bat
  `Ptal-N*01:01`, which has a 3-residue insertion at position 51) are poorly
  captured by fixed 39-aa pseudosequences. Always check alignment warnings and
  compare against an unrelated negative-control allele (such as `HLA-A*02:01`)
  when evaluating a small non-model dataset.
- **Class I weights do not transfer to Class II:** Testing structurally aligned
  Class II pseudosequences on frozen Class I weights gives `0.551` AUROC (worse
  than an `HLA-A*02:01` negative control and far below `NetMHCIIpan-4.3` at
  `0.923`). Use a dedicated Class II predictor for Class II alleles.

## Full whitepaper, figures & validation logs

- **Unified Technical Whitepaper:** {doc}`whitepaper_25x_cross_species`
  ([Download publication PDF](_static/whitepaper/mhcflurry_25x_cross_species_whitepaper.pdf)) —
  complete figures, tables, failure-mode analysis, and references.
- **Individual dated experiment logs:** {doc}`experiment_25x_cross_species_hitlist`,
  {doc}`experiment_25x_cattle_dog_hitlist`, and {doc}`experiment_25x_pan_class_ii`
  in {doc}`experiments`.
- **Reproducible benchmark scripts:**
  [`scripts/experiments/release_2_5_0/`](https://github.com/openvax/mhcflurry/tree/master/scripts/experiments/release_2_5_0).
