# Cross-species alleles, pseudosequences, and zero-shot validation

MHCflurry 2.5.0 integrates optional [`mhcseqs`](https://github.com/pirl-unc/mhcseqs)
support to extract Class I binding-groove pseudosequences directly from any
MHC Class I heavy-chain protein sequence — without external `clustalo` or
Biopython binaries. Existing `2.3.0` model weights and predictions on
previously supported alleles remain unchanged.

Install the optional `mhcseqs` extra to enable raw sequence decomposition and
full-database extension:

```shell
pip install "mhcflurry[mhcseqs]>=2.5,<2.6"
```

## Predicting on new species or binding-pocket mutants

Register a novel allele or engineered binding-pocket mutant on any loaded
pan-allele predictor at runtime:

```python
from mhcflurry import Class1AffinityPredictor

predictor = Class1AffinityPredictor.load()

# Register a novel heavy-chain sequence (signal peptide stripped automatically)
predictor.add_allele_from_sequence("FLA-E*003:01", cat_heavy_chain_seq)
predictor.predict(["APRGPGRAF"], allele="FLA-E*003:01")

# Evaluate a mature-numbered binding-pocket point mutant
predictor.add_allele_from_sequence(
    "FLA-E*018:01_W167S",
    cat_e018_seq,
    mutations=("W167S",),
)

# Strictly generalize all supported alleles from mhcseqs (preserving existing
# 2.3.0 non-X pseudosequences verbatim by default)
predictor.extend_alleles_from_mhcseqs(prefer_existing=True)
```

From the command line:

```shell
mhcflurry pseudosequences extract --sequence "$SEQ" --allele "FLA-E*018:01" --mutation W167S
mhcflurry pseudosequences build --existing-csv allele_sequences.csv --prefer-existing --out extended.csv
```

When `mhcseqs >= 2.9.0` is installed, MHCflurry delegates binding-groove
alignment to `mhcseqs.align_domain_to_imgt` / `mhcseqs.align_groove_to_imgt`
and logs a warning whenever a groove contains internal insertions omitted by
fixed 39-aa pseudosequences or when a requested point mutation does not alter
any of the 39 extracted positions. Call
`mhcflurry.pseudosequences.extract_class1_pseudosequence_with_alignment(...)`
to inspect the structured `GrooveAlignment` object directly.

## Taxonomic expansion and groove-indel landscape

Across the 19,886 non-`X` overlapping Class I alleles in MHCflurry 2.3.0's
legacy `allele_sequences.csv`, `mhcseqs 2.9.0` IMGT extraction achieves
**99.45% exact 39-for-39 match (19,777 / 19,886)** and **99.98% per-residue
identity (775,385 / 775,554)** while expanding complete Class I groove coverage
from **47 species (20,205 alleles)** in `2.3.0` to **241 distinct vertebrate
species (27,243 complete Class I grooves)** in `mhcseqs 2.9.0`.

```{figure} _static/release-2.5.0/species_coverage.svg
:alt: Taxonomic expansion of Class I MHC groove coverage from 47 species in MHCflurry 2.3.0 to 241 vertebrate species in mhcseqs 2.9.0, alongside the distribution of binding-groove insertions and deletions across 27,243 Class I alleles.

Left: distinct vertebrate species with complete Class I grooves in MHCflurry 2.3.0 versus MHCflurry 2.5.0 + mhcseqs 2.9.0. Right: IMGT-aligned insertion and deletion counts by canonical mature Class I position across all 27,243 complete grooves in mhcseqs 2.9.0.
```

Non-canonical insertions and deletions cluster into six recurring structural
loops rather than occurring uniformly across the groove: the α1 CD-turn
(positions 39–42, `-2 aa` in birds and frogs), the α1 N-helix hinge (positions
51–58, `+1` to `+3 aa` in bats and marsupials), the α2 AB-turn (positions
103–109, `-1` to `-2 aa`), and the α2 helix kink (positions 145–155, `+1` to
`+2 aa`).

## Zero-shot cross-species generalization of frozen 2.3.0 weights

To test how well MHCflurry's existing pan-allele `2.3.0` weights generalize to
**previously unsupported species and clades (0 training rows in 2.3.0)** using
`mhcseqs 2.9.0` 39-aa pseudosequences, we evaluated frozen `2.3.0` pan-allele
affinity ensembles against monoallelic mass-spectrometry ligands, multi-allelic
MS elutions, and IEDB binding assays across marsupials (~160 Myr diverged from
placentals), felines, birds (~310 Myr diverged), equines, amphibians (~350 Myr
diverged), and bats.

For every ligand set, we paired true ligands against 10:1 length-matched
background decoys (seed 42) and also scored the identical peptides with an
unrelated human negative control (`HLA-A*02:01`) to separate genuine
groove-specific recognition from peptide-composition bias.

```{figure} _static/release-2.5.0/zero_shot_cross_species.svg
:alt: Zero-shot ligand ranking AUROC with 95 percent bootstrap intervals versus HLA-A*02:01 negative control, and median predicted IC50 separation between true ligands and background decoys across ten previously unsupported vertebrate Class I alleles.

Zero-shot generalization of frozen MHCflurry 2.3.0 weights using mhcseqs-extracted 39-aa grooves (every allele has 0 training rows in 2.3.0). Left: AUROC with 95% stratified bootstrap CI compared to an HLA-A*02:01 negative control on identical peptides. Right: median predicted IC50 (nM) for true ligands versus length-matched background decoys.
```

| Allele (clade) | Positive ligands | Groove alignment (`mhcseqs 2.9.0`) | Zero-shot AUROC (95% CI) | `HLA-A*02:01` ctrl AUROC | PPV@N | Median IC50 pos / dec (nM) |
|---|---:|---|---:|---:|---:|---:|
| Cat `FLA-E*018:01` (Carnivora) | 24 | canonical | **0.978** [0.950, 0.996] | 0.648 | 0.833 | 94 / 21,630 (230×) |
| Horse `Eqca-N*006:02` (Perissodactyla) | 7 | canonical | **0.965** [0.914, 1.000] | 0.527 | 0.714 | 1,258 / 26,389 (21×) |
| Duck `Anpl-UAA*01` (Aves) | 31 | `+1aa @52(V), -2aa @40–41; +1aa @148(Q), -2aa @107–108` | **0.894** [0.847, 0.938] | 0.852 | 0.387 | 25 / 600 (24×) |
| Cat `FLA-K*007:01` (Carnivora) | 491 | canonical | **0.880** [0.863, 0.895] | 0.659 | 0.464 | 1,440 / 21,670 (15×) |
| Possum `Trvu-UB*01:01` (Marsupialia, MS) | 5,607 | `+3aa @53(RVG); +1aa @149(D)` | **0.852** [0.847, 0.857] | 0.551 | 0.409 | 164 / 2,958 (18×) |
| Cat `FLA-E*003:01` (Carnivora, MS) | 1,612 | canonical | **0.830** [0.821, 0.839] | 0.562 | 0.372 | 6,004 / 23,365 (3.9×) |
| Duck `Anpl-UAA*76` (Aves) | 33 | `+1aa @52(A), -2aa @40–41; +1aa @152(D), -1aa @109` | **0.824** [0.737, 0.896] | 0.593 | 0.394 | 183 / 1,032 (5.6×) |
| Frog `Xela-UAAg` (Amphibia) | 26 | `+1aa @54(K), -2aa @40–41; +2aa @152(NA), -1aa @107` | 0.818 [0.708, 0.904] | 0.815 | 0.538 | 139 / 9,733 (70×) |
| Tas. devil `Saha` (Marsupialia, 10 alleles, MS) | 981 | multi-allelic UniProt grooves | **0.764** [0.750, 0.779] | 0.651 | 0.317 | 124 / 847 (6.8×) |
| Bat `Ptal-N*01:01` (Chiroptera) | 38 | `+3aa @51(MDL)` *(A-pocket ins)* | **0.625** [0.529, 0.717] | 0.490 | 0.184 | 26,031 / 28,998 (1.1×) |

On experimental positive-versus-negative binding assays (no synthetic decoys)
with at least 10 positives and 10 negatives, Cat `FLA-E*018:01` achieves
**AUROC 0.684** (24+ / 12−, median predicted IC50 94 vs 383 nM, vs `HLA-A*02:01`
control 0.417), whereas Bat `Ptal-N*01:01` achieves **AUROC 0.599** (38+ / 32−,
median IC50 26,031 vs 29,560 nM).

## Zero-shot motif recovery and insertion failure mode

Comparing empirical 9-mer MS ligands against the top 1% of 30,000 random 9-mers
scored zero-shot by frozen `2.3.0` weights shows both why transfer works and
where fixed-length pseudosequences reach their structural limit:

```{figure} _static/release-2.5.0/zero_shot_motifs.svg
:alt: Empirical versus zero-shot predicted 9-mer information-content sequence logos for Brushtail possum Trvu-UB*01:01, Cat FLA-E*003:01, Tasmanian devil Saha, and Bat Ptal-N*01:01.

Empirical 9-mer ligands (top row) compared against the top 1% zero-shot predicted 9-mers from 30,000 random background peptides (bottom row).
```

- **Cat `FLA-E*003:01` (canonical groove, n = 1,610 9-mers):** zero-shot
  predictions recover both the P2-Pro and P9-Arg/Lys anchor preferences without
  any feline training data.
- **Brushtail possum `Trvu-UB*01:01` (n = 5,597 9-mers) and Tasmanian devil
  `Saha`:** despite ~160 Myr of divergence and zero marsupial training data,
  frozen `2.3.0` weights recover the aromatic/hydrophobic P9 anchor chemistry
  and achieve AUROC 0.852 (vs 0.551 for `HLA-A*02:01`).
- **Bat `Ptal-N*01:01` (honest failure mode):** `Ptal-N*01:01` carries a 3-aa
  `MDL` insertion at mature position 51 that remodels the A-pocket (driving a
  strong empirical P1-Asp and P9-Pro preference) together with a rare `Y171H`
  substitution. Because a fixed 39-aa pseudosequence omits the `MDL` insertion
  and `2.3.0` has zero bat training data, predicted affinities collapse to flat
  ~26–30 µM IC50s (1.1× separation). `mhcseqs >= 2.9.0` explicitly flags these
  insertions via `logging.warning`.
- **Amphibian `Xela-UAAg` and Duck `Anpl-UAA*01` caveat:** on `Xela-UAAg`
  (carrying both `+1 aa K` and `+2 aa NA` pocket insertions), the target groove
  AUROC (0.818) matches the `HLA-A*02:01` negative control (0.815), indicating
  that ranking performance on those 26 peptides reflects shared hydrophobic
  composition rather than frog-specific groove accuracy. Always inspect the
  negative control and alignment warnings when predicting on deeply diverged
  clades.

## Binding-pocket point-mutant sensitivity

Using `mhcseqs` mature-numbered `mutations=...`, single-residue binding-pocket
substitutions improve discrimination over the corresponding wild-type grooves
on mutant-specific IEDB ligands:

```{figure} _static/release-2.5.0/pocket_mutants.svg
:alt: Wild-type versus point-mutant groove AUROC and paired per-peptide log10 IC50 scatter plots for Pig SLA-1*04:01 R156A and Cat FLA-E*018:01 W167S.

Single-residue binding-pocket substitutions extracted via mhcseqs mutations=(...) outperform their wild-type grooves on mutant-specific ligands for both Pig SLA-1*04:01 R156A (ΔAUROC = +0.010, 95% paired CI [+0.007, +0.013]) and Cat FLA-E*018:01 W167S (ΔAUROC = +0.006, 95% paired CI [+0.000, +0.015]).
```

## Reproducing the benchmark and figures

The complete evaluation and plotting pipeline lives in
[`scripts/benchmark_cross_species_mhcseqs.py`](https://github.com/openvax/mhcflurry/blob/master/scripts/benchmark_cross_species_mhcseqs.py):

```shell
python scripts/benchmark_cross_species_mhcseqs.py --collect --plot --out-dir ./cross_species_250
```

It uses frozen MHCflurry `2.3.0` weights (`models_class1_presentation`),
`mhcseqs 2.9.0` (`mhc-full-seqs.csv`), PDB heavy chains `6A2B` (`Xela-UAAg`)
and `4ZUU` (`Eqca-N*006:02`), `curated_training_data.csv.bz2` (to enforce zero
training overlap), and live IEDB Class I queries cached locally under
`--out-dir/cache/`.
