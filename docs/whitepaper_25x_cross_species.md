# Whitepaper: Pan-Vertebrate MHC Class I Generalization and Exploratory Pan-Class Transfer in MHCflurry 2.5

> **Companion PDF:** [Download Publication-Ready PDF (16 standalone vector figures, tables & references)](_static/whitepaper/mhcflurry_25x_cross_species_whitepaper.pdf)
>
> **Software & Data:** `mhcflurry 2.5.x` · frozen `2.3.0` PyTorch weights (`models_class1_presentation`) · [`mhcseqs`](https://github.com/pirl-unc/mhcseqs) `2.9–2.14` · [`mhcgnomes`](https://github.com/pirl-unc/mhcgnomes) `3.79–3.96` · [`hitlist`](https://github.com/pirl-unc/hitlist) (Nov 2026 bundles), IEDB Class I & II, ProteomeXchange `PXD008151` · **Date:** November 2026

---

## Abstract & Key Findings

MHCflurry's pan-allele models predict peptide binding from a compact **pseudosequence** — the 39 amino-acid positions of an MHC class I protein that line its peptide-binding pocket and contact the peptide. Because MHCflurry 2.5 integrates the optional [`mhcseqs`](https://github.com/pirl-unc/mhcseqs) sequence database and structural aligner alongside [`mhcgnomes`](https://github.com/pirl-unc/mhcgnomes) allele-name parsing, complete Class I coverage expands nearly five-fold — from **47 vertebrate species** (`20,205` alleles; `10` trained species / `308` trained alleles in `2.3.0`) to **223–231 vertebrate species** (`26,156–26,549` complete binding-groove sequences across 14 vertebrate clades) — **without retraining or altering existing `2.3.0` model weights**.

This whitepaper synthesizes all three `2.5.x` validation campaigns into one empirical evaluation with rigorous negative controls (**50-allele mismatched real-allele null distributions**, **30-groove column-permuted pseudosequences**, and **within-peptide residue-scrambled positive controls**): (1) zero-shot Class I binding affinity and antigen presentation on newly added species and engineered pocket mutants, (2) a mandatory training-overlap audit and unseen-peptide evaluation on cattle (`BoLA-I`) and dog (`DLA-88`) mass-spectrometry datasets, and (3) an exploratory test of whether frozen Class I weights can predict Class II binding.

- **Broad zero-shot Class I transfer verified against proper null distributions:** Across `15` `hitlist` unseen-species and binding-pocket mutant cohorts (`0` training rows), frozen `2.3.0` weights achieve **`0.77–0.98` AUROC** across carnivores (cat `0.977`, giant panda `0.883`), horses (`0.967`), birds (mallard duck `0.893–0.938`), marsupials (brushtail possum `0.869`, Tasmanian devil `0.806` with flanks), and single-residue pocket mutants (pig `R156A` `0.951`, `p = 0.020`, `50/50` null alleles beaten; cat `W167S` `0.976`). Holding amino-acid composition strictly constant via **within-peptide scrambled ligands** confirms genuine positional anchor recognition (`+0.20` to `+0.29` AUROC over mismatched alleles on cat, duck `Anpl-UAA*76`, possum, cynomolgus, and pig `R156A`), while honestly exposing composition-dominated or indel-limited cohorts (rabbit `RLA-A*01`, frog `Xela-UAAg`, bat `Ptal-N*01:01`).
- **Human-trained antigen processing transfers across species:** On Tasmanian devil (*Sarcophilus harrisii*, `~160 Myr` diverged from placental mammals), MHCflurry's human-trained processing model alone achieves **`0.750` AUROC** using real Tasmanian devil source-protein flanking sequences, lifting combined Presentation from **`0.749 → 0.806` AUROC** (`+0.057`, `+30%` relative PPV@N; Affinity beats `50/50` size-matched 11-allele mismatched null draws, `0.749` vs `0.695 [0.674, 0.718]`).
- **Mandatory three-table overlap audit on Cattle `BoLA-I` & Dog `DLA-88`:** Checking all three training tables bundled inside `2.3.0` (`811,134` affinity + `399,392` processing rows) catches **`100%` in-sample overlap** on `hitlist` Dog `DLA-88` and small IEDB Cattle `BoLA-I` cohorts, while isolating **`21,332` strictly unseen flanked Cattle `BoLA-I` MS ligands** (`PXD008151`) reaching **`0.902–0.960` Presentation (+F) AUROC** (`0.833–0.922` on real vs. residue-scrambled ligands vs. `0.604–0.731` null) and confirming strong ultra-few-shot transfer (`4–12` training rows).
- **Clean negative result on zero-shot Class I → Class II transfer:** Mapping `36 / 39` structurally equivalent Class II `DR`/`DQ`/`DP` contact positions onto frozen Class I weights yields **`0.551` AUROC** (9-mer core scanning) — indistinguishable from a **30-allele mismatched Class I null (`0.549`)**, below **column-permuted pseudosequences (`0.571`)**, and far below `NetMHCIIpan-4.3 BA` (`0.923`). On **within-peptide scrambled Class II binders**, frozen Class I collapses to **`0.502`** (exact coin flip) while `NetMHCIIpan-4.3 BA` maintains **`0.725`** (`+0.223`). Pan-class prediction requires **joint pan-class training**.
- **Bit-identical training parity:** Locking existing trained alleles via `--lock-existing` reproduces legacy `clustalo` pan-allele training bit-for-bit (**`Max |Δ| = 0.0`**, **`Pearson r = 1.000000`** across all `308` trained alleles).

---

## Companion Packages & Key Terms

| Term / Package | Plain-English Meaning |
| :--- | :--- |
| **[`mhcseqs`](https://github.com/pirl-unc/mhcseqs)** | Optional companion library providing curated MHC protein sequences across >220 vertebrate species and a structural aligner that maps any MHC protein onto standard IMGT numbering. |
| **[`mhcgnomes`](https://github.com/pirl-unc/mhcgnomes)** | Companion MHC allele-name parser used by MHCflurry to normalize naming conventions across species (e.g., `HLA-A0201` → `HLA-A*02:01`, plus UniProt-backed names for non-model vertebrates). |
| **[`hitlist`](https://github.com/pirl-unc/hitlist)** | Open benchmark corpus of curated MHC peptide-binding and mass-spectrometry elution datasets paired with matched species reference proteomes. |
| **39-aa pseudosequence** | The 39 amino-acid positions of an MHC class I heavy chain that line its peptide-binding pocket (groove) and directly contact the bound peptide. |
| **Flanking sequences (`+F` vs `noF`)** | The source-protein amino acids immediately upstream (`n_flank`) and downstream (`c_flank`) of a peptide, used by MHCflurry's antigen-processing model to estimate cleavage and transport likelihood. |
| **AUROC & PPV@N** | Ranking metrics against length-matched background decoys (`10:1` ratio here). AUROC (`0.5` = random, `1.0` = perfect) measures overall separation; PPV@N is precision among the top `N` predictions (where `N` = number of true ligands). |
| **Mismatched real-allele null (`N = 50`)** | Scoring the identical test peptides using 50 diverse, unrelated real Class I alleles (excluding the target species; size-matched `k`-allele random draws for multi-allelic haplotypes). Gives a true null distribution (median, 5th–95th percentile band, and empirical *p*-value) without the motif bias of a single control allele like `HLA-A*02:01`. |
| **Column-permuted pseudosequences (`N = 30`)** | Synthetic 39-aa grooves created by independently shuffling each of the 39 positions across real Class I alleles, keeping per-position amino-acid distributions in-distribution while breaking pocket identity. |
| **Within-peptide scrambled positives** | Scoring real positive ligands against `10×` residue-shuffled copies of those exact same ligands (identical length and exact amino-acid composition). Eliminates hydrophobicity/composition bias so AUROC above `0.50` directly measures **positional anchor specificity**. |
| **Closed vs. open binding groove** | MHC class I holds peptides in a *closed* pocket that pins both peptide ends (favoring 8–11mers); MHC class II has an *open* groove where a 9-mer core binds internally while both ends of a longer peptide (typically 15+ aa) overhang freely. |
| **Groove insertion** | Extra amino acids inserted into the peptide-binding region of some non-model species (e.g., a 3-aa insertion in bat `Ptal-N*01:01`) that cannot be represented in a fixed 39-position vector. |

---

## 1. Pan-Vertebrate Architecture in MHCflurry 2.5

In MHCflurry `2.0–2.3`, building the 39-residue pseudosequence table required downloading bulk FASTA databases and running external `clustalo` multiple-sequence alignments. That workflow covered `47` species (`20,205` alleles; training measurements concentrated in only `10` species and `308` alleles) and could not register novel heavy chains, uncurated UniProt proteins, or engineered pocket mutants at runtime.

MHCflurry `2.5` adds four capabilities without altering any released `2.3.0` weights (see {doc}`cross_species_validation` for installation and usage examples):

1. **Direct structural pseudosequence extraction via [`mhcseqs`](https://github.com/pirl-unc/mhcseqs):** Signal-peptide trimming, binding-domain extraction, and mature-numbered 39-residue extraction are delegated to `mhcseqs`'s structural aligner (`add_allele_from_sequence`, `extend_alleles_from_mhcseqs`, and `mhcflurry pseudosequences`).
2. **Automatic insertion warnings:** Whenever a sequence contains extra inserted amino acids inside its peptide-binding region (which a fixed 39-position pseudosequence must omit), MHCflurry logs a warning and exposes the full `GrooveAlignment` object so callers know before trusting a prediction.
3. **Sequence-backed naming for non-model species:** Through [`mhcgnomes`](https://github.com/pirl-unc/mhcgnomes) (`AccessionedAllele` support in `mhcflurry >= 2.5.9`), uncurated vertebrate sequences with UniProt accessions (such as shark `TriakisScyllium-UAA*uniprot:O46817` or Tasmanian devil `Saha-classI*uniprot:G3V1L5`) flow cleanly through `normalize_allele_name` and `Class1AffinityPredictor` (`26,156` alleles across `223` canonical vertebrate species, `2,884` accessioned alleles, `0` unparsed rows).
4. **Zero-regression parity gate:** Across `19,886` overlapping non-`X` alleles in `2.3.0`, `mhcseqs` achieves **`99.45%` exact 39/39 match** (`19,777 / 19,886`) and **`99.98%` per-residue identity** (`775,385 / 775,554`). `downloads-generation/allele_sequences/GENERATE.sh` uses `--lock-existing` plus `verify_trained_allele_parity.py` to enforce `0` character differences across all `308` trained alleles.

Complete Class I coverage expands from `47` species in `2.3.0` to `227` species in `mhcseqs 2.12` (`231` union, `26,549` complete grooves, **`+184` net new vertebrate species** across 14 clades): bony fish `2→53`, birds `1→42`, other mammals `0→28`, reptiles `0→17`, amphibians `0→17`, rodents `2→15`, marsupials `0→7`, sharks & rays `0→4`, bats `0→4`, and cetaceans `0→2` (`Figure 1`). Experimental validation cohorts across these clades in [`hitlist`](https://github.com/pirl-unc/hitlist) span cellular MS elutions, direct ligands, positive/negative binding assays, and binding-pocket mutants (`Figure 2`).

```{figure} _static/whitepaper/figures/fig01_taxonomic_expansion.svg
:alt: Horizontal grouped bar chart comparing distinct vertebrate species with complete Class I grooves in MHCflurry 2.3.0 versus MHCflurry 2.5.x + mhcseqs across 14 vertebrate clades.
:width: 92%

**Figure 1. Taxonomic expansion of complete MHC class I binding grooves across 14 vertebrate clades.** Distinct vertebrate species with complete 39-aa pseudosequences increase from `47` in MHCflurry `2.3.0` (`10` trained) to `227` in `mhcseqs 2.12` (`231` union; `+184` net new vertebrate species), led by bony fish (`+51`), birds (`+41`), other mammals (`+28`), reptiles (`+17`), and amphibians (`+17`).
```

```{figure} _static/whitepaper/figures/fig02_hitlist_validation_landscape.svg
:alt: Heatmap of hitlist Class I validation peptide counts by species and experimental modality across 10 unseen vertebrate species, 2 unseen allele/mutant cohorts, and 2 trained reference controls.
:width: 92%

**Figure 2. `hitlist` Class I validation data availability by species and experimental modality.** Log-scaled peptide counts across four assay types (proteome-flanked cellular MS elutions, direct/unflanked ligands, experimental `+/-` binding assays, and engineered pocket mutants) for `10` unseen vertebrate species, `2` unseen allele/mutant cohorts in trained species, and `2` trained MS reference controls (below dashed line).
```

---

## 2. Zero-Shot Class I Affinity & Antigen Presentation Across Unseen Clades

### 2.1 Zero-Shot Presentation (+ Real Proteome Flanks) on Cellular MS Elutions

MHCflurry's presentation model combines pan-allele binding affinity with a human-trained antigen-processing network that examines each peptide and its neighboring source-protein flanking residues. To test whether this human-trained processing model generalizes to distant vertebrates, we took cellular mass-spectrometry (MS) datasets in [`hitlist`](https://github.com/pirl-unc/hitlist) with matched host proteomes (*Sarcophilus harrisii*, *Canis lupus*, *Macaca mulatta*), paired eluted ligands with `10:1` length-matched decoys carrying real flanking sequences from the same host proteome, and compared frozen `2.3.0` weights against a **size-matched mismatched real-allele null distribution** (`50` random 11-allele draws for Tasmanian devil `Saha`, single-allele null for monoallelic Dog and Rhesus controls; `Figures 3 and 4`).

```{figure} _static/whitepaper/figures/fig03_ms_presentation_auroc.svg
:alt: Grouped horizontal bar chart comparing AUROC of Presentation with real proteome flanks, Presentation without flanks, Affinity alone, Processing alone, and size-matched mismatched real-allele null on hitlist MS elutions.
:width: 92%

**Figure 3. Zero-shot Presentation (+ real species proteome flanks) versus Affinity and mismatched-allele null AUROC on `hitlist` MS elutions.** AUROC (95% bootstrap CI) against `10:1` same-proteome length-matched decoys for zero-shot Tasmanian devil (*Sarcophilus harrisii*, `0` training rows, `11` UniProt grooves, `15,301` mapped ligands) alongside trained Dog (`DLA-88`) and Rhesus macaque (`Mamu-B`) reference controls. Gray bars show the size-matched mismatched real-allele Affinity null (median and 5th–95th percentile range). Even across `~160 Myr` to marsupials, human-trained processing alone reaches `0.750` AUROC and lifts Presentation from `0.749` to `0.806` (`+0.057`).
```

```{figure} _static/whitepaper/figures/fig04_ms_presentation_ppv.svg
:alt: Grouped horizontal bar chart comparing PPV@N precision among top N predictions for Presentation with and without real flanks, Affinity alone, and Processing alone on hitlist MS elutions.
:width: 92%

**Figure 4. Top-N enrichment gain (`PPV@N`) from human-trained antigen processing on `hitlist` MS elutions.** Precision among the top `N` ranked peptides (dashed line = `0.091` random baseline at `10:1` decoy ratio). Adding human-trained processing and real *S. harrisii* flanks lifts Tasmanian devil `PPV@N` from `0.273` to `0.356` (`+30%` relative enrichment).
```

| Dataset (`hitlist` MS) | Status | Mapped Pos (Eval / Total) | Affinity AUROC | Mismatched Null (5–95%) | Processing (+F) | Pres (noF) | **Pres (+F)** | PPV@N (Aff → +F) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Tasmanian devil `Saha`** (11 grooves) | **0-shot Marsupialia** | 2,500 / 15,301 | 0.749 | `0.695 [0.674, 0.718]` | **0.750** | 0.791 | **0.806** (`+0.057`) | **0.273 → 0.356** (+30%) |
| Dog `DLA-88*501:01` | Trained ref* | 1,301 / 1,301 | 0.933 | `0.794 [0.703, 0.860]` | 0.837 | 0.942 | **0.947** (`+0.014`) | 0.711 → 0.746 |
| Dog `DLA-88*034:01` | Trained ref | 355 / 355 | 0.959 | `0.831 [0.683, 0.916]` | 0.875 | 0.966 | **0.970** (`+0.011`) | 0.752 → 0.780 |
| Rhesus `Mamu-B*008:01` | Trained ref | 874 / 874 | 0.968 | `0.721 [0.476, 0.942]` | 0.866 | 0.976 | **0.979** (`+0.011`) | 0.852 → 0.874 |

*\*As shown in Section 4, IEDB `DLA-88*501:01` overlaps `2.3.0`'s bundled affinity training table; these rows serve strictly as trained positive-control ceilings.*

### 2.2 Experimental `+/-` Binding Assays & 15-Cohort Zero-Shot Ranking

In-vitro binding assays (refolding, purified-MHC fluorescence screens, and synthetic peptide panels) bypass cellular antigen processing and rarely map to a host proteome (`19 / 448 = 4.2%` in `hitlist/binding.parquet`; `0%` for the `5,607` synthetic random-library `Trvu-UB*01:01` peptides). For these datasets, **Affinity** is the primary metric — evaluated both on measured positive-versus-negative binders with no synthetic decoys (`Figure 5`) and on `10:1` background ligand ranking (`Figure 6`), benchmarked against a **50-allele mismatched real-allele null distribution** (stratified across `HLA-A`, `HLA-B`, `HLA-C`, and non-human Class I alleles, excluding the target species) rather than a single arbitrary human allele.

```{figure} _static/whitepaper/figures/fig05_experimental_binding_assays.svg
:alt: Horizontal bar chart of zero-shot Affinity AUROC on experimental positive-versus-negative binding assays with no synthetic decoys from hitlist/binding.parquet, with 50-allele mismatched null medians and 5th-95th percentile intervals.
:width: 92%

**Figure 5. Zero-shot performance on experimental positive-versus-negative binding assays (no synthetic decoys) versus 50-allele mismatched null.** AUROC (95% bootstrap CI) on measured binders vs. measured non-binders from `hitlist/binding.parquet`, colored by clade, alongside the 50-allele mismatched real-allele null (black diamond = median; horizontal bar = 5th–95th percentile range). Cynomolgus `Mafa-A1*063:01` achieves `0.889` vs. `0.185 [0.032, 0.750]` null (`p = 0.020`), and Brushtail possum `Trvu-UB*01:01` reaches `0.655` vs. `0.534` null, while tiny single-mutant negative panels (`3+/13-` cat, `6+/10-` horse) have wide intervals.
```

```{figure} _static/whitepaper/figures/fig06_zeroshot_ligand_ranking.svg
:alt: Horizontal bar chart of zero-shot Affinity (solid) and no-flank Presentation (hatched) ligand ranking AUROC against 10:1 length-matched background decoys across 11 unseen vertebrate species and 3 binding-pocket mutants, overlaid with 50-allele mismatched real-allele null intervals.
:width: 96%

**Figure 6. Zero-shot Affinity and no-flank Presentation ligand ranking across 15 `hitlist` unseen-species and pocket-mutant cohorts.** Solid bars show zero-shot Affinity AUROC (95% CI); hatched bars show no-flank Presentation AUROC; black diamonds and horizontal whiskers mark the 50-allele mismatched real-allele null (median and 5th–95th percentile range; size-matched 11-allele draws for Tasmanian devil `Saha`). Asterisks (`*`) indicate alleles carrying internal groove insertions or deletions flagged by `mhcseqs`.
```

| Unseen Species / Allele (Clade) | Validation Data | Aff / Pres (noF) | 50-Allele Null Med `[5–95%]` (`beats/50`, *p*) | Col-Perm Null | Real vs. Scrambled *(Null)* | Exp. `+/-` *(no decoys)* |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Cat `FLA-E*018:01`** (Carnivora) | 24 direct/bind | **0.977** / **0.966** | `0.663 [0.283, 0.978]` (`46/50`, `0.098`) | 0.662 | **0.957** *(0.703)* | 0.308 *(3+/13-)* |
| **Cat `FLA-E*018:01 W167S`** (Mutant) | 12 mut. ligands | **0.976** / **0.981** | `0.663 [0.325, 0.987]` (`45/50`, `0.118`) | 0.677 | **0.957** *(0.664)* | — |
| **Horse `Eqca-N*006:02`** (Unseen allele) | 6+ / 10− bind | **0.967** / **0.956** | `0.833 [0.570, 0.988]` (`41/50`, `0.196`) | 0.876 | **0.919** *(0.821)* | 0.217 *(6+/10-)* |
| **Pig `SLA-1*04:01 R156A`** (Mutant) | 141 mut. ligands | **0.951** / **0.946** | `0.804 [0.534, 0.930]` (**`50/50`**, **`0.020`**) | 0.807 | **0.890** *(0.688)* | — |
| **Mallard duck `Anpl-UAA*01*`** (Aves) | 31 binding+ | **0.938** / **0.920** | `0.862 [0.705, 0.926]` (**`50/50`**, **`0.020`**) | 0.900 | **0.876** *(0.804)* | — |
| **Pig `SLA-1*13:01 F99Y`** (Mutant) | 60 mut. ligands | **0.898** / **0.892** | `0.780 [0.477, 0.886]` (**`49/50`**, **`0.039`**) | 0.740 | **0.852** *(0.706)* | — |
| **Mallard duck `Anpl-UAA*76*`** (Aves) | 30 direct ligands | **0.893** / **0.856** | `0.636 [0.365, 0.901]` (`46/50`, `0.098`) | 0.637 | **0.848** *(0.620)* | — |
| **Giant panda `Aime-128`** (Carnivora) | 21 direct/bind | **0.883** / **0.906** | `0.819 [0.746, 0.911]` (`37/50`, `0.275`) | 0.869 | **0.866** *(0.769)* | — |
| **Brushtail possum `Trvu-UB*01:01*`** (Marsup.) | 5,607 + 31 `+/-` | **0.869** / **0.849** | `0.715 [0.530, 0.838]` (**`49/50`**, **`0.039`**) | 0.725 | **0.754** *(0.542)* | **0.655** *(14+/17-)* |
| **Cynomolgus `Mafa-A1*063:01`** (NHP) | 56 `+/-` (`54+/2-`) | **0.868** / **0.830** | `0.599 [0.403, 0.842]` (`48/50`, `0.059`) | 0.645 | **0.763** *(0.536)* | **0.889** *(p=0.020)* |
| **African clawed frog `Xela-UAAg*`** (Amph.) | 26 + 19 `+/-` | 0.826 / 0.828 | `0.808 [0.742, 0.873]` (`36/50`, `0.294`) | 0.832 | 0.812 *(0.769)* | 0.357 *(14+/5-)* |
| **Rabbit `RLA-A*01`** (Lagomorpha) | 27 `+/-` (`11+/16-`) | 0.787 / 0.792 | `0.876 [0.708, 0.954]` (`8/50`, `0.843`) | 0.905 | 0.734 *(0.819)* | 0.562 *(11+/16-)* |
| **Tasmanian devil `Saha*`** (Marsup., 11 gr.) | 23,973 MS (`15k` +F) | **0.772** / **0.803** | `0.725 [0.703, 0.745]` (**`50/50`**, **`0.020`**) | 0.732 | 0.676 *(0.668)* | — |
| **Norway rat `RT1-A*a`** (Rodentia) | 5 binding+ | 0.720 / **0.940** | `0.582 [0.337, 0.888]` (`36/50`, `0.294`) | 0.522 | 0.752 *(0.548)* | 0.286 *(7+/1-)* |
| **Black flying fox `Ptal-N*01:01*`** (Bat) | 86 `+/-` (`37+/43-`) | 0.593 / 0.717 | `0.568 [0.483, 0.696]` (`31/50`, `0.392`) | 0.597 | 0.540 *(0.529)* | **0.606** *(37+/43-)* |

*(In addition, IEDB feline MS cohorts `FLA-E*003:01` (`n=1,612`) and `FLA-K*007:01` (`n=491`) achieve `0.830` and `0.880` AUROC zero-shot and recover empirical P2-Pro / P9-Arg/Lys anchor motifs; see `Figure 9`.)*

---

## 3. Groove Specificity, Positional Anchor Controls, Motifs & Training Parity

### 3.1 Mismatched-Allele Nulls & Within-Peptide Scrambled Controls

Using a single fixed negative-control allele (such as `HLA-A*02:01`) has a known pitfall: if a rare species allele happens to share `HLA-A*02:01`'s hydrophobic P2/P9 anchor preference, the control looks artificially high; if its motif is anti-correlated with `HLA-A*02:01`, the control drops below `0.50` and inflates the apparent lift.

Evaluating every unseen cohort against **two complementary null controls** gives a much cleaner picture (`Figures 7 and 8`):

1. **Mismatched real-allele & column-permuted groove nulls (`Figure 7`):** Fourteen of fifteen cohorts exceed the 50-allele mismatched median, and **five large/clean cohorts achieve empirical `p ≤ 0.05` (`49–50 / 50` unrelated real alleles beaten)**: Pig `SLA-1*04:01 R156A` (`p = 0.020`), Duck `Anpl-UAA*01` (`p = 0.020`), Tasmanian devil `Saha` (`p = 0.020` against size-matched 11-allele draws), Brushtail possum `Trvu-UB*01:01` (`p = 0.039`), and Pig `SLA-1*13:01 F99Y` (`p = 0.039`), with Cynomolgus `Mafa-A1*063:01` at `48/50` (`p = 0.059`, and `p = 0.020` on experimental `+/-` binding). Column-permuted synthetic pseudosequences (`N = 30`, open squares) track the real-allele null closely, confirming that the target groove's advantage is not an out-of-distribution artifact.
2. **Within-peptide residue-scrambled positive controls (`Figure 8`):** Scoring real ligands against `10×` residue-shuffled copies of those exact same ligands holds amino-acid composition and peptide length strictly identical. Genuine zero-shot anchor decoders separate real ligands from their own scrambled copies far better than mismatched alleles — led by **Cat `FLA-E*018:01 W167S` (`0.957` vs `0.664`, `+0.293`)**, **Cat `FLA-E*018:01` (`0.957` vs `0.703`, `+0.254`)**, **Duck `Anpl-UAA*76` (`0.848` vs `0.620`, `+0.228`)**, **Cynomolgus `Mafa-A1*063:01` (`0.763` vs `0.536`, `+0.227`)**, **Brushtail possum `Trvu-UB*01:01` (`0.754` vs `0.542`, `+0.212`)**, and **Pig `SLA-1*04:01 R156A` (`0.890` vs `0.688`, `+0.202`)**.
3. **Honest caveats clarified by proper controls:** Three cohorts show little or no groove-specific lift over the null distribution: **Rabbit `RLA-A*01`** sits *below* the 50-allele null (`0.787` vs `0.876`, `8/50`, scrambled `0.734` vs `0.819`), **African clawed frog `Xela-UAAg`** (`~350 Myr`, 2 groove indels) sits inside the null (`0.826` vs `0.808`, `p = 0.294`, `col-perm = 0.832`), and **Bat `Ptal-N*01:01`** (3-aa `MDL` A-pocket insertion) collapses to `0.540` vs `0.529` null on scrambled peptides.

```{figure} _static/whitepaper/figures/fig07_groove_specificity_vs_control.svg
:alt: Dumbbell plot comparing zero-shot target-groove AUROC against the 50-allele mismatched real-allele null distribution (median and 5th-95th percentile band) and 30 column-permuted pseudosequences across all 15 hitlist unseen-species and mutant cohorts.
:width: 96%

**Figure 7. Groove-specific zero-shot transfer versus 50-allele mismatched real-allele and 30 column-permuted pseudosequence nulls.** Colored circles show target-allele zero-shot AUROC; gray bars and black diamonds mark the 50-allele mismatched real-allele null (5th–95th percentile range and median); open squares mark the median of 30 column-permuted pseudosequences. Labels report target AUROC, margin over null median, and alleles beaten out of 50 (`*` denotes empirical `p ≤ 0.05`).
```

```{figure} _static/whitepaper/figures/fig08_scrambled_peptide_control.svg
:alt: Horizontal bar chart of real-ligand versus 10x within-peptide residue-scrambled copy AUROC (95% CI) for target grooves compared against the mismatched real-allele median on the exact same real-vs-scrambled task.
:width: 96%

**Figure 8. Positional anchor specificity: real ligands versus `10×` within-peptide residue-scrambled copies.** Holding peptide length and exact amino-acid composition constant eliminates generic hydrophobicity bias. Colored bars show target-groove AUROC (95% bootstrap CI); black diamonds show the mismatched real-allele median on the identical real-vs-scrambled task, sorted by lift `Δ`. Strong positive lifts (`+0.20` to `+0.29`) confirm true positional anchor decoding in feline, avian, marsupial, primate, and mutant porcine grooves, whereas Bat `Ptal-N*01:01` (`0.54` vs `0.53`) and Rabbit `RLA-A*01` (`0.73` vs `0.82`) show no groove-specific positional lift.
```

### 3.2 Empirical vs Zero-Shot Sequence Logos & Insertion Failure Mode

Comparing experimentally observed 9-mer ligands against the top 1% of `30,000` random 9-mers scored zero-shot (`Figure 9`) explains both why pan-allele transfer succeeds and where fixed 39-aa pseudosequences reach their physical limit:

- **Feline & marsupial anchor recovery:** With zero feline training data, `FLA-E*003:01` recovers both its empirical position-2 proline (`P2-Pro`) and position-9 basic (`P9-Arg/Lys`) preferences; brushtail possum `Trvu-UB*01:01` and Tasmanian devil `Saha` recover their aromatic/hydrophobic C-terminal (`P9`) anchors across `~160 Myr`.
- **Single-residue pocket mutants:** Extracting mature-numbered point mutations via `mhcseqs` improves discrimination over the corresponding wild-type allele on mutant-specific ligands for both Pig `SLA-1*04:01 R156A` (`ΔAUROC = +0.010`, 95% CI `[+0.007, +0.013]`) and Cat `FLA-E*018:01 W167S` (`ΔAUROC = +0.006`, 95% CI `[+0.000, +0.015]`).
- **Honest failure mode — Bat `Ptal-N*01:01` (3-aa `MDL` insertion):** This bat allele carries three extra amino acids (`MDL`) inserted at position 51 inside the N-terminal binding pocket (the A-pocket), reshaping its preference toward `P1-Asp` and `P9-Pro`. Because a fixed 39-position pseudosequence must skip those three inserted residues, predicted affinities flatten to `~26–30 µM` (`1.1×` separation; `0.540` on real vs. scrambled ligands). `mhcseqs >= 2.9` automatically flags such groove insertions with a warning.

```{figure} _static/whitepaper/figures/fig09_zero_shot_motifs.svg
:alt: Empirical versus zero-shot predicted 9-mer information-content sequence logos for Brushtail possum Trvu-UB*01:01, Cat FLA-E*003:01, Tasmanian devil Saha, and Bat Ptal-N*01:01.
:width: 100%

**Figure 9. Empirical versus zero-shot predicted 9-mer sequence logos.** Observed 9-mer ligands (top row) compared against the top 1% zero-shot predicted 9-mers from `30,000` random background peptides (bottom row) for Brushtail possum `Trvu-UB*01:01`, Cat `FLA-E*003:01`, Tasmanian devil `Saha`, and the Bat `Ptal-N*01:01` 3-aa insertion failure mode.
```

### 3.3 Controlled A/B Pan-Allele Training Parity

In a controlled A/B pan-allele training run (`[1024, 1024]` architecture, `30,760` train / `7,690` held-out test across all `307` trained alleles, seed `42`), `--lock-existing` reproduces legacy `clustalo` training bit-for-bit while adding all `+184` net new vertebrate species (`Figure 10`).

```{figure} _static/whitepaper/figures/fig10_training_parity.svg
:alt: Grouped bar chart comparing controlled A/B pan-allele training arms (Legacy clustalo, mhcseqs with --lock-existing, and unlocked mhcseqs prefer_existing) on held-out AUROC and Pearson correlation.
:width: 82%

**Figure 10. Controlled A/B pan-allele training parity across all 307 trained `2.3.0` alleles.** Locking existing alleles verbatim (`Arm B: mhcseqs + --lock-existing`) reproduces legacy `clustalo` training (`Arm A`) bit-for-bit (`Max |Δ| = 0.0`, `log10 r = 1.000000`, `AUROC = 0.80169`) while expanding the table by `+184` vertebrate species.
```

| Arm | Pseudosequence Table | Held-out AUROC (all) | Chicken `Gaga` AUROC | Max `|Δ|` vs Arm A | `log10 r` vs Arm A |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Arm A** | Legacy `2.3.0` `clustalo` | `0.80169` | `0.57025` | `0.0` | `1.000000` |
| **Arm B** | `mhcseqs` + `--lock-existing` | **`0.80169`** | **`0.57025`** | **`0.0`** | **`1.000000`** |
| **Arm C** | `mhcseqs` (`prefer_existing=True`, unlocking 7 `Gaga-BF`) | `0.83151` | `0.52342` | non-zero | `0.920` |

---

## 4. Audited Livestock & Companion-Animal Validation: Cattle `BoLA-I` & Dog `DLA-88`

Checking only `curated_training_data.affinity.csv.bz2` misses mass-spectrometry ligands that were folded into the training tables shipped inside `models_class1_presentation/models/`. Auditing all three bundled `2.3.0` training tables (`affinity_predictor_train_data.csv.bz2`: `811,134` rows; `processing_predictor_with_flanks_train_data.csv.bz2`: `399,392` rows; `processing_predictor_no_flank_train_data.csv.bz2`: `399,392` rows) cleanly separates in-sample training data from genuinely unseen peptides (`Figure 11`), confirms strong Presentation (+F) ranking on `21,332` strictly unseen Cattle `BoLA-I` ligands against size-matched mismatched-allele nulls and within-peptide scrambled controls (`Figure 12`), and quantifies ultra-few-shot transfer down to `4–12` training rows (`Figure 13`).

```{figure} _static/whitepaper/figures/fig11_cattle_dog_overlap_audit.svg
:alt: Stacked horizontal bar chart showing exact peptide overlap of Cattle BoLA-I and Dog DLA-88 hitlist cohorts against all three bundled 2.3.0 training tables.
:width: 100%

**Figure 11. Mandatory three-table `2.3.0` training-overlap audit for Cattle `BoLA-I` and Dog `DLA-88`.** Every `hitlist` Dog `DLA-88` MS peptide and small IEDB Cattle `BoLA-I` cohort (PMID 36423003) is `100%` in-sample in `affinity_predictor_train_data.csv.bz2` and excluded from held-out claims, whereas `79.5%–87.3%` (`21,606` unique; `21,332` flanked) of `PXD008151` Cattle `BoLA-I` ligands have zero overlap with any of the three bundled `2.3.0` training tables.
```

```{figure} _static/whitepaper/figures/fig12_cattle_bola_disjoint_auroc.svg
:alt: Grouped bar chart of ligand-ranking AUROC on strictly unseen PXD008151 Cattle BoLA-I cohorts comparing size-matched mismatched real-allele null, Affinity, Presentation without flanks, and Presentation with real Bos taurus flanks.
:width: 88%

**Figure 12. Strictly unseen `PXD008151` Cattle `BoLA-I` ligand ranking (+ real *Bos taurus* flanks).** Evaluated on `2,500` strictly unseen flanked ligands per cohort against `25,000` (`10×`) unseen bovine proteome decoys with symmetric real 5-aa flanks. All three cohorts reach `0.902–0.960` Presentation (+F) AUROC, far above their size-matched mismatched real-allele null distributions (`0.691–0.752`, gray bars with 5th–95th percentile whiskers) and achieving `0.833–0.922` AUROC on real vs. residue-scrambled bovine ligands (vs. `0.604–0.731` null).
```

```{figure} _static/whitepaper/figures/fig13_cattle_bola_fewshot_ablation.svg
:alt: Grouped bar chart ablating ultra-few-shot Cattle BoLA alleles (4 to 12 training rows) on strictly unseen bovine ligands.
:width: 86%

**Figure 13. Ultra-few-shot Cattle `BoLA-I` allele ablation (`4–12` training rows).** Adding `BoLA-3*002:01` (only `12` training rows in `2.3.0`) lifts the 2-allele haplotype by `+0.029` Presentation (+F) AUROC (`+0.041` Affinity) and captures `42.8%` of unseen bovine ligands as top binder; `BoLA-4*024:01` (`6` training rows) reaches `0.904` Presentation (+F) AUROC alone, and `BoLA-2*025:01` (`4` training rows) reaches `0.850`.
```

| Cohort (Source) | 2.3.0 Bundled Train Rows | Strictly Unseen (%) | Aff AUROC | Pres (noF) | **Pres (+F)** | Size-Matched Null `[5–95%]` | Real vs. Scrambled *(Null)* | PPV@N (`noF → +F`) |
| :--- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Cattle `BoLA-6*013:01`** (`PXD008151`) | `126` | **5,873 (87.3%)** | 0.950 | 0.957 | **0.960** | `0.691 [0.505, 0.901]` | **0.922** *(0.721)* | 0.726 → 0.725 |
| **Cattle `2*012:01; 3*002:01`** (`PXD008151`) | `64` + `12` | **8,304 (80.8%)** | 0.871 | 0.896 | **0.902** | `0.752 [0.653, 0.811]` | **0.833** *(0.731)* | 0.538 → **0.560** |
| **Cattle `1*023; 2*025; 4*024`** (`PXD008151`) | `24` + `4` + `6` | **7,429 (79.5%)** | 0.892 | 0.915 | **0.918** | `0.691 [0.596, 0.784]` | **0.857** *(0.604)* | 0.570 → **0.594** |
| Cattle `BoLA` (IEDB PMID 36423003) | `21`/`14`/`13` | 0 (0.0%) | — | — | *Excluded (`100%` in-sample)* | — | — | — |
| Dog `DLA-88*501:01` (PMID 27893789) | `1,963` | 0 (0.0%) | — | — | *Excluded (`100%` in-sample; `#490`)* | — | — | — |
| Dog `DLA-88*508:01` (PMID 29475511) | `96` | 0 (0.0%) | — | — | *Excluded (`100%` in-sample; `#490`)* | — | — | — |

---

## 5. Exploratory Boundary Study: Zero-Shot Class I → Class II Transfer

MHC class I (one heavy chain forming both halves of the binding groove) and MHC class II (separate `α` and `β` chains each contributing one half) share the same overall 3D protein fold. Using [`mhcseqs`](https://github.com/pirl-unc/mhcseqs)'s structural IMGT alignment, we mapped MHCflurry's 39 Class I contact positions directly onto their structurally equivalent positions in Class II `HLA-DR`, `HLA-DQ`, and `HLA-DP` heterodimers (`36 / 39` positions directly populated; `3` Class I-only positions padded with `X`).

We tested whether these aligned 39-aa Class II pseudosequences allow **frozen Class I MHCflurry `2.3.0` weights** to predict quantitative Class II 15-mer binding affinity zero-shot across `15` HLA-DR/DQ/DP alleles (`500` unique 15-mers per allele; `7,500` total IC50 nM pairs from [`hitlist`](https://github.com/pirl-unc/hitlist); binder threshold `IC50 ≤ 500 nM`), benchmarked against `NetMHCIIpan-4.3` (BA & EL), a **30-allele mismatched real Class I null distribution**, **20 column-permuted pseudosequences**, and **within-peptide residue-scrambled Class II binders** (`Figures 14–16`).

```{figure} _static/whitepaper/figures/fig14_class_ii_auroc.svg
:alt: Horizontal grouped bar chart comparing per-allele Class II AUROC (IC50 <= 500 nM) for NetMHCIIpan-4.3 BA, frozen Class I MHCflurry 9-mer core, direct 15-mer, and 30-allele mismatched Class I null (median and 5th-95th percentile range) across 15 HLA-DR, DQ, and DP alleles.
:width: 92%

**Figure 14. Exploratory zero-shot Class I → Class II AUROC versus 30-allele mismatched Class I null across 15 HLA-DR, DQ, and DP alleles.** Binary binder classification AUROC (`IC50 ≤ 500 nM`, `500` 15-mers per allele). Frozen Class I MHCflurry `2.3.0` with pan-class 9-mer core scanning averages `0.551` AUROC — statistically indistinguishable from the 30-allele mismatched Class I null (`0.549`, gray bars with 5th–95th percentile whiskers) and far below native `NetMHCIIpan-4.3 BA` (`0.923`). Direct 15-mer scoring drops to `0.235–0.255` on HLA-DP.
```

```{figure} _static/whitepaper/figures/fig15_class_ii_spearman.svg
:alt: Horizontal grouped bar chart comparing continuous Spearman rank correlation against experimental -log10(IC50) across 15 HLA-DR, DQ, and DP alleles alongside the 30-allele mismatched Class I null.
:width: 92%

**Figure 15. Quantitative Class II 15-mer IC50 rank correlation (`Spearman ρ`) versus 30-allele mismatched Class I null across 15 alleles.** Continuous rank correlation against experimental `-log10(IC50)`. Native `NetMHCIIpan-4.3 BA` achieves mean `ρ = +0.801`, whereas frozen Class I 9-mer core scanning achieves `+0.115` (matching the 30-allele mismatched Class I null at `+0.100`, below 20 column-permuted grooves at `+0.146`) and direct 15-mer scoring is systematically anti-correlated (`ρ = -0.224`, down to `-0.521` on `DPA1*01:03/DPB1*04:01`).
```

```{figure} _static/whitepaper/figures/fig16_class_ii_specificity_matrix.svg
:alt: 15x15 cross-allele specificity heatmap showing row-centered AUROC differences when scoring each Class II allele test set with every other Class II pseudosequence on frozen Class I weights.
:width: 78%

**Figure 16. `15 × 15` Class I → Class II cross-allele specificity matrix.** Row-centered AUROC (`AUROC − row mean`) when scoring each true Class II allele test set (rows) with each pan-class Class II pseudosequence (columns) using frozen Class I weights. Strong vertical column stripes rather than a bright diagonal (`mean diag 0.551` vs `mean off-diag 0.556`, `Δ = -0.005`) confirm zero Class II groove specificity.
```

| Method | Scoring mode | Mean AUROC (`≤ 500 nM`) | Mean Spearman `ρ` (`-log10 IC50`) | Real vs. Scrambled Binders AUROC | Interpretation |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **`NetMHCIIpan-4.3 (BA)`** | Native Class II BA | **0.923** | **+0.801** | **0.725** | True Class II positional motif recognition |
| **`NetMHCIIpan-4.3 (EL)`** | Native Class II EL | **0.763** | **+0.493** | — | Eluted-ligand score on binding-affinity assay |
| `20` Column-permuted grooves | 9-mer core (`min IC50`) | `0.571` | `+0.146` | — | Synthetic column-shuffled grooves beat pan-class Class II |
| **Frozen Class I MHCflurry `2.3.0`** | **Pan-class 9-mer core** | `0.551` | `+0.115` | **`0.502`** | Matches mismatched Class I null; coin flip on scrambled binders |
| `30` Mismatched Class I alleles | 9-mer core (`min IC50`) | `0.549` | `+0.100` | `0.514` | Shared hydrophobic-core composition baseline |
| `30` Mismatched Class I alleles | Direct 15-mer | `0.417` | `-0.179` | — | Closed-groove N/C terminus mismatch |
| **Frozen Class I MHCflurry `2.3.0`** | **Pan-class direct 15-mer** | `0.394` | `-0.224` | — | Anti-correlated (down to `0.235 / -0.521` on DP) |

*(See {doc}`experiment_25x_pan_class_ii` for the complete 15-allele breakdown.)*

### Why Zero-Shot Class I → Class II Fails

1. **Within-peptide scrambled binders prove 9-mer core scanning is pure hydrophobicity bias (`0.502` vs `0.725`):** Scanning 9-mer cores lifts AUROC from `0.394` to `0.551` only because Class II binders tend to contain hydrophobic internal 9-mer cores that *any* Class I groove prefers (`30`-allele mismatched Class I null = `0.549`; `20` column-permuted grooves = `0.571`). When amino-acid composition is held constant by comparing real Class II binders (`IC50 ≤ 500 nM`) against `3×` residue-scrambled copies of those exact 15-mers, frozen Class I collapses to **`0.502`** (coin flip across all 15 alleles), while native `NetMHCIIpan-4.3 BA` achieves **`0.725`** (`+0.223`).
2. **Closed-groove vs. open-groove geometry causes strong anti-correlation on HLA-DP (`0.235–0.255`, `ρ = -0.40 to -0.52`):** MHC class I binds 15-mers in a *closed* pocket by pinning both peptide ends and bulging the middle out, whereas MHC class II binds an *open* groove where the 9-mer core sits anywhere inside the peptide with polar/charged ends overhanging freely.
3. **Zero diagonal specificity proves joint training is required:** In `Figure 16`, matching the right Class II pseudosequence (`0.551`) performs slightly *worse* than mismatched Class II alleles (`0.556`, `Δ = -0.005`). While `mhcseqs` provides a clean shared structural alignment across both MHC classes, predicting Class II accurately requires training a future architecture **jointly** on both Class I and Class II data.

---

## 6. Practical Guidance & Reproducibility

### Checklist for Predicting on New Vertebrate Species

1. **Watch for insertion warnings:** Use `extract_class1_pseudosequence_with_alignment()` (or check `mhcflurry` logs). Canonical grooves and small deletions transfer reliably across mammals and birds; multi-residue pocket insertions (like bat `Ptal-N*01:01`) degrade fixed-39-aa accuracy.
2. **Use mismatched-allele null distributions and within-peptide scrambled controls:** Rather than comparing against a single arbitrary human allele (which can accidentally share or oppose the target motif), benchmark new non-model datasets against a diverse panel of unrelated real alleles (size-matched `k`-allele sets for multi-allelic data) and residue-scrambled copies of the positive ligands.
3. **Supply real flanking sequences for MS/cellular tasks:** Real species proteome flanks consistently improve Presentation AUROC and PPV@N even in marsupials (`+0.057` AUROC on Tasmanian devil) and cattle (`+4%` PPV@N).
4. **Audit all three bundled training tables before claiming held-out accuracy:** Check `affinity_predictor_train_data.csv.bz2` and both `processing_predictor_*_train_data.csv.bz2` files inside `models_class1_presentation/models/`.
5. **Do not apply frozen Class I weights to Class II alleles:** Use dedicated Class II predictors (such as `NetMHCIIpan-4.3`) until a jointly trained pan-class MHCflurry model is released.

### Reproducible Scripts & Companion Pages

- **How-to user guide:** {doc}`cross_species_validation`
- **Individual dated experiment logs:** {doc}`experiment_25x_cross_species_hitlist`, {doc}`experiment_25x_cattle_dog_hitlist`, {doc}`experiment_25x_pan_class_ii`, {doc}`dla_validation`.
- **Reproducible benchmark scripts:** [`scripts/experiments/release_2_5_0/`](https://github.com/openvax/mhcflurry/tree/master/scripts/experiments/release_2_5_0).
- **Rebuild the companion PDF:** `python docs/build_whitepaper_pdf.py` (requires `rsvg-convert` and `xelatex`).

---

## 7. References

1. O'Donnell, T. J., Rubinsteyn, A., & Laserson, U. (2020). MHCflurry 2.0: Improved pan-allele prediction of MHC class I-presented peptides by incorporating antigen processing. *Cell Systems*, 11(1), 42–48.e7. <https://doi.org/10.1016/j.cels.2020.06.010>
2. O'Donnell, T. J., Rubinsteyn, A., Bonsack, M., Riemer, A. B., Laserson, U., & Hammerbacher, J. (2018). MHCflurry: Open-source Class I MHC binding affinity prediction. *Cell Systems*, 7(1), 129–132.e4. <https://doi.org/10.1016/j.cels.2018.05.014>
3. Nielsen, M., Connelley, T., & Ternette, N. (2018). Improved prediction of bovine leucocyte antigens (BoLA) presented ligands by use of mass-spectrometry-determined ligand and in vitro binding data (`PXD008151`, PMID 29115832). *Journal of Proteome Research*, 17(1), 559–567.
4. Nilsson, J. B., Kaabinejadian, S., Yari, H., et al. (2023). Accurate prediction of HLA class II antigen presentation across all loci using tailored data acquisition and refined machine learning (`NetMHCIIpan-4.3`). *Science Advances*, 9(47), eadj6367.
5. Lefranc, M.-P., Pommié, C., Kaas, Q., et al. (2005). IMGT unique numbering for immunoglobulin and T cell receptor constant domains and Ig superfamily C-like domains (and G-DOMAINs). *Developmental & Comparative Immunology*, 29(3), 185–203.
6. Caldwell, A., Coleby, R., Tovar, C., et al. (2018). The newly-arisen Devil facial tumour disease 2 (DFT2) reveals a mechanism for the emergence of a contagious cancer (*Sarcophilus harrisii* immunopeptidome). *eLife*, 7, e35314.
7. Wynne, J. W., Weyer, J. L., Shiell, B. J., et al. (2016). Characterization of the MHC class I peptide repertoire of the black flying fox (*Pteropus alecto*, `Ptal-N*01:01`). *Journal of Immunology*, 196(11), 4445–4454.
8. OpenVax / PIRL-UNC immunogenomics ecosystem: [`mhcseqs`](https://github.com/pirl-unc/mhcseqs), [`mhcgnomes`](https://github.com/pirl-unc/mhcgnomes), and [`hitlist`](https://github.com/pirl-unc/hitlist).
