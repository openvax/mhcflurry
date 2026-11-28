# Whitepaper: Pan-Vertebrate MHC Class I Generalization and Exploratory Pan-Class Transfer in MHCflurry 2.5

> **Companion PDF:** [Download Publication-Ready PDF (vector figures, tables & references)](_static/whitepaper/mhcflurry_25x_cross_species_whitepaper.pdf)
>
> **Software & Data:** `mhcflurry 2.5.x` · frozen `2.3.0` PyTorch weights (`models_class1_presentation`) · [`mhcseqs`](https://github.com/pirl-unc/mhcseqs) `2.9–2.14` · [`mhcgnomes`](https://github.com/pirl-unc/mhcgnomes) `3.79–3.93` · [`hitlist`](https://github.com/pirl-unc/hitlist) (Nov 2026 bundles), IEDB Class I & II, ProteomeXchange `PXD008151` · **Date:** November 2026

---

## Abstract & Key Findings

MHCflurry's pan-allele models predict peptide binding from a compact **pseudosequence** — the 39 amino-acid positions of an MHC class I protein that line its peptide-binding pocket and contact the peptide. Because MHCflurry 2.5 integrates the optional [`mhcseqs`](https://github.com/pirl-unc/mhcseqs) sequence database and structural aligner alongside [`mhcgnomes`](https://github.com/pirl-unc/mhcgnomes) allele-name parsing, complete Class I coverage expands nearly five-fold — from **47 vertebrate species** (`20,205` alleles; `10` trained species / `308` trained alleles in `2.3.0`) to **223–231 vertebrate species** (`26,156–26,549` complete binding-groove sequences across 14 vertebrate clades) — **without retraining or altering existing `2.3.0` model weights**.

This whitepaper synthesizes all three `2.5.x` validation campaigns into one empirical evaluation: (1) zero-shot Class I binding affinity and antigen presentation on newly added species and engineered pocket mutants, (2) a mandatory training-overlap audit and unseen-peptide evaluation on cattle (`BoLA-I`) and dog (`DLA-88`) mass-spectrometry datasets, and (3) an exploratory test of whether frozen Class I weights can predict Class II binding.

- **Broad zero-shot Class I transfer:** Across `16` unseen-species and binding-pocket mutant cohorts (`0` training rows), frozen `2.3.0` weights achieve **`0.77–0.98` AUROC** across carnivores (cat `0.977`, giant panda `0.883`), horses (`0.967`), birds (mallard duck `0.893–0.938`), marsupials (brushtail possum `0.869`, Tasmanian devil `0.806` with flanks), and single-residue pocket mutants (pig `R156A` `Δ = +0.010`, cat `W167S` `Δ = +0.006`).
- **Human-trained antigen processing transfers across species:** On Tasmanian devil (*Sarcophilus harrisii*, `~160 Myr` diverged from placental mammals), MHCflurry's human-trained processing model alone achieves **`0.750` AUROC** using real Tasmanian devil source-protein flanking sequences, lifting combined Presentation from **`0.749 → 0.806` AUROC** (`+0.057`, `+30%` relative PPV@N).
- **Mandatory three-table overlap audit on Cattle `BoLA-I` & Dog `DLA-88`:** Checking all three training tables bundled inside `2.3.0` (`811,134` affinity + `399,392` processing rows) catches **`100%` in-sample overlap** on `hitlist` Dog `DLA-88` and small IEDB Cattle `BoLA-I` cohorts, while isolating **`21,332` strictly unseen flanked Cattle `BoLA-I` MS ligands** (`PXD008151`) reaching **`0.902–0.960` Presentation (+F) AUROC** and confirming strong ultra-few-shot transfer (`4–12` training rows).
- **Clean negative result on zero-shot Class I → Class II transfer:** Mapping `36 / 39` structurally equivalent Class II `DR`/`DQ`/`DP` contact positions onto frozen Class I weights yields **`0.551` AUROC** (9-mer core scanning) — **worse** than an unrelated `HLA-A*02:01` negative control (`0.598`) and far below `NetMHCIIpan-4.3 BA` (`0.923`), with anti-correlated direct 15-mer scores (`0.394`, down to `0.235` on HLA-DP) and zero allele specificity (`Δ = -0.005`). Pan-class prediction requires **joint pan-class training**.
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

Complete Class I coverage expands from `47` species in `2.3.0` to `227` species in `mhcseqs 2.12` (`231` union, `26,549` complete grooves, **`+184` net new vertebrate species** across 14 clades): bony fish `2→53`, birds `1→42`, other mammals `0→28`, reptiles `0→17`, amphibians `0→17`, rodents `2→15`, marsupials `0→7`, sharks & rays `0→4`, bats `0→4`, and cetaceans `0→2`.

```{figure} _static/release-2.5.7/fig1_species_and_validation_landscape.svg
:alt: Taxonomic expansion of complete Class I binding grooves (+184 net new species across 14 vertebrate clades) paired with a heatmap of hitlist Class I validation datasets by species and experimental modality.
:width: 100%

**Figure 1. Taxonomic expansion of complete Class I grooves and `hitlist` validation data landscape.** *Left:* Distinct vertebrate species with complete Class I grooves in MHCflurry `2.3.0` (`47` species, `10` trained) versus MHCflurry `2.5.x` + `mhcseqs 2.12` (`227` species / `231` union; `+184` net new species across 14 clades). *Right:* Experimental Class I validation data availability in [`hitlist`](https://github.com/pirl-unc/hitlist) by species and modality across 10 unseen vertebrate species, 2 unseen allele/mutant cohorts in trained species, and 2 trained MS reference controls.
```

---

## 2. Zero-Shot Class I Affinity & Antigen Presentation Across Unseen Clades

### 2.1 Zero-Shot Presentation (+ Real Proteome Flanks) on Cellular MS Elutions

MHCflurry's presentation model combines pan-allele binding affinity with a human-trained antigen-processing network that examines each peptide and its neighboring source-protein flanking residues. To test whether this human-trained processing model generalizes to distant vertebrates, we took cellular mass-spectrometry (MS) datasets in [`hitlist`](https://github.com/pirl-unc/hitlist) with matched host proteomes (*Sarcophilus harrisii*, *Canis lupus*, *Macaca mulatta*), paired eluted ligands with `10:1` length-matched decoys carrying real flanking sequences from the same host proteome, and scored frozen `2.3.0` weights with zero retraining.

```{figure} _static/release-2.5.7/fig2_hitlist_presentation_vs_affinity.svg
:alt: Zero-shot Presentation with and without real species proteome flanks versus Affinity and Processing alone on hitlist MS elutions for Tasmanian devil and trained Dog and Rhesus reference alleles.
:width: 100%

**Figure 2. Zero-shot Presentation (+ real species proteome flanks) versus Affinity on `hitlist` MS elutions.** AUROC with 95% stratified bootstrap confidence intervals against `10:1` same-proteome length-matched decoys for zero-shot Tasmanian devil (*Sarcophilus harrisii*, `0` training rows, `11` UniProt grooves, `15,301` mapped ligands) alongside trained Dog (`DLA-88`) and Rhesus macaque (`Mamu-B`) MS reference controls.
```

| Dataset (`hitlist` MS) | Status | Mapped Pos (Eval / Total) | Affinity AUROC | Processing (+F) AUROC | Pres (noF) AUROC | **Pres (+ real flanks)** | Δ (Pres+F − Aff) | PPV@N (Aff → Pres+F) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Tasmanian devil `Saha`** (11 grooves) | **0-shot Marsupialia** | 2,500 / 15,301 | 0.749 | **0.750** | 0.791 | **0.806** | **+0.057** | **0.273 → 0.356** (+30%) |
| Dog `DLA-88*501:01` | Trained ref* | 1,301 / 1,301 | 0.933 | 0.837 | 0.942 | **0.947** | +0.014 | 0.711 → 0.746 |
| Dog `DLA-88*034:01` | Trained ref | 355 / 355 | 0.959 | 0.875 | 0.966 | **0.970** | +0.011 | 0.752 → 0.780 |
| Rhesus `Mamu-B*008:01` | Trained ref | 874 / 874 | 0.968 | 0.866 | 0.976 | **0.979** | +0.011 | 0.852 → 0.874 |

*\*As shown in Section 4, IEDB `DLA-88*501:01` overlaps `2.3.0`'s bundled affinity training table; these rows serve strictly as trained positive-control ceilings.*

Even across `~160 Myr` of divergence from placental mammals to marsupials, MHCflurry's human-trained processing model alone achieves **`0.750` AUROC** on Tasmanian devil peptides with real *S. harrisii* flanks and lifts combined Presentation by **`+0.057` AUROC** (`+30%` relative PPV@N).

### 2.2 Experimental `+/-` Binding Assays & 16-Cohort Zero-Shot Ranking

In-vitro binding assays (refolding, purified-MHC fluorescence screens, and synthetic peptide panels) bypass cellular antigen processing and rarely map to a host proteome (`19 / 448 = 4.2%` in `hitlist/binding.parquet`; `0%` for the `5,607` synthetic random-library `Trvu-UB*01:01` peptides). For these datasets, **Affinity** is the primary metric — evaluated both on measured positive-versus-negative binders (no synthetic decoys) and on `10:1` background ligand ranking, paired with an unrelated human negative control (`HLA-A*02:01`) scored on identical peptides to catch peptide-composition bias.

```{figure} _static/release-2.5.7/fig3_hitlist_zeroshot_binding_and_ligands.svg
:alt: Experimental positive versus negative binding assay AUROC without synthetic decoys and zero-shot Affinity versus Presentation no-flank ligand ranking AUROC across unseen vertebrate species and pocket mutants.
:width: 100%

**Figure 3. Experimental `+/-` binding assays and broad zero-shot ligand ranking across unseen vertebrates.** *Left:* Experimental positive-versus-negative binding assay AUROC (no synthetic decoys) from `hitlist/binding.parquet`. *Right:* Zero-shot Affinity (solid) and no-flank Presentation (hatched) ligand ranking AUROC against `10:1` length-matched background decoys across 11 unseen vertebrate species and 3 engineered binding-pocket mutants.
```

| Unseen Species / Allele (Clade) | Validation Data Available | Ligand Ranking AUROC: Aff / Pres (noF) | `HLA-A*02:01` Ctrl | Exp. `+/-` Binding AUROC *(no decoys)* | Notes / Caveats |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **Cat `FLA-E*018:01`** (Carnivora) | 24 direct/bind + 16 `+/-` (`3+/13-`) | **0.977** / **0.966** | 0.636 | 0.308 *(3+/13-)* | `155 vs 16,992 nM`; full IEDB `24+/12-` gave `0.684` |
| **Cat `FLA-E*018:01 W167S`** (Mutant) | 12 mutant ligands | **0.976** / **0.981** | 0.642 | — | Mutant groove beats WT (`Δ = +0.006`) |
| **Cat `FLA-E*003:01` & `FLA-K*007:01`** | 1,612 & 491 MS ligands (IEDB) | **0.830** & **0.880** | 0.562 / 0.659 | — | Recovers P2-Pro & P9-Arg/Lys anchor motif zero-shot |
| **Horse `Eqca-N*006:02`** (Unseen allele) | 6+ / 10− binding | **0.967** / **0.956** | 0.611 | 0.217 *(6+/10-)* | Single-mutant scan negatives (`n=16`) |
| **Pig `SLA-1*04:01 R156A`** (Mutant) | 141 mutant ligands | **0.951** / **0.946** | 0.706 | — | Mutant groove beats WT (`Δ = +0.010`) |
| **Mallard duck `Anpl-UAA*01`** (Aves) | 31 binding+ | **0.938** / **0.920** | 0.862 | — | Indel-bearing avian groove; high ctrl baseline |
| **Pig `SLA-1*13:01 F99Y`** (Mutant) | 60 mutant ligands | **0.898** / **0.892** | 0.639 | — | `227 vs 5,543 nM` |
| **Mallard duck `Anpl-UAA*76`** (Aves) | 30 direct ligands | **0.893** / **0.856** | 0.564 | — | Strong `+0.329` lift over ctrl across `~310 Myr` |
| **Giant panda `Aime-128`** (Carnivora) | 21 direct/bind (PDB `7DC6`) | **0.883** / **0.906** | 0.718 | — | `432 vs 15,828 nM` |
| **Brushtail possum `Trvu-UB*01:01`** (Marsupial) | 5,607 direct/fluo + 31 `+/-` (`14+/17-`) | **0.869** / **0.849** | 0.570 | **0.655** *(14+/17-)* | `+0.299` over ctrl (`1,182 vs 5,237 nM` on `+/-`) |
| **Cynomolgus `Mafa-A1*063:01`** (NHP) | 56 `+/-` binding (`54+/2-`) | **0.868** / **0.830** | 0.666 | **0.889** *(54+/2-)* | Ctrl `0.157` on `+/-` (`1,325 vs 9,663 nM`) |
| **African clawed frog `Xela-UAAg`** (Amphibia) | 26 direct/bind + 19 `+/-` (`14+/5-`) | 0.826 / 0.828 | 0.810 | 0.357 *(14+/5-)* | Caveat: matches ctrl (composition bias + 2 indels) |
| **Rabbit `RLA-A*01`** (Lagomorpha) | 27 `+/-` binding (`11+/16-`) | 0.787 / 0.792 | 0.969 | 0.562 *(11+/16-)* | Shares P2/P9 hydrophobic motif with `HLA-A*02:01` |
| **Tasmanian devil `Saha`** (Marsupial) | 23,973 unique MS (`15,301` flanked) | **0.772** / **0.803** *(0.806 +F)* | 0.666 | — | Multi-allelic (`11` UniProt grooves) |
| **Norway rat `RT1-A*a`** (Rodentia) | 5 binding+ (24 obs) | 0.720 / **0.940** | 0.728 | — | Tiny sample (`n=5`) |
| **Black flying fox `Ptal-N*01:01`** (Bat) | 86 `+/-` binding (`37+/43-`) | 0.593 / 0.717 | 0.520 | **0.606** *(37+/43-)* | Failure mode: 3-aa `MDL` N-terminal pocket insertion (warned) |

---

## 3. Groove Specificity, Motif Recovery, Failure Modes & Training Parity

### 3.1 Negative-Control Separation & Empirical vs Zero-Shot Sequence Logos

Comparing each unseen cohort against the `HLA-A*02:01` negative control (`Figure 4`, left) separates true allele-specific transfer (`+0.356` horse, `+0.341` cat `FLA-E*018:01`, `+0.329` duck `Anpl-UAA*76`, `+0.299` possum) from peptide-composition artifacts (`Xela-UAAg` frog: `0.826` vs `0.810` control).

```{figure} _static/release-2.5.8/fig4_zeroshot_vs_control_and_parity.svg
:alt: Groove-specific zero-shot transfer compared to an HLA-A*02:01 negative control across all 16 unseen-species and mutant cohorts, paired with the controlled A/B pan-allele training parity comparison.
:width: 100%

**Figure 4. Groove-specific transfer across 16 unseen cohorts and controlled A/B pan-allele training parity.** *Left:* Zero-shot target-groove AUROC versus `HLA-A*02:01` negative control on identical peptides across all 16 unseen-species and pocket-mutant datasets, sorted by `Δ = Target − Control`. *Right:* Controlled A/B pan-allele training parity across all `307` trained `2.3.0` alleles comparing legacy `clustalo` (Arm A) against `mhcseqs --lock-existing` (Arm B: `Max |Δ| = 0.0`, `r = 1.000000`) and unlocked `Gaga-BF` resolution (Arm C).
```

Comparing experimentally observed 9-mer ligands against the top 1% of `30,000` random 9-mers scored zero-shot (`Figure 5`) shows both why pan-allele transfer works and where fixed 39-aa pseudosequences reach their limit:

- **Feline & marsupial anchor recovery:** With zero feline training data, `FLA-E*003:01` recovers both its empirical position-2 proline (`P2-Pro`) and position-9 basic (`P9-Arg/Lys`) preferences; brushtail possum `Trvu-UB*01:01` and Tasmanian devil `Saha` recover their aromatic/hydrophobic C-terminal (`P9`) anchors across `~160 Myr`.
- **Single-residue pocket mutants:** Extracting mature-numbered point mutations via `mhcseqs` improves discrimination over the corresponding wild-type allele on mutant-specific ligands for both Pig `SLA-1*04:01 R156A` (`ΔAUROC = +0.010`, 95% CI `[+0.007, +0.013]`) and Cat `FLA-E*018:01 W167S` (`ΔAUROC = +0.006`, 95% CI `[+0.000, +0.015]`).
- **Honest failure mode — Bat `Ptal-N*01:01` (3-aa `MDL` insertion):** This bat allele carries three extra amino acids (`MDL`) inserted at position 51 inside the N-terminal binding pocket (the A-pocket), reshaping its preference toward `P1-Asp` and `P9-Pro`. Because a fixed 39-position pseudosequence must skip those three inserted residues, predicted affinities flatten to `~26–30 µM` (`1.1×` separation). `mhcseqs >= 2.9` automatically flags such groove insertions with a warning.

```{figure} _static/release-2.5.0/zero_shot_motifs.svg
:alt: Empirical versus zero-shot predicted 9-mer information-content sequence logos for Brushtail possum Trvu-UB*01:01, Cat FLA-E*003:01, Tasmanian devil Saha, and Bat Ptal-N*01:01.
:width: 100%

**Figure 5. Empirical versus zero-shot predicted 9-mer sequence logos.** Observed 9-mer ligands (top row) compared against the top 1% zero-shot predicted 9-mers from `30,000` random background peptides (bottom row) for Brushtail possum `Trvu-UB*01:01`, Cat `FLA-E*003:01`, Tasmanian devil `Saha`, and the Bat `Ptal-N*01:01` insertion failure mode.
```

### 3.2 Controlled A/B Pan-Allele Training Parity

In a controlled A/B pan-allele training run (`[1024, 1024]` architecture, `30,760` train / `7,690` held-out test across all trained alleles, seed `42`), `--lock-existing` reproduces legacy `clustalo` training bit-for-bit while adding all `+184` net new vertebrate species:

| Arm | Pseudosequence Table | Held-out AUROC (all) | Chicken `Gaga` AUROC | Max `|Δ|` vs Arm A | `log10 r` vs Arm A |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Arm A** | Legacy `2.3.0` `clustalo` | `0.80169` | `0.57025` | `0.0` | `1.000000` |
| **Arm B** | `mhcseqs` + `--lock-existing` | **`0.80169`** | **`0.57025`** | **`0.0`** | **`1.000000`** |
| **Arm C** | `mhcseqs` (`prefer_existing=True`, unlocking 7 `Gaga-BF`) | `0.83151` | `0.52342` | non-zero | `0.920` |

---

## 4. Audited Livestock & Companion-Animal Validation: Cattle `BoLA-I` & Dog `DLA-88`

Checking only `curated_training_data.affinity.csv.bz2` misses mass-spectrometry ligands that were folded into the training tables shipped inside `models_class1_presentation/models/`. Auditing all three bundled `2.3.0` training tables (`affinity_predictor_train_data.csv.bz2`: `811,134` rows; `processing_predictor_with_flanks_train_data.csv.bz2`: `399,392` rows; `processing_predictor_no_flank_train_data.csv.bz2`: `399,392` rows) cleanly separates in-sample training data from genuinely unseen peptides (`Figure 6`).

```{figure} _static/release-2.5.13/fig_cattle_dog_hitlist_validation.svg
:alt: Three-panel Cattle BoLA-I and Dog DLA hitlist audit and validation figure.
:width: 100%

**Figure 6. Mandatory three-table overlap audit and strictly unseen Cattle `BoLA-I` / Dog `DLA-88` validation.** **A.** Exact peptide overlap against all three bundled `2.3.0` training tables: every `hitlist` Dog `DLA-88` MS peptide and small IEDB Cattle `BoLA-I` cohort (PMID 36423003) is `100%` in-sample and excluded from held-out claims, whereas `79.5%–87.3%` (`21,606` unique; `21,332` flanked) of `PXD008151` Cattle `BoLA-I` have zero sequence overlap with any `2.3.0` training table. **B.** Unseen `PXD008151` `BoLA-I` ranking against `10×` unseen *Bos taurus* proteome decoys with real 5-aa flanks. **C.** Ultra-few-shot `BoLA` allele ablation (`4–12` training rows).
```

| Cohort (Source) | 2.3.0 Bundled Train Rows | Strictly Unseen (%) | Aff AUROC | Pres (noF) AUROC | **Pres (+F) AUROC** | `HLA-A*02:01` Ctrl | PPV@N (`noF → +F`) |
| :--- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Cattle `BoLA-6*013:01`** (`PXD008151`) | `126` | **5,873 (87.3%)** | 0.950 | 0.957 | **0.960** (`+0.003`) | 0.789 | 0.726 → 0.725 |
| **Cattle `BoLA-2*012:01; 3*002:01`** (`PXD008151`) | `64` + `12` | **8,304 (80.8%)** | 0.871 | 0.896 | **0.902** (`+0.006`) | 0.648 | 0.538 → **0.560** |
| **Cattle `BoLA-1*023:01; 2*025:01; 4*024:01`** (`PXD008151`) | `24` + `4` + `6` | **7,429 (79.5%)** | 0.892 | 0.915 | **0.918** (`+0.004`) | 0.628 | 0.570 → **0.594** |
| Cattle `BoLA` (IEDB PMID 36423003) | `21`/`14`/`13` | 0 (0.0%) | — | — | *Excluded (`100%` in-sample)* | — | — |
| Dog `DLA-88*501:01` (IEDB PMID 27893789) | `1,963` | 0 (0.0%) | — | — | *Excluded (`100%` in-sample; `#490`)* | — | — |
| Dog `DLA-88*508:01` (IEDB PMID 29475511) | `96` | 0 (0.0%) | — | — | *Excluded (`100%` in-sample; `#490`)* | — | — |

In ultra-few-shot haplotype ablations (`Figure 6C`), adding `BoLA-3*002:01` (only `12` training rows in `2.3.0`) lifts the 2-allele haplotype by `+0.041` Affinity AUROC (`+0.029` Presentation +F) and captures `42.8%` of unseen bovine ligands as top binder; `BoLA-4*024:01` (only `6` training rows) reaches `0.904` Presentation (+F) AUROC on its own, and `BoLA-2*025:01` (`4` training rows) reaches `0.850`.

---

## 5. Exploratory Boundary Study: Zero-Shot Class I → Class II Transfer

MHC class I (one heavy chain forming both halves of the binding groove) and MHC class II (separate `α` and `β` chains each contributing one half) share the same overall 3D protein fold. Using [`mhcseqs`](https://github.com/pirl-unc/mhcseqs)'s structural IMGT alignment, we mapped MHCflurry's 39 Class I contact positions directly onto their structurally equivalent positions in Class II `HLA-DR`, `HLA-DQ`, and `HLA-DP` heterodimers (`36 / 39` positions directly populated; `3` Class I-only positions padded with `X`).

We tested whether these aligned 39-aa Class II pseudosequences allow **frozen Class I MHCflurry `2.3.0` weights** to predict quantitative Class II 15-mer binding affinity zero-shot across `15` HLA-DR/DQ/DP alleles (`500` unique 15-mers per allele; `7,500` total IC50 nM pairs from [`hitlist`](https://github.com/pirl-unc/hitlist); binder threshold `IC50 ≤ 500 nM`), benchmarked against `NetMHCIIpan-4.3` (BA & EL) and an unrelated `HLA-A*02:01` negative control in both **9-mer core scanning** (best predicted affinity across the 7 overlapping 9-mer substrings) and **direct 15-mer** scoring (`Figure 7`).

```{figure} _static/release-2.5.9/fig_pan_class_ii_experiment.svg
:alt: Three-panel benchmark comparing NetMHCIIpan-4.3 BA against frozen Class I MHCflurry 2.3 using mhcseqs pan-class IMGT 39-aa Class II pseudosequences (9-mer core and direct 15-mer) and an HLA-A*02:01 negative control across 15 HLA-DR, DQ, and DP alleles, plus the 15x15 cross-allele specificity matrix.
:width: 100%

**Figure 7. Exploratory zero-shot Class I → Class II pan-class transfer versus `NetMHCIIpan-4.3`.** **A, B.** Per-allele AUROC (`IC50 ≤ 500 nM`) and continuous Spearman rank correlation (`ρ` against `-log10 IC50`) across 15 HLA-DR, DQ, and DP alleles (`500` 15-mers each). **C.** `15 × 15` cross-allele specificity matrix (AUROC minus row mean under pan-class 9-mer core scoring), displaying column-dominated vertical bands rather than diagonal allele specificity (`mean diag 0.551` vs `mean off-diag 0.556`, `Δ = -0.005`).
```

| Method | Scoring mode | Mean AUROC (`≤ 500 nM`) | Mean Spearman `ρ` (`-log10 IC50`) | Interpretation |
| :--- | :--- | ---: | ---: | :--- |
| **`NetMHCIIpan-4.3 (BA)`** | Native Class II BA | **0.923** | **+0.801** | Reference ceiling on quantitative 15-mer IC50 |
| **`NetMHCIIpan-4.3 (EL)`** | Native Class II EL | **0.763** | **+0.493** | Eluted-ligand score on binding-affinity assay |
| `HLA-A*02:01` negative control | 9-mer core (`min IC50`) | `0.598` | `+0.187` | Shared hydrophobic-core composition baseline |
| **Frozen Class I MHCflurry `2.3.0`** | **Pan-class 9-mer core** | `0.551` | `+0.115` | Below `HLA-A*02:01` control (`Δ = -0.047`) |
| `HLA-A*02:01` negative control | Direct 15-mer | `0.430` | `-0.146` | Closed-groove N/C terminus mismatch |
| **Frozen Class I MHCflurry `2.3.0`** | **Pan-class direct 15-mer** | `0.394` | `-0.224` | Anti-correlated (down to `0.235 / -0.521` on DP) |

*(See {doc}`experiment_25x_pan_class_ii` for the complete 15-allele breakdown.)*

### Why Zero-Shot Class I → Class II Fails

1. **9-mer core scanning picks up generic hydrophobicity, not Class II allele specificity:** Scanning 9-mer cores lifts AUROC from `0.394` to `0.551` only because Class II binders tend to contain hydrophobic internal 9-mer cores that Class I models also prefer — yet the unrelated `HLA-A*02:01` negative control scores *higher* (`0.598`) on the exact same peptides.
2. **Closed-groove vs. open-groove geometry causes strong anti-correlation on HLA-DP (`0.235–0.255`, `ρ = -0.40 to -0.52`):** MHC class I binds 15-mers in a *closed* pocket by pinning both peptide ends and bulging the middle out, whereas MHC class II binds an *open* groove where the 9-mer core sits anywhere inside the peptide with polar/charged ends overhanging freely.
3. **Zero diagonal specificity proves joint training is required:** In `Figure 7C`, matching the right Class II pseudosequence (`0.551`) performs slightly *worse* than mismatched Class II alleles (`0.556`, `Δ = -0.005`). While `mhcseqs` provides a clean shared structural alignment across both MHC classes, predicting Class II accurately requires training a future architecture **jointly** on both Class I and Class II data.

---

## 6. Practical Guidance & Reproducibility

### Checklist for Predicting on New Vertebrate Species

1. **Watch for insertion warnings:** Use `extract_class1_pseudosequence_with_alignment()` (or check `mhcflurry` logs). Canonical grooves and small deletions transfer reliably across mammals and birds; multi-residue pocket insertions (like bat `Ptal-N*01:01`) degrade fixed-39-aa accuracy.
2. **Include an `HLA-A*02:01` negative control on small datasets:** Genuine allele-specific transfer yields a clear positive AUROC margin (`Δ > 0.10`), whereas tied scores (like frog `Xela-UAAg`) warn of peptide-composition bias.
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
