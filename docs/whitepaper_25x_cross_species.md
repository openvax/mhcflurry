# Whitepaper: Pan-Vertebrate MHC Class I Generalization & Exploratory Pan-Class Transfer in MHCflurry 2.5

> **Companion PDF:** [Download Publication-Ready PDF (vector figures, tables & references)](_static/whitepaper/mhcflurry_25x_cross_species_whitepaper.pdf)
>
> **Software & Data:** `mhcflurry 2.5.x` · frozen `2.3.0` PyTorch weights (`models_class1_presentation`) · `mhcseqs 2.9–2.14` · `mhcgnomes 3.79–3.93` · [`hitlist`](https://github.com/pirl-unc/hitlist) (Nov 2026 bundles), IEDB Class I & II, ProteomeXchange `PXD008151` · **Date:** November 2026

---

## Abstract & Key Findings

MHCflurry 2.5 integrates [`mhcseqs`](https://github.com/pirl-unc/mhcseqs) IMGT G-DOMAIN structural alignment and [`mhcgnomes`](https://github.com/pirl-unc/mhcgnomes) sequence-resolved nomenclature directly into MHCflurry's pan-allele pipeline, expanding complete Class I binding-groove coverage nearly five-fold—from **47 vertebrate species** (`20,205` alleles; `10` trained species / `308` alleles in `2.3.0`) to **223–231 vertebrate species** (`26,156–26,549` complete grooves across 14 vertebrate clades)—**without retraining or altering existing `2.3.0` model weights**.

This technical whitepaper synthesizes all three `2.5.x` empirical validation campaigns into a single unified evaluation of (1) zero-shot Class I binding affinity and antigen presentation on newly added vertebrate clades and engineered pocket mutants, (2) a mandatory three-table training-overlap audit and strictly peptide-disjoint evaluation of livestock (`BoLA-I`) and companion-animal (`DLA-88`) mass-spectrometry cohorts, and (3) an exploratory Class I → Class II structural boundary test against `NetMHCIIpan-4.3`:

- **Broad zero-shot Class I transfer:** Across `16` unseen-species and binding-pocket mutant cohorts (`0` training rows), frozen `2.3.0` weights achieve **`0.77–0.98` AUROC** across Carnivora (Cat `0.977`, Giant panda `0.883`), Perissodactyla (Horse `0.967`), Aves (Mallard duck `0.893–0.938`), Marsupialia (Brushtail possum `0.869`, Tasmanian devil `0.806 +F`), and engineered pocket mutants (Pig `R156A` `Δ = +0.010`, Cat `W167S` `Δ = +0.006`).
- **Cross-clade transfer of human-trained antigen processing:** On Tasmanian devil (*Sarcophilus harrisii*, `~160 Myr` diverged from placentals), MHCflurry's human-trained processing model alone achieves **`0.750` AUROC** with real marsupial flanks and lifts combined Presentation from **`0.749 → 0.806` AUROC** (`+0.057`, `+30%` relative PPV@N).
- **Mandatory three-table overlap audit on Cattle `BoLA-I` & Dog `DLA-88`:** Auditing against all three bundled `2.3.0` training tables (`811,134` affinity + `399,392` processing rows) catches **`100%` in-sample overlap** on `hitlist` Dog `DLA-88` and small IEDB `BoLA-I` cohorts, while isolating **`21,332` strictly peptide-disjoint flanked Cattle `BoLA-I` MS ligands** (`PXD008151`) reaching **`0.902–0.960` Presentation (+F) AUROC** and confirming strong ultra-few-shot transfer (`4–12` training rows).
- **Clean negative result on zero-shot Class I → Class II transfer:** Mapping `36 / 39` Class II `DR`/`DQ`/`DP` groove positions via pan-class IMGT alignments onto frozen Class I weights yields **`0.551` AUROC** (9-mer core)—**below** an unrelated `HLA-A*02:01` negative control (`0.598`) and `NetMHCIIpan-4.3 BA` (`0.923`)—with anti-correlated direct 15-mer scores (`0.394`, down to `0.235` on HLA-DP) and zero diagonal specificity (`Δ = -0.005`), proving that pan-class prediction requires **joint pan-class training**.
- **Bit-identical training parity:** Locking existing trained alleles via `--lock-existing` reproduces legacy `clustalo` pan-allele training with **`Max |Δ| = 0.0`** and **`Pearson r = 1.000000`** across all `308` trained alleles.

---

## 1. Introduction & Pan-Vertebrate Architecture in MHCflurry 2.5

Pan-allele Class I MHC predictors represent each MHC allele as a fixed-length *pseudosequence* of `39` polymorphic peptide-contacting residues lining the α1/α2 binding groove. In MHCflurry `2.0–2.3`, constructing this table required external `clustalo` alignments against curated anchors, covering `47` species (`20,205` alleles; training measurements concentrated in only `10` species and `308` alleles) with no runtime mechanism to score novel heavy chains, uncurated UniProt sequences, or binding-pocket mutants.

MHCflurry `2.5` adds four capabilities without altering any released `2.3.0` weights:

1. **Direct IMGT G-DOMAIN groove extraction (`mhcseqs`):** Installing `pip install "mhcflurry[mhcseqs]>=2.5,<2.6"` delegates signal-peptide trimming, α1/α2 domain decomposition, and mature-numbered 39-aa pseudosequence extraction to `mhcseqs`'s Kabsch-calibrated IMGT aligner (`add_allele_from_sequence`, `extend_alleles_from_mhcseqs`, and `mhcflurry pseudosequences`).
2. **Explicit structural indel diagnostics:** When `mhcseqs >= 2.9` detects non-canonical binding-groove insertions or deletions (such as bat or marsupial α1 loop insertions), MHCflurry exposes a structured `GrooveAlignment` object and emits a warning rather than silently returning an out-of-register pseudosequence.
3. **Unified sequence-resolved nomenclature (`mhcgnomes.AccessionedAllele`):** Since `mhcflurry 2.5.9`, canonical sequence-backed non-model identifiers (e.g., `TriakisScyllium-UAA*uniprot:O46817`, `Saha-classI*uniprot:G3V1L5`) flow cleanly through `normalize_allele_name` and `Class1AffinityPredictor` under `normalize_alleles=True` (`26,156` alleles across `223` canonical vertebrate species, including `2,884` `AccessionedAllele`s and `0` unparsed rows).
4. **Zero-regression parity gate:** Across `19,886` overlapping non-`X` alleles in `2.3.0`, `mhcseqs` achieves **`99.45%` exact 39/39 match** (`19,777 / 19,886`) and **`99.98%` per-residue identity** (`775,385 / 775,554`). `downloads-generation/allele_sequences/GENERATE.sh` defaults to `mhcseqs --lock-existing` paired with `verify_trained_allele_parity.py` enforcing `0` diffs across all `308` trained alleles.

Complete Class I groove coverage expands from `47` species in `2.3.0` to `227` species in `mhcseqs 2.12` (`231` union, `26,549` complete grooves, **`+184` net new vertebrate species** across 14 clades): bony fish `2→53`, birds `1→42`, other mammals `0→28`, reptiles `0→17`, amphibians `0→17`, rodents `2→15`, marsupials `0→7`, sharks & rays `0→4`, bats `0→4`, and cetaceans `0→2`.

```{figure} _static/release-2.5.7/fig1_species_and_validation_landscape.svg
:alt: Taxonomic expansion of complete Class I binding grooves (+184 net new species across 14 vertebrate clades) paired with a heatmap of hitlist Class I validation datasets by species and experimental modality.
:width: 100%

**Figure 1. Taxonomic expansion of complete Class I grooves and `hitlist` validation data landscape.** *Left:* Distinct vertebrate species with complete Class I grooves in MHCflurry `2.3.0` (`47` species, `10` trained) versus MHCflurry `2.5.x` + `mhcseqs 2.12` (`227` species / `231` union; `+184` net new species across 14 clades). *Right:* Experimental Class I validation data availability in `hitlist` by species and modality across 10 unseen vertebrate species, 2 unseen allele/mutant cohorts in trained species, and 2 trained MS reference controls.
```

---

## 2. Zero-Shot Class I Affinity & Antigen Presentation Across Unseen Clades

### 2.1 Zero-Shot Presentation (+ Real Proteome Flanks) on Cellular MS Elutions

MHCflurry's presentation model pairs pan-allele binding affinity with a human-trained convolutional antigen-processing predictor that uses upstream and downstream flanking sequences from the source protein. For `hitlist` MS immunopeptidomics datasets with verified host reference proteomes (*Sarcophilus harrisii*, *Canis lupus*, *Macaca mulatta*), we evaluated mapped eluted ligands against `10:1` length-matched decoys sampled from the same host proteome with symmetric real flanking sequences, scoring frozen `2.3.0` weights with zero retraining.

```{figure} _static/release-2.5.7/fig2_hitlist_presentation_vs_affinity.svg
:alt: Zero-shot Presentation with and without real species proteome flanks versus Affinity and Processing alone on hitlist MS elutions for Tasmanian devil and trained Dog and Rhesus reference alleles.
:width: 100%

**Figure 2. Zero-shot Presentation (+ real species proteome flanks) versus Affinity on `hitlist` MS elutions.** AUROC with 95% stratified bootstrap CI against `10:1` same-proteome length-matched decoys for zero-shot Tasmanian devil (*Sarcophilus harrisii*, `0` training rows, `11` UniProt grooves, `15,301` mapped ligands) alongside trained Dog (`DLA-88`) and Rhesus macaque (`Mamu-B`) MS reference controls.
```

| Dataset (`hitlist` MS) | Status | Mapped Pos (Eval / Total) | Affinity AUROC | Processing (+F) AUROC | Pres (noF) AUROC | **Pres (+ real flanks)** | Δ (Pres+F − Aff) | PPV@N (Aff → Pres+F) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Tasmanian devil `Saha`** (11 grooves) | **0-shot Marsupialia** | 2,500 / 15,301 | 0.749 | **0.750** | 0.791 | **0.806** | **+0.057** | **0.273 → 0.356** (+30%) |
| Dog `DLA-88*501:01` | Trained ref* | 1,301 / 1,301 | 0.933 | 0.837 | 0.942 | **0.947** | +0.014 | 0.711 → 0.746 |
| Dog `DLA-88*034:01` | Trained ref | 355 / 355 | 0.959 | 0.875 | 0.966 | **0.970** | +0.011 | 0.752 → 0.780 |
| Rhesus `Mamu-B*008:01` | Trained ref | 874 / 874 | 0.968 | 0.866 | 0.976 | **0.979** | +0.011 | 0.852 → 0.874 |

*\*As shown in Section 4, IEDB `DLA-88*501:01` overlaps `2.3.0`'s bundled affinity training table; these rows serve strictly as trained positive-control ceilings.*

Even across `~160 Myr` of divergence from placentals to marsupials, MHCflurry's human-trained processing model alone achieves **`0.750` AUROC** on Tasmanian devil peptides with real *S. harrisii* flanks and lifts combined Presentation by **`+0.057` AUROC** (`+30%` relative PPV@N).

### 2.2 Experimental `+/-` Binding Assays & 16-Cohort Zero-Shot Ranking

Non-MS datasets in `hitlist` and IEDB (in-vitro refolding, purified-MHC fluorescence screens, and synthetic viral/mutant binding panels) undergo no cellular proteasome/TAP processing and have minimal host-proteome mapping (`19 / 448 = 4.2%` in `binding.parquet`; `0%` for the `5,607` synthetic random-library `Trvu-UB*01:01` ligands). We therefore report **Affinity** as the primary metric—on both experimental positive-versus-negative binding assays (no synthetic decoys) and `10:1` length-matched background ligand ranking—paired with an unrelated `HLA-A*02:01` negative control on identical peptides.

```{figure} _static/release-2.5.7/fig3_hitlist_zeroshot_binding_and_ligands.svg
:alt: Experimental positive versus negative binding assay AUROC without synthetic decoys and zero-shot Affinity versus Presentation no-flank ligand ranking AUROC across unseen vertebrate species and pocket mutants.
:width: 100%

**Figure 3. Experimental `+/-` binding assays and broad zero-shot ligand ranking across unseen vertebrates.** *Left:* Experimental positive-versus-negative binding assay AUROC (no synthetic decoys) from `hitlist/binding.parquet`. *Right:* Zero-shot Affinity (solid) and no-flank Presentation (hatched) ligand ranking AUROC against `10:1` length-matched background decoys across 11 unseen vertebrate species and 3 engineered binding-pocket mutants.
```

| Unseen Species / Allele (Clade) | Validation Data Available | Ligand Ranking AUROC: Aff / Pres (noF) | `HLA-A*02:01` Ctrl | Exp. `+/-` Binding AUROC *(no decoys)* | Notes / Caveats |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **Cat `FLA-E*018:01`** (Carnivora) | 24 direct/bind + 16 `+/-` (`3+/13-`) | **0.977** / **0.966** | 0.636 | 0.308 *(3+/13-)* | `155 vs 16,992 nM`; full IEDB `24+/12-` gave `0.684` |
| **Cat `FLA-E*018:01 W167S`** (Mutant) | 12 mutant ligands | **0.976** / **0.981** | 0.642 | — | Mutant groove beats WT (`Δ = +0.006`) |
| **Cat `FLA-E*003:01` & `FLA-K*007:01`** | 1,612 & 491 MS ligands (IEDB) | **0.830** & **0.880** | 0.562 / 0.659 | — | Recovers P2-Pro & P9-Arg/Lys motif zero-shot |
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
| **Black flying fox `Ptal-N*01:01`** (Bat) | 86 `+/-` binding (`37+/43-`) | 0.593 / 0.717 | 0.520 | **0.606** *(37+/43-)* | Failure mode: `+3aa MDL` A-pocket insertion (auto-warned) |

---

## 3. Groove Specificity, Motif Recovery, Failure Modes & Training Parity

### 3.1 Negative-Control Separation & Empirical vs Zero-Shot Sequence Logos

Comparing each unseen cohort against the `HLA-A*02:01` negative control (`Figure 4`, left) cleanly separates true groove-specific generalization (`+0.356` Horse, `+0.341` Cat `FLA-E*018:01`, `+0.329` Duck `Anpl-UAA*76`, `+0.299` Possum) from composition-confounded cases (`Xela-UAAg` frog: `0.826` vs `0.810` control).

```{figure} _static/release-2.5.8/fig4_zeroshot_vs_control_and_parity.svg
:alt: Groove-specific zero-shot transfer compared to an HLA-A*02:01 negative control across all 16 unseen-species and mutant cohorts, paired with the controlled A/B pan-allele training parity comparison.
:width: 100%

**Figure 4. Groove-specific transfer across 16 unseen cohorts and controlled A/B pan-allele training parity.** *Left:* Zero-shot target-groove AUROC versus `HLA-A*02:01` negative control on identical peptides across all 16 unseen-species and pocket-mutant datasets, sorted by `Δ = Target − Control`. *Right:* Controlled A/B pan-allele training parity across all `307` trained `2.3.0` alleles comparing legacy `clustalo` (Arm A) against `mhcseqs --lock-existing` (Arm B: `Max |Δ| = 0.0`, `r = 1.000000`) and unlocked `Gaga-BF` resolution (Arm C).
```

Comparing empirical 9-mer MS ligands against the top 1% of `30,000` random 9-mers scored zero-shot (`Figure 5`) illustrates both anchor recovery and the structural limit of fixed 39-aa pseudosequences:

- **Feline & marsupial anchor recovery:** `FLA-E*003:01` recovers both P2-Pro and P9-Arg/Lys anchors with zero feline training data; Brushtail possum `Trvu-UB*01:01` and Tasmanian devil `Saha` recover aromatic/hydrophobic P9 anchors across `~160 Myr`.
- **Single-residue pocket mutants:** Extracting mature-numbered point mutations via `mhcseqs` improves discrimination over wild-type grooves for Pig `SLA-1*04:01 R156A` (`ΔAUROC = +0.010`, 95% paired CI `[+0.007, +0.013]`) and Cat `FLA-E*018:01 W167S` (`ΔAUROC = +0.006`, 95% CI `[+0.000, +0.015]`).
- **Honest failure mode — Bat `Ptal-N*01:01` (`+3aa MDL` A-pocket insertion):** The 3-aa `MDL` insertion at mature position 51 remodels the A-pocket (driving P1-Asp and P9-Pro specificity) alongside `Y171H`. Because a fixed 39-aa vector omits those three residues, predicted affinities flatten to `~26–30 µM` (`1.1×` separation). `mhcseqs >= 2.9` explicitly flags these insertions via `logging.warning`.

```{figure} _static/release-2.5.0/zero_shot_motifs.svg
:alt: Empirical versus zero-shot predicted 9-mer information-content sequence logos for Brushtail possum Trvu-UB*01:01, Cat FLA-E*003:01, Tasmanian devil Saha, and Bat Ptal-N*01:01.
:width: 100%

**Figure 5. Empirical versus zero-shot predicted 9-mer sequence logos.** Empirical 9-mer ligands (top row) compared against the top 1% zero-shot predicted 9-mers from `30,000` random background peptides (bottom row) for Brushtail possum `Trvu-UB*01:01`, Cat `FLA-E*003:01`, Tasmanian devil `Saha`, and the Bat `Ptal-N*01:01` `+3aa MDL` insertion failure mode.
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

Auditing released weights against `curated_training_data.affinity.csv.bz2` alone misses MS ligands embedded inside `models_class1_presentation/models/`. Auditing all three bundled `2.3.0` training tables (`affinity_predictor_train_data.csv.bz2`: `811,134` rows; `processing_predictor_with_flanks_train_data.csv.bz2`: `399,392` rows; `processing_predictor_no_flank_train_data.csv.bz2`: `399,392` rows) cleanly separates contaminated cohorts from genuinely unseen peptides (`Figure 6`).

```{figure} _static/release-2.5.13/fig_cattle_dog_hitlist_validation.svg
:alt: Three-panel Cattle BoLA-I and Dog DLA hitlist audit and validation figure.
:width: 100%

**Figure 6. Mandatory three-table overlap audit and strictly peptide-disjoint Cattle `BoLA-I` / Dog `DLA-88` validation.** **A.** Exact peptide overlap against all three bundled `2.3.0` training tables: every `hitlist` Dog `DLA-88` MS peptide and small IEDB Cattle `BoLA-I` cohort (PMID 36423003) is `100%` in-sample and excluded from held-out claims, whereas `79.5%–87.3%` (`21,606` unique; `21,332` flanked) of `PXD008151` Cattle `BoLA-I` are strictly peptide-disjoint. **B.** Strictly peptide-disjoint `PXD008151` `BoLA-I` ranking against `10×` disjoint *Bos taurus* proteome decoys with symmetric real 5-aa flanks. **C.** Ultra-few-shot `BoLA` allele ablation (`4–12` training rows).
```

| Cohort (Source) | 2.3.0 Bundled Train Rows | Strictly Disjoint (%) | Aff AUROC | Pres (noF) AUROC | **Pres (+F) AUROC** | `HLA-A*02:01` Ctrl | PPV@N (`noF → +F`) |
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

Because MHC Class I (`α1`/`α2` heavy chain) and Class II (`α1` from `DRA`/`DQA1`/`DPA1`, `β1` from `DRB1`/`DQB1`/`DPB1`) share the homologous G-DOMAIN fold, `mhcseqs.align_domain_to_imgt(...)` places all four domains onto a single 3D Kabsch-calibrated IMGT coordinate system (`G-ALPHA1 ↔ G-ALPHA`, `G-ALPHA2 ↔ G-BETA`). Extracting MHCflurry's `39` groove positions populates **`36 / 39` positions** directly across Class II heterodimers, leaving only `3` Class I-specific α1 loop positions (`mature 16, 19, 59`) padded with `X`.

We tested whether these aligned 39-aa Class II pseudosequences enable **frozen Class I MHCflurry `2.3.0` weights** to predict quantitative Class II 15-mer IC50 binding zero-shot across `15` HLA-DR, HLA-DQ, and HLA-DP alleles (`500` unique 15-mers per allele; `7,500` total pairs from `hitlist`; binder threshold `IC50 ≤ 500 nM`), benchmarked head-to-head against `NetMHCIIpan-4.3` (BA & EL) and an `HLA-A*02:01` negative control in both **9-mer core scanning** (`min IC50` over the 7 overlapping 9-mers) and **direct 15-mer** scoring (`Figure 7`).

```{figure} _static/release-2.5.9/fig_pan_class_ii_experiment.svg
:alt: Three-panel benchmark comparing NetMHCIIpan-4.3 BA against frozen Class I MHCflurry 2.3 using mhcseqs pan-class IMGT 39-aa Class II pseudosequences (9-mer core and direct 15-mer) and an HLA-A*02:01 negative control across 15 HLA-DR, DQ, and DP alleles, plus the 15x15 cross-allele specificity matrix.
:width: 100%

**Figure 7. Exploratory zero-shot Class I → Class II pan-class transfer versus `NetMHCIIpan-4.3`.** **A, B.** Per-allele AUROC (`IC50 ≤ 500 nM`) and continuous Spearman rank correlation (`ρ` against `-log10 IC50`) across 15 HLA-DR, DQ, and DP alleles (`500` 15-mers each). **C.** `15 × 15` cross-allele specificity matrix (AUROC minus row mean under pan-class 9-mer core scoring), displaying column-dominated vertical bands rather than diagonal groove specificity (`mean diag 0.551` vs `mean off-diag 0.556`, `Δ = -0.005`).
```

| Method | Scoring mode | Mean AUROC (`≤ 500 nM`) | Mean Spearman `ρ` (`-log10 IC50`) | Interpretation |
| :--- | :--- | ---: | ---: | :--- |
| **`NetMHCIIpan-4.3 (BA)`** | Native Class II BA | **0.923** | **+0.801** | Reference ceiling on quantitative 15-mer IC50 |
| **`NetMHCIIpan-4.3 (EL)`** | Native Class II EL | **0.763** | **+0.493** | Eluted-ligand score on binding-affinity assay |
| `HLA-A*02:01` negative control | 9-mer core (`min IC50`) | `0.598` | `+0.187` | Shared hydrophobic-core composition baseline |
| **Frozen Class I MHCflurry `2.3.0`** | **Pan-class 9-mer core** | `0.551` | `+0.115` | Below `HLA-A*02:01` control (`Δ = -0.047`) |
| `HLA-A*02:01` negative control | Direct 15-mer | `0.430` | `-0.146` | Closed-groove N/C terminus mismatch |
| **Frozen Class I MHCflurry `2.3.0`** | **Pan-class direct 15-mer** | `0.394` | `-0.224` | Anti-correlated (down to `0.235 / -0.521` on DP) |

*(See {doc}`experiment_25x_pan_class_ii` for the full 15-row per-allele table.)*

### Three Mechanistic Takeaways

1. **9-mer core scanning recovers peptide-level hydrophobicity, not Class II groove specificity:** Scanning 9-mer cores lifts AUROC from `0.394` to `0.551` because Class II 15-mer binders contain hydrophobic internal P1 cores favored by Class I predictors too, yet the unrelated `HLA-A*02:01` control scores *higher* (`0.598`) on the exact same peptides.
2. **Closed-groove vs open-groove geometry causes severe anti-correlation on HLA-DP (`AUROC 0.235–0.255`, `ρ = -0.40 to -0.52`):** Class I binds 15-mers in a *closed groove* with pinned N/C termini and a central bulge, whereas Class II binds an *open groove* with a floating internal 9-mer core and polar/charged overhangs extending beyond both ends.
3. **Zero diagonal transfer proves joint pan-class training is required:** Mean on-diagonal AUROC (`0.551`) is slightly below mean off-diagonal AUROC (`0.556`, `Δ = -0.005`), confirming that frozen Class I weights respond to global sequence-distribution shifts rather than Class II pocket chemistry. `mhcseqs`'s unified IMGT coordinate system provides the right structural alignment for a **jointly trained pan-class architecture** in a future release, not zero-shot weight reuse.

---

## 6. Practical Guidance & Reproducibility

### Checklist for Predicting on New Vertebrate Species

1. **Inspect groove alignment warnings:** Use `extract_class1_pseudosequence_with_alignment()` (or watch `mhcflurry` logs). Canonical grooves and small deletions transfer robustly across mammals and birds; multi-residue pocket insertions (e.g., bat `Ptal-N*01:01` `+3aa MDL`) degrade fixed-39-aa accuracy.
2. **Compare against an `HLA-A*02:01` negative control:** On small non-model ligand sets, genuine groove transfer produces a clear positive AUROC margin (`Δ > 0.10`), whereas tied scores (like frog `Xela-UAAg`) warn of peptide-composition bias.
3. **Supply real host flanking sequences for cellular MS tasks:** Real species proteome flanks consistently improve Presentation AUROC and PPV@N even across `~160 Myr` to marsupials (`+0.057` AUROC on Tasmanian devil) and on audited bovine MS (`+4%` PPV@N).
4. **Audit all three bundled training tables before claiming held-out accuracy:** Check `affinity_predictor_train_data.csv.bz2` and both `processing_predictor_*_train_data.csv.bz2` files inside `models_class1_presentation/models/`.
5. **Do not apply frozen Class I weights to Class II alleles:** Use dedicated Class II predictors (such as `NetMHCIIpan-4.3`) until a jointly trained pan-class MHCflurry model is released.

### Reproducible Scripts & Companion Pages

- **Individual detailed logs:** {doc}`cross_species_validation`, {doc}`experiment_25x_cross_species_hitlist`, {doc}`experiment_25x_cattle_dog_hitlist`, {doc}`experiment_25x_pan_class_ii`, {doc}`dla_validation`.
- **Reproducible benchmark scripts:** [`scripts/experiments/release_2_5_0/`](https://github.com/openvax/mhcflurry/tree/master/scripts/experiments/release_2_5_0) (`benchmark_cross_species_mhcseqs.py`, `benchmark_hitlist_cross_species.py`, `plot_hitlist_cross_species_figures.py`, `benchmark_cattle_dog_hitlist.py`, `benchmark_pan_class_ii_transfer.py`).
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
