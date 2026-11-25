# Experiment 2026-11-25: Cross-species `hitlist` validation & `mhcseqs` training parity

**Date:** 2026-11-25
**Software:** `mhcflurry 2.5.x`, frozen `2.3.0` weights (`models_class1_presentation`), `mhcseqs 2.12.x`, `mhcgnomes 3.77–3.81`
**Goal:** Quantify (1) vertebrate species expansion from `mhcseqs`, (2) zero-shot Class I Affinity and Presentation (with real species proteome flanks) accuracy on previously unseen species in [`hitlist`](https://github.com/pirl-unc/hitlist) and IEDB, and (3) controlled A/B pan-allele training parity before migrating `downloads-generation/allele_sequences/GENERATE.sh` to `mhcseqs`.

---

## 1. Taxonomic expansion & `hitlist` validation data landscape

Complete Class I binding-groove coverage expands from **47 species** (only **10 species / 308 alleles** trained in `2.3.0`) to **227 species** in `mhcseqs 2.12` (**231 union**, **26,549 complete grooves**, **+184 net new vertebrate species** across 14 clades): bony fish `2→53 (+51)`, birds `1→42 (+41)`, other mammals `0→28 (+28)`, reptiles `0→17 (+17)`, amphibians `0→17 (+17)`, rodents `2→15 (+13)`, marsupials `0→7 (+7)`, sharks & rays `0→4 (+4)`, bats `0→4 (+4)`, and cetaceans `0→2 (+2)`.

```{figure} _static/release-2.5.7/fig1_species_and_validation_landscape.svg
:alt: Taxonomic expansion of complete Class I grooves (+184 net new species across 14 vertebrate clades) paired with a heatmap of hitlist Class I validation datasets by species and experimental modality.

Left: distinct vertebrate species with complete Class I grooves in MHCflurry 2.3.0 versus MHCflurry 2.5.x + mhcseqs 2.12. Right: hitlist Class I validation data availability across 10 unseen vertebrate species, 2 unseen allele/mutant cohorts in trained species, and 2 trained MS reference controls.
```

---

## 2. Zero-shot Presentation (+ real proteome flanks) vs Affinity on `hitlist` MS elutions

For cellular MS immunopeptidomics datasets with mapped host proteomes (`sarcophilus_harrisii.fasta`, `canis_lupus.fasta`, `macaca_mulatta.fasta`), we paired true eluted ligands against 10:1 length-matched decoys with real N- and C-terminal flanking sequences sampled from the same species proteome and scored frozen `2.3.0` weights with zero retraining:

```{figure} _static/release-2.5.7/fig2_hitlist_presentation_vs_affinity.svg
:alt: Presentation with and without real species proteome flanks versus Affinity and Processing alone on hitlist MS elutions for zero-shot Tasmanian devil and trained Dog and Rhesus reference alleles.

Zero-shot Presentation (+ real species proteome flanks) versus Affinity on hitlist MS elutions (10:1 same-proteome length-matched decoys, 95% stratified bootstrap CI).
```

| Dataset (`hitlist` MS) | Status | Mapped Pos (Eval / Total) | Affinity AUROC | Processing (+F) AUROC | Pres (noF) AUROC | **Pres (+ real flanks)** | Δ (Pres+F − Aff) | PPV@N (Aff → Pres+F) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Tasmanian devil `Saha`** (11 grooves) | **0-shot Marsupialia** | 2,500 / 15,301 | 0.749 | **0.750** | 0.791 | **0.806** | **+0.057** | **0.273 → 0.356** (+30%) |
| Dog `DLA-88*501:01` | Trained ref | 1,301 / 1,301 | 0.933 | 0.837 | 0.942 | **0.947** | +0.014 | 0.711 → 0.746 |
| Dog `DLA-88*034:01` | Trained ref | 355 / 355 | 0.959 | 0.875 | 0.966 | **0.970** | +0.011 | 0.752 → 0.780 |
| Rhesus `Mamu-B*008:01` | Trained ref | 874 / 874 | 0.968 | 0.866 | 0.976 | **0.979** | +0.011 | 0.852 → 0.874 |

Even across ~160 Myr of divergence to marsupials, MHCflurry's human-trained processing model alone achieves `0.750` AUROC on Tasmanian devil peptides with real *S. harrisii* flanks and lifts combined Presentation by `+0.057` AUROC (`+30%` relative PPV@N).

---

## 3. Experimental `+/-` binding & broad zero-shot ranking across 16 unseen-species & mutant sets

Non-MS datasets in `hitlist` (in-vitro refolding, purified-MHC fluorescence screens, and synthetic viral/mutant binding panels) have minimal host proteome coverage (`19 / 448 = 4.2%` mapped in `peptide_mappings.parquet`, and `0%` for the `5,607` synthetic random-library `Trvu-UB*01:01` peptides) and undergo no cellular proteasome/TAP processing, so **Affinity** is the primary metric (shown alongside no-flank Presentation and an `HLA-A*02:01` negative control on identical peptides):

```{figure} _static/release-2.5.7/fig3_hitlist_zeroshot_binding_and_ligands.svg
:alt: Experimental positive versus negative binding assay AUROC without synthetic decoys and zero-shot Affinity versus Presentation no-flank ligand ranking AUROC across unseen vertebrate species and pocket mutants.

Left: experimental positive-versus-negative binding assays (no synthetic decoys) from hitlist/binding.parquet. Right: zero-shot Affinity (solid) and no-flank Presentation (hatched) ligand ranking against 10:1 length-matched background decoys.
```

```{figure} _static/release-2.5.8/fig4_zeroshot_vs_control_and_parity.svg
:alt: Groove-specific zero-shot transfer compared to an HLA-A*02:01 negative control across all 16 unseen-species and mutant cohorts, paired with the controlled A/B pan-allele training parity comparison.

Left: groove-specific zero-shot AUROC versus HLA-A*02:01 negative control across all 16 unseen-species and binding-pocket mutant datasets (sorted by Δ = Target − Control). Right: controlled A/B pan-allele training parity across all 307 trained 2.3.0 alleles.
```

| Unseen Species / Allele (Clade) | Validation Data Available | Ligand Ranking AUROC: Aff / Pres (noF) | `HLA-A*02:01` Ctrl | Exp. `+/-` Binding AUROC *(no decoys)* | Notes / Caveats |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **Cat `FLA-E*018:01`** (Carnivora) | 24 direct/bind + 16 `+/-` (`3+/13-`) | **0.977** / **0.966** | 0.636 | 0.308 *(3+/13-)* | `155 vs 16,992 nM`; `2.5.0` full IEDB `24+/12-` gave `0.684` |
| **Cat `FLA-E*018:01 W167S`** (Mutant) | 12 mutant ligands | **0.976** / **0.981** | 0.642 | — | Mutant groove beats WT (`Δ = +0.006`) |
| **Cat `FLA-E*003:01` & `FLA-K*007:01`** | 1,612 & 491 MS ligands (`2.5.0` IEDB) | **0.830** & **0.880** | 0.562 / 0.659 | — | Recovers P2-Pro & P9-Arg/Lys motif zero-shot |
| **Horse `Eqca-N*006:02`** (Unseen allele) | 6+ / 10− binding | **0.967** / **0.956** | 0.611 | 0.217 *(6+/10-)* | Single-mutant scan negatives (`n=16`) |
| **Pig `SLA-1*04:01 R156A`** (Mutant) | 141 mutant ligands | **0.951** / **0.946** | 0.706 | — | Mutant groove beats WT (`Δ = +0.010`) |
| **Mallard duck `Anpl-UAA*01`** (Aves) | 31 binding+ | **0.938** / **0.920** | 0.862 | — | Indel-bearing avian groove; high ctrl baseline |
| **Pig `SLA-1*13:01 F99Y`** (Mutant) | 60 mutant ligands | **0.898** / **0.892** | 0.639 | — | `227 vs 5,543 nM` |
| **Mallard duck `Anpl-UAA*76`** (Aves) | 30 direct ligands | **0.893** / **0.856** | 0.564 | — | Strong `+0.329` lift over ctrl across ~310 Myr |
| **Giant panda `Aime-128`** (Carnivora) | 21 direct/bind (PDB `7DC6`) | **0.883** / **0.906** | 0.718 | — | `432 vs 15,828 nM` |
| **Brushtail possum `Trvu-UB*01:01`** (Marsupial) | 5,607 direct/fluo + 31 `+/-` (`14+/17-`) | **0.869** / **0.849** | 0.570 | **0.655** *(14+/17-)* | `+0.299` over ctrl (`1,182 vs 5,237 nM` on `+/-`) |
| **Cynomolgus `Mafa-A1*063:01`** (NHP) | 56 `+/-` binding (`54+/2-`) | **0.868** / **0.830** | 0.666 | **0.889** *(54+/2-)* | Ctrl `0.157` on `+/-` (`1,325 vs 9,663 nM`) |
| **African clawed frog `Xela-UAAg`** (Amphibia) | 26 direct/bind + 19 `+/-` (`14+/5-`) | 0.826 / 0.828 | 0.810 | 0.357 *(14+/5-)* | Caveat: matches ctrl (composition bias + 2 pocket insertions) |
| **Rabbit `RLA-A*01`** (Lagomorpha) | 27 `+/-` binding (`11+/16-`) | 0.787 / 0.792 | 0.969 | 0.562 *(11+/16-)* | Shares P2/P9 hydrophobic motif with `HLA-A*02:01` |
| **Tasmanian devil `Saha`** (Marsupial) | 23,973 unique MS (`15,301` flanked) | **0.772** / **0.803** *(0.806 +F)* | 0.666 | — | Multi-allelic (11 UniProt grooves) |
| **Norway rat `RT1-A*a`** (Rodentia) | 5 binding+ (24 obs) | 0.720 / **0.940** | 0.728 | — | Tiny sample (`n=5`) |
| **Black flying fox `Ptal-N*01:01`** (Bat) | 86 `+/-` binding (`37+/43-`) | 0.593 / 0.717 | 0.520 | **0.606** *(37+/43-)* | Mechanistic failure mode: `+3aa MDL` A-pocket insertion omitted by 39-aa pseudoseq (auto-warned by `mhcseqs`) |

---

## 4. Controlled A/B pan-allele training parity & `GENERATE.sh` gate

Auditing all `308` trained alleles in `curated_training_data.csv.bz2` against `mhcseqs 2.12` showed `297 / 297` mammalian trained alleles match `2.3.0`'s legacy `clustalo` pseudosequences 39-for-39, while `7` chicken `Gaga-BF1/BF2` alleles (`176` rows) differ at an indel alignment site and `4` alleles (`H2-D*q`, `H2-L*q`, `Gogo-B*01:01`, `HLA-A*30:14L`) were filed upstream as `pirl-unc/mhcseqs#141` (closed in `mhcseqs 2.12.5`).

In a controlled A/B pan-allele training run (`[1024, 1024]` architecture, `30,760` train / `7,690` held-out test across all trained alleles, seed `42`):

| Arm | Pseudosequence Table | Held-out AUROC (all) | Chicken `Gaga` AUROC | Max `|Δ|` vs Arm A | `log10 r` vs Arm A |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Arm A** | Legacy `2.3.0` `clustalo` | `0.80169` | `0.57025` | `0.0` | `1.000000` |
| **Arm B** | `mhcseqs` + `--lock-existing` | **`0.80169`** | **`0.57025`** | **`0.0`** | **`1.000000`** |
| **Arm C** | `mhcseqs` (`prefer_existing=True`, unlocking 7 `Gaga-BF`) | `0.83151` | `0.52342` | non-zero | `0.920` |

This directly validated migrating `downloads-generation/allele_sequences/GENERATE.sh` to `mhcflurry pseudosequences build --lock-existing` with `verify_trained_allele_parity.py` enforcing `308 / 308` bit-for-bit identical trained alleles.

## Upstream data hygiene issues discovered & filed during this campaign

- [`pirl-unc/mhcseqs#141`](https://github.com/pirl-unc/mhcseqs/issues/141) *(closed in `mhcseqs 2.12.5`)* — canonical names for `H2-D*q` (`Q31145`), `H2-L*q` (`Q31151`), `Gogo-B*01:01`, `HLA-A*30:14L`.
- [`pirl-unc/mhcgnomes#306`](https://github.com/pirl-unc/mhcgnomes/issues/306) — bare UniProt accessions (`H2LR16..44` Japanese medaka fish → mouse `H2-L*r16..44`; `P01898..` → human pseudogene `HLA-P*`); defensively guarded in `mhcflurry 2.5.8`.
- [`pirl-unc/hitlist#977`](https://github.com/pirl-unc/hitlist/issues/977) — 9 hardcoded UniProt proteome UPIDs in `hitlist/downloads.py` pointing to wrong organisms (e.g., `trichosurus_vulpecula.fasta` → sesame seed `Sesamum indicum`, `hpv16` → 45 MB poplar tree `Populus trichocarpa`).
