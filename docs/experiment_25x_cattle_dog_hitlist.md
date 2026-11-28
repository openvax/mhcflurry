# Experiment: Audited Cattle BoLA-I & Dog DLA `hitlist` Validation (2.5.x)

*Date: 2026-11-26 · Package: MHCflurry 2.5.x (frozen `2.3.0` presentation weights) · Data: refreshed [`hitlist`](https://github.com/pirl-unc/hitlist) benchmark corpus `2026-11-26` (PXD008151 Cattle BoLA-I + IEDB)*

This experiment evaluates MHCflurry's frozen `2.3.0` Class I Affinity and
Presentation predictors (with and without real source-protein flanking
sequences) on the Cattle (`Bos taurus`, `BoLA-I`) and Dog
(`Canis lupus familiaris`, `DLA-88`) mass-spectrometry cohorts in the refreshed
[`hitlist`](https://github.com/pirl-unc/hitlist) corpus, paired with a
**mandatory three-table `2.3.0` training-overlap audit** to separate genuinely
unseen peptides from in-sample training data. *(See also the unified
{doc}`whitepaper_25x_cross_species`.)*

```{figure} _static/release-2.5.13/fig_cattle_dog_hitlist_validation.svg
:alt: Three-panel Cattle BoLA-I and Dog DLA hitlist audit and validation figure
:width: 100%

**Audited Cattle BoLA-I and Dog DLA validation.** **A.** Exact peptide-overlap
audit against all three bundled `2.3.0` training tables (`811,134` affinity
rows + `399,392` with-flank and no-flank processing rows). All `hitlist` Dog
`DLA-88` MS peptides and small IEDB Cattle `BoLA-I` cohorts from PMID 36423003
are `100%` in-sample in `affinity_predictor_train_data.csv.bz2` and are
excluded from held-out claims, whereas `79.5%–87.3%` (`21,606` unique
peptides) of the PXD008151 Cattle `BoLA-I` MS dataset are strictly
peptide-disjoint from all three tables. **B.** Strictly peptide-disjoint
PXD008151 Cattle `BoLA-I` ligand ranking against `10×` disjoint,
length-matched `Bos taurus` proteome decoys with symmetric real 5-aa N/C
flanks. **C.** Ultra-few-shot BoLA allele ablation (`4–12` training rows) on
strictly disjoint bovine ligands.
```

---

## 1. Mandatory Three-Table `2.3.0` Training-Overlap Audit

A critical pitfall in evaluating released MHCflurry `2.3.0` weights is auditing
against `curated_training_data.affinity.csv.bz2` alone (which contains only
quantitative/qualitative binding measurements) instead of the actual bundled
training tables shipped inside `models_class1_presentation/models/`:

- `affinity_predictor_train_data.csv.bz2` (`811,134` rows — binding + MS)
- `processing_predictor_with_flanks_train_data.csv.bz2` (`399,392` rows)
- `processing_predictor_no_flank_train_data.csv.bz2` (`399,392` rows)

Auditing every Class I Cattle and Dog cohort from the refreshed `hitlist`
bundle (`2026-11-26`, 8–15mer standard-AA peptides) against all three bundled
tables immediately separates genuinely out-of-sample cohorts from in-sample
training data:

| Cohort (Source) | 2.3.0 Bundled Affinity Train Rows (per allele) | Unique 8–15mers | Same-Allele Train Overlap | Any Affinity / Processing Table Overlap | Strictly Peptide-Disjoint (%) | Held-Out Eligibility |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **Cattle `BoLA-6*013:01`** (PXD008151 + IEDB) | `126` | 6,724 | 36 (0.5%) | 587 / 550 | **5,873 (87.3%)** | **Eligible (5,811 flanked)** |
| **Cattle `BoLA-2*012:01; 3*002:01`** (PXD008151) | `64` + `12` | 10,273 | 5 (0.0%) | 1,507 / 1,237 | **8,304 (80.8%)** | **Eligible (8,197 flanked)** |
| **Cattle `BoLA-1*023:01; 2*025:01; 4*024:01`** (PXD008151) | `24` + `4` + `6` | 9,345 | 2 (0.0%) | 1,453 / 1,254 | **7,429 (79.5%)** | **Eligible (7,324 flanked)** |
| **Cattle `BoLA-3*027:01` / `1*009:01` / `2*026:01`** (IEDB PMID 36423003) | `21` / `14` / `13` | 48 | **48 (100.0%)** | 48 / 0 | **0 (0.0%)** | **Excluded (in-sample)** |
| **Dog `DLA-88*501:01`** (IEDB PMID 27893789) | `1,963` | 1,959 | **1,959 (100.0%)** | 1,959 / 799 | **0 (0.0%)** | **Excluded (in-sample)** |
| **Dog `DLA-88*508:01`** (IEDB PMID 29475511) | `96` | 50 | **50 (100.0%)** | 50 / 13 | **0 (0.0%)** | **Excluded (in-sample)** |

### Connection to Canine DLA Provenance (`#490`)

This audit independently confirms the central conclusion of
{doc}`dla_validation` (issue `#490`): every `DLA-88*501:01` and `DLA-88*508:01`
MS peptide currently in IEDB/`hitlist` is already present in `2.3.0`'s bundled
`affinity_predictor_train_data.csv.bz2` (`1,963` and `96` rows respectively),
so those cohorts **cannot** support held-out canine accuracy claims. Inspecting
executable vs. trained vs. validated status for any allele remains the role of
`mhcflurry eval allele-capabilities`.

---

## 2. Strictly Peptide-Disjoint Cattle `BoLA-I` Evaluation (PXD008151)

For the three PXD008151 Cattle `BoLA-I` cohorts (Nielsen/Ternette, PMID
29115832), we retained **only** peptides with zero sequence overlap against all
three bundled `2.3.0` training tables (`21,332` strictly disjoint peptides with
mapped `Bos taurus` UniProt reference proteome flanks, `51,509` proteins).
Each cohort was evaluated by sampling `2,500` disjoint flanked positives against
`25,000` (`10×`) length-matched, training-disjoint `Bos taurus` proteome decoys
carrying symmetric real 5-aa N- and C-flanking sequences (seed `42`):

| Strictly Disjoint Cohort | 2.3.0 Train Rows | Disjoint Flanked (Eval Pos / Dec) | Aff AUROC | Pres (noF) AUROC | Pres (+F) AUROC (`Δ`) | HLA-A*02:01 Ctrl | PPV@N (`noF → +F`) | Med IC50 (Pos vs Dec) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Cattle `BoLA-6*013:01`** (monoallelic) | `126` | 5,811 (2,500 / 25,000) | 0.950 | 0.957 | **0.960** (`+0.003`) | 0.789 | 0.726 → 0.725 | 50 vs 13,912 nM |
| **Cattle `BoLA-2*012:01; 3*002:01`** (2-allele) | `64` + `12` | 8,197 (2,500 / 25,000) | 0.871 | 0.896 | **0.902** (`+0.006`) | 0.648 | 0.538 → **0.560** | 1,055 vs 17,991 nM |
| **Cattle `BoLA-1*023:01; 2*025:01; 4*024:01`** (3-allele) | `24` + `4` + `6` | 7,324 (2,500 / 25,000) | 0.892 | 0.915 | **0.918** (`+0.004`) | 0.628 | 0.570 → **0.594** | 333 vs 12,117 nM |

Key observations:

1. **Strong groove-specific transfer on 21.3k unseen bovine MS ligands:**
   All three cohorts achieve `0.902–0.960` Presentation (+F) AUROC with large
   margins (`+0.171` to `+0.290` AUROC) over the `HLA-A*02:01` negative control
   and `17×–278×` tighter median predicted IC50 than length-matched bovine
   proteome decoys.
2. **Real `Bos taurus` flanking sequences consistently improve Presentation:**
   Adding symmetric real bovine N/C flanks improves AUROC on every cohort
   (`+0.003` to `+0.006`) and lifts multi-allelic PPV@N by `+4.1%`
   (`0.538 → 0.560`) and `+4.2%` (`0.570 → 0.594`), confirming cross-species
   transfer of MHCflurry's human-trained antigen-processing model to Artiodactyla.

---

## 3. Ultra-Few-Shot BoLA Allele Ablation (`4–12` Training Rows)

Because the two multi-allelic PXD008151 haplotypes combine modestly trained
alleles (`24–64` rows) with ultra-few-shot alleles (`4–12` rows), ablating
individual alleles on the exact same `2,500` strictly disjoint peptides tests
whether MHCflurry's pan-allele groove representation transfers effectively to
nearly unseen bovine alleles:

| Haplotype Cohort (2,500 strictly disjoint pos) | Allele Subset Evaluated | 2.3.0 Train Rows | Aff AUROC | Pres (+F) AUROC | Share of Disjoint Ligands Claimed (`best_allele`) |
| --- | --- | --- | ---: | ---: | ---: |
| `BoLA-2*012:01; 3*002:01` | `BoLA-2*012:01` alone | `64` | 0.830 | 0.873 | 57.2% (`1,430 / 2,500`) |
| `BoLA-2*012:01; 3*002:01` | **`BoLA-3*002:01` alone** | **`12`** | **0.812** | **0.858** | **42.8% (`1,070 / 2,500`)** |
| `BoLA-2*012:01; 3*002:01` | **Full 2-allele haplotype** | `64 + 12` | **0.871** (`+0.041`) | **0.902** (`+0.029`) | 100% |
| `BoLA-1*023:01; 2*025:01; 4*024:01` | `BoLA-1*023:01` + `4*024:01` | `24 + 6` | 0.888 | 0.913 | 69.0% + 19.3% (`1,724 + 483`) |
| `BoLA-1*023:01; 2*025:01; 4*024:01` | **`BoLA-4*024:01` alone** | **`6`** | **0.868** | **0.904** | **19.3% (`483 / 2,500`)** |
| `BoLA-1*023:01; 2*025:01; 4*024:01` | **`BoLA-2*025:01` alone** | **`4`** | **0.717** | **0.850** | **11.7% (`293 / 2,500`)** |
| `BoLA-1*023:01; 2*025:01; 4*024:01` | **Full 3-allele haplotype** | `24 + 4 + 6` | **0.892** (`+0.004`) | **0.918** (`+0.005`) | 100% |

Adding ultra-few-shot `BoLA-3*002:01` (only `12` rows in `2.3.0`) lifts the
2-allele haplotype by `+0.041` Affinity AUROC (`+0.029` Presentation +F AUROC)
and captures `42.8%` of the unseen bovine ligands as top binder; `BoLA-4*024:01`
(only `6` training rows) reaches `0.904` Presentation (+F) AUROC on its own.

---

## 4. Methodological Caveats

- **Peptide-disjoint ≠ complete donor/study lineage:** Stripping every exact
  peptide present in all three bundled `2.3.0` training tables prevents direct
  memorization of evaluation peptides, but historical `2.3.0` training tables
  lack full study/donor provenance columns (tracked under `#444` and `#494`).
- **Haplotype deconvolution:** Multi-allelic PXD008151 cohorts are scored via
  sample-level best-allele presentation (`alleles={"sample": [...]}`);
  `best_allele` counts reflect model attribution, not experimental monoallelic
  restriction.
- **No retraining or weight changes:** All scores use unmodified `2.3.0`
  weights and default `2.3.0` pseudosequences.
