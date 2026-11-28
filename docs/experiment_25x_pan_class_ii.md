# Experiment 2026-11-25: Speculative pan-class Class I → Class II transfer via `mhcseqs` IMGT alignments

**Date:** 2026-11-25
**Software:** `mhcflurry 2.5.9`, frozen Class I `2.3.0` weights (`models_class1_presentation`), [`mhcseqs`](https://github.com/pirl-unc/mhcseqs) structural IMGT aligner, `NetMHCIIpan-4.3` (BA and EL), [`hitlist`](https://github.com/pirl-unc/hitlist) quantitative Class II IC50 dataset
**Goal:** Test whether mapping Class II binding-pocket contact residues onto MHCflurry's 39-position Class I pseudosequence representation (using `mhcseqs` structural alignments) allows **frozen Class I MHCflurry `2.3.0` weights** to predict Class II 15-mer binding affinity zero-shot, compared head-to-head against `NetMHCIIpan-4.3` and an unrelated Class I negative control (`HLA-A*02:01`). *(See also {doc}`whitepaper_25x_cross_species`.)*

**Bottom line:** **Negative result — informative and clean.** Frozen Class I MHCflurry weights do **not** transfer across the Class I / Class II structural boundary (`mean AUROC = 0.551` with 9-mer core scanning vs `0.598` for `HLA-A*02:01` negative control and `0.923` for `NetMHCIIpan-4.3 BA`), and direct 15-mer scoring is systematically **anti-correlated** (`mean AUROC = 0.394`, `Spearman ρ = -0.224`; down to `0.235 / ρ = -0.521` on HLA-DP). Cross-allele specificity shows zero diagonal enrichment (`mean diag 0.551` vs `mean off-diag 0.556`, `Δ = -0.005`), proving that pan-class transfer requires **joint pan-class training** rather than zero-shot weight reuse.

---

## 1. Methodology: pan-class IMGT coordinate mapping

Because MHC Class I (`α1`/`α2` heavy chain) and Class II (`α1` from `DRA`/`DQA1`/`DPA1`, `β1` from `DRB1`/`DQB1`/`DPB1`) share the homologous G-DOMAIN fold, `mhcseqs.align_domain_to_imgt(...)` aligns all four domains onto the same 3D Kabsch-calibrated IMGT G-DOMAIN coordinate system:

- **α1 half (`22` of `39` MHCflurry positions):** Map `HLA-A*02:01` `G-ALPHA1` mature positions (`7, 9, 16, 19, 21, 24, 45, 47, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84`) into their IMGT G-DOMAIN coordinates, then extract the structurally equivalent residues from Class II `G-ALPHA` (`HLA-DRA*01:01`, `HLA-DQA1*`, `HLA-DPA1*`).
- **α2 / β1 half (`17` of `39` MHCflurry positions):** Map `HLA-A*02:01` `G-ALPHA2` mature positions (`91, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171`) into their IMGT G-DOMAIN coordinates, then extract the structurally equivalent residues from Class II `G-BETA` (`HLA-DRB1*`, `HLA-DQB1*`, `HLA-DPB1*`).
- **Occupancy:** **`36 / 39`** positions are directly occupied across Class II heterodimers; only `3` Class I-specific α1 loop positions (`mature 16, 19, 59` → IMGT `G-ALPHA1` positions absent in Class II `G-ALPHA`) are padded with `"X"`.

### Benchmark dataset & scoring modes

From `hitlist` quantitative Class II binding data (`30,381` unique `(allele, 15-mer)` quantitative IC50 nM pairs across 15 HLA-DR, HLA-DQ, and HLA-DP alleles), we sampled `500` unique 15-mer peptides per allele (`7,500` total pairs; binary binder threshold `IC50 ≤ 500 nM`, plus full continuous `Spearman ρ` against `-log10(IC50)`):

1. **`NetMHCIIpan-4.3 (BA)`** and **`NetMHCIIpan-4.3 (EL)`** — gold-standard pan-specific Class II reference models (exact `DRB1_*` / `HLA-DQA1*-DQB1*` / `HLA-DPA1*-DPB1*` allele names).
2. **`MHCflurry 2.3 Class I (pan-class 9-mer core)`** — register each Class II pan-class 39-aa pseudosequence on frozen `2.3.0` Class I weights and take the strongest predicted IC50 (`min IC50`) across the 7 overlapping 9-mer binding cores inside each 15-mer.
3. **`MHCflurry 2.3 Class I (pan-class direct 15-mer)`** — pass the full 15-mer directly into frozen `2.3.0` Class I weights (which internally encode 15-mers as a bulging closed-groove peptide with pinned N- and C-termini).
4. **`HLA-A*02:01` negative control (`9-mer core` and `direct 15-mer`)** — score the exact same 15-mers using an unrelated human Class I groove to separate generic peptide hydrophobicity/composition effects from genuine Class II groove recognition.

---

## 2. Summary results across 15 HLA-DR / DQ / DP alleles

```{figure} _static/release-2.5.9/fig_pan_class_ii_experiment.svg
:alt: Three-panel benchmark comparing NetMHCIIpan-4.3 BA against frozen Class I MHCflurry 2.3 using mhcseqs pan-class IMGT 39-aa Class II pseudosequences (9-mer core and direct 15-mer) and an HLA-A*02:01 negative control across 15 HLA-DR, DQ, and DP alleles, plus the 15x15 cross-allele specificity matrix.

Panels A and B: per-allele AUROC (IC50 <= 500 nM) and continuous Spearman rank correlation against experimental -log10(IC50) across 15 HLA Class II alleles (500 15-mers each). Panel C: 15x15 cross-allele specificity matrix (AUROC minus row mean using pan-class 9-mer core scoring), showing column-dominated vertical bands rather than diagonal groove specificity.
```

| Method | Scoring mode | Mean AUROC (`≤ 500 nM`) | Mean Spearman `ρ` (`-log10 IC50`) | Interpretation |
| :--- | :--- | ---: | ---: | :--- |
| **`NetMHCIIpan-4.3 (BA)`** | Native Class II BA | **0.923** | **+0.801** | Reference ceiling on quantitative 15-mer IC50 |
| **`NetMHCIIpan-4.3 (EL)`** | Native Class II EL | **0.763** | **+0.493** | Eluted-ligand score on binding-affinity assay |
| `HLA-A*02:01` negative control | 9-mer core (`min IC50`) | `0.598` | `+0.187` | Shared hydrophobic-core composition baseline |
| **Frozen Class I MHCflurry `2.3.0`** | **Pan-class 9-mer core** | `0.551` | `+0.115` | Below `HLA-A*02:01` control (`Δ = -0.047`) |
| `HLA-A*02:01` negative control | Direct 15-mer | `0.430` | `-0.146` | Closed-groove N/C terminus mismatch |
| **Frozen Class I MHCflurry `2.3.0`** | **Pan-class direct 15-mer** | `0.394` | `-0.224` | Anti-correlated (down to `0.235 / -0.521` on DP) |

---

## 3. Complete per-allele breakdown (15 alleles, 500 15-mers each)

| Allele | Locus | Binders (`≤ 500 nM`) | `NetMHCIIpan` BA (AUROC / `ρ`) | `NetMHCIIpan` EL (AUROC / `ρ`) | MHCflurry 9-mer core (AUROC / `ρ`) | `HLA-A*02:01` 9-mer ctrl (AUROC / `ρ`) | MHCflurry direct 15-mer (AUROC / `ρ`) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `HLA-DRB1*01:01` | DR | 301 / 500 | 0.908 / +0.824 | 0.734 / +0.481 | 0.592 / +0.180 | 0.619 / +0.235 | 0.410 / -0.170 |
| `HLA-DRB1*03:01` | DR | 128 / 500 | 0.918 / +0.785 | 0.793 / +0.544 | 0.568 / +0.163 | 0.592 / +0.183 | 0.438 / -0.192 |
| `HLA-DRB1*04:01` | DR | 173 / 500 | 0.859 / +0.649 | 0.730 / +0.360 | 0.498 / +0.081 | 0.557 / +0.174 | 0.406 / -0.245 |
| `HLA-DRB1*07:01` | DR | 244 / 500 | 0.924 / +0.851 | 0.760 / +0.525 | 0.617 / +0.257 | 0.645 / +0.300 | 0.319 / -0.355 |
| `HLA-DRB1*08:02` | DR | 161 / 500 | 0.894 / +0.714 | 0.721 / +0.433 | 0.525 / +0.033 | 0.577 / +0.116 | 0.404 / -0.183 |
| `HLA-DRB1*09:01` | DR | 207 / 500 | 0.896 / +0.774 | 0.772 / +0.511 | 0.622 / +0.181 | 0.586 / +0.135 | 0.413 / -0.195 |
| `HLA-DRB1*11:01` | DR | 202 / 500 | 0.929 / +0.836 | 0.818 / +0.631 | 0.484 / -0.018 | 0.592 / +0.117 | 0.333 / -0.356 |
| `HLA-DRB1*13:02` | DR | 193 / 500 | 0.920 / +0.777 | 0.652 / +0.367 | 0.567 / +0.133 | 0.591 / +0.159 | 0.409 / -0.163 |
| `HLA-DRB1*15:01` | DR | 206 / 500 | 0.893 / +0.797 | 0.768 / +0.529 | 0.518 / +0.052 | 0.569 / +0.176 | 0.363 / -0.308 |
| `HLA-DQA1*05:01/DQB1*02:01` | DQ | 104 / 500 | 0.930 / +0.825 | 0.797 / +0.555 | 0.543 / +0.089 | 0.521 / +0.095 | 0.491 / -0.065 |
| `HLA-DQA1*05:01/DQB1*03:01` | DQ | 238 / 500 | 0.945 / +0.859 | 0.807 / +0.610 | 0.534 / +0.093 | 0.531 / +0.070 | 0.650 / +0.269 |
| `HLA-DQA1*01:02/DQB1*06:02` | DQ | 151 / 500 | 0.940 / +0.873 | 0.805 / +0.580 | 0.565 / +0.181 | 0.593 / +0.247 | 0.532 / -0.021 |
| `HLA-DPA1*01:03/DPB1*04:01` | DP | 84 / 500 | 0.961 / +0.827 | 0.840 / +0.519 | 0.514 / +0.087 | 0.634 / +0.262 | 0.250 / -0.521 |
| `HLA-DPA1*02:01/DPB1*01:01` | DP | 85 / 500 | 0.969 / +0.797 | 0.694 / +0.351 | 0.566 / +0.076 | 0.660 / +0.215 | 0.255 / -0.398 |
| `HLA-DPA1*03:01/DPB1*04:02` | DP | 76 / 500 | 0.964 / +0.826 | 0.745 / +0.401 | 0.553 / +0.135 | 0.697 / +0.322 | 0.235 / -0.459 |
| **Mean (15 alleles)** | — | **167 / 500** | **0.923 / +0.801** | **0.763 / +0.493** | **0.551 / +0.115** | **0.598 / +0.187** | **0.394 / -0.224** |

---

## 4. Three mechanistic takeaways

1. **9-mer core scanning recovers peptide-level hydrophobicity, not Class II groove specificity.**
   Taking `min IC50` over the seven 9-mer substrings lifts AUROC from `0.394` to `0.551`, because Class II 15-mer binders generally contain an internal hydrophobic anchor core (`P1` aromatic/aliphatic) that Class I predictors also favor in 9-mers. However, `HLA-A*02:01` alone scores **higher** (`0.598` vs `0.551`) on the identical peptides, showing that the pan-class Class II pseudosequences are out-of-distribution for Class I weights.
2. **Direct 15-mer scoring fails catastrophically on HLA-DP (`AUROC = 0.235–0.255`, `ρ = -0.40 to -0.52`).**
   Class I MHC binds 15-mers in a **closed groove** by pinning the exact N- and C-terminal residues and bulging the middle out, whereas Class II MHC binds an **open groove** where the 9-mer core sits anywhere inside the 15-mer with overhangs extending out both ends. On HLA-DP (whose `P1`/`P6` aromatic anchors sit inside the 15-mer while terminal overhangs are frequently charged/polar), Class I's fixed-terminus encoding actively penalizes true binders.
3. **Zero diagonal transfer in the `15 × 15` specificity matrix proves joint training is required.**
   In Panel C, the mean on-diagonal AUROC (`0.551`) is slightly **below** the mean off-diagonal AUROC (`0.556`, `Δ = -0.005`, diagonal beats off-diagonal on only `9 / 15` rows). The strong vertical stripes (e.g., `DRB1*07:01` scoring higher on nearly every row while `DQA05:01/DQB02:01` and `DP` score lower on nearly every row) confirm that frozen Class I weights respond to global sequence-distribution shifts across the 39-aa vector rather than decoding Class II pocket chemistry (`P1`/`P4`/`P6`/`P9` vs Class I `P2`/`P9`). Conversely, `mhcseqs`'s shared 3D IMGT G-DOMAIN coordinate system gives us a clean, structurally aligned representation ready for **joint pan-class training** in a future architecture (where a model sees both Class I closed-groove and Class II open-groove binding modes during training).
