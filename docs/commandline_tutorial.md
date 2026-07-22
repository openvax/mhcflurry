(commandline_tutorial)=

# Command-line tutorial

(downloading)=
(downloading-models)=

## Download models

Most users need only the presentation bundle. It includes the binding-affinity
and antigen-processing components:

```shell
$ mhcflurry downloads fetch models_class1_presentation
```

Downloads are stored outside the Python package in a platform-specific data
directory. Use `info` to list available bundles and `path` to locate one:

```{command-output} mhcflurry downloads path models_class1_presentation
:nostderr:
```

## Predict peptides

`mhcflurry predict` scores individual peptides with the downloaded models:

Allele names are parsed as sequence-resolved MHC class I alleles. Invalid,
ambiguous, class-II, pseudogene, null, or unsupported names produce a specific
error. Add `--no-throw` when processing mixed-quality tables to keep those rows
with `NaN` predictions instead.

```{command-output} mhcflurry predict --alleles HLA-A0201 HLA-A0301 --peptides SIINFEKL SIINFEKD SIINFEKQ --out /tmp/predictions.csv
:nostderr:
```

```{command-output} cat /tmp/predictions.csv
```

| Output | Interpretation |
|---|---|
| `mhcflurry_affinity` | Predicted nM affinity; lower is stronger. |
| `mhcflurry_affinity_percentile` | Allele-specific rank from 0–100; lower is stronger. |
| `mhcflurry_processing_score` | Allele-independent processing score; higher is stronger. |
| `mhcflurry_presentation_score` | Combined binding and processing score; higher is stronger. |

Affinity thresholds of 500 nM or 2nd percentile are common screening choices.
Presentation scores are useful for ranking candidates, but there is no
universal presentation-score threshold.

(allele-input-semantics)=

### Alleles, genotypes, and samples

MHCflurry treats each allele argument or CSV cell as one query. Delimiters
inside a query (`;`, `,`, or whitespace) combine alleles into one genotype;
separate command-line arguments remain separate queries.

| Input | Meaning |
|---|---|
| `--alleles A0201 A0301 --peptides P1 P2` | Four independent allele–peptide rows. |
| `--alleles 'A0201;A0301' --peptides P1 P2` | Two genotype–peptide rows; `best_allele` identifies the stronger allele. |
| CSV rows `P1,A0201` and `P1,A0301` | Two independent rows. |
| CSV row `P1,A0201;A0301` | One genotype row with the strongest allele reported. |

`mhcflurry predict-scan` uses the same rule: each `--alleles` argument names
one sample. A quoted comma-separated panel is scored as one group and reports
the best allele across that group; separate arguments keep per-allele or
per-genotype results. A large population panel is therefore not the same thing
as one person's genotype.

For CSV prediction, optional `n_flank` and `c_flank` columns provide source
protein context for cleavage prediction. See the
{ref}`command reference <ref-mhcflurry-predict>` for the complete input schema.


## Scanning protein sequences for predicted MHC I ligands

Use `mhcflurry predict-scan` to score every supported peptide window in a
protein sequence.

We'll generate predictions across `example.fasta`, a FASTA file with two short
sequences:

```{literalinclude} /example.fasta
```

This invocation keeps peptides predicted to bind at 100 nM or tighter:

```shell
$ mhcflurry predict-scan example.fasta \
    --alleles HLA-A*02:01 \
    --threshold-affinity 100
```

See the {ref}`command reference <ref-mhcflurry-predict-scan>` for FASTA/CSV
input, presentation-score filtering, peptide lengths, and output options.


## Training models

Training is an advanced workflow; most users should use the released models.
If you have custom measurements, choose the smallest workflow that matches the
data:

- allele-specific affinity models for one or a few well-covered alleles;
- pan-allele affinity models for measurements spanning many alleles; or
- `mhcflurry train pan-allele-release` for a complete retrain, selection,
  calibration, and evaluation run.

The {doc}`training` guide covers input schemas, hyperparameters, output bundles,
and release-style training without interrupting this prediction tutorial.


## Evaluating trained models

After fitting a model, compare it with a released predictor before using it as a
default. A local comparison and diagnostic PDF take two commands:

```shell
$ mhcflurry eval compare-models \
    --a results/new_run/ \
    --b public \
    --out results/new_run/eval_comparison/

$ mhcflurry eval plot-comparison \
    --input results/new_run/eval_comparison/ \
    --summary-pdf results/new_run/eval_comparison/plots/model_comparison_figures.pdf
```

The {doc}`evaluation` guide explains the output layers, saved-prediction schema,
paper-style figures, and remote release behavior.

## Using older allele-specific models

MHCflurry still distributes the allele-specific predictors described in the
2018 paper. Download them and pass their model directory explicitly:

```shell
$ mhcflurry downloads fetch models_class1
$ mhcflurry predict \
    --alleles HLA-A0201 HLA-A0301 \
    --peptides SIINFEKL SIINFEKD SIINFEKQ \
    --models "$(mhcflurry downloads path models_class1)/models" \
    --out /tmp/predictions.csv
```

Use the current pan-allele presentation bundle unless you specifically need
these historical models.

## Configuration and command reference

See {doc}`configuration` for prediction batches, hardware autosizing,
reproducibility, and unified command aliases. The complete generated argument
reference is in {doc}`commandline_tools`.
