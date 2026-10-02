(commandline_tutorial)=

# Command-line tutorial

(downloading)=
(downloading-models)=

## Download models

The presentation bundle includes the binding-affinity and antigen-processing
components:

```shell
$ mhcflurry downloads fetch models_class1_presentation
```

Downloads are stored outside the Python package in a platform-specific data
directory. Use `info` to list available bundles and `path` to locate one:

```{command-output} mhcflurry downloads path models_class1_presentation
:nostderr:
```

See {doc}`model_downloads` to browse releases and select historical weights
with `--model-release`.

## Predict peptides

`mhcflurry predict` scores peptides with their MHC alleles and N/C source-protein
flanks. The example CSV contains windows from `example.fasta`:

```{literalinclude} /example-peptides.csv
:language: text
```

Flanks are used by default when both columns are provided:

```{command-output} mhcflurry predict example-peptides.csv --out /tmp/predictions.csv
:nostderr:
```

```{command-output} cat /tmp/predictions.csv
```

To compare the same rows without flanks:

```shell
mhcflurry predict example-peptides.csv --no-flanking --out predictions-no-flanks.csv
```

When context is unavailable, omit the flank columns or supply peptide arguments:

```shell
mhcflurry predict --alleles HLA-A0201 HLA-A0301 --peptides SIINFEKL SIINFEKD --out predictions.csv
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

Allele names are parsed as sequence-resolved MHC class I alleles. Invalid,
ambiguous, class-II, pseudogene, null, or unsupported names produce a specific
error. Add `--no-throw` when processing mixed-quality tables to keep those rows
with `NaN` predictions instead.

(allele-input-semantics)=

### MHC alleles and samples

MHCflurry treats each allele argument or CSV cell as one query. Delimiters
inside a query (`;`, `,`, or whitespace) combine alleles into one MHC allele set;
separate command-line arguments remain separate queries.

| Input | Meaning |
|---|---|
| `--alleles A0201 A0301 --peptides P1 P2` | Four independent allele–peptide rows. |
| `--alleles 'A0201;A0301' --peptides P1 P2` | Two MHC allele set–peptide rows; `best_allele` identifies the stronger allele. |
| CSV rows `P1,A0201` and `P1,A0301` | Two independent rows. |
| CSV row `P1,A0201;A0301` | One MHC allele set row with the strongest allele reported. |

`mhcflurry predict-scan` uses the same rule: each `--alleles` argument names
one sample. A quoted comma-separated panel is scored as one group and reports
the best allele across that group; separate arguments keep per-allele or
per-sample results. A large population panel is therefore not the same thing
as one person's MHC allele set.

For CSV prediction, optional `n_flank` and `c_flank` columns provide source
protein context for cleavage prediction. See the
{ref}`command reference <ref-mhcflurry-predict>` for the complete input schema.


## Scanning protein sequences for predicted MHC I ligands

Use `mhcflurry predict-scan` to score peptide windows in a protein sequence.
The default lengths are 8–11 amino acids; set `--peptide-lengths` to change them.
By default, the output keeps rows with affinity percentile at most 2. Use
`--results-all` to return every scored window, or `--threshold-*` to choose a
different filter. Scanning supplies N/C source-protein flanks automatically;
use `--no-flanking` for predictions without that context.

`example.fasta` contains two short sequences:

```{literalinclude} /example.fasta
```

This invocation keeps peptides with predicted affinity at most 100 nM:

```shell
$ mhcflurry predict-scan example.fasta \
    --alleles 'HLA-A*02:01' \
    --threshold-affinity 100
```

See the {ref}`command reference <ref-mhcflurry-predict-scan>` for FASTA/CSV
input, presentation-score filtering, peptide lengths, and output options.


## Next steps

- {doc}`model_downloads` shows how to select other weight releases, including
  the older allele-specific models.
- {doc}`training` and {doc}`evaluation` cover fitting custom models and
  comparing them with the released ones.
- {doc}`configuration` covers prediction batch sizes, hardware autosizing, and
  reproducibility.
- {doc}`commandline_tools` is the full command and option reference.
