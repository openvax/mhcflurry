# Downloading and selecting model weights

Code and weights have separate versions. A code patch can keep the same default
weights: the current package uses the **2.3.0 weight release**. New weight
releases identify changed model artifacts; they need not accompany every code
release. Historical catalogue names remain available for reproducibility.

## Browse what is available

```shell
mhcflurry downloads releases models_class1_presentation
mhcflurry downloads list --kind models
mhcflurry downloads info models_class1_presentation
```

`releases` lists valid catalogue identifiers. With a bundle name, it shows only
releases containing that bundle and groups entries with the same archive URLs.
For example, catalogues `2.2.0` and `2.0.0` point to the same presentation archive.
`list` groups prediction models, legacy models, experiments, and supporting data.
The first table shows the latest weights, distinct older archive versions and
installed catalogue directories, with the recommended presentation bundle first,
followed by standalone affinity and processing bundles.
Affinity and processing rows also show components found inside installed
presentation bundles as `RELEASE via presentation`; standalone bundle installs
remain separate entries. Components must have a model directory and
`manifest.csv` to appear in the table. A bundle containing only one processing
variant is marked `with flanks only` or `without flanks only`. The presentation
bundle's `?` or `!` marker indicates unknown or different source URLs.
Aliases sharing the same archive are grouped in the availability columns;
`releases DOWNLOAD` lists every valid identifier. Historical resources appear
below the main predictors. Terminal output uses restrained color; redirected
output and `NO_COLOR=1` remain plain.
Use `--kind data` for data only, or `--release 2.2.0` to inspect an older catalogue.
`info DOWNLOAD` adds descriptions, archive locations, fetch/use commands and
exact embedded component paths. These presence checks do not verify weight
file integrity or prove that embedded and standalone ensembles are identical.
All three commands support `--json` and read the installed package's catalogue
offline. Update the package to obtain a newer catalogue.

| Bundle | Contents and use |
|---|---|
| `models_class1_presentation` | Full presentation predictor, including affinity, processing, and their combiner. The normal prediction bundle. |
| `models_class1_pan` | Selected pan-allele affinity ensemble, for standalone affinity prediction. |
| `models_class1_processing` | Standalone processing ensembles with and without N/C flanks. |
| `models_class1` | Legacy allele-specific affinity models. The name does not mean the current general-purpose class I bundle. |
| `*_unselected` | Candidate networks before ensemble selection. |
| `*_variants`, `*_refined`, `*_experiments1` | Historical experimental configurations. |
| `*_with_mass_spec`, `*_no_mass_spec`, `*_minimal` | Historical training variants or smaller subsets. |
| `data_*`, `allele_sequences`, `analysis_predictor_info`, `random_peptide_predictions`, `cross_validation_class1` | Supporting data and analyses. |

The presentation bundle contains its own affinity and processing components.
The standalone bundles can be absent while full presentation prediction is
ready to use. `Class1AffinityPredictor.load()` can fall back to the default
presentation bundle when standalone affinity weights are absent and no
affinity-path override is set. `Class1ProcessingPredictor.load()` has no such
fallback: pass an embedded component path explicitly, or use
`Class1PresentationPredictor.load().processing_predictor_with_flanks` (or
`processing_predictor_without_flanks`). Explicit paths and release selection
retain their existing precedence.

In JSON output, `downloaded`, `status` and `path` still describe the named
bundle. The additive `presentation_components` entries describe embedded paths,
directory/manifest presence and the presentation bundle's source status.

## Compare new and historical weights

The public **2.1.5, 2.2.0, and 2.2.1 packages used the same 2020 model archives**,
registered under catalogue `2.2.0`. There is no separate `2.1.5` or `2.2.1`
weight catalogue. The `2.3.0` models use the 2023 curated affinity snapshot and an updated
processing training recipe; this does not mean every component gained new
2023 observations. See {doc}`release_training_recipe`.

Fetch both presentation bundles once:

```shell
mhcflurry downloads fetch models_class1_presentation --release 2.3.0
mhcflurry downloads fetch models_class1_presentation --release 2.2.0
```

Use a CSV with `peptide`, `allele`, `n_flank`, and `c_flank` columns. An `allele`
cell can contain a semicolon-separated MHC allele set. Run both models on the
same rows; available N/C flanks are used by default:

```shell
mhcflurry predict eval.csv --model-release 2.3.0 --out new.csv
mhcflurry predict eval.csv --model-release 2.2.0 --out old.csv
```

For the corresponding comparison without flanks:

```shell
mhcflurry predict eval.csv --model-release 2.3.0 --no-flanking --out new-no-flanks.csv
mhcflurry predict eval.csv --model-release 2.2.0 --no-flanking --out old-no-flanks.csv
```

`predict-scan` also accepts `--model-release`. The selector uses a presentation
bundle, including when `predict --affinity-only` requests only its affinity
component. It selects installed weights; it does not download them automatically.
`--models DIR` instead selects an arbitrary local predictor and cannot be
combined with `--model-release`.

These commands compare weights using the installed code. Reproducing a
historical software version also requires that version in its own environment.
See {doc}`release_model_evaluation` for the completed comparison across public
software versions, NetMHCpan BA/EL outputs, and MixMHCpred on identical rows.

## Older allele-specific models

MHCflurry still distributes the allele-specific predictors described in the
2018 paper for reproducing earlier results; for new work, use the current
presentation bundle. They are a separate, affinity-only bundle selected with
`--models` rather than `--model-release`:

```shell
mhcflurry downloads fetch models_class1
mhcflurry predict \
    --alleles HLA-A0201 HLA-A0301 \
    --peptides SIINFEKL SIINFEKD SIINFEKQ \
    --models "$(mhcflurry downloads path models_class1)/models" \
    --affinity-only \
    --out predictions.csv
```

## Local storage and overrides

```shell
mhcflurry downloads info
mhcflurry downloads path models_class1_presentation --release 2.3.0
mhcflurry downloads url models_class1_presentation --release 2.3.0
```

`info` starts with model availability, followed by historical resources and
resolved configuration. Use `mhcflurry downloads --verbose info` for configured
default predictor paths, components inside the default presentation predictor,
and environment variables. The variables are **optional
overrides**: `unset` means
the default is in use, not that the local path is missing. On macOS the default
root is `~/Library/Application Support/mhcflurry/4/`; each weight release has
its own subdirectory. The `4` is the cache-layout version, not a model version.
The presentation predictor is inside `models_class1_presentation/models`.

`MHCFLURRY_DATA_DIR` changes the parent of the release directories.
`MHCFLURRY_DOWNLOADS_CURRENT_RELEASE` changes the active catalogue.
`MHCFLURRY_DOWNLOADS_DIR` selects an unversioned custom download directory.
The `MHCFLURRY_DEFAULT_CLASS1_*` variables shown by `info` override individual
predictor defaults. Explicit `--model-release` selects from the requested
catalogue instead of those individual model-path overrides.

With an unversioned custom directory, `--model-release` requires recorded
source URLs matching the requested bundle; otherwise it asks you to use a
versioned cache or select the local model explicitly with `--models`. A
release selector must not silently load another release's weights.

“Installed” means a directory exists. “Source matches” means its
`DOWNLOAD_INFO.csv` URLs match the catalogue; it does not verify every file's
integrity. Public model checksums are attached to the corresponding GitHub
release. The `--release` option is supported consistently by `fetch`, `list`,
`info`, `path`, and `url`.

## Why archive tags differ

A catalogue collects many resources. Unchanged older resources keep their
original URLs, so a current catalogue can include storage tags such as
`pre-2.0` alongside `2.3.0`. Those tags describe where an archive was uploaded;
they do not require prerelease code. Several catalogues can reference the same
archive, and a single catalogue can include model and data archives from
different dates. Inspect `releases DOWNLOAD` or `info DOWNLOAD` to see this
mapping explicitly.

## Percentile calibration

Released affinity and presentation predictors include percentile calibration;
released processing predictors do not. For custom models, or for processing
percentiles, {doc}`shared_percent_rank_transforms` documents compact calibration, which uses
a small validated knot representation instead of dense histogram tables.
Calibration changes percentile mappings, not raw model scores or weights.
Use a separate model copy and an appropriate background reference; do not fit
calibration to evaluation labels.
