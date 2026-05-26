# Command-line reference

See also the {ref}`tutorial <commandline_tutorial>`.

Starting in 2.3.0, MHCflurry installs a unified `mhcflurry` parent
command whose subcommands share one help surface (`mhcflurry --help`).
The historical standalone scripts (`mhcflurry-predict`,
`mhcflurry-downloads`, `mhcflurry-class1-*`, etc.) remain installed
unchanged — they are listed below the parent-command section. Migration
of the legacy entry points under the unified dispatcher is tracked in
[#291](https://github.com/openvax/mhcflurry/issues/291).

## Unified `mhcflurry` command

(mhcflurry-compare-models)=

```{autoprogram} mhcflurry.cli.compare_models:parser
:prog: mhcflurry compare-models
```

(mhcflurry-plot-model-comparison)=

```{autoprogram} mhcflurry.cli.plot_model_comparison:parser
:prog: mhcflurry plot-model-comparison
```

## Legacy standalone commands

These are the historical `mhcflurry-*` entry points. They are still
installed and supported; new functionality may land here, under the
parent `mhcflurry` command, or both.

(mhcflurry-predict)=

```{autoprogram} mhcflurry.predict_command:parser
:prog: mhcflurry-predict
```

(mhcflurry-predict-scan)=

```{autoprogram} mhcflurry.predict_scan_command:parser
:prog: mhcflurry-predict-scan
```

(mhcflurry-downloads)=

```{autoprogram} mhcflurry.downloads_command:parser
:prog: mhcflurry-downloads
```

(mhcflurry-class1-train-allele-specific-models)=

```{autoprogram} mhcflurry.train_allele_specific_models_command:parser
:prog: mhcflurry-class1-train-allele-specific-models
```

(mhcflurry-class1-select-allele-specific-models)=

```{autoprogram} mhcflurry.select_allele_specific_models_command:parser
:prog: mhcflurry-class1-select-allele-specific-models
```

(mhcflurry-class1-train-pan-allele-models)=

```{autoprogram} mhcflurry.train_pan_allele_models_command:parser
:prog: mhcflurry-class1-train-pan-allele-models
```

(mhcflurry-class1-select-pan-allele-models)=

```{autoprogram} mhcflurry.select_pan_allele_models_command:parser
:prog: mhcflurry-class1-select-pan-allele-models
```

(mhcflurry-class1-train-processing-models)=

```{autoprogram} mhcflurry.train_processing_models_command:parser
:prog: mhcflurry-class1-train-processing-models
```

(mhcflurry-class1-select-processing-models)=

```{autoprogram} mhcflurry.select_processing_models_command:parser
:prog: mhcflurry-class1-select-processing-models
```

(mhcflurry-class1-train-presentation-models)=

```{autoprogram} mhcflurry.train_presentation_models_command:parser
:prog: mhcflurry-class1-train-presentation-models
```

(mhcflurry-calibrate-percentile-ranks)=

```{autoprogram} mhcflurry.calibrate_percentile_ranks_command:parser
:prog: mhcflurry-calibrate-percentile-ranks
```

```{note}
`mhcflurry-pseudosequences` is a shell-helper CLI for the
pseudosequence CSV registry (filename / path / list subcommands).
Run `mhcflurry-pseudosequences --help` for its argument forms.
```
