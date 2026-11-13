[![Build Status](https://github.com/openvax/mhcflurry/actions/workflows/ci.yml/badge.svg)](https://github.com/openvax/mhcflurry/actions/workflows/ci.yml)
[![Docs](https://github.com/openvax/mhcflurry/actions/workflows/docs.yml/badge.svg)](https://openvax.github.io/mhcflurry/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvax/mhcflurry/blob/master/notebooks/mhcflurry-colab.ipynb)

# MHCflurry

MHCflurry predicts which peptides are likely to be displayed by MHC class I
molecules. Given peptides or protein sequences and MHC class I alleles, its
pretrained models predict:

- Binding affinity: how strongly a peptide binds an MHC allele, in nM.
- Antigen processing: whether cellular processing favors the peptide.
- Presentation: a combined score using binding and processing predictions.

You can use the released models from the command line or Python, scan proteins
for candidate epitopes, or train models on your own data.

In the openvax stack, MHCflurry is the built-in class I predictor:
[mhcgnomes](https://github.com/pirl-unc/mhcgnomes) normalizes its allele names,
and [mhctools](https://github.com/openvax/mhctools) wraps it alongside
NetMHCpan and other tools behind one interface.

## Install

MHCflurry needs Python 3.11 or later on Linux or macOS. Install the latest
stable 2.3 patch release and download the pretrained presentation models:

```shell
pip install --upgrade "mhcflurry>=2.3,<2.4"
mhcflurry downloads fetch models_class1_presentation
```

The presentation bundle includes the affinity and processing components.
Conda users should follow
[using conda](https://openvax.github.io/mhcflurry/intro.html#using-conda).
<a id="docker"></a>There is also a [Docker image](https://openvax.github.io/mhcflurry/docker.html),
and a [Colab notebook](https://colab.research.google.com/github/openvax/mhcflurry/blob/master/notebooks/mhcflurry-colab.ipynb)
that needs no installation at all.

## Quick start

```python
from mhcflurry import Class1PresentationPredictor

predictor = Class1PresentationPredictor.load()
df = predictor.predict(
    peptides=["SIINFEKL", "NLVPMVATV"],
    alleles=["HLA-A*02:01", "HLA-B*07:02"],
    verbose=0,
)
print(df[["peptide", "best_allele", "affinity", "presentation_score"]])
```

```
     peptide  best_allele     affinity  presentation_score
0   SIINFEKL  HLA-A*02:01  6249.061165            0.033052
1  NLVPMVATV  HLA-A*02:01    34.716625            0.899710
```

The allele list is one MHC allele set, so each peptide gets one row reporting
its strongest-binding allele. Pass a dict of sample name to allele list for
several genotypes. See the
[Python tutorial](https://openvax.github.io/mhcflurry/python_tutorial.html).

## Command line

Predict for peptides you supply:

```shell
mhcflurry predict \
    --alleles HLA-A0201 HLA-A0301 \
    --peptides SIINFEKL SIINFEKD SIINFEKQ \
    --out predictions.csv
```

`predictions.csv` then holds one row per peptide and allele:

```
allele,peptide,mhcflurry_affinity,mhcflurry_affinity_percentile,mhcflurry_processing_score,mhcflurry_presentation_score,mhcflurry_presentation_percentile
HLA-A0201,SIINFEKL,6249.065499493528,7.964165930516958,0.40422916412353516,0.033052283100532154,6.944031455219743
HLA-A0201,SIINFEKD,23909.568760090307,37.47775351146445,0.37816470861434937,0.0073666587891107905,37.09948787355323
HLA-A0201,SIINFEKQ,18116.302643992014,24.233721813265326,0.38493812084198,0.010121858361444535,27.123365633634172
HLA-A0301,SIINFEKL,33742.20326754587,33.1477898986373,0.40422916412353516,0.005571558837167078,47.903918632744286
HLA-A0301,SIINFEKD,37091.75804533898,70.70629825509532,0.37816470861434937,0.004612722441857152,55.74056664248984
HLA-A0301,SIINFEKQ,32357.81853037747,25.848763661316145,0.38493812084198,0.00545944147265313,48.749722985590026
```

Or scan a protein sequence for candidate ligands:

```shell
mhcflurry predict-scan \
    --sequences MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHS \
    --alleles 'HLA-A*02:01' \
    --out scan.csv
```

`mhcflurry --help` lists every subcommand, including the training, selection
and calibration workflows. The
[introduction](https://openvax.github.io/mhcflurry/intro.html#understand-the-results)
explains how to read the output columns.

To complete subcommands and options with Tab, run
`activate-global-python-argcomplete` once, or add
`eval "$(register-python-argcomplete mhcflurry)"` to your shell startup file.

Scripts that use `mhcflurry-predict`, `mhcflurry-predict-scan` or
`mhcflurry-downloads` keep working. The other older `mhcflurry-*` names are
deprecated in favour of `mhcflurry <subcommand>`; see the
[2.3.0 release notes](https://github.com/openvax/mhcflurry/blob/master/RELEASE_NOTES_2.3.0.md).

## Documentation

Full documentation: [openvax.github.io/mhcflurry](https://openvax.github.io/mhcflurry/).

- [Introduction and installation](https://openvax.github.io/mhcflurry/intro.html): start here
- [Command-line tutorial](https://openvax.github.io/mhcflurry/commandline_tutorial.html) and [Python tutorial](https://openvax.github.io/mhcflurry/python_tutorial.html)
- [Docker](https://openvax.github.io/mhcflurry/docker.html): published images, notebooks, and building your own
- [Model weights and older releases](https://openvax.github.io/mhcflurry/model_downloads.html): browse weights with `mhcflurry downloads info` and select one with `--model-release`
- [Evaluation of the 2.3.0 weights](https://openvax.github.io/mhcflurry/release_model_evaluation.html) and [training your own models](https://openvax.github.io/mhcflurry/training.html)
- [Command reference](https://openvax.github.io/mhcflurry/commandline_tools.html) and [API reference](https://openvax.github.io/mhcflurry/api.html)
- [Predicted binding motifs](https://openvax.github.io/mhcflurry-motifs/)

Please [file an issue](https://github.com/openvax/mhcflurry/issues) if you have
questions or encounter problems.

## Development

Contributions are welcome. Start with [CONTRIBUTING.md](https://github.com/openvax/mhcflurry/blob/master/CONTRIBUTING.md):

```shell
source develop.sh          # create/activate .venv and editable-install
./lint.sh                  # ruff check mhcflurry/ test/
python -m pytest test/     # tests live in test/, singular
```

The [testing guide](https://openvax.github.io/mhcflurry/testing.html) describes
the fast local checks and the full suite.

<a id="citing-mhcflurry"></a>

## Citation

If you use MHCflurry in your research, please cite:

> T. O'Donnell, A. Rubinsteyn, U. Laserson. "MHCflurry 2.0: Improved
> pan-allele prediction of MHC I-presented peptides by incorporating antigen
> processing," *Cell Systems*, 2020.
> <https://doi.org/10.1016/j.cels.2020.06.010>

> T. O'Donnell, A. Rubinsteyn, M. Bonsack, A. B. Riemer, U. Laserson, and
> J. Hammerbacher, "MHCflurry: Open-Source Class I MHC Binding Affinity
> Prediction," *Cell Systems*, 2018.
> <https://doi.org/10.1016/j.cels.2018.05.014>

## License

MHCflurry is released under the
[Apache License 2.0](https://github.com/openvax/mhcflurry/blob/master/LICENSE).
