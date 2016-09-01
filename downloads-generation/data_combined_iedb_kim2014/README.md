# The combined training set

This download contains the data used to train the production class1 MHCflurry models. This data is derived from a recent [IEDB](http://www.iedb.org/home_v3.php) export as well as the data from [Kim 2014](http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-15-241). 

The latest IEDB data is downloaded as part of generating this dataset. The Kim 2014 data is in its own MHCflurry download [here](../data_kim2014). 

Since affinity is measured using a variety of assays, some of which are incompatible, the `create-combined-class1-dataset.py` script filters the available Class I binding assays in IEDB by only retaining those with high correlation to overlapping measurements in BD2013. 

To generate this download run:

```
./GENERATE.sh
```