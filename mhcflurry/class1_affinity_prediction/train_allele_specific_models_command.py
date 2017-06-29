"""
Train Class1 single allele models.

"""
import sys
import argparse
import json

import pandas

from mhcflurry import Class1AffinityPredictor, downloads
from mhcflurry.common import configure_logging


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--data",
    metavar="FILE.csv",
    required=True,
    help=(
        "Training data CSV. Expected columns: "
        "allele, peptide, measurement_value"))
parser.add_argument(
    "--out-models-dir",
    metavar="DIR",
    required=True,
    help="Directory to write models and manifest")
parser.add_argument(
    "--hyperparameters",
    metavar="FILE.json",
    required=True,
    help="JSON of hyperparameters")
parser.add_argument(
    "--allele",
    default=None,
    nargs="+",
    help="Alleles to train models for. If not specified, all alleles with "
    "enough measurements will be used.")
parser.add_argument(
    "--min-measurements-per-allele",
    type=int,
    metavar="N",
    default=50,
    help="Train models for alleles with >=N measurements.")
parser.add_argument(
    "--verbosity",
    type=int,
    help="Keras verbosity. Default: %(default)s",
    default=1)
parser.add_argument(
    "--use-kubeface",
    type=bool,
    help="Use Kubeface: %(default)s",
    default=False)

try:
    import kubeface
    kubeface.Client.add_args(parser)
except Exception as e:
    pass
    

def run(argv=sys.argv[1:]):
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbosity > 1)

    hyperparameters_lst = json.load(open(args.hyperparameters))
    assert isinstance(hyperparameters_lst, list)
    print("Loaded hyperparameters list: %s" % str(hyperparameters_lst))


    data_path = downloads.get_path("data_curated", args.data)
    df = pandas.read_csv(data_path)
    
    print("Loaded training data: %s" % (str(df.shape)))

    df = df.ix[
        (df.peptide.str.len() >= 8) & (df.peptide.str.len() <= 15)
    ]
    print("Subselected to 8-15mers: %s" % (str(df.shape)))

    allele_counts = df.allele.value_counts()

    if args.allele:
        alleles = args.allelle
        df = df.ix[df.allele.isin(alleles)]
    else:
        alleles = list(allele_counts.ix[
            allele_counts > args.min_measurements_per_allele
        ].index)

    print("Selected %d alleles: %s" % (len(alleles), ' '.join(alleles)))
    print("Training data: %s" % (str(df.shape)))

    predictor = Class1AffinityPredictor()

    def train_fn(arguments):
        n_models, hyperparameters, allele, peptides, affinities, Class1AffinityPredictor = arguments
        predictor = Class1AffinityPredictor()
        for model_group in range(n_models):
            print(
                "[%2d / %2d hyperparameters] "
                "[%2d / %2d replicates] "
                "[%4d / %4d alleles]: %s" % (
                    h + 1,
                    len(hyperparameters_lst),
                    model_group + 1,
                    n_models,
                    i + 1,
                    len(alleles), allele))

            train_data = df.ix[df.allele == allele].dropna().sample(
                frac=1.0)

            model = predictor.fit_allele_specific_predictors(
                n_models=n_models,
                architecture_hyperparameters=hyperparameters,
                allele=allele,
                peptides=train_data.peptide.values,
                affinities=train_data.measurement_value.values)
            
            return predictor 


    for (h, hyperparameters) in enumerate(hyperparameters_lst):
        n_models = hyperparameters.pop("n_models")

        inputs = []
        for (i, allele) in enumerate(alleles):
            train_data = df.ix[df.allele == allele].dropna().sample(
                frac=1.0)
            inputs.append((n_models, hyperparameters, 
                            allele,
                            train_data.peptide.values,
                            train_data.measurement_value.values,
                            Class1AffinityPredictor))
        
        map_fn = map
        
        if args.use_kubeface:
            client = kubeface.Client.from_args(args)            
            map_fn = client.map

        inputs = inputs[:3]
        results = map_fn(train_fn, inputs)

        final_predictor = Class1AffinityPredictor()
        final_predictor.merge(results)
        final_predictor.save(args.out_models_dir)



if __name__ == '__main__':
    run()
