from copy import copy

import pandas
from numpy import log, exp, nanmean

from ...dataset import Dataset
from ...class1_allele_specific import Class1BindingPredictor
from ...common import normalize_allele_name

from .mhc_binding_component_model_base import MHCBindingComponentModelBase


MHCFLURRY_DEFAULT_HYPERPARAMETERS = dict(
    embedding_output_dim=8,
    dropout_probability=0.25)


class MHCflurryTrainedOnHits(MHCBindingComponentModelBase):
    """
    Final model input that is a mhcflurry predictor trained on mass-spec
    hits and, optionally, affinity measurements (for example from IEDB).

    Parameters
    ------------
    iedb_dataset : mhcflurry.Dataset
        IEDB data for this allele. If not specified no iedb data is used.

    mhcflurry_hyperparameters : dict

    hit_affinity : float
        nM affinity to use for hits

    decoy_affinity : float
        nM affinity to use for decoys

    **kwargs : dict
        Passed to MHCBindingComponentModel()
    """

    def __init__(
            self,
            iedb_dataset=None,
            mhcflurry_hyperparameters=MHCFLURRY_DEFAULT_HYPERPARAMETERS,
            hit_affinity=100,
            decoy_affinity=20000,
            **kwargs):

        MHCBindingComponentModelBase.__init__(self, **kwargs)
        self.iedb_dataset = iedb_dataset
        self.mhcflurry_hyperparameters = mhcflurry_hyperparameters
        self.hit_affinity = hit_affinity
        self.decoy_affinity = decoy_affinity

        self.allele_to_model = {}

    def combine_ensemble_predictions(self, column_name, values):
        # Geometric mean
        return exp(nanmean(log(values), axis=1))

    def supports_predicting_allele(self, allele):
        return allele in self.allele_to_model

    def fit_allele(self, allele, hit_list, decoys_list):
        if self.allele_to_model is None:
            self.allele_to_model = {}

        assert allele not in self.allele_to_model, \
            "TODO: Support training on >1 experiments with same allele " \
            + str(self.allele_to_model)

        mhcflurry_allele = normalize_allele_name(allele)

        extra_hits = hit_list = set(hit_list)

        iedb_dataset_df = None
        if self.iedb_dataset is not None:
            iedb_dataset_df = (
                self.iedb_dataset.get_allele(mhcflurry_allele).to_dataframe())
            extra_hits = hit_list.difference(set(iedb_dataset_df.peptide))
            print("Using %d / %d ms hits not in iedb in augmented model" % (
                len(extra_hits),
                len(hit_list)))

        df = pandas.DataFrame({
            "peptide": sorted(set(hit_list).union(decoys_list))
        })
        df["allele"] = mhcflurry_allele
        df["species"] = "human"
        df["affinity"] = ((
            ~df.peptide.isin(hit_list))
            .astype(float) * (
                self.decoy_affinity - self.hit_affinity) + self.hit_affinity)
        df["sample_weight"] = 1.0
        df["peptide_length"] = 9

        if self.iedb_dataset is not None:
            df = df.append(iedb_dataset_df, ignore_index=True)

        dataset = Dataset(
            df.sample(frac=1))  # shuffle dataframe
        print("Train data: ", dataset)
        model = Class1BindingPredictor(
            **self.mhcflurry_hyperparameters)
        model.fit_dataset(dataset, verbose=True)
        self.allele_to_model[allele] = model

    def predict_allele(self, allele, peptides_list):
        assert self.allele_to_model, "Must fit first"
        return self.allele_to_model[allele].predict(peptides_list)

    def get_fit(self):
        return {
            'model': 'MHCflurryTrainedOnMassSpec',
            'allele_to_model': self.allele_to_model,
        }

    def restore_fit(self, fit_info):
        fit_info = dict(fit_info)
        self.allele_to_model = fit_info.pop('allele_to_model')

        model = fit_info.pop('model')
        assert model == 'MHCflurryTrainedOnMassSpec', model
        assert not fit_info, "Extra info in fit: %s" % str(fit_info)

    def clone(self):
        result = copy(self)
        result.reset_cache()
        result.allele_to_model = copy(result.allele_to_model)
        return result
