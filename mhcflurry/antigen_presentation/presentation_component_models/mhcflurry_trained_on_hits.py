from copy import copy

import pandas
from numpy import log, exp, nanmean

from ...class1_affinity_prediction import Class1AffinityPredictor
from mhcnames import normalize_allele_name

from .mhc_binding_component_model_base import MHCBindingComponentModelBase


class MHCflurryTrainedOnHits(MHCBindingComponentModelBase):
    """
    Final model input that is a mhcflurry predictor trained on mass-spec
    hits and, optionally, affinity measurements (for example from IEDB).

    Parameters
    ------------
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
            mhcflurry_hyperparameters={},
            hit_affinity=100,
            decoy_affinity=20000,
            ensemble_size=1,
            **kwargs):

        MHCBindingComponentModelBase.__init__(self, **kwargs)
        self.mhcflurry_hyperparameters = mhcflurry_hyperparameters
        self.hit_affinity = hit_affinity
        self.decoy_affinity = decoy_affinity
        self.ensemble_size = ensemble_size
        self.predictor = Class1AffinityPredictor()

    def combine_ensemble_predictions(self, column_name, values):
        # Geometric mean
        return exp(nanmean(log(values), axis=1))

    def supports_predicting_allele(self, allele):
        return allele in self.predictor.supported_alleles

    def fit_allele(self, allele, hit_list, decoys_list):
        allele = normalize_allele_name(allele)
        hit_list = set(hit_list)
        df = pandas.DataFrame({
            "peptide": sorted(set(hit_list).union(decoys_list))
        })
        df["allele"] = allele
        df["species"] = "human"
        df["affinity"] = ((
            ~df.peptide.isin(hit_list))
            .astype(float) * (
                self.decoy_affinity - self.hit_affinity) + self.hit_affinity)
        df["sample_weight"] = 1.0
        df["peptide_length"] = 9
        self.predictor.fit_allele_specific_predictors(
            n_models=self.ensemble_size,
            architecture_hyperparameters=self.mhcflurry_hyperparameters,
            allele=allele,
            peptides=df.peptide.values,
            affinities=df.affinity.values,
        )

    def predict_allele(self, allele, peptides_list):
        return self.predictor.predict(peptides=peptides_list, allele=allele)

    def get_fit(self):
        return {
            'model': 'MHCflurryTrainedOnMassSpec',
            'predictor': self.predictor,
        }

    def restore_fit(self, fit_info):
        fit_info = dict(fit_info)
        self.predictor = fit_info.pop('predictor')

        model = fit_info.pop('model')
        assert model == 'MHCflurryTrainedOnMassSpec', model
        assert not fit_info, "Extra info in fit: %s" % str(fit_info)

    def clone(self):
        result = copy(self)
        result.reset_cache()
        result.predictor = copy(result.predictor)
        return result
