import weakref

from copy import copy

import numpy
import pandas

from ...common import (
    dataframe_cryptographic_hash, assert_no_null, freeze_object)


def cache_dict_for_policy(policy):
    if policy == "weak":
        return weakref.WeakValueDictionary()
    elif policy == "strong":
        return {}
    elif policy == "none":
        return None
    else:
        raise ValueError("Unsupported cache policy: %s" % policy)


class PresentationComponentModel(object):
    '''
    Base class for component models to a presentation model.

    The component models are things like mhc binding affinity and cleavage,
    and the presentation model is typically a logistic regression model
    over these.
    '''
    def __init__(
            self, fit_cache_policy="weak", predictions_cache_policy="weak"):
        self.fit_cache_policy = fit_cache_policy
        self.predictions_cache_policy = predictions_cache_policy
        self.reset_cache()

    def reset_cache(self):
        self.cached_fits = cache_dict_for_policy(self.fit_cache_policy)
        self.cached_predictions = cache_dict_for_policy(
            self.predictions_cache_policy)

    def __getstate__(self):
        d = dict(self.__dict__)
        d["cached_fits"] = None
        d["cached_predictions"] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.reset_cache()

    def combine_ensemble_predictions(self, column_name, values):
        return numpy.nanmean(values, axis=1)

    def stratification_groups(self, hits_df):
        return hits_df.experiment_name

    def column_names(self):
        """
        Names for the values this final model input emits.
        Some final model inputs emit multiple related quantities, such as
        "binding affinity" and "binding percentile rank".
        """
        raise NotImplementedError(str(self))

    def requires_fitting(self):
        """
        Does this model require fitting to mass-spec data?

        For example, the 'expression' componenet models don't need to be
        fit, but some cleavage predictors and binding predictors can be
        trained on the ms data.
        """
        raise NotImplementedError(str(self))

    def clone_and_fit(self, hits_df):
        """
        Clone the object and fit to given dataset with a weakref cache.
        """
        if not self.requires_fitting():
            return self

        if self.cached_fits is None:
            key = None
            result = None
        else:
            key = dataframe_cryptographic_hash(
                hits_df[["experiment_name", "peptide"]])
            result = self.cached_fits.get(key)
        if result is None:
            print("Cache miss in clone_and_fit: %s" % str(self))
            result = self.clone()
            result.fit(hits_df)
            if self.cached_fits is not None:
                self.cached_fits[key] = result
        else:
            print("Cache hit in clone_and_fit: %s" % str(self))
        return result

    def clone_and_restore_fit(self, fit_info):
        if not self.requires_fitting():
            assert fit_info is None
            return self

        if self.cached_fits is None:
            key = None
            result = None
        else:
            key = freeze_object(fit_info)
            result = self.cached_fits.get(key)
        if result is None:
            print("Cache miss in clone_and_restore_fit: %s" % str(self))
            result = self.clone()
            result.restore_fit(fit_info)
            if self.cached_fits is not None:
                self.cached_fits[key] = result
        else:
            print("Cache hit in clone_and_restore_fit: %s" % str(self))
        return result

    def fit(self, hits_df):
        """
        Train the model.

        Parameters
        -----------
        hits_df : pandas.DataFrame
            dataframe of hits with columns 'experiment_name' and 'peptide'

        """
        if self.requires_fitting():
            raise NotImplementedError(str(self))

    def predict_for_experiment(self, experiment_name, peptides):
        """
        A more convenient prediction method to implement.

        Subclasses should override this method or predict().

        Returns
        ------------

        A dict of column name -> list of predictions for each peptide

        """
        assert self.predict != PresentationComponentModel.predict, (
            "Must override predict_for_experiment() or predict()")
        peptides_df = pandas.DataFrame({
            'peptide': peptides,
        })
        peptides_df["experiment_name"] = experiment_name
        return self.predict(peptides_df)

    def predict(self, peptides_df):
        """
        Subclasses can override either this or predict_for_experiment.

        This is the high-level predict method that users should call.

        This convenience method groups the peptides_df by experiment
        and calls predict_for_experiment on each experiment.
        """
        assert (
            self.predict_for_experiment !=
            PresentationComponentModel.predict_for_experiment)
        assert 'experiment_name' in peptides_df.columns
        assert 'peptide' in peptides_df.columns

        if self.cached_predictions is None:
            cache_key = None
            cached_result = None
        else:
            cache_key = dataframe_cryptographic_hash(peptides_df)
            cached_result = self.cached_predictions.get(cache_key)
        if cached_result is not None:
            print("Cache hit in predict: %s" % str(self))
            return cached_result
        else:
            print("Cache miss in predict: %s" % str(self))

        grouped = peptides_df.groupby("experiment_name")

        if len(grouped) == 1:
            print("%s : using single-experiment predict optimization" % (
                str(self)))
            return_value = pandas.DataFrame(
                self.predict_for_experiment(
                    str(peptides_df.iloc[0].experiment_name),
                    peptides_df.peptide.values))
            assert len(return_value) == len(peptides_df), (
                "%d != %d" % (len(return_value), len(peptides_df)),
                str(self),
                peptides_df.peptide.nunique(),
                return_value,
                peptides_df)
            assert_no_null(return_value, str(self))
        else:
            peptides_df = (
                peptides_df[["experiment_name", "peptide"]]
                .reset_index(drop=True))
            columns = self.column_names()
            result_df = peptides_df.copy()
            for col in columns:
                result_df[col] = numpy.nan
            for (experiment_name, sub_df) in grouped:
                assert (
                    result_df.loc[sub_df.index, "experiment_name"] ==
                    experiment_name).all()

                unique_peptides = list(set(sub_df.peptide))
                if len(unique_peptides) == 0:
                    continue

                result_dict = self.predict_for_experiment(
                    experiment_name, unique_peptides)

                for col in columns:
                    assert len(result_dict[col]) == len(unique_peptides), (
                        "Final model input %s: wrong number of predictions "
                        "%d (expected %d) for col %s:\n%s\n"
                        "Input was: experiment: %s, peptides:\n%s" % (
                            str(self),
                            len(result_dict[col]),
                            len(unique_peptides),
                            col,
                            result_dict[col],
                            experiment_name,
                            unique_peptides))
                    prediction_series = pandas.Series(
                        result_dict[col],
                        index=unique_peptides)
                    prediction_values = (
                        prediction_series.ix[sub_df.peptide.values]).values
                    result_df.loc[
                        sub_df.index, col
                    ] = prediction_values

            assert len(result_df) == len(peptides_df), "%s != %s" % (
                len(result_df),
                len(peptides_df))
            return_value = result_df[columns]
        if self.cached_predictions is not None:
            self.cached_predictions[cache_key] = return_value
        return dict(
            (col, return_value[col].values) for col in self.column_names())

    def clone(self):
        """
        Copy this object so that the original and copy can be fit
        independently.
        """
        if self.requires_fitting():
            # shallow copy won't work here, subclass must override.
            raise NotImplementedError(str(self))
        result = copy(self)

        # We do not want to share a cache with the clone.
        result.reset_cache()
        return result

    def get_fit(self):
        if self.requires_fitting():
            raise NotImplementedError(str(self))
        return None

    def restore_fit(self, fit_info):
        if self.requires_fitting():
            raise NotImplementedError(str(self))
        assert fit_info is None, (str(self), str(fit_info))
