"""
Hyperparameter (neural network options) management
"""
from __future__ import (
    print_function,
    division,
    absolute_import,
)
import itertools


class HyperparameterDefaults(object):
    """
    Class for managing hyperparameters. Thin wrapper around a dict.

    Instances of this class are a specification of the hyperparameters
    *supported* by a model and their defaults. The particular
    hyperparameter settings to be used, for example, to train a model
    are kept in plain dicts.
    """
    def __init__(self, **defaults):
        self.defaults = dict(defaults)

    def extend(self, other):
        """
        Return a new HyperparameterDefaults instance containing the
        hyperparameters from the current instance combined with
        those from other.

        It is an error if self and other have any hyperparameters in
        common.
        """
        overlap = [key for key in other.defaults if key in self.defaults]
        if overlap:
            raise ValueError(
                "Duplicate hyperparameter(s): %s" % " ".join(overlap))
        new = dict(self.defaults)
        new.update(other.defaults)
        return HyperparameterDefaults(**new)

    def with_defaults(self, obj):
        """
        Given a dict of hyperparameter settings, return a dict containing
        those settings augmented by the defaults for any keys missing from
        the dict.
        """
        self.check_valid_keys(obj)
        obj = dict(obj)
        for (key, value) in self.defaults.items():
            if key not in obj:
                obj[key] = value
        return obj

    def subselect(self, obj):
        """
        Filter a dict of hyperparameter settings to only those keys defined
        in this HyperparameterDefaults  .
        """
        return dict(
            (key, value) for (key, value)
            in obj.items()
            if key in self.defaults)

    def check_valid_keys(self, obj):
        """
        Given a dict of hyperparameter settings, throw an exception if any
        keys are not defined in this HyperparameterDefaults instance.
        """
        invalid_keys = [
            x for x in obj if x not in self.defaults
        ]
        if invalid_keys:
            raise ValueError(
                "No such model parameters: %s. Valid parameters are: %s"
                % (" ".join(invalid_keys), " ".join(self.defaults)))

    def models_grid(self, **kwargs):
        '''
        Make a grid of models by taking the cartesian product of all specified
        model parameter lists.

        Parameters
        -----------
        The valid kwarg parameters are the entries of this
        HyperparameterDefaults instance. Each parameter must be a list
        giving the values to search across.

        Returns
        -----------
        list of dict giving the parameters for each model. The length of the
        list is the product of the lengths of the input lists.
        '''

        # Check parameters
        self.check_valid_keys(kwargs)
        for (key, value) in kwargs.items():
            if not isinstance(value, list):
                raise ValueError(
                    "All parameters must be lists, but %s is %s"
                    % (key, str(type(value))))

        # Make models, using defaults.
        parameters = dict(
            (key, [value]) for (key, value) in self.defaults.items())
        parameters.update(kwargs)
        parameter_names = list(parameters)
        parameter_values = [parameters[name] for name in parameter_names]

        models = [
            dict(zip(parameter_names, model_values))
            for model_values in itertools.product(*parameter_values)
        ]
        return models
