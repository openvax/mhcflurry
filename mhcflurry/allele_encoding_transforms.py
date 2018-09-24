import time

import pandas
import sklearn.decomposition


class AlleleEncodingTransform(object):
    def transform(self, allele_encoding, argument=None):
        raise NotImplementedError()

    def get_fit(self):
        """
        Get the fit for serialization, which must be in the form of one or more
        dataframes.

        Returns
        -------
        dict : string to DataFrame
        """
        raise NotImplementedError()

    def restore_fit(self, fit):
        """
        Restore a serialized fit.

        Parameters
        ----------
        fit : string to array
        """


class PCATransform(AlleleEncodingTransform):
    name = 'pca'
    serialization_keys = ['mean', 'components']

    def __init__(self):
        self.model = None

    def is_fit(self):
        return self.model is not None

    def fit(self, allele_representations):
        self.model = sklearn.decomposition.PCA()
        shape = allele_representations.shape
        flattened = allele_representations.reshape(
            (shape[0], shape[1] * shape[2]))
        print("Fitting PCA allele encoding transform on data of shape: %s" % (
            str(flattened.shape)))
        start = time.time()
        self.model.fit(flattened)
        print("Fit complete in %0.3f sec." % (time.time() - start))

    def get_fit(self):
        return {
            'mean': self.model.mean_,
            'components': self.model.components_,
        }

    def restore_fit(self, fit):
        self.model = sklearn.decomposition.PCA()
        self.model.mean_ = fit["mean"]
        self.model.components_ = fit["components"]

    def transform(self, allele_encoding, underlying_representation):
        allele_representations = allele_encoding.allele_representations(
            underlying_representation)
        if not self.is_fit():
            self.fit(allele_representations)
        flattened = allele_representations.reshape(
            (allele_representations.shape[0],
             allele_representations.shape[1] * allele_representations.shape[2]))
        return self.model.transform(flattened)


TRANSFORMS = dict((klass.name, klass) for klass in [PCATransform])
