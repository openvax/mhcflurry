import time

import pandas
import sklearn.decomposition


class AlleleEncodingTransform(object):
    def transform(self, data):
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
        fit : string to DataFrame
        """


class PCATransform(AlleleEncodingTransform):
    name = 'pca'
    serialization_keys = ['data']

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
        df = pandas.DataFrame(self.model.components_)
        df.columns = ["pca_%s" % c for c in df.columns]
        df["mean"] = self.model.mean_
        return {
            'data': df
        }

    def restore_fit(self, fit):
        assert list(fit) == ['data']
        data = fit["data"]
        self.model = sklearn.decomposition.PCA()
        self.model.mean_ = data["mean"].values
        self.model.components_ = data.drop(columns="mean").values

    def transform(self, allele_representations):
        if not self.is_fit():
            self.fit(allele_representations)
        flattened = allele_representations.reshape(
            (allele_representations.shape[0],
             allele_representations.shape[1] * allele_representations.shape[2]))
        return self.model.transform(flattened)


TRANSFORMS = dict((klass.name, klass) for klass in [PCATransform])
