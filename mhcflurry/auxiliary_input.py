import numpy
from numpy.testing import assert_equal
import pandas

AUXILIARY_FEATURES = {}


class AuxiliaryInputEncoder(object):
    def __init__(self, alleles=None, peptides=None):
        if alleles is not None:
            alleles = numpy.array(
                alleles, copy=False).reshape((len(alleles), -1))
            assert_equal(alleles.ndim, 2)
        if peptides is not None:
            peptides = numpy.array(peptides, copy=False)
            assert_equal(peptides.ndim, 1)
        if alleles is not None and peptides is not None:
            assert_equal(alleles.shape[0], len(peptides))

        self.alleles_shape = alleles.shape if alleles is not None else None
        self.alleles_flat = alleles.flatten() if alleles is not None else None
        self.peptides = numpy.repeat(
            peptides,
            self.alleles_shape[1] if alleles is not None else 1
        ) if peptides is not None else None

    @staticmethod
    def fill_dataframe(result_df, features, feature_parameters={}):
        for feature in features:
            obj = AUXILIARY_FEATURES[feature](
                **feature_parameters.get(feature, {}))
            obj(result_df)

    @classmethod
    def get_columns(cls, features, feature_parameters={}):
        result_df = pandas.DataFrame(
            {"allele": [], "peptide": []}, dtype=str)
        cls.fill_dataframe(result_df, features, feature_parameters)
        del result_df["allele"]
        del result_df["peptide"]
        return result_df.columns.tolist()

    @staticmethod
    def split_features(how, features, feature_parameters={}):
        predicate = None
        if how == "peptide_independent":
            predicate = lambda obj: not obj.requires_peptides
        elif how == "allele_independent":
            predicate = lambda obj: not obj.requires_alleles
        else:
            raise NotImplementedError("Unsupported 'how' value", how )

        matching = []
        non_matching = []
        for feature in features:
            obj = AUXILIARY_FEATURES[feature](
                **feature_parameters.get(feature, {}))
            if predicate(obj):
                matching.append(feature)
            else:
                non_matching.append(feature)
        return (matching, non_matching)

    def get_array(self, features, feature_parameters={}):
        result_df = pandas.DataFrame()
        if self.alleles_flat is not None:
            result_df["allele"] = self.alleles_flat
        if self.peptides is not None:
            result_df["peptide"] = self.peptides

        for feature in features:
            obj = AUXILIARY_FEATURES[feature](
                **feature_parameters.get(feature, {}))
            if obj.requires_alleles and "allele" not in result_df.columns:
                raise ValueError("%s requires alleles" % obj.name)
            if obj.requires_peptides and "peptide" not in result_df.columns:
                raise ValueError("%s requires peptides" % obj.name)
            obj(result_df)
        if "allele" in result_df.columns:
            del result_df["allele"]
        if "peptide" in result_df.columns:
            del result_df["peptide"]
        result = numpy.reshape(
            result_df.values, self.alleles_shape + (-1,)).astype("float32")
        assert not numpy.isnan(result).any()
        return result


class AuxiliaryInputFeature(object):
    name = None
    requires_alleles = False
    requires_peptides = False

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class Gene(AuxiliaryInputFeature):
    name = "gene"
    requires_alleles = True

    def __call__(self, result_df):
        result_df["gene:HLA-A"] = (
            result_df.allele.fillna("").str.startswith("HLA-A"))
        result_df["gene:HLA-B"] = (
            result_df.allele.fillna("").str.startswith("HLA-B"))


AUXILIARY_FEATURES[Gene.name] = Gene
