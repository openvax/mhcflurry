import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True

import pandas
import tempfile
import pickle

from numpy.testing import assert_, assert_equal, assert_allclose, assert_array_equal
from nose.tools import assert_greater, assert_less
import numpy

from mhcflurry.class1_processing_predictor import Class1ProcessingPredictor

from mhcflurry.common import random_peptides

from mhcflurry.testing_utils import cleanup, startup

from .test_class1_processing_neural_network import train_basic_network

AFFINITY_PREDICTOR = None

def setup():
    pass


def teardown():
    pass


def test_basic():
    network = train_basic_network(num=10000, do_assertions=False, max_epochs=10)
    predictor = Class1ProcessingPredictor(models=[network])

    num=10000
    df = pandas.DataFrame({
        "n_flank": random_peptides(num, 10),
        "c_flank": random_peptides(num, 10),
        "peptide": random_peptides(num, 9),
    })
    df["score"] = predictor.predict(df.peptide, df.n_flank, df.c_flank)

    # Test predictions are deterministic
    df1b = predictor.predict_to_dataframe(
        peptides=df.peptide.values,
        n_flanks=df.n_flank.values,
        c_flanks=df.c_flank.values)
    assert_array_equal(df.score.values, df1b.score.values)

    # Test saving and loading
    models_dir = tempfile.mkdtemp("_models")
    print(models_dir)
    predictor.save(models_dir)
    predictor2 = Class1ProcessingPredictor.load(models_dir)

    df2 = predictor2.predict_to_dataframe(
        peptides=df.peptide.values,
        n_flanks=df.n_flank.values,
        c_flanks=df.c_flank.values)
    assert_array_equal(df.score.values, df2.score.values)

    # Test pickling
    predictor3 = pickle.loads(
        pickle.dumps(predictor, protocol=pickle.HIGHEST_PROTOCOL))
    df3 = predictor3.predict_to_dataframe(
        peptides=df.peptide.values,
        n_flanks=df.n_flank.values,
        c_flanks=df.c_flank.values)
    assert_array_equal(df.score.values, df3.score.values)

