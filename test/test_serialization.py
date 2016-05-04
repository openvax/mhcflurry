from mhcflurry import Class1BindingPredictor
from tempfile import NamedTemporaryFile
import numpy as np
from os import remove

def test_predict_after_saving_model_to_disk():
    # don't even both fitting the model, just save its random weights
    # and check we get the same predictions back afterward
    model = Class1BindingPredictor.from_hyperparameters(name="rando")
    peptides = ["A" * 9, "C" * 9]
    original_predictions = model.predict_peptides_ic50(peptides)
    json_filename = NamedTemporaryFile("w", delete=False).name
    hdf_filename = NamedTemporaryFile("w+b", delete=False).name
    print("JSON: %s" % json_filename)
    print("HDF5: %s" % hdf_filename)

    model.to_disk(json_filename, hdf_filename, overwrite=True)

    deserialized_model = Class1BindingPredictor.from_disk(json_filename, hdf_filename)
    assert deserialized_model.model is not None

    deserialized_predictions = deserialized_model.predict_peptides_ic50(peptides)

    assert np.allclose(original_predictions, deserialized_predictions), (
        peptides, original_predictions, deserialized_predictions)
    remove(json_filename)
    remove(hdf_filename)

if __name__ == "__main__":
    test_predict_after_saving_model_to_disk()
