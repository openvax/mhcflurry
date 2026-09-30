"""Regression coverage for reproducible preparation, metrics and subset saves."""

import bz2
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy
import pandas
import pytest

from mhcflurry import Class1ProcessingNeuralNetwork, Class1ProcessingPredictor
from mhcflurry.cli.compare_models import _ppv_at_n
from mhcflurry.percentile_calibration import fit_percent_rank_transform


ANNOTATOR = Path(__file__).resolve().parents[1] / (
    "downloads-generation/models_class1_processing/annotate_hits_with_expression.py")


@pytest.mark.parametrize("scores,expected", [
    ([1, 1, 1, 1], .5),
    ([3, 2, 2, 1], .75),
    ([4, 3, 2, 1], .5),
    ([3, 3, 1, 1], .5),
])
def test_ppv_cutoff_ties_are_permutation_invariant(scores, expected):
    y = numpy.array([1, 0, 1, 0])
    scores = numpy.array(scores)
    for order in itertools.permutations(range(4)):
        index = list(order)
        assert _ppv_at_n(y[index], scores[index], 2) == expected


def test_ppv_empty_and_full_selection():
    assert numpy.isnan(_ppv_at_n([], [], 0))
    assert _ppv_at_n([1, 0, 1], [1, 1, 1], 3) == 2 / 3
    with pytest.raises(ValueError, match="exceeds"):
        _ppv_at_n([1], [1], 2)


def annotation_rows():
    return pandas.DataFrame([
        dict(hit_id="hit.%02d" % hit, sample_id="001", peptide="SIINFEKL",
             protein_accession="protein%d" % source, n_flank="AC"[source] * 3,
             c_flank="GG", protein_ensembl="gene%d" % source,
             expression_dataset="sample", tpm=1.0)
        for hit in range(12) for source in range(2)
    ])


def test_annotation_seed_ignores_global_rng_and_row_order():
    spec = importlib.util.spec_from_file_location("annotator", ANNOTATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = annotation_rows()
    expected = module.select_annotations(rows, 42)
    numpy.random.seed(999)
    shuffled = rows.sample(frac=1, random_state=1)
    shuffled = pandas.concat([shuffled, shuffled.iloc[:3]], ignore_index=True)
    pandas.testing.assert_frame_equal(expected, module.select_annotations(shuffled, 42))
    assert not expected.equals(module.select_annotations(rows, 43))
    rows.loc[rows.protein_accession.eq("protein1"), "tpm"] = 2
    assert module.select_annotations(rows, 42).protein_accession.eq("protein1").all()


def test_annotation_process_replay_and_provenance_validation(tmp_path):
    hits, expression = tmp_path / "hits.csv", tmp_path / "expression.csv"
    annotation_rows().drop(columns="tpm").to_csv(hits, index=False)
    pandas.DataFrame({"sample": [1., 1.]}, index=["gene0", "gene1"]).to_csv(expression)
    base = [sys.executable, str(ANNOTATOR), "--hits", str(hits),
            "--expression", str(expression), "--random-seed", "42"]
    outputs = [tmp_path / "first.csv", tmp_path / "second.csv"]
    for index, output in enumerate(outputs):
        subprocess.run(base + ["--out", str(output)], check=True, capture_output=True,
                       env=dict(os.environ, PYTHONHASHSEED=str(index + 1)))
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    recorded = json.loads(Path(str(outputs[0]) + ".provenance.json").read_text())
    assert recorded["random_seed"] == 42
    assert recorded["selected_rows"] == 12
    assert len(recorded["hits_sha256"]) == 64
    compressed = tmp_path / "first.csv.bz2"
    compressed.write_bytes(bz2.compress(outputs[0].read_bytes()))
    validate = base + ["--out", str(compressed), "--provenance",
                       str(outputs[0]) + ".provenance.json", "--validate-existing"]
    subprocess.run(validate, check=True, capture_output=True)
    failure = subprocess.run(validate + ["--random-seed", "43"], capture_output=True, text=True)
    assert failure.returncode and "random_seed" in failure.stderr
    with expression.open("a") as stream:
        stream.write("extra,0\n")
    failure = subprocess.run(validate, capture_output=True, text=True)
    assert failure.returncode and "expression_sha256" in failure.stderr
    compressed.write_bytes(bz2.compress(b"corrupt table\n"))
    failure = subprocess.run(validate, capture_output=True, text=True)
    assert failure.returncode and "output_csv_sha256" in failure.stderr


def processing_predictor():
    models = []
    for seed in (1, 2, 3):
        model = Class1ProcessingNeuralNetwork(
            convolutional_filters=2, convolutional_kernel_size=3,
            n_flank_length=2, c_flank_length=2, dropout_rate=0)
        import torch
        torch.manual_seed(seed)
        model._network = model.make_network(
            **model.network_hyperparameter_defaults.subselect(model.hyperparameters))
        models.append(model)
    predictor = Class1ProcessingPredictor(models)
    predictor.percent_rank_transform = fit_percent_rank_transform(
        numpy.linspace(0, 1, 100), method="histogram", bins=10)
    return predictor


def test_processing_subset_roundtrip_and_incremental_save(tmp_path):
    predictor = processing_predictor()
    names = predictor.manifest_df.model_name.tolist()
    chosen = [names[2], names[0]]
    subset = tmp_path / "subset"
    predictor.save(str(subset), model_names_to_write=chosen)
    loaded = Class1ProcessingPredictor.load(str(subset))
    assert loaded.manifest_df.model_name.tolist() == chosen
    assert loaded.percent_rank_transform is None
    expected = Class1ProcessingPredictor([predictor.models[2], predictor.models[0]])
    numpy.testing.assert_array_equal(
        loaded.predict(["SIINFEKL"], ["AA"], ["GG"]),
        expected.predict(["SIINFEKL"], ["AA"], ["GG"]))
    # The next training completion must retain both already-written members.
    predictor.save(str(subset), model_names_to_write=[names[1]])
    loaded = Class1ProcessingPredictor.load(str(subset))
    assert loaded.manifest_df.model_name.tolist() == names
    assert loaded.percent_rank_transform is not None
    numpy.testing.assert_array_equal(
        loaded.predict(["SIINFEKL"], ["AA"], ["GG"]),
        predictor.predict(["SIINFEKL"], ["AA"], ["GG"]))


def test_processing_partial_save_validates_names_and_retained_files(tmp_path):
    predictor = processing_predictor()
    names = predictor.manifest_df.model_name.tolist()
    for selected, message in [(["missing"], "Unknown"), ([names[0]] * 2, "Duplicate")]:
        with pytest.raises(ValueError, match=message):
            predictor.save(str(tmp_path / "invalid"), model_names_to_write=selected)
        assert not (tmp_path / "invalid").exists()
    predictor.save(str(tmp_path))
    Path(predictor.weights_path(str(tmp_path), names[0])).unlink()
    before = (tmp_path / "manifest.csv").read_bytes()
    with pytest.raises(ValueError, match="Missing previously saved"):
        predictor.save(str(tmp_path), model_names_to_write=[names[1]])
    assert (tmp_path / "manifest.csv").read_bytes() == before


def test_processing_partial_save_keeps_unsaved_configs_and_clears_subset_calibration(tmp_path):
    predictor = processing_predictor()
    names = predictor.manifest_df.model_name.tolist()
    predictor.save(str(tmp_path))
    before = pandas.read_csv(tmp_path / "manifest.csv").set_index("model_name")
    predictor.models[0].hyperparameters["dropout_rate"] = .3
    predictor.save(str(tmp_path), model_names_to_write=[names[1]])
    after = pandas.read_csv(tmp_path / "manifest.csv").set_index("model_name")
    assert before.loc[names[0], "config_json"] == after.loc[names[0], "config_json"]
    # A fresh subset must not inherit stale calibration files left in its target.
    subset = tmp_path / "subset"
    subset.mkdir()
    (subset / "percent_ranks.json").write_text("stale")
    predictor.save(str(subset), model_names_to_write=[names[1]], write_percent_ranks=False)
    assert not (subset / "percent_ranks.json").exists()
    assert Class1ProcessingPredictor.load(str(subset)).percent_rank_transform is None
