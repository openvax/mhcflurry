"""Regression tests for 2.5.3 audit bugfixes (#550-#563, #570, #571, #573, #574, #576)."""
from argparse import Namespace
import hashlib
import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest

from mhcflurry import (
    Class1AffinityPredictor,
    Class1NeuralNetwork,
    Class1PresentationPredictor,
    Class1ProcessingPredictor,
    downloads,
)
from mhcflurry.affinity.persistence import (
    read_supported_alleles,
    read_supported_peptide_lengths,
)
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.class1_processing_neural_network import Class1ProcessingNeuralNetwork
from mhcflurry.cli import downloads_command
from mhcflurry.cli.reassign_mass_spec_training_data import (
    reassign_mass_spec_training_data,
)
from mhcflurry.encodable_sequences import EncodableSequences, EncodingError
from mhcflurry.flanking_encoding import FlankingEncoding
from mhcflurry.mhcseqs_pseudosequences import (
    HLA_A0201_GROOVE1,
    HLA_A0201_GROOVE2,
    _GROOVE_INDICES_BY_LENGTH,
    _align_domain_to_reference,
    _extract_groove1_residues,
    _extract_groove2_residues,
    build_pseudosequences_from_mhcseqs,
    extend_allele_to_sequence_from_mhcseqs,
)
from mhcflurry.training_folds import read_processing_training_data
from mhcflurry.training_provenance import decode_sources, deduplicate_measurements


def _make_tar_bz2(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for name, content in files:
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def test_550_read_supported_metadata_handles_large_config_json(tmp_path):
    """#550: manifest.csv config_json > 128 KB must not raise _csv.Error."""
    net = Class1NeuralNetwork(layer_sizes=[4], max_epochs=1, validation_split=0.0)
    net.fit(["SIINFEKL", "NLVPMVATV"], [50.0, 5000.0], verbose=0)
    cfg = net.get_config()
    cfg["padding_blob"] = "x" * 200_000  # > 131,072 default csv.field_size_limit
    (tmp_path / "manifest.csv").write_text(
        "model_name,allele,config_json\n"
        f'm0,HLA-A*02:01,"{json.dumps(cfg).replace(chr(34), chr(34) * 2)}"\n',
        encoding="utf-8",
    )
    assert read_supported_alleles(str(tmp_path)) == ["HLA-A*02:01"]
    min_l, max_l = read_supported_peptide_lengths(str(tmp_path))
    assert (min_l, max_l) == net.supported_peptide_lengths


def test_551_and_552_downloads_verify_text_and_invalid_store_handling(
    tmp_path, monkeypatch, capsys
):
    """#551 & #552: verify text reports corruption; invalid managed store never falls back."""
    release = "2.3.0"
    downloads_root = tmp_path / "cache" / release
    archives_dir = tmp_path / "archives"
    archives_dir.mkdir(parents=True)

    entry = downloads.get_release_downloads(release)["models_class1_presentation"]["metadata"]
    archive_name = entry["url"].rsplit("/", 1)[-1]
    archive_bytes = _make_tar_bz2([("models/weights.csv", b"intercept\n0.0\n")])
    (archives_dir / archive_name).write_bytes(archive_bytes)

    meta_copy = json.loads(json.dumps(downloads.get_downloads_metadata()))
    for item in meta_copy["releases"][release]["downloads"]:
        if item["name"] == "models_class1_presentation":
            item["sha256"] = hashlib.sha256(archive_bytes).hexdigest()
            item["size"] = len(archive_bytes)
    monkeypatch.setattr(downloads, "_METADATA", meta_copy)
    monkeypatch.setattr(downloads, "_DOWNLOADS_DIR", str(downloads_root))
    monkeypatch.setattr(downloads, "_CURRENT_RELEASE", release)

    # Create a legacy dir alongside the managed store.
    legacy_dir = downloads_root / "models_class1_presentation" / "models"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "weights.csv").write_text("legacy\n", encoding="utf-8")

    downloads.install_download(
        "models_class1_presentation",
        release=release,
        already_downloaded_dir=str(archives_dir),
        force=True,
    )

    # #552: relative override prefers managed store over legacy dir.
    monkeypatch.setattr(
        downloads,
        "_MHCFLURRY_DEFAULT_CLASS1_PRESENTATION_MODELS_DIR",
        "models_class1_presentation/models",
    )
    resolved = downloads.get_default_class1_presentation_models_dir(test_exists=True)
    assert ".datacache" in resolved

    # Corrupt an extracted file inside the managed store.
    managed_models = downloads.get_path("models_class1_presentation", "models", release=release)
    with open(f"{managed_models}/weights.csv", "wb") as fd:
        fd.write(b"TAMPERED")

    # #551: `downloads info --verify` text must surface invalid status, not clean 2.3.0.
    downloads_command.info_subcommand(
        Namespace(download_name=None, release=release, json=False, verbose=False, verify=True)
    )
    out = capsys.readouterr().out
    assert "invalid" in out.lower()
    assert "not file integrity" not in out

    # Corrupt manifest so even verify_files=False detects invalid managed store.
    bundle_dir = downloads.get_path("models_class1_presentation", "", release=release)
    with open(f"{bundle_dir}/.datacache-manifest.json", "w", encoding="utf-8") as fd:
        fd.write("{corrupt")

    info = downloads.inspect_download("models_class1_presentation", release=release)
    assert info["install_status"] == "invalid"
    assert ".datacache" in info["path"]

    # #552: get_path and get_model_release_dir must raise, never silently return stale legacy dir.
    with pytest.raises(RuntimeError, match="--force"):
        downloads.get_path("models_class1_presentation", "models", release=release)
    with pytest.raises(RuntimeError, match="--force"):
        downloads.get_model_release_dir(release)


def test_553_and_554_pseudosequence_fallback_and_nan_handling():
    """#553 & #554: fallback groove fast paths and NaN-safe row handling."""
    g1_pos, g2_pos = _GROOVE_INDICES_BY_LENGTH[39]
    assert HLA_A0201_GROOVE1[59] == "W"
    assert _extract_groove1_residues(HLA_A0201_GROOVE1, g1_pos, imgt_aligner=None) == (
        "".join(HLA_A0201_GROOVE1[p] for p in g1_pos)
    )
    # Compensating indel in groove2 (1-aa del near pos 17 + 1-aa ins near pos 59)
    # keeps len=92 and Cys at indices 10 & 73; fallback must align rather than slice.
    comp_g2 = (
        HLA_A0201_GROOVE2[:17]
        + HLA_A0201_GROOVE2[18:59]
        + "A"
        + HLA_A0201_GROOVE2[59:]
    )
    assert len(comp_g2) == len(HLA_A0201_GROOVE2) and comp_g2[10] == "C" and comp_g2[73] == "C"
    assert _extract_groove2_residues(comp_g2, g2_pos, imgt_aligner=None) == (
        _align_domain_to_reference(comp_g2, HLA_A0201_GROOVE2, g2_pos)
    )

    # #554: NaN two_field_allele falls back to allele; NaN existing sequence is skipped.
    rows = [{
        "mhc_class": "I",
        "two_field_allele": float("nan"),
        "allele": "HLA-A*02:01",
        "groove1": HLA_A0201_GROOVE1,
        "groove2": HLA_A0201_GROOVE2,
    }]
    built = build_pseudosequences_from_mhcseqs(rows=rows, normalize_alleles=True)
    assert "HLA-A*02:01" in built
    extended = extend_allele_to_sequence_from_mhcseqs(
        {"HLA-A*02:01": "Y" * 39, "HLA-B*07:02": float("nan")}, rows=[]
    )
    assert extended == {"HLA-A*02:01": "Y" * 39}


def test_556_numpy2_list_sample_weights_and_output_indices():
    """#556: Class1NeuralNetwork.fit accepts Python lists on NumPy 2.x."""
    m = Class1NeuralNetwork(
        max_epochs=1,
        layer_sizes=[4],
        locally_connected_layers=[],
        validation_split=0.0,
        num_outputs=2,
        loss="custom:mse_with_inequalities_and_multiple_outputs",
    )
    m.fit(
        ["SIINFEKL", "SIINFEKM", "SIINFEKA", "SIINFEKP"],
        [10.0, 20.0, 5000.0, 10000.0],
        sample_weights=[1.0, 1.0, 1.0, 1.0],
        output_indices=[0, 1, 0, 1],
        verbose=0,
    )


def test_557_allele_encoding_handles_none_on_pandas3():
    """#557: AlleleEncoding supports None mixed with strings under Pandas 3."""
    enc = AlleleEncoding([None, "HLA-A*02:01"], {"HLA-A*02:01": "Y" * 34})
    assert enc.indices.tolist() == [0, 1]
    compacted = enc.compact()
    assert compacted.indices.tolist() == [0, 1]


def test_558_and_559_pandas3_checkpoint_provenance_and_reassign_dtypes(tmp_path):
    """#558 & #559: Pandas 3 strict-dtype assignment for checkpoints, provenance, and MS values."""
    # #558: Processing predictor reload with all-NaN checkpoint_best_weights column.
    proc_dir = tmp_path / "proc"
    net = Class1ProcessingNeuralNetwork(
        max_epochs=1, convolutional_filters=4, n_flank_length=5, c_flank_length=5
    )
    net._network = net.make_network(
        **net.network_hyperparameter_defaults.subselect(net.hyperparameters)
    )
    net.checkpoint_weights["best"] = net.get_weights()
    pred = Class1ProcessingPredictor([net])
    pred.save(str(proc_dir))
    pred.models[0].checkpoint_weights.clear()
    pred.save(str(proc_dir))
    reloaded = Class1ProcessingPredictor.load(str(proc_dir))
    reloaded.models[0].checkpoint_weights["best"] = reloaded.models[0].get_weights()
    reloaded.save(str(proc_dir))

    # #559: training_provenance empty column & pandas.NA; reassign float into int64 CSV.
    prov_df = pd.read_csv(
        io.StringIO("allele,peptide,source_provenance\nHLA-A*02:01,SIINFEKL,\nHLA-A*02:01,SIINFEKL,\n")
    )
    deduped = deduplicate_measurements(prov_df, ["allele", "peptide"])
    assert len(deduped) == 1
    assert decode_sources(pd.NA) == []

    ms_csv = tmp_path / "ms.csv"
    ms_csv.write_text(
        "allele,peptide,measurement_value,measurement_kind,measurement_inequality\n"
        "HLA-A*02:01,SIINFEKL,100,mass_spec,<\n",
        encoding="utf-8",
    )
    reassigned = reassign_mass_spec_training_data(str(ms_csv), set_measurement_value=100.5)
    assert reassigned["measurement_value"].iloc[0] == pytest.approx(100.5)


def test_555_560_561_562_563_570_affinity_and_nn_bugs(tmp_path):
    """#555, #560, #561, #562, #563, #570: pan-allele merge/optimize/cache/activation fixes."""
    a2s = {
        "HLA-A*01:01": "A" * 34,
        "HLA-A*02:01": "C" * 34,
        "HLA-B*07:02": "D" * 34,
        "HLA-C*01:02": "E" * 34,
    }
    enc = AlleleEncoding(["HLA-A*01:01", "HLA-A*02:01", "HLA-B*07:02", "HLA-C*01:02"], a2s)
    m1 = Class1NeuralNetwork(
        max_epochs=1,
        layer_sizes=[4],
        locally_connected_layers=[{"filters": 4, "kernel_size": 3, "activation": "relu"}],
        peptide_allele_merge_method="concatenate",
        validation_split=0.0,
    )
    m2 = Class1NeuralNetwork(
        max_epochs=1,
        layer_sizes=[4],
        locally_connected_layers=[{"filters": 4, "kernel_size": 3, "activation": "relu"}],
        peptide_allele_merge_method="concatenate",
        validation_split=0.0,
    )
    peps = ["SIINFEKL", "NLVPMVATV", "GILGFVFTL", "YLQPRTFLL"]
    m1.fit(peps, [50.0, 500.0, 100.0, 5000.0], allele_encoding=enc, verbose=0)
    m2.fit(peps, [60.0, 400.0, 120.0, 4000.0], allele_encoding=enc, verbose=0)

    # #561: merged network preserves LocallyConnected1D activation on reload.
    merged_nn = Class1NeuralNetwork.merge([m1, m2], merge_method="concatenate")
    reloaded_nn = Class1NeuralNetwork.from_config(
        merged_nn.get_config(), weights=merged_nn.get_weights()
    )
    assert reloaded_nn.network().networks[0].lc_layers[0].activation_name == "relu"

    # #560 & #570: optimize preserves models_dir; merge rejects optimized pan predictors.
    p1 = Class1AffinityPredictor(
        class1_pan_allele_models=[m1, m2], allele_to_sequence=a2s, models_dir=str(tmp_path)
    )
    assert p1.optimize() is True
    assert p1.models_dir == str(tmp_path)
    with pytest.raises(ValueError, match="optimization_level=0"):
        Class1AffinityPredictor.merge([p1, p1])
    with pytest.raises(ValueError, match="optimization_level=0"):
        p1.add_pan_allele_model(m1)

    # #555: calibrate_percentile_ranks_fast cache hit still syncs allele embeddings after predict().
    cal_peps = ["SIINFEKL", "NLVPMVATV", "GILGFVFTL", "YLQPRTFLL", "AAADDDCCC", "DDDEEEFFF"]
    p1.calibrate_percentile_ranks_fast(
        cal_peps, alleles=["HLA-A*01:01"], method="histogram", device="cpu", verbose=False
    )
    p1.predict(["SIINFEKL"], allele="HLA-C*01:02")
    p1.calibrate_percentile_ranks_fast(
        cal_peps, alleles=["HLA-A*02:01", "HLA-B*07:02"], method="histogram", device="cpu", verbose=False
    )
    assert "HLA-B*07:02" in p1.allele_to_percent_rank_transform

    # #562: optimize skips multi-output pan-allele ensembles instead of corrupting outputs.
    mo1 = Class1NeuralNetwork(
        max_epochs=1, layer_sizes=[4], locally_connected_layers=[], validation_split=0.0,
        peptide_allele_merge_method="concatenate",
        num_outputs=2, loss="custom:mse_with_inequalities_and_multiple_outputs",
    )
    mo2 = Class1NeuralNetwork(
        max_epochs=1, layer_sizes=[4], locally_connected_layers=[], validation_split=0.0,
        peptide_allele_merge_method="concatenate",
        num_outputs=2, loss="custom:mse_with_inequalities_and_multiple_outputs",
    )
    mo1.fit(peps, [50.0, 500.0, 100.0, 5000.0], allele_encoding=enc, output_indices=[0, 1, 0, 1], verbose=0)
    mo2.fit(peps, [60.0, 400.0, 120.0, 4000.0], allele_encoding=enc, output_indices=[0, 1, 0, 1], verbose=0)
    p_mo = Class1AffinityPredictor(class1_pan_allele_models=[mo1, mo2], allele_to_sequence=a2s)
    assert p_mo.optimize() is False
    assert p_mo.predict(["SIINFEKL"], allele="HLA-A*01:01").shape == (1,)

    # #563: prediction_cache keys on output_index, copies on miss, clears on set_weights_list/fit.
    sa_mo = Class1NeuralNetwork(
        max_epochs=1, layer_sizes=[4], locally_connected_layers=[], validation_split=0.0,
        num_outputs=2, loss="custom:mse_with_inequalities_and_multiple_outputs",
    )
    sa_mo.fit(peps, [50.0, 500.0, 100.0, 5000.0], output_indices=[0, 1, 0, 1], verbose=0)
    enc_p = EncodableSequences.create(["SIINFEKL", "NLVPMVATV"])
    out0 = sa_mo.predict(enc_p, output_index=0)
    out0[:] = -999.0  # mutate returned array on cache miss
    assert sa_mo.predict(enc_p, output_index=0)[0] > 0  # cache not corrupted
    out_all = sa_mo.predict(enc_p, output_index=None)
    assert out_all.shape == (2, 2)
    old_val = sa_mo.predict(enc_p, output_index=0).copy()
    sa_mo.set_weights_list([np.zeros_like(w) for w in sa_mo.get_weights()], auto_convert_keras=False)
    assert not np.allclose(sa_mo.predict(enc_p, output_index=0), old_val)


def test_571_processing_fit_preserves_restored_weights():
    """#571: Class1ProcessingNeuralNetwork.fit() must not wipe from_config() weights."""
    fe = FlankingEncoding(["SIINFEKL", "AAADDDCCC"], ["NNNNN", "MMMMM"], ["CCCCC", "DDDDD"])
    net = Class1ProcessingNeuralNetwork(
        max_epochs=0, convolutional_filters=4, n_flank_length=5, c_flank_length=5, validation_split=0.0
    )
    net.fit(fe, [1.0, 0.0], verbose=-1, seed=42)
    orig_w = [w.copy() for w in net.get_weights()]
    restored = Class1ProcessingNeuralNetwork.from_config(net.get_config(), weights=net.get_weights())
    restored.fit(fe, [1.0, 0.0], verbose=-1, seed=999)
    np.testing.assert_allclose(orig_w[0], restored.get_weights()[0], rtol=0, atol=0)


def test_573_574_576_presentation_and_flanking_bugs(tmp_path):
    """#573, #574, #576: flank Series alignment, empty alleles/peptides, NaN/'NA' flanks."""
    # #576: FlankingEncoding handles None/NaN flanks, misaligned Series indices, and off-by-one EncodingError.
    fe = FlankingEncoding(
        pd.Series(["AAADDDCCC", "SIINFEKL"], index=[1, 0]),
        pd.Series(["MMMMM", None], index=[0, 1]),
        pd.Series([np.nan, "CCCCC"], index=[0, 1]),
    )
    assert fe.dataframe["peptide"].tolist() == ["AAADDDCCC", "SIINFEKL"]
    assert fe.dataframe["n_flank"].tolist() == ["MMMMM", ""]
    assert fe.dataframe["c_flank"].tolist() == ["", "CCCCC"]
    fe.categorical_encode(15, 5, 5)

    with pytest.raises(EncodingError) as exc_info:
        FlankingEncoding(["A" * 16], [""], [""]).categorical_encode(15, 5, 5, throw=True)
    assert exc_info.value.supported_peptide_lengths == (1, 15)

    # #576: read_processing_training_data preserves empty and literal "NA" flanks.
    proc_csv = tmp_path / "proc.csv"
    proc_csv.write_text(
        "sample_id,peptide,n_flank,c_flank,hit\nS1,SIINFEKL,,CCC,1\nS1,AAAAAAAA,NA,NAN,0\n",
        encoding="utf-8",
    )
    proc_df = read_processing_training_data(str(proc_csv))
    assert proc_df["n_flank"].tolist() == ["", "NA"]
    assert proc_df["c_flank"].tolist() == ["CCC", "NAN"]

    # Build a minimal presentation predictor for #573 & #574.
    aff_net = Class1NeuralNetwork(layer_sizes=[4], max_epochs=1, validation_split=0.0)
    aff_net.fit(["SIINFEKL", "AAADDDCCC"], [50.0, 5000.0], verbose=0)
    aff = Class1AffinityPredictor(allele_to_allele_specific_models={"HLA-A*02:01": [aff_net]})
    proc_net = Class1ProcessingNeuralNetwork(
        max_epochs=1, convolutional_filters=4, n_flank_length=5, c_flank_length=5, validation_split=0.0
    )
    proc_net.fit(
        FlankingEncoding(["SIINFEKL", "AAADDDCCC"], ["NNNNN", "MMMMM"], ["CCCCC", "DDDDD"]),
        [1.0, 0.0],
        verbose=-1,
    )
    proc = Class1ProcessingPredictor([proc_net])
    pres = Class1PresentationPredictor(
        affinity_predictor=aff,
        processing_predictor_with_flanks=proc,
        processing_predictor_without_flanks=proc,
        weights_dataframe=pd.DataFrame(
            {"affinity_score": [1.0, 1.0], "processing_score": [1.0, 1.0], "intercept": [0.0, 0.0]},
            index=["with_flanks", "without_flanks"],
        ),
    )

    # #573: non-RangeIndex Series must not swap or NaN-corrupt flanks.
    inp = pd.DataFrame({
        "peptide": ["SIINFEKL", "AAADDDCCC"],
        "n_flank": ["NNNNN", "MMMMM"],
        "c_flank": ["CCCCC", "DDDDD"],
    }).sort_values("peptide")  # index [1, 0]
    out = pres.predict(
        peptides=inp.peptide,
        n_flanks=inp.n_flank,
        c_flanks=inp.c_flank,
        alleles=["HLA-A*02:01"],
        include_affinity_percentile=False,
        verbose=0,
    )
    assert out.loc[0, "peptide"] == "AAADDDCCC"
    assert out.loc[0, "n_flank"] == "MMMMM"
    assert out.loc[0, "c_flank"] == "DDDDD"

    # #574: alleles=[] (processing-only), alleles={"s1": []}, alleles={}, empty peptides + sample_names.
    proc_only = pres.predict(peptides=["SIINFEKL"], alleles=[], verbose=0)
    assert "processing_score" in proc_only.columns and "affinity" not in proc_only.columns

    empty_sample_alleles = pres.predict(
        peptides=["SIINFEKL"], alleles={"s1": []}, include_affinity_percentile=False, verbose=0
    )
    assert np.isnan(empty_sample_alleles["presentation_score"].iloc[0])

    empty_aff = pres.predict_affinity(peptides=["SIINFEKL"], alleles={}, include_affinity_percentile=False, verbose=0)
    assert len(empty_aff) == 0 and "affinity" in empty_aff.columns

    empty_with_samples = pres.predict(
        peptides=[], alleles={"s1": ["HLA-A*02:01"]}, sample_names=[], include_affinity_percentile=False, verbose=0
    )
    assert len(empty_with_samples) == 0

    short_seq = pres.predict_sequences(
        sequences=["SHORT"], alleles=[["HLA-A*02:01"]], include_affinity_percentile=False, verbose=0
    )
    assert len(short_seq) == 0
