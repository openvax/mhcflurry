"""Regression tests for 2.5.5 training, selection, evaluation, and CLI bugfixes (#588-#595)."""

import os
from unittest.mock import MagicMock

import numpy
import pandas
import pytest

from mhcflurry import Class1AffinityPredictor, downloads
from mhcflurry.cli import (
    compare_models,
    downloads_command,
    predict_command,
    predict_scan_command,
    select_allele_specific_models_command as sel_as,
    train_allele_specific_models_command as tas,
    train_pan_allele_models_command as tpan,
)
from mhcflurry.random_negative_peptides import RandomNegativePeptides
from mhcflurry.ranking_metrics import ppv_at_n


def test_588_train_rounds_aligned_after_shuffle_and_list_inputs(monkeypatch):
    df = pandas.DataFrame({
        "allele": ["HLA-A*02:01"] * 5 + ["HLA-A*01:01"] * 15,
        "peptide": (
            [f"TARGET{i:02d}" for i in range(5)]
            + [f"OTHER_{i:02d}" for i in range(15)]
        ),
        "measurement_value": [100.0] * 20,
        "measurement_type": ["quantitative"] * 20,
    })
    sim = pandas.DataFrame(
        [[1.0, 0.9], [0.9, 1.0]],
        index=["HLA-A*02:01", "HLA-A*01:01"],
        columns=["HLA-A*02:01", "HLA-A*01:01"],
    )
    monkeypatch.setitem(tas.WORKER_CONTEXT, "allele_similarity_matrix", sim)
    mock_pred = MagicMock()
    tas.train_model(
        n_models=1,
        allele_num=0,
        n_alleles=2,
        model_num=0,
        hyperparameter_set_num=0,
        num_hyperparameter_sets=1,
        allele="HLA-A*02:01",
        hyperparameters={"train_data": {"subset": "all", "pretrain_min_points": 10}},
        verbose=0,
        progress_print_interval=None,
        predictor=mock_pred,
        save_to=None,
        constant_data={"train_data": df, "seed": 42},
    )
    kw = mock_pred.fit_allele_specific_predictors.call_args.kwargs
    round1_peptides = set(kw["peptides"][kw["train_rounds"] > 0])
    assert round1_peptides == {f"TARGET{i:02d}" for i in range(5)}

    # Python list inputs must work without TypeError in fit_allele_specific_predictors.
    pred = Class1AffinityPredictor()
    models = pred.fit_allele_specific_predictors(
        n_models=1,
        architecture_hyperparameters_list=[{
            "max_epochs": 1,
            "layer_sizes": [4],
            "locally_connected_layers": [],
            "validation_split": 0.0,
            "random_negative_rate": 0.0,
            "random_negative_constant": 0,
            "minibatch_size": 2,
        }],
        allele="HLA-A*02:01",
        peptides=["SIINFEKL", "AAADDDCCC", "LLLEEEFFF"],
        affinities=[50.0, 5000.0, 100.0],
        inequalities=["=", "=", "<"],
        train_rounds=[0, 0, 1],
        verbose=0,
    )
    assert len(models) == 1


def test_589_train_pan_allele_optional_inequality_and_default_peptide_encoding(
        tmp_path, monkeypatch):
    # 1. Pretrain branch works when measurement_inequality is absent and
    # peptide_encoding is omitted from hyperparameters (uses model default).
    allele_seq = {"HLA-A*02:01": "A" * 37}
    from mhcflurry.allele_encoding import AlleleEncoding

    enc = AlleleEncoding(["HLA-A*02:01"], allele_to_sequence=allele_seq)
    train_df = pandas.DataFrame({
        "allele": ["HLA-A*02:01", "HLA-A*02:01"],
        "peptide": ["SIINFEKL", "AAADDDCCC"],
        "measurement_value": [50.0, 5000.0],
    })
    pretrain_calls = []

    def fake_fit_streaming(self, **kwargs):
        pretrain_calls.append(kwargs)
        self.fit_info.append({"learning_rate": 1e-3, "val_loss": [0.1]})

    def fake_fit(self, **kwargs):
        self.fit_info.append({"learning_rate": 1e-4, "val_loss": [0.1]})

    monkeypatch.setattr(
        tpan.Class1NeuralNetwork, "fit_streaming_batches", fake_fit_streaming
    )
    monkeypatch.setattr(tpan.Class1NeuralNetwork, "fit", fake_fit)
    monkeypatch.setattr(
        tpan, "pretrain_network_input_iterator", lambda **kw: iter([])
    )

    model = tpan.train_model(
        work_item_name="item0",
        work_item_num=0,
        num_work_items=1,
        architecture_num=0,
        num_architectures=1,
        fold_num=0,
        num_folds=1,
        replicate_num=0,
        num_replicates=1,
        hyperparameters={
            "train_data": {
                "pretrain": True,
                "pretrain_steps_per_epoch": 1,
                "pretrain_max_epochs": 1,
            },
            "layer_sizes": [4],
            "locally_connected_layers": [],
        },
        pretrain_data_filename=str(tmp_path / "dummy.csv"),
        verbose=0,
        progress_print_interval=None,
        predictor=None,
        save_to=None,
        constant_data={
            "allele_encoding": enc,
            "train_data": train_df,
            "folds_df": pandas.DataFrame({"fold_0": [True, True]}),
            "seed": 42,
        },
    )
    assert model is not None
    assert len(pretrain_calls) == 1
    assert pretrain_calls[0]["validation_inequalities"] is None


def test_590_random_negative_peptides_keeps_zero_binder_alleles():
    rnp = RandomNegativePeptides(
        random_negative_rate=1.0,
        random_negative_constant=10,
        random_negative_method="by_allele_equalize_nonbinders",
        random_negative_binder_threshold=500.0,
        random_negative_lengths=[8, 9],
    )
    plan = rnp.plan(
        peptides=["SIINFEKL", "AAAAAAAA", "CCCCCCCC", "DDDDDDDDD"],
        affinities=[100.0, 5000.0, 10000.0, 10000.0],
        alleles=["HLA-A*02:01", "HLA-A*02:01", "HLA-B*07:02", "HLA-B*07:02"],
    )
    assert "HLA-B*07:02" in plan.index
    assert (plan.loc["HLA-B*07:02"] >= 5).all()

    rnp_len = RandomNegativePeptides(
        random_negative_rate=1.0,
        random_negative_constant=4,
        random_negative_method="by_length",
        random_negative_binder_threshold=500.0,
        random_negative_lengths=[8, 9],
    )
    plan_len = rnp_len.plan(
        peptides=["AAAAAAAA", "DDDDDDDDD"],
        affinities=[10000.0, 10000.0],
        alleles=None,
    )
    assert list(plan_len.index) == [""]
    assert int(plan_len.loc["", 8]) == 4
    assert int(plan_len.loc["", 9]) == 4


def test_591_ppv_at_n_handles_nan_scores():
    # Only 1 positive in 4 finite rows -> max possible PPV@2 is 0.5.
    val = ppv_at_n([0, 1, 0, 0, 0], [numpy.nan, 0.9, 0.5, 0.5, 0.1], n=2)
    assert val == pytest.approx(0.5)
    assert numpy.isnan(ppv_at_n([1, 0, 0], [numpy.nan, 0.9, 0.1]))
    with pytest.raises(ValueError, match="N exceeds"):
        ppv_at_n([1, 0], [numpy.nan, 0.5], n=2)


def test_592_compare_models_sample_id_degenerate_length_and_early_validation(tmp_path):
    # 1. Preserve leading-zero numeric and "NA" sample_ids.
    bench = tmp_path / "bench.csv"
    bench.write_text(
        "peptide,sample_id,hla,hit\n"
        "SIINFEKL,00123,HLA-A*02:01,1\n"
        "AAAAAAAA,00123,HLA-A*02:01,0\n"
        "CCCCCCCC,NA,HLA-A*02:01,1\n"
        "DDDDDDDD,NA,HLA-A*02:01,0\n"
    )
    loaded = compare_models._read_holdout_benchmark_files(
        [str(bench)], {"00123", "NA"}, "test"
    )
    assert len(loaded) == 4
    assert set(loaded["sample_id"]) == {"00123", "NA"}

    # 2. _affinity_per_length must skip all-positive degenerate groups.
    n = 35
    test_df = pandas.DataFrame({
        "hla": ["HLA-A*02:01"] * n + ["HLA-B*07:02"] * n,
        "peptide_len": [9] * (2 * n),
        "hit": [1] * n + ([1] * 10 + [0] * (n - 10)),
        "a_score": numpy.linspace(0.1, 0.9, 2 * n),
        "b_score": numpy.linspace(0.2, 0.8, 2 * n),
    })
    per_len, per_allele_len = compare_models._affinity_per_length(test_df)
    assert int(per_len.iloc[0]["n_alleles_reported"]) == 1
    assert list(per_allele_len["allele"]) == ["HLA-B*07:02"]

    # 3. Invalid --presentation-score-kinds and processing cohort/policy mismatch
    # must fail in _validate_component_configuration before wiping --out.
    from types import SimpleNamespace

    args_bad_pres = SimpleNamespace(
        presentation_modes="with_flanks",
        presentation_score_kinds="bogus_kind",
    )
    with pytest.raises(SystemExit):
        compare_models._validate_component_configuration(
            args_bad_pres, ["presentation"], {}, {}
        )

    proc_dir = tmp_path / "proc"
    (proc_dir / "models.selected.with_flanks").mkdir(parents=True)
    args_bad_proc = SimpleNamespace(
        processing_modes="with_flanks",
        processing_negative_policy="unmatched",
        processing_matched_cohort=str(tmp_path / "cohort.csv"),
    )
    side = {"paths": {"processing": str(proc_dir)}, "label": "side"}
    with pytest.raises(ValueError, match="matched policy"):
        compare_models._validate_component_configuration(
            args_bad_proc, ["processing"], side, side
        )


def test_593_select_allele_specific_mse_inequalities_and_dropna():
    df_ineq = pandas.DataFrame({
        "allele": ["HLA-A*02:01"] * 4,
        "peptide": ["SIINFEKL", "AAADDDCCC", "LLLEEEFFF", "GGGHHHIII"],
        "measurement_value": [50.0, 500.0, 5000.0, 10000.0],
        "measurement_inequality": ["<", ">", "<", ">"],
    })
    fake_pred = MagicMock()
    fake_pred.predict.return_value = numpy.array([40.0, 600.0, 4000.0, 12000.0])
    selector = sel_as.MSEModelSelector(
        df=df_ineq, predictor=fake_pred, min_measurements=1
    )
    score_fn = selector.score_function("HLA-A*02:01")
    meta = {}
    val = score_fn(fake_pred, additional_metadata_out=meta)
    assert numpy.isfinite(val)
    assert numpy.isnan(meta["score_pearsonr"])


def test_594_predict_scan_input_parsing_and_thresholds(tmp_path):
    # Inverted and non-positive ranges rejected.
    with pytest.raises(ValueError):
        predict_scan_command.parse_peptide_lengths("11-8")
    with pytest.raises(ValueError):
        predict_scan_command.parse_peptide_lengths("0-9")
    assert predict_scan_command.parse_peptide_lengths("8-10") == [8, 9, 10]

    # Explicit incompatible thresholds fail early (before loading predictor).
    with pytest.raises(SystemExit):
        predict_scan_command.run([
            "--models", str(tmp_path),
            "--sequences", "SIINFEKLGGGNLVPMVATV",
            "--alleles", "HLA-A*02:01",
            "--no-affinity-percentile",
            "--threshold-affinity-percentile", "0.5",
        ])
    with pytest.raises(SystemExit):
        predict_scan_command.run([
            "--models", str(tmp_path),
            "--sequences", "SIINFEKLGGGNLVPMVATV",
            "--threshold-affinity", "50",
        ])

    # Dotless filename 'gz' fails cleanly via parser.error (SystemExit), not IndexError.
    with pytest.raises(SystemExit):
        predict_scan_command.run([
            "--models", str(tmp_path),
            "gz",
            "--alleles", "HLA-A*02:01",
            "--results-all",
        ])

    # .csv.bz2 guessed as csv, "NA" sequence_id preserved, trailing allele delimiter cleaned.
    csv_bz2 = tmp_path / "proteins.csv.bz2"
    pandas.DataFrame({
        "sequence_id": ["NA"],
        "sequence": ["SIINFEKLGG"],
    }).to_csv(csv_bz2, index=False)

    captured = {}

    class _FakePres:
        def predict_sequences(self, sequences, alleles, **kwargs):
            captured["sequences"] = sequences
            captured["alleles"] = alleles
            return pandas.DataFrame({
                "sequence_name": ["NA"],
                "pos": [0],
                "peptide": ["SIINFEKL"],
                "n_flank": [""],
                "c_flank": ["GG"],
                "sample_name": ["genotype_00"],
                "affinity": [10.0],
                "best_allele": ["HLA-A*02:01"],
                "affinity_percentile": [0.1],
                "processing_score": [0.9],
                "presentation_score": [0.95],
                "presentation_percentile": [0.1],
            })

    out_csv = tmp_path / "out.csv"
    orig_loader = predict_scan_command._load_predictor_for_command
    predict_scan_command._load_predictor_for_command = lambda _d: _FakePres()
    try:
        predict_scan_command.run([
            "--models", str(tmp_path),
            str(csv_bz2),
            "--alleles", "HLA-A*02:01; ",
            "--peptide-lengths", "8",
            "--num-jobs", "0",
            "--out", str(out_csv),
        ])
    finally:
        predict_scan_command._load_predictor_for_command = orig_loader

    assert captured["sequences"] == {"NA": "SIINFEKLGG"}
    assert captured["alleles"] == {"genotype_00": ["HLA-A*02:01"]}


def test_595_predict_custom_columns_and_dynamic_fetch_release(
        tmp_path, monkeypatch):
    # 1. predict --alleles --peptides with custom --allele-column / --peptide-column.
    class _FakePres:
        def predict_affinity(self, peptides, alleles, sample_names, **kwargs):
            return pandas.DataFrame({
                "peptide": list(peptides),
                "affinity": [42.0] * len(peptides),
                "best_allele": ["HLA-A*02:01"] * len(peptides),
            })

    out_csv = tmp_path / "pred.csv"
    monkeypatch.setattr(
        predict_command,
        "_load_predictor_for_command",
        lambda *_a, **_kw: (_FakePres(), "presentation"),
    )
    predict_command.run([
        "--models", str(tmp_path),
        "--alleles", "HLA-A*02:01",
        "--peptides", "SIINFEKL",
        "--allele-column", "mhc",
        "--peptide-column", "pep",
        "--affinity-only",
        "--no-affinity-percentile",
        "--num-jobs", "0",
        "--out", str(out_csv),
    ])
    res = pandas.read_csv(out_csv)
    assert list(res["mhc"]) == ["HLA-A*02:01"]
    assert list(res["pep"]) == ["SIINFEKL"]

    # 2. parser_fetch does not freeze --release at import time.
    old_env = os.environ.get("MHCFLURRY_DOWNLOADS_CURRENT_RELEASE")
    try:
        monkeypatch.setenv("MHCFLURRY_DOWNLOADS_CURRENT_RELEASE", "2.2.0")
        downloads.configure()
        parsed = downloads_command.parser.parse_args(["fetch"])
        assert parsed.release is None
        # Verify fetch_subcommand resolves dynamically to 2.2.0.
        captured_release = {}

        def fake_get_release_downloads(rel):
            captured_release["release"] = rel
            raise ValueError("stop")

        monkeypatch.setattr(
            downloads_command, "get_release_downloads", fake_get_release_downloads
        )
        with pytest.raises(ValueError, match="stop"):
            downloads_command.fetch_subcommand(parsed)
        assert captured_release["release"] == "2.2.0"
    finally:
        if old_env is None:
            os.environ.pop("MHCFLURRY_DOWNLOADS_CURRENT_RELEASE", None)
        else:
            os.environ["MHCFLURRY_DOWNLOADS_CURRENT_RELEASE"] = old_env
        downloads.configure()
