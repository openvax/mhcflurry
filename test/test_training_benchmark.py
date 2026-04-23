import json

import pandas

from mhcflurry.training_benchmark import analyze_training_run


def test_analyze_training_run_summarizes_phases_and_logs(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    fit_info = [
        {
            "training_info": {"phase": "pretrain"},
            "time": 30.0,
            "loss": [0.4, 0.3],
            "epoch_fetch_time": [1.0, 1.5],
            "epoch_train_time": [6.0, 5.0],
            "epoch_validation_time": [2.0, 2.5],
            "epoch_total_time": [10.0, 10.5],
            "epoch_num_train_batches": [2, 2],
            "epoch_num_train_rows": [64, 64],
            "epoch_num_validation_batches": [2, 2],
            "validation_rows": 32,
            "effective_validation_batch_size": 16,
            "dataloader_num_workers": 2,
            "first_batch_time": 4.0,
        },
        {
            "training_info": {"phase": "finetune"},
            "time": 12.0,
            "loss": [0.2],
            "epoch_input_build_time": [1.0],
            "epoch_dataloader_setup_time": [0.25],
            "epoch_train_time": [3.0],
            "epoch_validation_materialize_time": [0.75],
            "epoch_validation_time": [1.25],
            "epoch_total_time": [6.5],
            "epoch_num_train_batches": [3],
            "epoch_num_train_rows": [96],
            "epoch_tail_train_rows": [4],
            "epoch_num_validation_batches": [4],
            "validation_rows": 48,
            "validation_cache_reused": True,
            "effective_validation_batch_size": 16,
            "dataloader_num_workers": 0,
            "first_batch_time": 1.5,
        },
    ]
    manifest_df = pandas.DataFrame(
        [
            {
                "model_name": "pan-class1-0",
                "allele": "pan-class1",
                "config_json": json.dumps({"fit_info": fit_info}),
            }
        ]
    )
    manifest_df.to_csv(models_dir / "manifest.csv", index=False)

    train_log = tmp_path / "train.log"
    train_log.write_text(
        "\n".join(
            [
                "TIMING_MARKER start 100.0",
                "TIMING_MARKER data_loaded 115.0",
                "TIMING_MARKER setup_done 130.0",
                "TIMING_MARKER training_done 190.0",
                "PROCESS_TELEMETRY pid=11 marker=START rss_mb=100.0 num_fds=20",
                "PROCESS_TELEMETRY pid=11 marker=END rss_mb=140.0 num_fds=26",
                "GPU_MEMORY_TELEMETRY pid=11 task=1/2 marker=START allocated_gb=1.0 reserved_gb=2.0 max_allocated_gb=3.0",
            ]
        )
        + "\n"
    )
    selection_log = tmp_path / "select.log"
    selection_log.write_text(
        "\n".join(
            [
                "TIMING_MARKER selection_start 200.0",
                "TIMING_MARKER selection_done 214.5",
            ]
        )
        + "\n"
    )

    result = analyze_training_run(
        models_dir,
        train_log=train_log,
        selection_log=selection_log,
    )

    assert result["top_level_wall_times_s"]["data_load_s"] == 15.0
    assert result["top_level_wall_times_s"]["setup_s"] == 15.0
    assert result["top_level_wall_times_s"]["training_s"] == 60.0
    assert result["top_level_wall_times_s"]["selection_s"] == 14.5

    pretrain = result["phase_summaries"]["pretrain"]
    assert pretrain["num_fits"] == 1
    assert pretrain["train_time_s"] == 11.0
    assert pretrain["fetch_time_s"] == 2.5
    assert pretrain["validation_total_time_s"] == 4.5
    assert pretrain["total_train_batches"] == 4
    assert pretrain["validation_rows_seen"] == 64

    finetune = result["phase_summaries"]["finetune"]
    assert finetune["input_build_time_s"] == 1.0
    assert finetune["dataloader_setup_time_s"] == 0.25
    assert finetune["validation_total_time_s"] == 2.0
    assert finetune["validation_cache_reused_fit_count"] == 1
    assert finetune["total_tail_train_rows"] == 4

    telemetry = result["telemetry"]["train"]
    assert telemetry["max_rss_mb"] == 140.0
    assert telemetry["max_gpu_reserved_gb"] == 2.0
    assert telemetry["rss_growth_by_pid"] == [{"pid": 11, "rss_mb_delta": 40.0}]
