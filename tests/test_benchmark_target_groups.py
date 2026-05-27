import csv
import json
from pathlib import Path

from training.experiments.benchmark_target_groups import (
    benchmark_target_groups,
    grouped_mae,
    load_target_strategy,
    promotion_gate_for_group,
    select_best_model_per_target,
)


def test_target_strategy_config_loads() -> None:
    config = load_target_strategy("training/configs/target_strategy_phase_3v.json")

    assert config["strategy_version"] == "phase_3v_target_strategy_v1"
    assert "chest_cm" in config["target_groups"]["silhouette_learnable"]
    assert "height_cm" in config["target_groups"]["manual_or_user_input"]


def test_grouped_mae_calculation_is_correct() -> None:
    assert grouped_mae([1.0, 2.0, 6.0]) == 3.0


def test_promotion_gate_output_is_deterministic() -> None:
    assert promotion_gate_for_group("silhouette_learnable", 6.0)["gate"] == "research_only"
    assert promotion_gate_for_group("silhouette_learnable", 4.0)["gate"] == "assisted_manual_confirmation"
    assert promotion_gate_for_group("silhouette_learnable", 2.0)["gate"] == "stronger_candidate"
    assert promotion_gate_for_group("landmark_or_proportion_required", 2.0)["gate"] == "landmark_or_manual_required"
    assert promotion_gate_for_group("manual_or_user_input", 2.0)["gate"] == "manual_or_user_input_required"


def test_best_target_model_selection_is_deterministic() -> None:
    rows = {
        "run_b": {
            "waist_cm": {"run_name": "run_b", "target": "waist_cm", "test_mae": 3.0},
        },
        "run_a": {
            "waist_cm": {"run_name": "run_a", "target": "waist_cm", "test_mae": 3.0},
        },
    }

    best = select_best_model_per_target(rows, ["run_b", "run_a"])

    assert best["waist_cm"]["run_name"] == "run_a"


def test_grouped_benchmark_writes_outputs_and_warns_on_missing_targets(tmp_path: Path) -> None:
    config_path = _write_fixture_config(tmp_path)
    output_dir = tmp_path / "out"

    result = benchmark_target_groups(config_path, output_dir)

    assert Path(result["target_groups_json"]).exists()
    assert Path(result["grouped_results_json"]).exists()
    assert Path(result["grouped_results_csv"]).exists()
    assert Path(result["per_target_recommendations_csv"]).exists()
    assert Path(result["promotion_gates_md"]).exists()
    summary = json.loads(Path(result["grouped_results_json"]).read_text(encoding="utf-8"))
    assert any("missing per-target metrics" in warning for warning in summary["warnings"])
    silhouette = [
        row for row in summary["grouped_results"]
        if row["run_name"] == "run_a" and row["group_name"] == "silhouette_learnable"
    ][0]
    assert silhouette["available_target_count"] == 2
    assert silhouette["group_mae"] == 4.0


def _write_fixture_config(tmp_path: Path) -> Path:
    benchmark_csv = tmp_path / "benchmark.csv"
    per_target_csv = tmp_path / "per_target.csv"
    with benchmark_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["run_name", "source", "dataset", "feature_group", "model_type", "train_mae", "val_mae", "test_mae", "sample_count_test", "notes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_name": "run_a",
                "source": "fixture",
                "dataset": "fixture",
                "feature_group": "raw_scale_camera",
                "model_type": "ridge",
                "train_mae": "3.0",
                "val_mae": "4.0",
                "test_mae": "5.0",
                "sample_count_test": "3",
                "notes": "",
            }
        )
    with per_target_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["run_name", "source", "dataset", "feature_group", "model_type", "target", "test_mae"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_name": "run_a",
                "source": "fixture",
                "dataset": "fixture",
                "feature_group": "raw_scale_camera",
                "model_type": "ridge",
                "target": "chest_cm",
                "test_mae": "3.0",
            }
        )
        writer.writerow(
            {
                "run_name": "run_a",
                "source": "fixture",
                "dataset": "fixture",
                "feature_group": "raw_scale_camera",
                "model_type": "ridge",
                "target": "waist_cm",
                "test_mae": "5.0",
            }
        )
    config = {
        "strategy_version": "fixture",
        "target_groups": {
            "all_targets": ["chest_cm", "waist_cm", "hip_cm"],
            "silhouette_learnable": ["chest_cm", "waist_cm"],
            "landmark_or_proportion_required": ["height_cm"],
            "manual_or_user_input": ["height_cm"],
            "mass_proxy_uncertain": ["weight_kg"],
        },
        "candidate_runs": [{"run_name": "run_a", "description": "fixture run"}],
        "artifact_sources": {
            "benchmark_csvs": [str(benchmark_csv)],
            "per_target_csvs": [str(per_target_csv)],
            "metrics_jsons": [],
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path
