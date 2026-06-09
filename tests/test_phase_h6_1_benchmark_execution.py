import json
from pathlib import Path

from scripts.execute_phase_h6_vision_benchmark import execute_phase_h6_vision_benchmark


def test_phase_h6_1_execution_persists_blocked_status_without_fabricating_reports(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    result = execute_phase_h6_vision_benchmark(
        repo_root,
        "reports/phase_h6_1_vision_benchmark",
        generated_at="2026-06-09T00:00:00Z",
    )

    status_path = Path(result["status_path"])
    markdown_path = status_path.with_suffix(".md")
    status = json.loads(status_path.read_text(encoding="utf-8"))

    assert status_path.exists()
    assert markdown_path.exists()
    assert status["status"] == "blocked"
    assert status["benchmarkExecuted"] is False
    assert status["productionModelUpdated"] is False
    assert status["liveApiBehaviorChanged"] is False
    assert {blocker["kind"] for blocker in status["blockers"]} == {
        "missing_dataset",
        "missing_metadata_candidate_artifact",
        "missing_vision_candidate_artifact",
        "missing_vision_weights",
    }
    assert status["expectedOutputs"] == [
        "vision_candidate_evaluation_metrics.json",
        "vision_candidate_benchmark_report.md",
        "vision_ablation_report.json",
        "vision_view_contribution_report.json",
        "vision_confidence_calibration_report.json",
        "vision_promotion_recommendation.json",
    ]
    assert not (status_path.parent / "vision_candidate_evaluation_metrics.json").exists()
