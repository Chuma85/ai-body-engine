from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

DEFAULT_OUTPUT_DIR = Path("reports") / "phase_h6_1_vision_benchmark"
EXPECTED_OUTPUTS = (
    "vision_candidate_evaluation_metrics.json",
    "vision_candidate_benchmark_report.md",
    "vision_ablation_report.json",
    "vision_view_contribution_report.json",
    "vision_confidence_calibration_report.json",
    "vision_promotion_recommendation.json",
)
DEFAULT_RECORD_FILES = (
    "records.jsonl",
    "verified_measurements.jsonl",
    "records.json",
    "verified_measurements.json",
    "manifest.json",
)
EXCLUDED_DIRS = {".git", ".pytest_cache", "__pycache__"}


def execute_phase_h6_vision_benchmark(
    repo_root: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    dataset: str | Path | None = None,
    metadata_candidate_model: str | Path | None = None,
    vision_candidate_model: str | Path | None = None,
    vision_weights: str | Path | None = None,
    dataset_version: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    output_path = root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    inventory = build_inventory(root)
    selected_dataset = Path(dataset) if dataset is not None else single_or_none(inventory["verifiedDatasets"])
    selected_metadata = Path(metadata_candidate_model) if metadata_candidate_model is not None else single_or_none(inventory["metadataCandidateModels"])
    selected_vision = Path(vision_candidate_model) if vision_candidate_model is not None else single_or_none(inventory["visionCandidateModels"])
    selected_weights = Path(vision_weights) if vision_weights is not None else infer_weights_path(selected_vision)
    blockers = build_blockers(selected_dataset, selected_metadata, selected_vision, selected_weights, inventory)

    if blockers:
        status = {
            "schemaVersion": "phase_h6_1_benchmark_execution_status_v1",
            "generatedAt": generated,
            "status": "blocked",
            "benchmarkExecuted": False,
            "productionModelUpdated": False,
            "liveApiBehaviorChanged": False,
            "expectedOutputs": list(EXPECTED_OUTPUTS),
            "blockers": blockers,
            "inventory": {
                key: [str(path) for path in paths]
                for key, paths in inventory.items()
            },
            "nextAction": "Provide a verified FashionApp dataset plus matching H.2 metadata and H.5 vision candidate artifacts, then rerun this script.",
        }
        write_status(output_path, status)
        return {"status": status, "status_path": str(output_path / "benchmark_execution_status.json")}

    from training.evaluate_vision_candidate_model import evaluate_vision_candidate_model

    result = evaluate_vision_candidate_model(
        selected_dataset,
        selected_metadata,
        selected_vision,
        output_path,
        vision_weights_path=selected_weights,
        dataset_version=dataset_version,
        generated_at=generated,
    )
    status = {
        "schemaVersion": "phase_h6_1_benchmark_execution_status_v1",
        "generatedAt": generated,
        "status": "completed",
        "benchmarkExecuted": True,
        "productionModelUpdated": False,
        "liveApiBehaviorChanged": False,
        "dataset": str(selected_dataset),
        "metadataCandidateModel": str(selected_metadata),
        "visionCandidateModel": str(selected_vision),
        "visionWeights": str(selected_weights),
        "outputs": {name: str(output_path / name) for name in EXPECTED_OUTPUTS},
        "recommendation": result["recommendation"],
    }
    write_status(output_path, status)
    result["status"] = status
    result["status_path"] = str(output_path / "benchmark_execution_status.json")
    return result


def build_inventory(root: Path) -> dict[str, list[Path]]:
    inventory = {
        "verifiedDatasets": [],
        "metadataCandidateModels": [],
        "visionCandidateModels": [],
        "visionWeights": [],
    }
    for path in iter_files(root):
        if path.name in DEFAULT_RECORD_FILES and looks_like_verified_dataset(path):
            inventory["verifiedDatasets"].append(path.parent)
        elif path.name == "model.json" and json_contains(path, "candidate_body_ai_measurement_model"):
            inventory["metadataCandidateModels"].append(path)
        elif path.name == "vision_model.json" and json_contains(path, "vision_multimodal_body_ai_candidate"):
            inventory["visionCandidateModels"].append(path)
        elif path.name == "vision_model.pt":
            inventory["visionWeights"].append(path)
    return {key: sorted(set(paths)) for key, paths in inventory.items()}


def iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        if current.name in EXCLUDED_DIRS or (current.name == "tmp" and current.parent.name == "artifacts"):
            continue
        try:
            for child in current.iterdir():
                if child.is_dir():
                    stack.append(child)
                elif child.is_file():
                    files.append(child)
        except OSError:
            continue
    return files


def looks_like_verified_dataset(path: Path) -> bool:
    if path.suffix not in {".json", ".jsonl"}:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "dataset_version" in text and "front_image" in text and "final_approved" in text


def json_contains(path: Path, token: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return token in text


def single_or_none(paths: list[Path]) -> Path | None:
    return paths[0] if len(paths) == 1 else None


def infer_weights_path(vision_candidate_model: Path | None) -> Path | None:
    if vision_candidate_model is None:
        return None
    return vision_candidate_model.parent / "vision_model.pt"


def build_blockers(
    dataset: Path | None,
    metadata_candidate: Path | None,
    vision_candidate: Path | None,
    vision_weights: Path | None,
    inventory: dict[str, list[Path]],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if dataset is None:
        blockers.append(blocker("missing_dataset", "No verified FashionApp dataset root was found.", inventory["verifiedDatasets"]))
    elif not dataset.exists():
        blockers.append(blocker("missing_dataset", f"Verified dataset root does not exist: {dataset}", []))
    if metadata_candidate is None:
        blockers.append(blocker("missing_metadata_candidate_artifact", "No H.2 metadata candidate model.json was found.", inventory["metadataCandidateModels"]))
    elif not metadata_candidate.exists():
        blockers.append(blocker("missing_metadata_candidate_artifact", f"Metadata candidate artifact does not exist: {metadata_candidate}", []))
    if vision_candidate is None:
        blockers.append(blocker("missing_vision_candidate_artifact", "No H.5 vision candidate vision_model.json was found.", inventory["visionCandidateModels"]))
    elif not vision_candidate.exists():
        blockers.append(blocker("missing_vision_candidate_artifact", f"Vision candidate artifact does not exist: {vision_candidate}", []))
    if vision_weights is None:
        blockers.append(blocker("missing_vision_weights", "No H.5 vision candidate vision_model.pt was found.", inventory["visionWeights"]))
    elif not vision_weights.exists():
        blockers.append(blocker("missing_vision_weights", f"Vision candidate weights do not exist: {vision_weights}", []))
    return blockers


def blocker(kind: str, message: str, candidates: list[Path]) -> dict[str, Any]:
    return {
        "kind": kind,
        "message": message,
        "discoveredCandidates": [str(path) for path in candidates],
    }


def write_status(output_path: Path, status: dict[str, Any]) -> None:
    status_path = output_path / "benchmark_execution_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Phase H.6.1 Vision Benchmark Execution Status",
        "",
        f"Status: `{status['status']}`",
        f"Benchmark executed: `{status['benchmarkExecuted']}`",
        f"Production model updated: `{status['productionModelUpdated']}`",
        f"Live API behavior changed: `{status['liveApiBehaviorChanged']}`",
        "",
    ]
    if status["status"] == "blocked":
        lines.extend(["## Blockers", ""])
        lines.extend(f"- `{row['kind']}`: {row['message']}" for row in status["blockers"])
    else:
        lines.extend(["## Outputs", ""])
        lines.extend(f"- `{name}`: `{path}`" for name, path in status["outputs"].items())
        lines.extend(["", "## Recommendation", "", f"`{status['recommendation']['decision']}`"])
    lines.append("")
    (output_path / "benchmark_execution_status.md").write_text("\n".join(lines), encoding="utf-8")


def format_execution_summary(result: dict[str, Any]) -> str:
    status = result["status"]
    if status["status"] == "blocked":
        lines = [
            f"Benchmark execution status: {result['status_path']}",
            "Benchmark executed: false",
            "Blockers:",
        ]
        lines.extend(f"- {row['kind']}: {row['message']}" for row in status["blockers"])
        return "\n".join(lines)
    from training.evaluate_vision_candidate_model import format_evaluation_summary

    return "\n".join([format_evaluation_summary(result), f"Benchmark execution status: {result['status_path']}"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute Phase H.6 vision benchmark when required artifacts are available.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset")
    parser.add_argument("--metadata-candidate-model")
    parser.add_argument("--vision-candidate-model")
    parser.add_argument("--vision-weights")
    parser.add_argument("--dataset-version")
    args = parser.parse_args(argv)
    result = execute_phase_h6_vision_benchmark(
        args.repo_root,
        args.output,
        dataset=args.dataset,
        metadata_candidate_model=args.metadata_candidate_model,
        vision_candidate_model=args.vision_candidate_model,
        vision_weights=args.vision_weights,
        dataset_version=args.dataset_version,
    )
    print(format_execution_summary(result))
    return 0 if result["status"]["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
