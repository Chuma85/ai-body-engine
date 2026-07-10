from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "containers/runtime_entrypoint.py"


def run_entrypoint(workload: str, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    clean_env = os.environ.copy()
    for name in ("DATASET_URI", "MODEL_INPUT_URI", "MODEL_OUTPUT_URI", "REPORT_OUTPUT_URI"):
        clean_env.pop(name, None)
    clean_env.update(env or {})
    return subprocess.run(
        [sys.executable, str(ENTRYPOINT), workload, *args],
        cwd=ROOT,
        env=clean_env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_every_workload_help_delegates_to_existing_command() -> None:
    for workload in ("training", "evaluation", "inference", "dataset-validator"):
        result = run_entrypoint(workload, "--help")
        assert result.returncode == 0, result.stderr
        assert "workload_exit" in result.stdout


def test_training_missing_input_fails_nonzero_without_sensitive_output() -> None:
    result = run_entrypoint("training")
    assert result.returncode == 2
    assert "DATASET_URI" in result.stderr
    assert "configuration_error" in result.stderr


def test_raw_gcs_uri_requires_mount_or_download() -> None:
    result = run_entrypoint(
        "training",
        env={"DATASET_URI": "gs://private/data", "MODEL_OUTPUT_URI": "/outputs/model"},
    )
    assert result.returncode == 2
    assert "mount or download" in result.stderr


def test_container_definitions_are_non_root_and_do_not_copy_entire_context() -> None:
    for name in ("training", "evaluation", "inference", "dataset-validator"):
        text = (ROOT / f"containers/{name}/Dockerfile").read_text(encoding="utf-8")
        assert "USER bodyengine" in text
        assert "COPY . ." not in text


def test_cloud_build_has_all_immutable_workload_images() -> None:
    config = yaml.safe_load((ROOT / "cloudbuild/ai-body-containers.yaml").read_text(encoding="utf-8"))
    script = "\n".join(str(step) for step in config["steps"])
    for image in ("body-engine-training", "body-engine-evaluation", "body-engine-inference", "dataset-validator"):
        assert image in script
    assert "COMMIT_SHA" in script
    assert "Refusing to overwrite" in script
    assert "immutable SHA tag" in script


def test_dockerignore_blocks_sensitive_asset_classes() -> None:
    rules = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for required in (".git", ".env", ".venv", "venv", "artifacts", "data", "models", "checkpoints", "*.dump", "*service_account*.json"):
        assert required in rules
