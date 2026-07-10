#!/usr/bin/env python3
"""Validated container launcher for existing AI Body Engine commands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

WORKLOADS = {"training", "evaluation", "inference", "dataset-validator"}
SAFE_ENV_NAMES = (
    "GCP_PROJECT_ID", "GCP_REGION", "DATASET_URI", "MODEL_INPUT_URI",
    "MODEL_OUTPUT_URI", "REPORT_OUTPUT_URI", "TRAINING_RUN_ID",
    "MODEL_VERSION", "CANDIDATE_ID",
)


def log(event: str, **fields: object) -> None:
    safe = {key: value for key, value in fields.items() if key not in {"participant", "user_id", "token", "secret"}}
    print(json.dumps({"severity": "INFO", "event": event, **safe}, sort_keys=True), flush=True)


def fail(message: str) -> int:
    print(json.dumps({"severity": "ERROR", "event": "configuration_error", "message": message}), file=sys.stderr, flush=True)
    return 2


def local_path(name: str, *, required: bool = True) -> str | None:
    value = os.getenv(name)
    if not value:
        if required:
            raise ValueError(f"required environment variable {name} is missing")
        return None
    parsed = urlparse(value)
    if parsed.scheme == "file":
        value = parsed.path
    elif parsed.scheme == "gs":
        raise ValueError(f"{name} is a gs:// URI; mount or download it first and pass the local container path")
    elif parsed.scheme:
        raise ValueError(f"{name} uses unsupported URI scheme {parsed.scheme!r}")
    return value


def ensure_input(name: str) -> str:
    value = local_path(name)
    if not Path(value).exists():
        raise ValueError(f"{name} does not exist at the mounted container path")
    return value


def command_for(workload: str, extra: list[str]) -> list[str]:
    if extra:
        modules = {
            "training": [sys.executable, "-m", "training.train_candidate_model"],
            "evaluation": [sys.executable, "-m", "training.evaluate_candidate_model"],
            "dataset-validator": [sys.executable, "-m", "synthetic.validate_synthetic_dataset"],
        }
        if workload == "inference":
            return ["uvicorn", "app.main:app", *extra]
        return [*modules[workload], *extra]
    if workload == "training":
        dataset = ensure_input("DATASET_URI")
        output = local_path("MODEL_OUTPUT_URI")
        command = [sys.executable, "-m", "training.train_candidate_model", "--dataset", dataset, "--output", output]
        if os.getenv("MODEL_VERSION"):
            command += ["--model-version", os.environ["MODEL_VERSION"]]
        return command
    if workload == "evaluation":
        return [
            sys.executable, "-m", "training.evaluate_candidate_model",
            "--dataset", ensure_input("DATASET_URI"),
            "--candidate-model", ensure_input("MODEL_INPUT_URI"),
            "--output", local_path("REPORT_OUTPUT_URI"),
        ]
    if workload == "dataset-validator":
        dataset = ensure_input("DATASET_URI")
        kind = os.getenv("DATASET_KIND", "synthetic")
        if kind == "synthetic":
            return [sys.executable, "-m", "synthetic.validate_synthetic_dataset", "--dataset", dataset]
        if kind == "verified":
            command = [sys.executable, "-m", "training.datasets.verified_measurement_dataset", "--dataset", dataset]
            if os.getenv("REPORT_OUTPUT_URI"):
                command += ["--output", local_path("REPORT_OUTPUT_URI")]
            return command
        raise ValueError("DATASET_KIND must be 'synthetic' or 'verified'")
    return ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", os.getenv("PORT", "8080")]


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in WORKLOADS:
        return fail(f"first argument must be one of {sorted(WORKLOADS)}")
    workload, extra = argv[0], argv[1:]
    try:
        command = command_for(workload, extra)
    except ValueError as exc:
        return fail(str(exc))
    log(
        "workload_start",
        workload=workload,
        training_run_id=os.getenv("TRAINING_RUN_ID"),
        model_version=os.getenv("MODEL_VERSION"),
        candidate_id=os.getenv("CANDIDATE_ID"),
        configured_environment_variables=sorted(name for name in SAFE_ENV_NAMES if os.getenv(name)),
    )
    result = subprocess.run(command, check=False)
    log("workload_exit", workload=workload, exit_code=result.returncode)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
