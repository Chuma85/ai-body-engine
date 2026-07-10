from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_validation_pipeline_runs_tests_configuration_and_container_smoke() -> None:
    config = yaml.safe_load((ROOT / "cloudbuild/validate-ai-body-engine.yaml").read_text(encoding="utf-8"))
    text = str(config)
    assert "python -m pytest -q" in text
    assert "validate-cloud-configuration.py" in text
    assert "container-smoke-tests" in text
    assert "docker push" not in text


def test_build_pipeline_publishes_only_after_validation_and_smoke() -> None:
    config = yaml.safe_load((ROOT / "cloudbuild/build-ai-body-containers.yaml").read_text(encoding="utf-8"))
    ids = [step["id"] for step in config["steps"]]
    assert ids.index("validate-before-build") < ids.index("build-images")
    assert ids.index("smoke-before-publish") < ids.index("publish-immutable-images")
    text = str(config)
    assert "COMMIT_SHA" in text
    assert "Refusing to overwrite immutable image" in text
    assert "latest" not in text
    assert "deploy" not in text.lower()
    assert "promote" not in text.lower()


def test_remote_setup_never_changes_origin_or_force_pushes() -> None:
    text = (ROOT / "scripts/gcp/setup-google-git-remote.ps1").read_text(encoding="utf-8")
    assert "remote add google" in text
    assert "remote set-url google" in text
    assert "set-url origin" not in text
    assert "--force" not in text
    assert "git push google --all" in text
    assert "git push google --tags" in text


def test_mirror_verifier_checks_refs_default_branch_and_files() -> None:
    text = (ROOT / "scripts/gcp/verify-google-git-mirror.ps1").read_text(encoding="utf-8")
    assert "ls-remote --heads google" in text
    assert "ls-remote --tags google" in text
    assert "ls-remote --symref google HEAD" in text
    assert "ls-tree -r --name-only" in text
    assert "origin_unchanged" in text
