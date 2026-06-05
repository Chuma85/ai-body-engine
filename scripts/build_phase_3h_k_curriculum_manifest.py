from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthetic.blender.blend_dataset import BLEND_LABEL_COLUMNS, CAMERA_VIEWS

DEFAULT_CLEAN_DATASET = "data/synthetic/phase_3h_i_coupled_1000"
DEFAULT_MOBILE_REALISM_DATASET = "data/synthetic/phase_3h_j_mobile_realism_1000"
DEFAULT_OUTPUT = "artifacts/phase_3h_k_curriculum_manifest"
DEFAULT_EXPECTED_SAMPLES = 1000
TARGET_COLUMNS = ("height_cm", "chest_cm", "waist_cm", "hip_cm", "shoulder_cm", "inseam_cm")
MANIFEST_COLUMNS = [
    "curriculum_stage",
    "curriculum_split",
    "dataset_source",
    "dataset_path",
    "curriculum_sample_id",
    "sample_id",
    "front_image",
    "side_image",
    "back_image",
    *TARGET_COLUMNS,
    "label_generation_mode",
    "variation_source",
    "synthetic_labels",
    "real_world_validated",
    "recommended_use",
]
ARCHIVED_DATASET_MARKERS = ("_archived_old_mannequin", "archived_old_mannequin", "old_mannequin")


def build_curriculum_manifests(
    *,
    clean_dataset: str | Path = DEFAULT_CLEAN_DATASET,
    mobile_realism_dataset: str | Path = DEFAULT_MOBILE_REALISM_DATASET,
    output_dir: str | Path = DEFAULT_OUTPUT,
    expected_samples: int = DEFAULT_EXPECTED_SAMPLES,
) -> dict[str, Any]:
    clean_path = Path(clean_dataset)
    mobile_path = Path(mobile_realism_dataset)
    out = Path(output_dir)
    clean_summary = validate_blend_dataset(clean_path, expected_samples=expected_samples)
    mobile_summary = validate_blend_dataset(mobile_path, expected_samples=expected_samples)

    clean_rows = read_labels(clean_path / "labels.csv")
    mobile_rows = read_labels(mobile_path / "labels.csv")
    mobile_train_rows = [row for row in mobile_rows if not is_mobile_evaluation_sample(row["sample_id"])]
    mobile_eval_rows = [row for row in mobile_rows if is_mobile_evaluation_sample(row["sample_id"])]

    out.mkdir(parents=True, exist_ok=True)
    clean_manifest = write_manifest(
        out / "clean_train_manifest.csv",
        manifest_rows(
            clean_rows,
            dataset_path=clean_path,
            dataset_source="phase_3h_i_clean",
            curriculum_stage="stage_1_clean_pretrain",
            curriculum_split="train",
            recommended_use="clean synthetic baseline pretraining",
        ),
    )
    mobile_manifest = write_manifest(
        out / "mobile_realism_train_manifest.csv",
        manifest_rows(
            mobile_train_rows,
            dataset_path=mobile_path,
            dataset_source="phase_3h_j_mobile_realism",
            curriculum_stage="stage_2_mobile_adaptation",
            curriculum_split="train",
            recommended_use="mobile-realistic adaptation after clean baseline",
        ),
    )
    mixed_manifest = write_manifest(
        out / "mixed_curriculum_manifest.csv",
        [
            *manifest_rows(
                clean_rows,
                dataset_path=clean_path,
                dataset_source="phase_3h_i_clean",
                curriculum_stage="stage_1_clean_pretrain",
                curriculum_split="train",
                recommended_use="clean synthetic baseline pretraining",
            ),
            *manifest_rows(
                mobile_train_rows,
                dataset_path=mobile_path,
                dataset_source="phase_3h_j_mobile_realism",
                curriculum_stage="stage_2_mobile_adaptation",
                curriculum_split="train",
                recommended_use="mobile-realistic adaptation after clean baseline",
            ),
        ],
    )
    evaluation_manifest = write_manifest(
        out / "evaluation_manifest.csv",
        manifest_rows(
            mobile_eval_rows,
            dataset_path=mobile_path,
            dataset_source="phase_3h_j_mobile_realism",
            curriculum_stage="stage_3_mobile_holdout",
            curriculum_split="evaluation",
            recommended_use="mobile-realistic holdout evaluation",
        ),
    )

    summary = {
        "phase": "phase_3h_k_clean_vs_mobile_realism_curriculum",
        "output_dir": out.as_posix(),
        "manifests": {
            "clean_train_manifest": clean_manifest,
            "mobile_realism_train_manifest": mobile_manifest,
            "mixed_curriculum_manifest": mixed_manifest,
            "evaluation_manifest": evaluation_manifest,
        },
        "datasets": {
            "clean": clean_summary,
            "mobile_realism": mobile_summary,
        },
        "row_counts": {
            "clean_train": len(clean_rows),
            "mobile_realism_train": len(mobile_train_rows),
            "mixed_curriculum": len(clean_rows) + len(mobile_train_rows),
            "evaluation": len(mobile_eval_rows),
        },
        "strategy": [
            "Stage 1: train baseline on clean synthetic Phase 3H-I.",
            "Stage 2: fine-tune or retrain with mixed clean plus mobile-realistic Phase 3H-I/J manifests.",
            "Stage 3: evaluate on the Phase 3H-J mobile-realistic holdout manifest.",
            "Stage 4: prepare a small real mobile-photo validation set later before production claims.",
        ],
        "synthetic_labels": True,
        "real_world_validated": False,
        "copies_images": False,
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_json"] = summary_path.as_posix()
    return summary


def validate_blend_dataset(dataset_path: Path, *, expected_samples: int) -> dict[str, Any]:
    ensure_not_archived_dataset(dataset_path)
    labels_path = dataset_path / "labels.csv"
    metadata_path = dataset_path / "metadata.json"
    images_dir = dataset_path / "images"
    view_dirs = {view: images_dir / view for view in CAMERA_VIEWS}
    for path in (dataset_path, labels_path, metadata_path, images_dir, *view_dirs.values()):
        if not path.exists():
            raise FileNotFoundError(f"Missing required Phase 3H-K dataset input path: {path}")

    rows = read_labels(labels_path)
    if len(rows) != expected_samples:
        raise ValueError(f"Expected {expected_samples} labels in {labels_path}, found {len(rows)}")
    missing_columns = [column for column in BLEND_LABEL_COLUMNS if column not in rows[0]]
    if missing_columns:
        raise ValueError(f"{labels_path} missing columns: {', '.join(missing_columns)}")
    missing_images = missing_images_for_rows(dataset_path, rows)
    if missing_images:
        raise ValueError("Missing referenced images: " + "; ".join(missing_images[:10]))
    png_count = sum(1 for _ in images_dir.rglob("*.png"))
    expected_png_count = expected_samples * len(CAMERA_VIEWS)
    if png_count != expected_png_count:
        raise ValueError(f"Expected {expected_png_count} PNGs in {images_dir}, found {png_count}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "dataset_path": dataset_path.as_posix(),
        "labels": len(rows),
        "pngs": png_count,
        "metadata_json": True,
        "front_images": view_dirs["front"].exists(),
        "side_images": view_dirs["side"].exists(),
        "back_images": view_dirs["back"].exists(),
        "variation_source": metadata.get("variation_source", ""),
        "synthetic_labels": bool(metadata.get("synthetic_labels")),
        "real_world_validated": bool(metadata.get("real_world_validated")),
    }


def manifest_rows(
    rows: list[dict[str, str]],
    *,
    dataset_path: Path,
    dataset_source: str,
    curriculum_stage: str,
    curriculum_split: str,
    recommended_use: str,
) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    for row in rows:
        sample_id = row["sample_id"]
        manifest.append(
            {
                "curriculum_stage": curriculum_stage,
                "curriculum_split": curriculum_split,
                "dataset_source": dataset_source,
                "dataset_path": dataset_path.as_posix(),
                "curriculum_sample_id": f"{dataset_source}:{sample_id}",
                "sample_id": sample_id,
                "front_image": dataset_reference(dataset_path, row["front_image"]),
                "side_image": dataset_reference(dataset_path, row["side_image"]),
                "back_image": dataset_reference(dataset_path, row["back_image"]),
                **{target: row[target] for target in TARGET_COLUMNS},
                "label_generation_mode": row.get("label_generation_mode", ""),
                "variation_source": row.get("variation_source", ""),
                "synthetic_labels": row.get("synthetic_labels", ""),
                "real_world_validated": row.get("real_world_validated", ""),
                "recommended_use": recommended_use,
            }
        )
    return manifest


def write_manifest(path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "path": path.as_posix(),
        "row_count": len(rows),
        "columns": MANIFEST_COLUMNS,
    }


def read_labels(labels_path: Path) -> list[dict[str, str]]:
    with labels_path.open("r", newline="", encoding="utf-8") as labels_file:
        return list(csv.DictReader(labels_file))


def missing_images_for_rows(dataset_path: Path, rows: list[dict[str, str]]) -> list[str]:
    missing: list[str] = []
    for row in rows:
        for view in CAMERA_VIEWS:
            relative = row.get(f"{view}_image", "")
            if not relative or not (dataset_path / relative).exists():
                missing.append(f"{row.get('sample_id', '<unknown>')}:{view}:{relative}")
    return missing


def dataset_reference(dataset_path: Path, relative_image: str) -> str:
    return (dataset_path / relative_image).as_posix()


def is_mobile_evaluation_sample(sample_id: str) -> bool:
    match = re.search(r"(\d+)$", sample_id)
    if not match:
        return False
    return int(match.group(1)) % 5 == 0


def ensure_not_archived_dataset(path: str | Path) -> None:
    normalized = str(path).replace("\\", "/").lower()
    if any(marker in normalized for marker in ARCHIVED_DATASET_MARKERS):
        raise ValueError(f"Phase 3H-K must not use archived old mannequin datasets: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phase 3H-K clean/mobile-realism curriculum manifests.")
    parser.add_argument("--clean-dataset", default=DEFAULT_CLEAN_DATASET)
    parser.add_argument("--mobile-realism-dataset", default=DEFAULT_MOBILE_REALISM_DATASET)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-samples", type=int, default=DEFAULT_EXPECTED_SAMPLES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = build_curriculum_manifests(
            clean_dataset=args.clean_dataset,
            mobile_realism_dataset=args.mobile_realism_dataset,
            output_dir=args.out,
            expected_samples=args.expected_samples,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Phase 3H-K curriculum manifest build failed: {exc}")
        return 1
    print("Phase 3H-K curriculum manifests created.")
    print(f"Output: {summary['output_dir']}")
    for name, manifest in summary["manifests"].items():
        print(f"{name}: {manifest['path']} ({manifest['row_count']} rows)")
    print(f"Summary: {summary['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
