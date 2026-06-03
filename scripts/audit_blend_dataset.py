from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthetic.blender.blend_dataset_audit import audit_blend_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a Blender blend-file synthetic dataset before training.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--max-contact-sheet-samples", type=int, default=12)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_blend_dataset(
            dataset=args.dataset,
            out=args.out,
            expected_samples=args.expected_samples,
            max_contact_sheet_samples=args.max_contact_sheet_samples,
            strict=args.strict,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Blend dataset audit failed: {exc}")
        return 1

    print(f"Audit report: {report['outputs']['audit_report_json']}")
    print(f"Audit summary: {report['outputs']['audit_summary_md']}")
    print(f"Contact sheet: {report['outputs']['sample_contact_sheet_png']}")
    print(f"Passed: {report['passed']}")
    print(f"Warnings: {len(report['warnings'])}")
    print(f"Errors: {len(report['errors'])}")
    print(f"Strict failures: {len(report['strict_failures'])}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
