from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality gate for augmented dataset + detector grade")
    parser.add_argument("--integrity-report", type=Path, required=True)
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--grade-report", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, default=None, help="Optional split audit JSON from generator")
    parser.add_argument("--max-geometry-outlier-rate", type=float, default=0.005)
    parser.add_argument("--min-run-grade", type=float, default=None, help="Optional absolute grade floor")
    args = parser.parse_args()

    integrity = _load_json(args.integrity_report)
    geometry = _load_json(args.geometry_report)
    grade = _load_json(args.grade_report)
    split_audit = _load_json(args.split_audit) if args.split_audit is not None else None

    integrity_issues = int(integrity.get("summary", {}).get("total_issues", 0))
    outlier_rate = float(geometry.get("summary", {}).get("outlier_rate", 1.0) or 0.0)
    run_grade = float(grade.get("aggregate", {}).get("run_grade_0_100", 0.0) or 0.0)
    overlap_count = int(split_audit.get("overlap_count", 0)) if isinstance(split_audit, dict) else None

    failed: list[str] = []
    if integrity_issues != 0:
        failed.append(f"integrity issues != 0 (got {integrity_issues})")
    if outlier_rate > args.max_geometry_outlier_rate:
        failed.append(
            f"geometry outlier rate too high (got {outlier_rate:.6f}, allowed {args.max_geometry_outlier_rate:.6f})"
        )
    if args.min_run_grade is not None and run_grade < args.min_run_grade:
        failed.append(f"run grade below threshold (got {run_grade:.4f}, required {args.min_run_grade:.4f})")
    if overlap_count is not None and overlap_count != 0:
        failed.append(f"split overlap must be 0 (got {overlap_count})")

    print(f"integrity_issues={integrity_issues}")
    print(f"geometry_outlier_rate={outlier_rate:.6f}")
    print(f"run_grade_0_100={run_grade:.4f}")
    if overlap_count is not None:
        print(f"split_overlap_count={overlap_count}")
    if failed:
        for msg in failed:
            print(f"FAIL: {msg}")
        raise SystemExit(1)
    print("PASS: quality gate satisfied")


if __name__ == "__main__":
    main()
