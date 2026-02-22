from __future__ import annotations

from .types import IntegrityIssue, SampleRecord
from .yolo import load_yolo_box, validate_yolo_box


def run_integrity_checks(records: list[SampleRecord]) -> tuple[list[IntegrityIssue], dict]:
    issues: list[IntegrityIssue] = []

    for rec in records:
        if rec.image_path is None:
            issues.append(IntegrityIssue(rec.split, rec.stem, "missing_image", "Image file missing"))
        if rec.label_path is None:
            issues.append(IntegrityIssue(rec.split, rec.stem, "missing_label", "Label file missing"))

        if rec.label_path is not None:
            try:
                box = load_yolo_box(rec.label_path)
                yolo_issues = validate_yolo_box(box)
                for msg in yolo_issues:
                    issues.append(IntegrityIssue(rec.split, rec.stem, "invalid_label", msg))
            except Exception as exc:
                issues.append(IntegrityIssue(rec.split, rec.stem, "label_parse_error", str(exc)))

    summary = {
        "total_samples": len(records),
        "total_issues": len(issues),
        "issue_counts": {},
    }

    for issue in issues:
        summary["issue_counts"][issue.code] = summary["issue_counts"].get(issue.code, 0) + 1

    return issues, summary
