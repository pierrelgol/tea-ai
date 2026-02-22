from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .types import DetectionSummary, GeometrySummary, StabilitySummary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_model_report(
    reports_dir: Path,
    model_name: str,
    config: dict,
    detection: DetectionSummary,
    stability: StabilitySummary,
    geometry: GeometrySummary,
    per_sample_rows: list[dict],
) -> Path:
    payload = {
        "model_name": model_name,
        "config": config,
        "detection": asdict(detection),
        "stability": asdict(stability),
        "geometry": asdict(geometry),
        "samples": per_sample_rows,
    }
    path = reports_dir / f"model_{model_name}_report.json"
    _write_json(path, payload)
    return path


def write_summary(reports_dir: Path, rows: list[dict]) -> tuple[Path, Path]:
    summary_json = reports_dir / "summary.json"
    _write_json(summary_json, {"models": rows})

    lines = ["# Detector Evaluator Summary", ""]
    for row in rows:
        lines.append(
            "- {name}: P={p:.4f} R={r:.4f} miss={m:.4f} meanIoU={iou} AP={ap} drift={drift}".format(
                name=row["model_name"],
                p=row["precision"],
                r=row["recall"],
                m=row["miss_rate"],
                iou=row["mean_iou"],
                ap=row["ap_at_iou"],
                drift=row["mean_center_drift_px"],
            )
        )

    summary_md = reports_dir / "summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_json, summary_md
